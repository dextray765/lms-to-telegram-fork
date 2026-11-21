#!/usr/bin/env python3
"""Прослойка LM Studio <-> Telegram: стриминг, чаты, vision и выбор любой скачанной модели (запущена всегда одна)."""
import asyncio
import base64
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

load_dotenv()

# --- LOGGING ---
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").strip().upper(), logging.INFO),
    handlers=[
        RotatingFileHandler("bot_debug.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
# httpx на уровне INFO пишет в лог каждый запрос — это шум
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- CONFIG FROM .ENV ---
def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_num(name: str, default, cast=float):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return cast(default)
    try:
        return cast(raw.strip())
    except ValueError:
        logger.warning(f"⚠️ Некорректное значение {name}={raw!r}, используем {default}")
        return cast(default)


def parse_admin_ids(raw: str) -> List[int]:
    """'123, 456;789' -> [123, 456, 789] (без повторов, только положительные числа)."""
    ids: List[int] = []
    for part in re.split(r"[,\s;]+", raw.strip()):
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            logger.warning(f"⚠️ Некорректный ID администратора пропущен: {part!r}")
            continue
        if value > 0 and value not in ids:
            ids.append(value)
    return ids


CFG = {
    "TOKEN": os.getenv("TELEGRAM_TOKEN", "").strip(),
    # ADMIN_IDS — список через запятую; старый ADMIN_ID тоже поддерживается
    "ADMIN_IDS": parse_admin_ids(f"{os.getenv('ADMIN_IDS', '')},{os.getenv('ADMIN_ID', '')}"),
    "BASE_URL": os.getenv("LM_STUDIO_URL", "http://localhost:1234").strip().rstrip("/"),
    "API_KEY": os.getenv("LM_API_TOKEN", "lm-studio").strip() or "lm-studio",
    # числовые параметры ограничены разумными пределами (нулевой MAX_HISTORY или интервал, например, ломали бы работу)
    "TIMEOUT": max(10.0, env_num("TIMEOUT", 600, float)),            # ожидание данных от модели
    "LOAD_TIMEOUT": max(30.0, env_num("LOAD_TIMEOUT", 600, float)),  # ожидание загрузки модели
    "MAX_HISTORY": max(2, env_num("MAX_HISTORY", 40, int)),
    "STREAM_UPDATE_INTERVAL": max(0.5, env_num("STREAM_UPDATE_INTERVAL", 1.5, float)),  # лимиты Telegram на правки
    "TEMPERATURE": min(2.0, max(0.0, env_num("TEMPERATURE", 0.8, float))),
    "CONTEXT_LENGTH": max(0, env_num("CONTEXT_LENGTH", 0, int)),  # 0 = как настроено в LM Studio
    "ENABLE_VISION": env_bool("ENABLE_VISION", True),
    "ENABLE_STREAMING": env_bool("ENABLE_STREAMING", True),
}

# --- CONFIG FROM config.json (необязательно: только системный промпт) ---
FALLBACK_PROMPT = "You are a helpful assistant. Reply in the language of the user."


def load_system_prompt() -> str:
    if not os.path.exists("config.json"):
        return FALLBACK_PROMPT
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(cfg.get("system_prompt") or "").strip() or FALLBACK_PROMPT
    except Exception as e:
        logger.error(f"❌ Не удалось прочитать config.json: {e}")
        return FALLBACK_PROMPT


SYSTEM_PROMPT = load_system_prompt()

# Эндпоинты API
API_URL_NATIVE = f"{CFG['BASE_URL']}/api/v1"
API_URL_OAI = f"{CFG['BASE_URL']}/v1"
HEADERS = {
    "Authorization": f"Bearer {CFG['API_KEY']}",
    "Content-Type": "application/json",
}

STATE_FILE = "sessions.json"
CHUNK_LIMIT = 3500  # безопасный размер сообщения (лимит Telegram — 4096)
THINK_LIMIT = 500
DEFAULT_IMAGE_PROMPT = "Опиши это изображение."

MODEL_LOCK = asyncio.Lock()          # одна операция загрузки/выгрузки за раз
active_model: Optional[str] = None   # модель, выбранная кнопкой «Модели» (общая для всех администраторов)


class ModelError(Exception):
    """Ошибка работы с LM Studio, текст предназначен для показа пользователю."""


# --- CHAT STATE ---
MAX_CHAT_NAME = 32


def clean_chat_name(raw: str) -> str:
    return re.sub(r"\s+", " ", raw or "").strip()


class ChatSession:
    def __init__(self, name: str = "Main"):
        self.id = uuid.uuid4().hex[:8]  # стабильный короткий id для кнопок (имя может быть длинным и меняться)
        self.name = name
        self.messages: List[Dict[str, str]] = []
        self.show_thinking = False
        self.created: Optional[float] = time.time()
        self.updated: Optional[float] = self.created

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.updated = time.time()
        if len(self.messages) > CFG["MAX_HISTORY"]:
            self.messages = self.messages[-CFG["MAX_HISTORY"]:]
        # история должна начинаться с сообщения пользователя (требование многих chat-шаблонов)
        while self.messages and self.messages[0]["role"] != "user":
            self.messages.pop(0)

    def last_user_preview(self, limit: int = 40) -> str:
        for m in reversed(self.messages):
            if m["role"] == "user":
                text = re.sub(r"\s+", " ", str(m["content"])).strip()
                return text if len(text) <= limit else text[:limit].rstrip() + "…"
        return ""

    def get_context(self, new_input: str, image_b64: Optional[str] = None) -> List[Dict[str, Any]]:
        sys_prompt = f"{SYSTEM_PROMPT}\n[System Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}]"
        ctx: List[Dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        ctx.extend(self.messages)
        if image_b64:
            content: Any = [
                {"type": "text", "text": new_input},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        else:
            content = new_input
        ctx.append({"role": "user", "content": content})
        return ctx


class UserState:
    """Чаты одного администратора. Каждый чат — отдельный контекст со своим именем."""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {"Main": ChatSession("Main")}
        self.current = "Main"
        # ожидание текстового ввода (имя нового/переименуемого чата); в файл не сохраняется
        self.awaiting: Optional[Dict[str, Any]] = None

    def by_id(self, chat_id: str) -> Optional[ChatSession]:
        return next((c for c in self.sessions.values() if c.id == chat_id), None)

    def _most_recent(self) -> str:
        return max(self.sessions, key=lambda n: self.sessions[n].updated or 0)

    def session(self) -> ChatSession:
        if not self.sessions:
            self.sessions["Main"] = ChatSession("Main")
        if self.current not in self.sessions:
            self.current = self._most_recent()
        return self.sessions[self.current]

    def new_name(self) -> str:
        taken = {n.lower() for n in self.sessions}
        i = len(self.sessions) + 1
        while f"chat {i}" in taken:  # имя не должно совпасть с существующим (иначе чат затрётся)
            i += 1
        return f"Chat {i}"

    def validate_name(self, name: str, ignore: Optional[str] = None) -> Optional[str]:
        """Возвращает текст ошибки или None, если имя подходит."""
        if not name:
            return "Имя не может быть пустым."
        if len(name) > MAX_CHAT_NAME:
            return f"Имя слишком длинное ({len(name)} симв., максимум {MAX_CHAT_NAME})."
        if name.isdigit():
            return "Имя не может состоять только из цифр."
        if any(n.lower() == name.lower() and n != ignore for n in self.sessions):
            return f"Чат «{name}» уже существует."
        return None

    def delete(self, name: str):
        del self.sessions[name]
        if not self.sessions:
            self.sessions["Main"] = ChatSession("Main")
            self.current = "Main"
        elif self.current == name:
            self.current = self._most_recent()

    def rename(self, old: str, new: str):
        # пересобираем словарь, чтобы сохранить порядок (номера в списке чатов не меняются)
        self.sessions = {(new if k == old else k): v for k, v in self.sessions.items()}
        self.sessions[new].name = new
        if self.current == old:
            self.current = new


def ago(ts: Optional[float]) -> str:
    if not ts:
        return "—"
    d = int(time.time() - ts)
    if d < 60:
        return "только что"
    if d < 3600:
        return f"{d // 60} мин назад"
    if d < 86400:
        return f"{d // 3600} ч назад"
    return f"{d // 86400} дн назад"


users: Dict[int, UserState] = {}


def get_user(user_id: int) -> UserState:
    if user_id not in users:
        users[user_id] = UserState()
    return users[user_id]


# --- PERSISTENCE ---
def save_state():
    try:
        data = {
            "model": active_model,
            "users": {
                str(uid): {
                    "current": st.current,
                    "sessions": {
                        k: {
                            "id": s.id, "name": s.name, "msgs": s.messages, "thinking": s.show_thinking,
                            "created": s.created, "updated": s.updated,
                        }
                        for k, s in st.sessions.items()
                    },
                }
                for uid, st in users.items()
            },
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(tmp, 0o600)  # в файле личная переписка — закрываем от других пользователей системы
        except OSError:
            pass
        os.replace(tmp, STATE_FILE)  # атомарно: файл не повредится при падении посреди записи
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения состояния: {e}")


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _clean_messages(raw: Any) -> List[Dict[str, str]]:
    """Оставляет только корректные сообщения (защита от ручных правок и битых файлов)."""
    if not isinstance(raw, list):
        return []
    out = [
        {"role": m["role"], "content": m["content"]}
        for m in raw
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ][-CFG["MAX_HISTORY"]:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def _load_sessions(raw: Any) -> Dict[str, ChatSession]:
    result: Dict[str, ChatSession] = {}
    seen_ids: set = set()
    if not isinstance(raw, dict):
        return result
    for k, v in raw.items():
        if not isinstance(v, dict) or not str(k).strip():
            continue
        s = ChatSession(str(k))  # ключ — источник истины для имени (иначе «текущий чат» мог не находиться)
        chat_id = str(v.get("id") or "")
        if chat_id and chat_id not in seen_ids:
            s.id = chat_id
        seen_ids.add(s.id)
        s.messages = _clean_messages(v.get("msgs"))
        s.show_thinking = bool(v.get("thinking", False))
        # у чатов из старых версий нет меток времени — показываем «—»
        s.created = _num(v.get("created"))
        s.updated = _num(v.get("updated"))
        result[str(k)] = s
    return result


def load_state():
    global active_model
    users.clear()
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError("корневой элемент не объект")
            model = d.get("model")
            active_model = model if isinstance(model, str) and model else None
            if "users" in d:
                for uid, ud in (d["users"] if isinstance(d["users"], dict) else {}).items():
                    try:
                        st = UserState()
                        loaded = _load_sessions(ud.get("sessions"))
                        if loaded:
                            st.sessions = loaded
                        st.current = ud.get("current") if isinstance(ud.get("current"), str) else "Main"
                        users[int(uid)] = st
                    except (ValueError, AttributeError, TypeError) as e:  # один битый пользователь не рушит остальных
                        logger.warning(f"⚠️ Пропущены данные пользователя {uid!r}: {e}")
            elif CFG["ADMIN_IDS"]:
                # старый формат (один администратор): переносим чаты первому администратору
                current = d.pop("current", "Main")
                loaded = _load_sessions(d)
                if loaded:
                    st = UserState()
                    st.sessions, st.current = loaded, current if isinstance(current, str) else "Main"
                    users[CFG["ADMIN_IDS"][0]] = st
            for st in users.values():
                st.session()  # нормализует текущий чат
            logger.info(f"📂 Состояние загружено: пользователей {len(users)}, модель: {active_model or '—'}")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки состояния: {e}")
        try:
            os.replace(STATE_FILE, STATE_FILE + ".bak")  # не затираем повреждённый файл
        except OSError:
            pass
        users.clear()


# --- LM STUDIO MODEL CATALOG ---
def model_key(m: Dict[str, Any]) -> str:
    return str(m.get("key") or m.get("id") or m.get("path") or "")


def is_llm(m: Dict[str, Any]) -> bool:
    return (m.get("type") or "llm") == "llm"


def loaded_instances(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    return m.get("loaded_instances") or []


def instance_id(inst: Dict[str, Any]) -> str:
    return str(inst.get("id") or inst.get("instance_id") or "")


def has_vision(m: Dict[str, Any]) -> bool:
    return bool((m.get("capabilities") or {}).get("vision"))


def sorted_llms(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((m for m in models if is_llm(m)), key=lambda m: model_key(m).lower())


def match_models(models: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    """Точное совпадение ключа/названия или все частичные совпадения (загруженные — первыми)."""
    n = (name or "").strip().lower()
    if not n:
        return []
    llms = sorted_llms(models)
    for m in llms:
        if model_key(m).lower() == n:
            return [m]
    for m in llms:
        if str(m.get("display_name") or "").lower() == n:
            return [m]
    partial = [
        m for m in llms
        if n in model_key(m).lower() or n in str(m.get("display_name") or "").lower()
    ]
    partial.sort(key=lambda m: not loaded_instances(m))
    return partial


def find_model(models: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    found = match_models(models, name)
    return found[0] if found else None


def _json_list(resp, key: str) -> List[Dict[str, Any]]:
    """Достаёт список словарей из JSON-ответа LM Studio."""
    try:
        data = resp.json()
    except ValueError as e:
        raise ModelError("LM Studio вернул некорректный ответ (не JSON).") from e
    items = data.get(key) if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ModelError("LM Studio вернул неожиданный формат списка моделей.")
    return [m for m in items if isinstance(m, dict)]


async def fetch_models(client: Optional[httpx.AsyncClient] = None) -> List[Dict[str, Any]]:
    """Список скачанных моделей из LM Studio (Native API, при его отсутствии — OpenAI-совместимый)."""
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=15.0)
    try:
        try:
            r = await client.get(f"{API_URL_NATIVE}/models", headers=HEADERS, timeout=15.0)
            if r.status_code == 200:
                return _json_list(r, "models")
            logger.warning(f"⚠️ Native API вернул {r.status_code}, пробуем /v1/models")
            r = await client.get(f"{API_URL_OAI}/models", headers=HEADERS, timeout=15.0)
            if r.status_code != 200:
                raise ModelError(f"LM Studio ответил {r.status_code}. Проверьте LM_API_TOKEN.")
            return [
                {"key": m["id"], "type": "llm", "loaded_instances": [], "_openai_only": True}
                for m in _json_list(r, "data") if m.get("id")
            ]
        except httpx.HTTPError as e:
            raise ModelError(
                f"LM Studio недоступен ({type(e).__name__}). Проверьте, что сервер запущен: {CFG['BASE_URL']}"
            ) from e
    finally:
        if own:
            await client.aclose()


async def ensure_model_loaded(target: str) -> str:
    """
    Запускает модель так, чтобы в LM Studio была загружена только она (остальные LLM выгружаются).
    Возвращает идентификатор для chat/completions. Бросает ModelError с понятным текстом.
    """
    async with MODEL_LOCK:
        async with httpx.AsyncClient(timeout=CFG["LOAD_TIMEOUT"]) as client:
            models = await fetch_models(client)
            info = find_model(models, target)
            if info is None:
                raise ModelError(f"Модель «{target}» не найдена в LM Studio. Выберите другую кнопкой «{MENU_MODELS}».")
            key = model_key(info)
            if info.get("_openai_only"):
                return key  # Native API недоступен — LM Studio сам загрузит модель (JIT)

            # Выгружаем всё лишнее: другие LLM и дубли экземпляров целевой (embedding-модели не трогаем)
            keep = loaded_instances(info)[:1]
            keep_ids = {instance_id(i) for i in keep}
            for m in models:
                if not is_llm(m):
                    continue
                for inst in loaded_instances(m):
                    iid = instance_id(inst)
                    if not iid or iid in keep_ids:
                        continue
                    logger.info(f"📤 Выгружаем '{iid}'")
                    try:
                        resp = await client.post(
                            f"{API_URL_NATIVE}/models/unload",
                            json={"instance_id": iid}, headers=HEADERS, timeout=30.0,
                        )
                        if resp.status_code != 200:
                            logger.warning(f"⚠️ Не удалось выгрузить {iid}: {resp.status_code} {resp.text[:150]}")
                    except httpx.HTTPError as e:
                        logger.warning(f"⚠️ Ошибка выгрузки {iid}: {e}")

            if keep:
                logger.info(f"✅ Модель '{key}' уже загружена")
                return instance_id(keep[0]) or key

            logger.info(f"📥 Загружаем модель: {key}")
            payload: Dict[str, Any] = {"model": key, "flash_attention": True}
            if CFG["CONTEXT_LENGTH"] > 0:
                payload["context_length"] = CFG["CONTEXT_LENGTH"]
            try:
                res = await client.post(
                    f"{API_URL_NATIVE}/models/load", json=payload, headers=HEADERS,
                    timeout=CFG["LOAD_TIMEOUT"],
                )
                if res.status_code in (400, 422) and "flash_attention" in payload:
                    # не все модели/движки поддерживают flash attention — пробуем без него
                    # (при 5xx, например нехватке памяти, повторная тяжёлая загрузка не нужна)
                    logger.warning(f"⚠️ Загрузка с flash_attention не удалась ({res.status_code}), повторяем без него")
                    payload.pop("flash_attention")
                    res = await client.post(
                        f"{API_URL_NATIVE}/models/load", json=payload, headers=HEADERS,
                        timeout=CFG["LOAD_TIMEOUT"],
                    )
            except httpx.HTTPError as e:
                raise ModelError(
                    f"Не удалось загрузить модель ({type(e).__name__}). Возможно, не хватило времени: LOAD_TIMEOUT."
                ) from e
            if res.status_code != 200:
                raise ModelError(f"LM Studio не смог загрузить «{key}»: {res.status_code} {res.text[:200]}")
            # Запрос синхронный: ответ приходит, когда модель уже загружена
            try:
                iid = res.json().get("instance_id") or key
            except (ValueError, AttributeError):
                iid = key
            logger.info(f"✅ Модель загружена: {iid}")
            return iid


async def resolve_target() -> str:
    """Модель для ответа: выбранная кнопкой «Модели», иначе любая уже запущенная в LM Studio."""
    if active_model:
        return active_model
    for m in await fetch_models():
        if is_llm(m) and loaded_instances(m):
            return model_key(m)
    raise ModelError(f"Модель не выбрана. Нажмите «{MENU_MODELS}» и выберите модель.")


async def check_vision(target: str):
    """Проверяет, что модель умеет работать с изображениями (если LM Studio сообщает такие данные)."""
    info = find_model(await fetch_models(), target)
    if info and not info.get("_openai_only") and not has_vision(info):
        raise ModelError(
            f"Модель «{model_key(info)}» не умеет работать с изображениями. "
            f"Выберите модель с пометкой 👁 в разделе «{MENU_MODELS}»."
        )


# --- TEXT PROCESSING ---
# Форматы «размышлений», которые модели оставляют прямо в тексте ответа: (открывающий тег, закрывающий тег).
# Если LM Studio сам выделяет размышления в reasoning_content (настройка «Reasoning Section Parsing»), до этих
# регулярных выражений дело не доходит; они нужны для моделей, у которых разбор не настроен.
_THINK_FORMATS = (
    # Qwen3, DeepSeek-R1, GLM и большинство остальных
    (re.compile(r"<think>", re.I), re.compile(r"</think>", re.I)),
    # Gemma 4: <|channel>thought\n ...размышление... <channel|>ответ. Метка «thought» — служебная, не часть текста
    (re.compile(r"<\|channel>(?:thought)?[ \t]*\r?\n?"), re.compile(r"<channel\|>")),
)
# Начала тегов для функции strip_partial_tag (чтобы недописанный тег не мелькал при стриминге)
_PARTIAL_TAGS = ("</think>", "<think>", "<channel|>", "<|channel>")
_THOUGHT_LABEL_RE = re.compile(r"^\s*thought[ \t]*\r?\n")


def _extract_blocks(content: str, open_re, close_re) -> Tuple[str, List[str]]:
    """Вырезает из текста блоки размышлений одного формата. Возвращает (текст без блоков, список размышлений)."""
    thoughts: List[str] = []
    parts: List[str] = []
    pos = 0
    while True:
        m = open_re.search(content, pos)
        if not m:
            break
        parts.append(content[pos:m.start()])
        c = close_re.search(content, m.end())
        if not c:  # размышление ещё не закончено (идёт стриминг или ответ оборвался)
            thoughts.append(content[m.end():])
            pos = len(content)
            break
        thoughts.append(content[m.end():c.start()])
        pos = c.end()
    parts.append(content[pos:])
    rest = "".join(parts)
    # Открывающий тег мог быть добавлен chat-шаблоном, тогда в ответе только закрывающий: всё до него — размышление
    c = close_re.search(rest)
    while c:
        thoughts.insert(0, rest[:c.start()])
        rest = rest[c.end():]
        c = close_re.search(rest)
    return rest, [t.strip() for t in thoughts if t.strip()]


def split_thinking(content: str, reasoning: str = "") -> Tuple[str, str]:
    """Отделяет «мысли» модели от ответа. Возвращает (ответ, мысли)."""
    thoughts: List[str] = []
    if reasoning and reasoning.strip():
        # если разбор в LM Studio настроен на «<|channel>» без «thought», метка остаётся в начале размышления
        thoughts.append(_THOUGHT_LABEL_RE.sub("", reasoning, count=1).strip())
    for open_re, close_re in _THINK_FORMATS:
        content, found = _extract_blocks(content, open_re, close_re)
        thoughts += found
    return content.strip(), "\n\n".join(t for t in thoughts if t)


def strip_partial_tag(text: str) -> str:
    """Во время стриминга прячет недописанный тег (`<thi`, `<|chan`) в конце текста, чтобы он не мелькал."""
    for tag in _PARTIAL_TAGS:
        for k in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:k]):
                return text[:-k]
    return text


def split_text(text: str, limit: int = CHUNK_LIMIT) -> List[str]:
    """Режет длинный текст на куски для Telegram, по возможности по абзацам; корректно закрывает ``` ."""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    fixed: List[str] = []
    open_fence = False
    for c in chunks:
        if open_fence:
            c = "```\n" + c
        if c.count("```") % 2 == 1:
            c += "\n```"
            open_fence = True
        else:
            open_fence = False
        fixed.append(c)
    return fixed


def md_to_html(text: str) -> str:
    """Простая конвертация Markdown (как его пишут LLM) в HTML-подмножество Telegram."""
    stash: List[str] = []

    def put(s: str) -> str:
        stash.append(s)
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(
        r"```[ \t]*[\w+#.-]*[ \t]*\n?(.*?)```",
        lambda m: put(f"<pre>{html.escape(m.group(1).strip(chr(10)))}</pre>"),
        text, flags=re.S,
    )
    text = re.sub(r"`([^`\n]+)`", lambda m: put(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = html.escape(text, quote=False)
    # [ \t] вместо \s: иначе регулярка «съедала» перевод строки после заголовка/маркера списка
    text = re.sub(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", r"<b>\1</b>", text, flags=re.M)
    text = re.sub(r"^([ \t]*)[*-][ \t]+", r"\1• ", text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.S)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"<b>\1</b>", text, flags=re.S)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)",
        lambda m: f'<a href="{m.group(2).replace(chr(34), "%22")}">{m.group(1)}</a>',
        text,
    )
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


# --- TELEGRAM SEND HELPERS ---
def _retry_seconds(e: RetryAfter) -> float:
    ra = e.retry_after
    return float(ra.total_seconds() if hasattr(ra, "total_seconds") else ra)


async def safe_edit(msg, text: str, reply_markup=None) -> bool:
    """Редактирование простым текстом: не падает на 'not modified' и FloodWait."""
    for _ in range(3):
        try:
            await msg.edit_text(text[:4096], reply_markup=reply_markup)
            return True
        except RetryAfter as e:
            await asyncio.sleep(min(_retry_seconds(e), 30) + 0.5)
        except BadRequest as e:
            if "not modified" in str(e).lower():
                return True
            logger.warning(f"⚠️ Ошибка редактирования: {e}")
            return False
        except TelegramError as e:
            logger.warning(f"⚠️ Telegram: {e}")
            return False
    return False


async def _tg_send(update: Update, target, text: str, parse_mode):
    for _ in range(3):
        try:
            if target is not None:
                await target.edit_text(text, parse_mode=parse_mode)
            else:
                await update.effective_message.reply_text(text, parse_mode=parse_mode)
            return
        except RetryAfter as e:
            await asyncio.sleep(min(_retry_seconds(e), 30) + 0.5)
        except BadRequest as e:
            if "not modified" in str(e).lower():
                return
            raise
    raise TelegramError("Flood control: попытки исчерпаны")


async def send_or_edit(update: Update, target, html_text: str, plain_text: str):
    """
    Доставка куска ответа. Порядок попыток: HTML → простой текст → (если редактировать нечего, например статусное
    сообщение удалили) новое сообщение. Так ответ не теряется ни при ошибке разметки, ни при удалённом сообщении.
    """
    if target is not None:
        attempts = [(target, html_text, ParseMode.HTML), (target, plain_text, None), (None, plain_text, None)]
    else:
        attempts = [(None, html_text, ParseMode.HTML), (None, plain_text, None)]
    for tgt, text, mode in attempts:
        try:
            await _tg_send(update, tgt, text, mode)
            return
        except BadRequest as e:
            logger.info(f"ℹ️ Отправка не удалась ({e}), пробуем запасной вариант")
        except TelegramError as e:
            logger.error(f"❌ Не удалось отправить ответ: {e}")
            return
    logger.error("❌ Не удалось доставить ответ ни одним способом")


async def send_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int, action: str = ChatAction.TYPING):
    """Индикатор «печатает»; сбой сети не должен срывать обработку сообщения."""
    try:
        await context.bot.send_chat_action(chat_id, action)
    except TelegramError as e:
        logger.debug(f"ℹ️ Не удалось отправить chat action: {e}")


async def deliver(update: Update, status_msg, visible: str, thoughts: str, show_thinking: bool):
    """Финальная доставка: разбивка на сообщения < 4096 символов + форматирование."""
    body = visible or "⚠️ Модель вернула пустой ответ (возможно, закончился лимит токенов или контекста)."
    prefix_html = prefix_plain = ""
    if show_thinking and thoughts:
        t = thoughts if len(thoughts) <= THINK_LIMIT else thoughts[:THINK_LIMIT].rstrip() + "…"
        prefix_html = f"<blockquote expandable>🧠 Мысли:\n{html.escape(t)}</blockquote>\n"
        prefix_plain = f"🧠 Мысли:\n{t}\n\n"
    for i, chunk in enumerate(split_text(body)):
        h = (prefix_html if i == 0 else "") + md_to_html(chunk)
        p = (prefix_plain if i == 0 else "") + chunk
        await send_or_edit(update, status_msg if i == 0 else None, h, p)


# --- GENERATION ---
def api_error(status: int, body: str) -> ModelError:
    text = f"API Error {status}: {body[:300]}"
    low = body.lower()
    if status == 400 and ("context" in low or "n_ctx" in low):
        text += "\nПохоже, переполнен контекст модели: очистите историю чата (⚙️ Настройки → Очистить) или увеличьте контекст в LM Studio."
    return ModelError(text)


async def request_completion(payload: Dict[str, Any], stream: bool, on_update=None) -> Tuple[str, str]:
    """Запрос chat/completions. Возвращает (content, reasoning)."""
    url = f"{API_URL_OAI}/chat/completions"
    timeout = httpx.Timeout(CFG["TIMEOUT"], connect=15.0)
    content, reasoning = "", ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        if not stream:
            r = await client.post(url, headers=HEADERS, json=payload)
            if r.status_code != 200:
                raise api_error(r.status_code, r.text)
            try:
                msg = r.json()["choices"][0]["message"]
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise ModelError("LM Studio вернул некорректный ответ модели.") from e
            return msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or ""

        async with client.stream("POST", url, headers=HEADERS, json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", "replace")
                raise api_error(response.status_code, body)
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    err = chunk["error"]
                    raise ModelError(f"Ошибка модели: {err.get('message', err) if isinstance(err, dict) else err}")
                choices = chunk.get("choices") or []  # последний чанк с usage приходит без choices
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                r_piece = delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    content += piece
                if r_piece:
                    reasoning += r_piece
                if (piece or r_piece) and on_update:
                    await on_update(content, reasoning)
    return content, reasoning


async def generate(update: Update, text: str, img_b64: Optional[str] = None, status_msg=None) -> str:
    """Единый конвейер генерации (со стримингом и без)."""
    msg = update.effective_message
    sess = get_user(update.effective_user.id).session()
    stream = CFG["ENABLE_STREAMING"]
    if status_msg is None:
        status_msg = await msg.reply_text("⏳ Подготовка...")
    started = time.monotonic()

    try:
        target = await resolve_target()
        if img_b64:
            await check_vision(target)

        await safe_edit(status_msg, "⏳ Подготовка модели...")
        model_id = await ensure_model_loaded(target)

        payload = {
            "model": model_id,
            "messages": sess.get_context(text, img_b64),
            "temperature": CFG["TEMPERATURE"],
            "stream": stream,
        }
        await safe_edit(status_msg, "⏳ Генерация...")

        last_edit = 0.0

        async def on_update(content: str, reasoning: str):
            nonlocal last_edit
            now = time.monotonic()
            if now - last_edit < CFG["STREAM_UPDATE_INTERVAL"]:
                return
            last_edit = now
            visible, thoughts = split_thinking(content, reasoning)
            visible = strip_partial_tag(visible).rstrip()
            if not visible and not thoughts:
                return
            preview = visible
            if thoughts and (sess.show_thinking or not visible):
                head = f"🧠 {thoughts[-200:]}" if sess.show_thinking else "🧠 Думаю..."
                preview = f"{head}\n\n{visible}" if visible else head
            await safe_edit(status_msg, preview[-3900:])

        content, reasoning = await request_completion(payload, stream, on_update if stream else None)
        visible, thoughts = split_thinking(content, reasoning)

        await deliver(update, status_msg, visible, thoughts, sess.show_thinking)
        logger.info(f"✅ Генерация завершена за {time.monotonic() - started:.1f} сек ({model_id})")

        if visible:
            # картинку в историю не кладём (она огромная) — оставляем пометку
            sess.add("user", f"[изображение] {text}" if img_b64 else text)
            sess.add("assistant", visible)
            save_state()
        return visible

    except ModelError as e:
        logger.error(f"❌ {e}")
        await safe_edit(status_msg, f"❌ {e}")
    except httpx.HTTPError as e:
        logger.error(f"❌ Сетевая ошибка: {e!r}")
        await safe_edit(status_msg, f"❌ Ошибка соединения с LM Studio ({type(e).__name__}). Проверьте, что сервер запущен.")
    except Exception as e:
        logger.exception("❌ Ошибка генерации")
        await safe_edit(status_msg, f"❌ Ошибка генерации: {str(e)[:200] or type(e).__name__}")
    return ""


# --- UI: КНОПКИ ---
MENU_MODELS, MENU_CHATS, MENU_SETTINGS = "🤖 Модели", "💬 Чаты", "⚙️ Настройки"
MENU_BUTTONS = {MENU_MODELS, MENU_CHATS, MENU_SETTINGS}
MODELS_PER_PAGE = 6
CHATS_PER_PAGE = 6
AWAIT_TTL = 300  # секунд: сколько бот ждёт имя чата после нажатия кнопки


def is_admin(update: Update) -> bool:
    user = update.effective_user
    return bool(user) and user.id in CFG["ADMIN_IDS"]


def btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text, callback_data=data)


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def short_hash(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def page_count(total: int, per_page: int) -> int:
    return max(1, (total + per_page - 1) // per_page)


def nav_row(prefix: str, page: int, pages: int) -> List[InlineKeyboardButton]:
    row = []
    if page > 0:
        row.append(btn("◀️", f"{prefix}:{page - 1}"))
    row.append(btn(f"{page + 1}/{pages}", "noop"))
    if page < pages - 1:
        row.append(btn("▶️", f"{prefix}:{page + 1}"))
    return row


def main_keyboard() -> ReplyKeyboardMarkup:
    """Постоянная клавиатура внизу экрана — главное меню."""
    return ReplyKeyboardMarkup(
        [[MENU_MODELS, MENU_CHATS, MENU_SETTINGS]],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Сообщение для модели…",
    )


def with_notice(notice: str, text: str) -> str:
    return f"{notice}\n\n{text}" if notice else text


# --- модели ---
async def render_models(page: int = 0, notice: str = "") -> Tuple[str, InlineKeyboardMarkup]:
    models = sorted_llms(await fetch_models())
    if not models:
        text = "В LM Studio не найдено ни одной LLM-модели. Скачайте модель и нажмите «Обновить»."
        return with_notice(notice, text), InlineKeyboardMarkup([[btn("🔄 Обновить", "m:0"), btn("✖ Закрыть", "x")]])
    sel = find_model(models, active_model) if active_model else None
    pages = page_count(len(models), MODELS_PER_PAGE)
    page = max(0, min(page, pages - 1))
    rows = []
    for m in models[page * MODELS_PER_PAGE:(page + 1) * MODELS_PER_PAGE]:
        key = model_key(m)
        mark = "🟢" if loaded_instances(m) else ("⭐" if m is sel else "⚪")
        rows.append([btn(clip(f"{mark} {'👁 ' if has_vision(m) else ''}{key}", 60), f"ms:{short_hash(key)}")])
    if pages > 1:
        rows.append(nav_row("m", page, pages))
    rows.append([btn("🔄 Обновить", f"m:{page}"), btn("✖ Закрыть", "x")])
    head = f"🤖 Модели LM Studio ({len(models)})" + (f", стр. {page + 1}/{pages}" if pages > 1 else "")
    text = (
        f"{head}\n🟢 запущена · ⭐ выбрана, но не запущена · ⚪ не запущена · 👁 видит изображения\n\n"
        f"Текущая: {active_model or 'не выбрана'}\n"
        "Нажмите на модель — она запустится, остальные будут выгружены."
    )
    return with_notice(notice, text), InlineKeyboardMarkup(rows)


async def launch_model(msg, key_hash: str) -> Tuple[str, InlineKeyboardMarkup]:
    """Запускает модель по хэшу ключа; возвращает панель моделей на странице с выбранной моделью."""
    global active_model
    models = sorted_llms(await fetch_models())
    chosen = next((m for m in models if short_hash(model_key(m)) == key_hash), None)
    if chosen is None:
        return await render_models(0, "⚠️ Список устарел, он обновлён.")
    page = models.index(chosen) // MODELS_PER_PAGE
    key = model_key(chosen)
    if active_model and find_model(models, active_model) is chosen and loaded_instances(chosen):
        return await render_models(page, f"ℹ️ Модель {key} уже запущена.")
    await safe_edit(msg, f"📥 Запускаю {key}...\nОстальные модели будут выгружены. Это может занять время.")
    try:
        await ensure_model_loaded(key)
    except ModelError as e:
        logger.error(f"❌ {e}")
        note = f"\nПрежняя модель ({active_model}) будет запущена заново при следующем запросе." if active_model else ""
        return await render_models(page, f"❌ {e}{note}")
    except httpx.HTTPError as e:
        return await render_models(page, f"❌ Ошибка соединения с LM Studio ({type(e).__name__}).")
    active_model = key
    save_state()
    return await render_models(page, f"✅ Запущена: {key}")


# --- чаты ---
def render_chats(st: UserState, page: Optional[int] = None, notice: str = "") -> Tuple[str, InlineKeyboardMarkup]:
    st.session()  # гарантирует валидный текущий чат
    items = list(st.sessions.values())
    pages = page_count(len(items), CHATS_PER_PAGE)
    if page is None:  # по умолчанию — страница с текущим чатом
        page = next((i for i, c in enumerate(items) if c.name == st.current), 0) // CHATS_PER_PAGE
    page = max(0, min(page, pages - 1))
    first = page * CHATS_PER_PAGE
    lines = [f"💬 Чаты ({len(items)}). ✅ — текущий\n"]
    rows = []
    for i, c in enumerate(items[first:first + CHATS_PER_PAGE], first + 1):
        mark = "✅ " if c.name == st.current else ""
        lines.append(f"{i}. {mark}{c.name} — {len(c.messages)} сообщ., {ago(c.updated)}")
        preview = c.last_user_preview()
        if preview:
            lines.append(f"    «{preview}»")
        rows.append([btn(clip(f"{mark}{c.name}", 28), f"cs:{c.id}"), btn("⚙️", f"ci:{c.id}")])
    if pages > 1:
        rows.append(nav_row("c", page, pages))
    rows.append([btn("➕ Новый чат", "cn"), btn("✖ Закрыть", "x")])
    lines.append("\nНажмите на чат — переключиться. ⚙️ — переименовать, очистить, удалить.")
    return with_notice(notice, "\n".join(lines)), InlineKeyboardMarkup(rows)


def render_chat_card(st: UserState, c: ChatSession, notice: str = "") -> Tuple[str, InlineKeyboardMarkup]:
    is_current = c.name == st.current
    lines = [
        f"💬 Чат «{c.name}»{' (текущий)' if is_current else ''}",
        f"Сообщений в истории: {len(c.messages)}",
        f"Последняя активность: {ago(c.updated)}",
        f"Создан: {ago(c.created)}",
    ]
    preview = c.last_user_preview(80)
    if preview:
        lines.append(f"Последний вопрос: «{preview}»")
    rows = []
    if not is_current:
        rows.append([btn("✅ Сделать текущим", f"cs:{c.id}")])
    rows.append([btn("✏️ Переименовать", f"cr:{c.id}"), btn("🧹 Очистить", f"ch:{c.id}")])
    rows.append([btn("🗑 Удалить", f"cd:{c.id}")])
    rows.append([btn("◀️ К чатам", "c:")])
    return with_notice(notice, "\n".join(lines)), InlineKeyboardMarkup(rows)


def render_confirm(kind: str, c: ChatSession) -> Tuple[str, InlineKeyboardMarkup]:
    if kind == "delete":
        text = f"⚠️ Удалить чат «{c.name}» ({len(c.messages)} сообщ.)? Это необратимо."
        yes = btn("🗑 Да, удалить", f"cdy:{c.id}")
    else:
        text = f"⚠️ Очистить историю чата «{c.name}» ({len(c.messages)} сообщ.)? Сам чат останется."
        yes = btn("🧹 Да, очистить", f"chy:{c.id}")
    return text, InlineKeyboardMarkup([[yes, btn("◀️ Нет", f"ci:{c.id}")]])


def render_settings(st: UserState, notice: str = "") -> Tuple[str, InlineKeyboardMarkup]:
    c = st.session()
    text = (
        "⚙️ Настройки\n\n"
        f"💬 Чат: {c.name} ({len(c.messages)} сообщ.)\n"
        f"🤖 Модель: {active_model or 'не выбрана'}\n"
        f"💭 Мысли модели: {'показывать' if c.show_thinking else 'скрывать'} (для этого чата)\n"
        f"⚡ Стриминг: {'вкл' if CFG['ENABLE_STREAMING'] else 'выкл'} · 👁 Фото: {'вкл' if CFG['ENABLE_VISION'] else 'выкл'}"
    )
    rows = [
        [btn(f"💭 Мысли: {'ВКЛ' if c.show_thinking else 'ВЫКЛ'}", "st")],
        [btn("🧹 Очистить историю чата", f"ch:{c.id}")],
        [btn("✖ Закрыть", "x")],
    ]
    return with_notice(notice, text), InlineKeyboardMarkup(rows)


def create_chat(st: UserState, name: str) -> ChatSession:
    c = ChatSession(name)
    st.sessions[name] = c
    st.current = name
    return c


# --- обработчики ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.effective_message.reply_text("🚫 Доступ запрещён")
        return
    st = get_user(update.effective_user.id)
    st.awaiting = None
    c = st.session()
    await update.effective_message.reply_text(
        "✨ Bridge Online\n\n"
        f"🤖 Модель: {active_model or 'не выбрана'}\n"
        f"💬 Чат: {c.name} ({len(c.messages)} сообщ.)\n\n"
        "Всё управление — кнопками внизу: модели, чаты, настройки.\n"
        "Просто напишите сообщение (или отправьте фото) — модель ответит в текущем чате.",
        reply_markup=main_keyboard(),
    )


async def open_panel(update: Update, st: UserState, which: str):
    """Открывает панель по нажатию кнопки нижней клавиатуры."""
    msg = update.effective_message
    try:
        if which == MENU_MODELS:
            text, markup = await render_models()
        elif which == MENU_CHATS:
            text, markup = render_chats(st)
        else:
            text, markup = render_settings(st)
    except ModelError as e:
        await msg.reply_text(f"❌ {e}")
        return
    await msg.reply_text(text, reply_markup=markup)


async def handle_awaiting(update: Update, context: ContextTypes.DEFAULT_TYPE, st: UserState, text: str):
    """Текст пользователя — это имя нового чата или новое имя существующего."""
    aw = st.awaiting
    name = clean_chat_name(text)
    notice = ""
    if aw["kind"] == "new":
        error = st.validate_name(name)
        if error:
            await update.effective_message.reply_text(f"❌ {error}\nОтправьте другое имя или нажмите «✖ Отмена» в сообщении выше.")
            return
        create_chat(st, name)
        notice = f"✅ Создан чат «{name}» с чистым контекстом и выбран текущим."
    else:
        c = st.by_id(aw["chat_id"])
        if c is None:
            notice = "⚠️ Чат уже удалён."
        else:
            error = st.validate_name(name, ignore=c.name)
            if error:
                await update.effective_message.reply_text(f"❌ {error}\nОтправьте другое имя или нажмите «✖ Отмена» в сообщении выше.")
                return
            old = c.name
            st.rename(old, name)
            notice = f"✅ Чат «{old}» переименован в «{name}»."
    st.awaiting = None
    save_state()
    text_out, markup = render_chats(st, None, notice)
    try:  # обновляем исходную панель, а не засоряем переписку
        await context.bot.edit_message_text(
            chat_id=aw["tg_chat"], message_id=aw["msg_id"], text=text_out, reply_markup=markup
        )
    except TelegramError:
        await update.effective_message.reply_text(text_out, reply_markup=markup)


async def cb_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not is_admin(update):
        await query.answer("🚫 Доступ запрещён", show_alert=True)
        return
    msg = query.message
    if msg is None or not hasattr(msg, "edit_text"):  # слишком старое («недоступное») сообщение
        await query.answer("Панель устарела. Откройте её заново кнопками внизу.", show_alert=True)
        return
    await query.answer()
    action, _, arg = (query.data or "").partition(":")
    if action == "noop":
        return
    st = get_user(update.effective_user.id)
    if action not in ("cn", "cr"):
        st.awaiting = None  # любое другое нажатие отменяет ожидание имени

    async def show(pair: Tuple[str, InlineKeyboardMarkup]):
        await safe_edit(msg, pair[0], pair[1])

    def page_arg() -> Optional[int]:
        return int(arg) if arg.isdigit() else None

    chat = st.by_id(arg) if action in ("cs", "ci", "cr", "cd", "cdy", "ch", "chy") else None
    if action in ("cs", "ci", "cr", "cd", "cdy", "ch", "chy") and chat is None:
        await show(render_chats(st, None, "⚠️ Чат не найден (возможно, уже удалён)."))
        return

    try:
        if action == "x":
            try:
                await msg.delete()
            except TelegramError:
                await safe_edit(msg, "Закрыто.")
        elif action == "m":
            await show(await render_models(page_arg() or 0))
        elif action == "ms":
            await show(await launch_model(msg, arg))
        elif action == "c":
            await show(render_chats(st, page_arg()))
        elif action == "cs":
            st.current = chat.name
            save_state()
            await show(render_chats(st, None, f"✅ Текущий чат: «{chat.name}»"))
        elif action == "ci":
            await show(render_chat_card(st, chat))
        elif action == "cn":
            st.awaiting = {"kind": "new", "chat_id": None, "tg_chat": msg.chat_id, "msg_id": msg.message_id, "ts": time.time()}
            await safe_edit(
                msg, "➕ Новый чат\n\nОтправьте сообщением имя нового чата (до 32 символов).",
                InlineKeyboardMarkup([[btn("Без имени", "cn0"), btn("✖ Отмена", "c:")]]),
            )
        elif action == "cn0":
            c = create_chat(st, st.new_name())
            save_state()
            await show(render_chats(st, None, f"✅ Создан чат «{c.name}» с чистым контекстом и выбран текущим."))
        elif action == "cr":
            st.awaiting = {"kind": "rename", "chat_id": chat.id, "tg_chat": msg.chat_id, "msg_id": msg.message_id, "ts": time.time()}
            await safe_edit(
                msg, f"✏️ Переименование чата «{chat.name}»\n\nОтправьте сообщением новое имя (до 32 символов).",
                InlineKeyboardMarkup([[btn("✖ Отмена", f"ci:{chat.id}")]]),
            )
        elif action == "cd":
            await show(render_confirm("delete", chat))
        elif action == "cdy":
            name = chat.name
            st.delete(name)
            save_state()
            await show(render_chats(st, None, f"🗑 Чат «{name}» удалён."))
        elif action == "ch":
            await show(render_confirm("clear", chat))
        elif action == "chy":
            chat.messages = []
            save_state()
            await show(render_chat_card(st, chat, "🧹 История очищена."))
        elif action == "s":
            await show(render_settings(st))
        elif action == "st":
            c = st.session()
            c.show_thinking = not c.show_thinking
            save_state()
            await show(render_settings(st))
    except ModelError as e:
        await safe_edit(msg, f"❌ {e}")
    except httpx.HTTPError as e:
        await safe_edit(msg, f"❌ Ошибка соединения с LM Studio ({type(e).__name__}).")
    except Exception:
        logger.exception("❌ Ошибка в обработчике кнопки")
        await safe_edit(msg, "❌ Внутренняя ошибка, подробности в bot_debug.log.")


async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    text = (update.effective_message.text or "").strip()
    if not text:
        return
    st = get_user(update.effective_user.id)
    if st.awaiting and time.time() - st.awaiting["ts"] > AWAIT_TTL:
        st.awaiting = None  # ввод имени «протух» — дальше это обычное сообщение
    if text in MENU_BUTTONS:
        st.awaiting = None
        await open_panel(update, st, text)
        return
    if st.awaiting:
        await handle_awaiting(update, context, st, text)
        return
    await send_typing(context, update.effective_chat.id)
    await generate(update, text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        return
    msg = update.effective_message
    if not CFG["ENABLE_VISION"]:
        await msg.reply_text("❌ Vision функция отключена в настройках (ENABLE_VISION).")
        return
    if not msg.photo:
        return
    get_user(update.effective_user.id).awaiting = None
    caption = msg.caption or DEFAULT_IMAGE_PROMPT
    await send_typing(context, update.effective_chat.id, ChatAction.UPLOAD_PHOTO)
    status_msg = await msg.reply_text("👁 Загружаю изображение...")
    try:
        file = await msg.photo[-1].get_file()
        file_bytes = await file.download_as_bytearray()
        b64 = base64.b64encode(bytes(file_bytes)).decode("utf-8")
    except Exception as e:
        logger.error(f"Ошибка загрузки фото: {e}")
        await safe_edit(status_msg, f"❌ Ошибка загрузки изображения: {e}")
        return
    await generate(update, caption, b64, status_msg)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    err = context.error
    if isinstance(err, NetworkError):  # обрывы связи с Telegram — штатная ситуация, трейсбек не нужен
        logger.warning(f"🌐 Проблема с сетью Telegram: {err}")
    else:
        logger.error("💥 Необработанная ошибка в обработчике", exc_info=err)


async def post_init(application: Application):
    load_state()
    try:
        await application.bot.set_my_commands([
            BotCommand("start", "Показать меню"),
            BotCommand("menu", "Показать меню"),
        ])
    except Exception as e:
        logger.debug(f"ℹ️ Не удалось установить команды: {e}")
    logger.info("🚀 Bridge Telegram Bot запущен!")
    logger.info(f"👤 ADMIN_IDS: {CFG['ADMIN_IDS']}")
    logger.info(f"🌐 LM Studio URL: {CFG['BASE_URL']}")


# --- MAIN ---
def main():
    if not CFG["TOKEN"]:
        logger.error("❌ TELEGRAM_TOKEN не задан в .env")
        sys.exit(1)
    if not CFG["ADMIN_IDS"]:
        logger.error("❌ ADMIN_IDS не задан в .env (числовые Telegram ID через запятую)")
        sys.exit(1)

    # Python 3.14: asyncio.get_event_loop() больше не создаёт loop сам — создаём явно (на старых версиях тоже безопасно)
    asyncio.set_event_loop(asyncio.new_event_loop())

    request = HTTPXRequest(
        connection_pool_size=10,
        read_timeout=30.0,
        connect_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )
    app = (
        ApplicationBuilder()
        .token(CFG["TOKEN"])
        .request(request)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler(["start", "menu", "help"], start))
    app.add_handler(CallbackQueryHandler(cb_handler))
    # UpdateType.MESSAGE — чтобы правка старого сообщения не вызывала обработчик (там нет update.message)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.UpdateType.MESSAGE, handle_msg))
    app.add_handler(MessageHandler(filters.PHOTO & filters.UpdateType.MESSAGE, handle_photo))
    app.add_error_handler(on_error)

    logger.info("▶️ Запуск бота...")
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.critical(f"💥 Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
