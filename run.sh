#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_$(date +%Y-%m-%d_%H-%M-%S).log"

# Храним только 20 последних логов запуска
ls -1t "$LOG_DIR"/launcher_*.log 2>/dev/null | tail -n +21 | xargs rm -f 2>/dev/null || true

exec > >(tee -a "$LOG_FILE") 2>&1

echo ">>> [$(date)] Launcher started"
echo ">>> Working directory: $SCRIPT_DIR"

# --- Конфигурация ---
# .env обязателен, только если токен не передан переменными окружения (например, через systemd)
if [ ! -f .env ] && [ -z "${TELEGRAM_TOKEN:-}" ]; then
    echo "❌ Файл .env не найден. Выполните: cp .env.example .env  — и заполните его."
    exit 1
fi
if [ ! -f config.json ] && [ -f config.json.example ]; then
    echo ">>> config.json не найден — копирую из config.json.example"
    cp config.json.example config.json
fi

# --- Проверка инструментов ---
if ! command -v python3 > /dev/null 2>&1; then
    echo "❌ python3 не найден. Установите Python 3.10 или новее."
    exit 1
fi
if ! command -v curl > /dev/null 2>&1; then
    echo "❌ curl не найден (нужен для проверки сервера LM Studio). Установите curl."
    exit 1
fi

# --- Виртуальное окружение (пересоздаётся только при смене версии Python) ---
PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ -d venv ] && [ "$(cat venv/.pyver 2>/dev/null || true)" != "$PYVER" ]; then
    echo ">>> Версия Python изменилась — пересоздаю venv..."
    rm -rf venv
fi
if [ ! -d venv ]; then
    echo ">>> Creating virtual environment..."
    if ! python3 -m venv venv; then
        rm -rf venv
        echo "❌ Не удалось создать venv. В Debian/Ubuntu установите пакет python3-venv."
        exit 1
    fi
    echo "$PYVER" > venv/.pyver
fi
# shellcheck disable=SC1091
source venv/bin/activate

# Зависимости ставятся только при первом запуске или после изменения requirements.txt
# (так бот стартует и без интернета, если всё уже установлено)
REQ_STAMP="$(cksum < requirements.txt | cut -d' ' -f1,2)"
if [ "$(cat venv/.req_stamp 2>/dev/null || true)" != "$REQ_STAMP" ]; then
    echo ">>> Installing Python dependencies..."
    pip install -q -r requirements.txt
    echo "$REQ_STAMP" > venv/.req_stamp
else
    echo ">>> Dependencies are up to date."
fi

# Значения читаем тем же парсером, что и сам бот (python-dotenv): понимает кавычки и комментарии.
# Переменные окружения имеют приоритет над .env — как и в боте.
read_env() {
    python -c '
import os, sys
from dotenv import dotenv_values
name, default = sys.argv[1], sys.argv[2]
print(os.environ.get(name) or dotenv_values(".env").get(name) or default)
' "$1" "$2"
}
LM_URL="$(read_env LM_STUDIO_URL http://localhost:1234)"
LM_URL="${LM_URL%/}"
LM_TOKEN="$(read_env LM_API_TOKEN lm-studio)"

# --- LM STUDIO ---
check_server() {
    curl -fsS --max-time 3 -H "Authorization: Bearer $LM_TOKEN" "$LM_URL/v1/models" > /dev/null 2>&1
}

echo ">>> Checking LM Studio server at $LM_URL ..."
if check_server; then
    echo "✅ LM Studio server is already running."
else
    case "$LM_URL" in
        *localhost*|*127.0.0.1*)
            if command -v lms > /dev/null 2>&1; then
                echo ">>> Server is down — starting via 'lms server start'..."
                lms server start > /dev/null 2>&1 || true
            else
                echo "⚠️ Server is down, and the 'lms' CLI was not found — start LM Studio server manually."
            fi
            ;;
    esac
    echo ">>> Waiting for the API (up to 45 s)..."
    for _ in $(seq 1 45); do
        check_server && break
        sleep 1
    done
    if ! check_server; then
        echo "❌ LM Studio API is unreachable at $LM_URL (проверьте, что сервер запущен и LM_API_TOKEN верный)."
        exit 1
    fi
    echo "✅ LM Studio server is up."
fi

# --- BOT ---
echo ">>> Starting Telegram bot..."
trap "echo '>>> Stopping bot...'" EXIT
python bot.py
echo ">>> Bot process stopped."
