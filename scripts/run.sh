#!/usr/bin/env bash
# Run the bot in the foreground. Used by both service managers.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

PYTHON="${PYTHON:-$(command -v python3)}"
export PYTHONUNBUFFERED=1
# Local npm installs (pi, node tools) are not on the systemd PATH.
export PATH="$HOME/.npm-global/bin:/usr/local/bin:$PATH"

exec "$PYTHON" "$APP_DIR/bot.py" "$@"
