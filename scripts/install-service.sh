#!/usr/bin/env bash
# Install/refresh Minamo as a systemd *user* service on Linux.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/minamo-bot.service"

mkdir -p "$UNIT_DIR"
chmod +x "$APP_DIR/scripts/run.sh"

cat > "$UNIT" <<EOF
[Unit]
Description=Minamo - Telegram bridge for local Ollama
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/scripts/run.sh
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=PATH=$HOME/.npm-global/bin:/usr/local/bin:/usr/bin:/bin

# pi runs as a child process; stopping the bot must stop it too (it would
# otherwise sit waiting for an approval that can no longer arrive).
TimeoutStopSec=20

# Keep the bot's memory use sane if things ever go sideways
MemoryHigh=1500M
MemoryMax=2500M

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable minamo-bot.service
systemctl --user restart minamo-bot.service
sleep 2
systemctl --user --no-pager status minamo-bot.service || true

cat <<'MSG'

Installed. Useful commands:
  systemctl --user status minamo-bot
  systemctl --user restart minamo-bot
  journalctl --user -u minamo-bot -f

The service starts when you log in. To keep it running after logout
(and across reboots without logging in), enable lingering once:
  sudo loginctl enable-linger "$USER"
MSG
