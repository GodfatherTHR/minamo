#!/usr/bin/env bash
# Install/refresh Minamo as a launchd agent on macOS (starts at login, auto-restarts).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="ai.minamo.bot"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PYTHON="${PYTHON:-$(command -v python3)}"
LOG_DIR="$APP_DIR/data"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$APP_DIR/scripts/run.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$APP_DIR/scripts/run.sh</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHON</key><string>$PYTHON</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>PATH</key><string>$HOME/.npm-global/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  <key>WorkingDirectory</key><string>$APP_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/launchd.err.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

cat <<EOF

Installed and started: $PLIST

Useful commands:
  launchctl print gui/$(id -u)/$LABEL | head -40
  tail -f "$LOG_DIR/launchd.err.log"
  launchctl kickstart -k gui/$(id -u)/$LABEL   # restart
EOF
