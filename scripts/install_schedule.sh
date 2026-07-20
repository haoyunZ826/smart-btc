#!/bin/bash
# Install a macOS LaunchAgent that runs the monitor daily and publishes.
#
# A fallback for the GitHub Actions workflow. Both paths run the identical
# scripts, so running both is harmless — publish.sh is a no-op when there is
# no new bar.
#
# macOS caveat, learned the hard way: if this repo lives under ~/Desktop,
# ~/Documents or ~/Downloads, the job will run but publish.sh dies with
# "Operation not permitted". Those directories are TCC-protected and a
# launchd agent does not inherit the Full Disk Access that your terminal has.
# The same script run by hand from a terminal works fine, which makes this
# look like a script bug when it is a sandbox boundary. Fixes, best first:
#   1. Use the GitHub Actions workflow instead — it runs in the cloud and
#      keeps updating while this machine is asleep.
#   2. Move the repo outside the protected directories (e.g. ~/projects).
#   3. Grant Full Disk Access to /bin/bash in System Settings → Privacy
#      & Security. This works but is a broad grant; prefer 1 or 2.
#
#   bash scripts/install_schedule.sh          # install and load
#   bash scripts/install_schedule.sh --remove # uninstall
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
LABEL="com.smartbtc.monitor"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "${1:-}" = "--remove" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

PYTHON=$(command -v python3)
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data/logs"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd '$ROOT' && '$PYTHON' scripts/monitor.py && bash scripts/publish.sh</string>
  </array>
  <!-- 01:20 UTC is 09:20 Asia/Shanghai; launchd schedules in LOCAL time. -->
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>20</integer></dict>
  <!-- Catch up after the laptop was asleep at the scheduled time. -->
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$ROOT/data/logs/monitor.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/logs/monitor.err</string>
</dict>
</plist>
PLIST_EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "installed $LABEL — runs daily at 09:20 local time"
echo "  logs:   $ROOT/data/logs/monitor.log"
echo "  test:   launchctl start $LABEL"
echo "  remove: bash scripts/install_schedule.sh --remove"
