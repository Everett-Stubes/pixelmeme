#!/usr/bin/env bash
# Start pixelmeme. Used by the systemd unit and fine to run by hand.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Belt and braces: the app re-asserts these itself every minute, but doing it
# here too covers the gap before the first frame.
command -v xset >/dev/null 2>&1 && { xset s off || true; xset -dpms || true; xset s noblank || true; }

exec python -m pixelmeme.main "$@"
