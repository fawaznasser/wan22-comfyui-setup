#!/usr/bin/env bash
# Wan 2.2 setup for Linux. Finds a suitable Python and runs install.py.
# Usage: ./setup.sh [install.py args]   e.g. ./setup.sh --yes --test
set -euo pipefail
cd "$(dirname "$0")"

command -v git >/dev/null || { echo "git not found (sudo apt install git)"; exit 1; }

PY=""
for p in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null &&
     "$p" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)'; then
    PY="$p"; break
  fi
done
[ -n "$PY" ] || { echo "Need Python 3.10-3.13 (e.g. sudo apt install python3.12 python3.12-venv)"; exit 1; }
"$PY" -c 'import venv, ensurepip' 2>/dev/null ||
  { echo "venv module missing: sudo apt install ${PY}-venv"; exit 1; }

echo "Using Python: $PY"
exec "$PY" install.py "$@"
