#!/usr/bin/env bash
# Launches the DNAO Analysis Tool. This repo's imports only resolve correctly
# when run from src/client/ (see client.py) - this wrapper exists so you don't
# have to remember that.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python3"
[ -x "$REPO_ROOT/.venv/bin/python3" ] && PYTHON="$REPO_ROOT/.venv/bin/python3"

cd "$REPO_ROOT/src/client"
exec "$PYTHON" client.py "$@"
