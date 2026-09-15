#!/usr/bin/env bash
# One-command setup: checks the neo-toolkit sibling checkout, installs this
# repo's Python deps, and fetches a SAM2 checkpoint. See README.md if you'd
# rather do these steps by hand (e.g. to pick a different checkpoint size).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEO_TOOLKIT="$REPO_ROOT/../neo-toolkit"
CHECKPOINT_SIZE="${1:-tiny}"

echo "== 1/3: checking for a neo-toolkit sibling checkout =="
if [ ! -d "$NEO_TOOLKIT/src/neo_toolkit" ]; then
    echo "error: expected a neo-toolkit checkout at $NEO_TOOLKIT (i.e. as a sibling of this repo)." >&2
    echo "       check it out there, or 'pip install -e ../neo-toolkit' if it's installed some other way." >&2
    exit 1
fi
echo "found $NEO_TOOLKIT"

echo
echo "== 2/3: installing Python dependencies =="
# Prefer an isolated venv (doesn't touch system/user Python at all). Falls back
# to installing into the current environment if venv isn't available (e.g. the
# `python3-venv` OS package isn't installed and you can't/don't want to add
# it) - some distros (Debian/Ubuntu 23.10+) then refuse a plain `pip install`
# ("externally-managed-environment", PEP 668), so that fallback tries again
# with --break-system-packages, which does modify your system/user Python
# environment - you'll see a note if that happens.
if [ ! -x "$REPO_ROOT/.venv/bin/pip" ]; then
    rm -rf "$REPO_ROOT/.venv"  # in case a previous failed attempt left a broken partial venv
    if python3 -m venv "$REPO_ROOT/.venv" >/dev/null 2>&1 && [ -x "$REPO_ROOT/.venv/bin/pip" ]; then
        echo "created .venv/"
    else
        rm -rf "$REPO_ROOT/.venv"  # venv module unusable here (e.g. missing python3-venv OS package)
    fi
fi
if [ -x "$REPO_ROOT/.venv/bin/pip" ]; then
    echo "installing into .venv/ (run with: source .venv/bin/activate, or just use ./run.sh)"
    "$REPO_ROOT/.venv/bin/pip" install -r "$REPO_ROOT/requirements.txt"
elif pip install -r "$REPO_ROOT/requirements.txt" 2>/tmp/dnao_pip_err.$$; then
    :
elif grep -q "externally-managed-environment" /tmp/dnao_pip_err.$$; then
    echo "note: no venv available and this Python is externally-managed; installing with --break-system-packages instead." >&2
    pip install --break-system-packages -r "$REPO_ROOT/requirements.txt"
else
    cat /tmp/dnao_pip_err.$$ >&2
    rm -f /tmp/dnao_pip_err.$$
    exit 1
fi
rm -f /tmp/dnao_pip_err.$$

echo
echo "== 3/3: fetching a SAM2 checkpoint (size: $CHECKPOINT_SIZE) =="
if compgen -G "$REPO_ROOT/checkpoints/*.pt" > /dev/null 2>&1; then
    echo "checkpoints/ already has a checkpoint, skipping download (pass a size arg to force: ./setup.sh <size>)"
else
    python3 "$NEO_TOOLKIT/scripts/download_sam2_checkpoint.py" "$CHECKPOINT_SIZE" --out-dir "$REPO_ROOT/checkpoints"
fi

echo
echo "Setup done. Run the tool with: ./run.sh"
