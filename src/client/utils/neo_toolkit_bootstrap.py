"""Make the sibling neo-toolkit checkout importable without a pip install.

Expects `neo-toolkit/` to be checked out next to this repo (both under the
same parent directory) -- the same layout produced by cloning NEO_CLEANUP's
repos side by side. If you've `pip install -e ../neo-toolkit`'d it instead,
this is a harmless no-op (the import already resolves).
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_neo_toolkit_on_path() -> None:
    try:
        import neo_toolkit  # noqa: F401
        return  # already importable (e.g. pip install -e)
    except ImportError:
        pass

    # this file: <repo_root>/src/client/utils/neo_toolkit_bootstrap.py
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root.parent / "neo-toolkit" / "src"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
    else:
        raise ImportError(
            f"neo_toolkit not found. Expected a sibling checkout at {candidate.parent}, "
            "or install it with `pip install -e ../neo-toolkit`. See README.md."
        )
