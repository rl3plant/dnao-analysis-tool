"""Guards against another `noahs_tools/load_save_afm.py`-style landmine: that
file hardcoded `/home/superhans/Master_Thesis/DNAO-Analysis-Tool/...` and
would only ever work on the machine it was written on. It was dead code (never
imported by the app) and has been removed, but nothing previously stopped a
hardcoded absolute path from being added to code that *is* on the app's real
import path - this test does.
"""
import re
from pathlib import Path

CLIENT_DIR = Path(__file__).resolve().parent.parent / "src" / "client"

# Matches /home/<user>/..., /Users/<user>/..., and a bare C:\ drive path -
# i.e. an absolute path baked in for one specific machine/account, not a
# relative path or an environment-variable-driven default (those are fine).
HARDCODED_ABSOLUTE_PATH = re.compile(r"""['"](/home/[\w.-]+/|/Users/[\w.-]+/|[A-Za-z]:\\)""")


def test_no_hardcoded_user_specific_paths_in_shipped_code():
    offenders = []
    for path in CLIENT_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if HARDCODED_ABSOLUTE_PATH.search(line):
                offenders.append(f"{path.relative_to(CLIENT_DIR.parent.parent)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found hardcoded, machine/user-specific absolute path(s) that will break "
        "for anyone else checking out this repo:\n" + "\n".join(offenders)
    )
