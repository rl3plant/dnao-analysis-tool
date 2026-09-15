"""Sanity checks for setup.sh/run.sh - not a full run (that needs network
access, a real venv, and takes minutes), just enough to catch a broken script
before someone else's first `./setup.sh` does."""
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _check_script(name):
    path = REPO_ROOT / name
    assert path.is_file(), f"{name} is missing"
    assert stat.S_IMODE(path.stat().st_mode) & stat.S_IXUSR, f"{name} is not executable"

    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, f"{name} has a bash syntax error:\n{result.stderr}"


def test_setup_sh_is_executable_and_syntactically_valid():
    _check_script("setup.sh")


def test_run_sh_is_executable_and_syntactically_valid():
    _check_script("run.sh")
