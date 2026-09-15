import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
CLIENT_DIR = os.path.join(PROJECT_ROOT, "src", "client")

# Match src/client/client.py's own sys.path setup: `project_root` is needed for
# the app's `from src.client....` absolute imports, `client_dir` for its
# `from ui.xxx` / `from network.xxx` style imports.
for path in (PROJECT_ROOT, CLIENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# Make the sibling neo-toolkit checkout importable too, same as local_sam_service.py.
from utils.neo_toolkit_bootstrap import ensure_neo_toolkit_on_path  # noqa: E402

ensure_neo_toolkit_on_path()
