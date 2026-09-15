"""Headless GUI smoke tests: key windows must construct and render without
crashing, under `QT_QPA_PLATFORM=offscreen` (no real display needed).

This formalizes the manual `QT_QPA_PLATFORM=offscreen` + screenshot check from
HANDOFF.md step 1-2 into something that runs on every change, instead of a
one-off check - "imports cleanly" is not the same bar as "the widgets you can
actually click all work."
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from data_handling.data_models import TaskType
from ui.main_window import DNAOApplication
from utils import config


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def main_window(qapp):
    window = DNAOApplication()
    window.setWindowTitle("DNAO Analysis Tool")
    yield window
    window.close()


def test_main_window_constructs(main_window):
    assert main_window.currentIndex() == 0  # starts on the session-list screen


def test_session_view_reachable_without_crashing(main_window, tmp_path, monkeypatch):
    # Sessions are stored under the session manager's own directory; point it at
    # a tmp dir so this test doesn't touch (or depend on) real user data.
    monkeypatch.setattr(main_window.session_ctrl.session_manager, "sessions_dir", str(tmp_path))

    main_window.session_ctrl.create_session("smoke_test_session", TaskType.YIELD)
    main_window.setCurrentIndex(1)

    assert main_window.currentIndex() == 1


def test_config_save_settings_does_not_raise():
    """Regression test for a real bug (surfaced while removing the Settings
    dialog that called this): config.save_settings() called logger.info()
    with `logger` never imported in that module - it raised NameError
    unconditionally, so the one thing this function did (log a message) was
    itself broken."""
    config.save_settings({'USE_PRECOMPUTED_CONTOURS': False, 'MASK_AREA_THRESHOLD': 1.0})
