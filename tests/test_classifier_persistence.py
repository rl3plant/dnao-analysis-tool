"""Regression tests for the two "found but not fixed" gaps from the
2026-09-12 README update, now fixed: the trained classifier silently never
reloaded on reopening a session, and its file lived at a bare filename
relative to CWD instead of next to the session's own data.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from data_handling.data_models import Annotation, Session, TaskType
from ui.main_window import DNAOApplication

SQUARE_CONTOUR = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], dtype=np.int32)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    # save_annotation() shows a modal QMessageBox on success/failure - .exec()
    # would otherwise block forever waiting for a click that can't happen headless.
    for method in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, method, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))


def _window(tmp_path, monkeypatch):
    window = DNAOApplication()
    monkeypatch.setattr(window.session_ctrl.session_manager, "sessions_dir", str(tmp_path))
    return window


def test_create_session_puts_the_classifier_next_to_the_session_data(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)

    window.session_ctrl.create_session("my_session", TaskType.YIELD)

    classifier_path = window.session_ctrl.current_session.get_classifier_path()
    assert os.path.dirname(classifier_path) == str(tmp_path)
    window.close()


def test_load_session_normalizes_an_old_bare_relative_classifier_path(qapp, tmp_path, monkeypatch):
    """An old-style session file (saved before this fix) has classifier_path
    as a bare filename; loading it must not perpetuate that."""
    window = _window(tmp_path, monkeypatch)
    session = Session(name="legacy_session", classifier_path="classifier_legacy_session.pkl")
    session_file = window.session_ctrl.session_manager.get_session_file_path("legacy_session")
    os.makedirs(os.path.dirname(session_file), exist_ok=True)
    with open(session_file, "w") as f:
        import json
        json.dump(session.to_dict(), f)

    window.session_ctrl.load_session(session_file)

    classifier_path = window.session_ctrl.current_session.get_classifier_path()
    assert os.path.dirname(classifier_path) == str(tmp_path)
    window.close()


def test_reopening_a_session_restores_its_trained_classifier(qapp, tmp_path, monkeypatch):
    """The actual end-to-end regression test: train a classifier, "close" the
    session (a fresh window against the same sessions_dir, simulating an app
    restart), reopen it, and confirm the classifier came back trained -
    before this fix it silently started untrained every time."""
    window = _window(tmp_path, monkeypatch)
    window.session_ctrl.create_session("persist_test", TaskType.YIELD)
    window.current_image_name = "image_1.png"
    window.annotation_ctrl.annotations = [
        Annotation(id="a1", source_image="image_1.png", contour=SQUARE_CONTOUR, class_label="intact"),
    ]
    window.annotation_ctrl.has_unsaved_changes = True
    window.session_ctrl.current_session.retrain_frequency = 1

    window.save_annotation()  # trains + saves the classifier to disk

    classifier_path = window.session_ctrl.current_session.get_classifier_path()
    assert os.path.exists(classifier_path)
    window.close()

    # simulate reopening the app / this session later
    reopened = _window(tmp_path, monkeypatch)
    session_file = reopened.session_ctrl.session_manager.get_session_file_path("persist_test")
    reopened.session_ctrl.load_session(session_file)  # fires session_loaded -> on_session_loaded -> reload_classifier

    classifier = reopened.session_ctrl.current_session._classifier_instance
    assert classifier.can_predict()
    reopened.close()
