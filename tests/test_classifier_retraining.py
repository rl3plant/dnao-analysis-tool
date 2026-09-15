"""Tests for the classifier-retraining path: main_window.py's
save_annotation() -> session-wide SVMClassifier.fit().
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from data_handling.data_models import Annotation, TaskType
from ui.main_window import DNAOApplication

SQUARE_CONTOUR = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], dtype=np.int32)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window_with_session(qapp, tmp_path, monkeypatch):
    # save_annotation() shows a modal QMessageBox on success/failure - .exec()
    # would otherwise block forever waiting for a click that can't happen headless.
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

    window = DNAOApplication()
    monkeypatch.setattr(window.session_ctrl.session_manager, "sessions_dir", str(tmp_path))
    window.session_ctrl.create_session("retrain_test_session", TaskType.YIELD)

    window.current_image_name = "image_1.png"
    window.annotation_ctrl.annotations = [
        Annotation(id="a1", source_image="image_1.png", contour=SQUARE_CONTOUR, class_label="intact"),
    ]
    window.annotation_ctrl.has_unsaved_changes = True
    yield window
    window.close()


def test_saving_annotations_increments_the_retrain_counter(window_with_session):
    """Regression test for a real bug: increment_retrain_counter() existed but
    was never called, so should_retrain_classifier()'s "retrain every k
    images" throttle never actually throttled anything once k > 0 (it was
    only masked by the k=0 "always retrain" default)."""
    session = window_with_session.session_ctrl.current_session
    session.retrain_frequency = 5  # so this single save should NOT trigger a retrain
    assert session.images_since_retrain == 0

    window_with_session.save_annotation()

    assert session.images_since_retrain == 1
    assert not session.should_retrain_classifier()


def test_saving_annotations_retrains_and_resets_counter_at_the_threshold(window_with_session, tmp_path):
    session = window_with_session.session_ctrl.current_session
    session.retrain_frequency = 1
    # get_classifier_path() returns a bare filename relative to CWD by default
    # (see README's "Found but not fixed" notes) - pin it into tmp_path so this
    # test doesn't write a stray .pkl into wherever pytest happens to run from.
    session.set_classifier_path(str(tmp_path / "classifier.pkl"))

    window_with_session.save_annotation()

    assert session.images_since_retrain == 0  # reset after retraining
    assert os.path.exists(session.get_classifier_path())


def test_retraining_trains_on_every_annotation_regardless_of_label(window_with_session, monkeypatch):
    """The mask-validity classifier trains on every KEPT annotation
    (intact or damaged alike), not just
    "intact" ones - filtering to intact-only would teach it that damaged (but
    correctly segmented) shapes look invalid, which is the opposite of what
    we want. Deletion, not labeling, is how a user excludes a bad detection."""
    session = window_with_session.session_ctrl.current_session
    session.retrain_frequency = 1
    window_with_session.annotation_ctrl.annotations.append(
        Annotation(id="a2", source_image="image_1.png", contour=SQUARE_CONTOUR, class_label="damaged")
    )

    fit_calls = []
    monkeypatch.setattr(
        "src.client.pipeline.instance_segmentation.segmentation_classification.svm_classifier.SVMClassifier.fit",
        lambda self, features, labels=None: fit_calls.append((len(features), list(labels or [])))
    )
    monkeypatch.setattr(
        "src.client.pipeline.instance_segmentation.segmentation_classification.svm_classifier.SVMClassifier.save",
        lambda self, path: None,
    )

    window_with_session.save_annotation()

    assert len(fit_calls) == 1
    n_features, labels = fit_calls[0]
    assert n_features == 2
    assert set(labels) == {"intact", "damaged"}
