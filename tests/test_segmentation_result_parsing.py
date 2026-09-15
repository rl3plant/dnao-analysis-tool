"""Regression tests for the single-candidate point/bbox segmentation bug fixed in
commit a0edfe6 (see README.md "Known issues fixed").

Bug: `annotation_controller.py`'s `_process_segmentation_result` reads
`result["prediction"]` (singular), but `image_handler.py`'s
`request_segmentation`/`request_bbox_segmentation` only ever returned
`result["predictions"]` (plural) - so every normal, single-candidate point or
box click silently produced "No prediction in segmentation result" and no
annotation, unless the multi-candidate picker dialog happened to fire and
explicitly set the singular key. Fixed by having the request_* methods set
`response['prediction'] = response['predictions'][0]['masks']` themselves.

These tests use a fake local_sam_service (no real SAM2/GPU/checkpoint) so they
run in well under a second, and no QApplication is needed - none of the code
under test touches Qt widgets.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from network.image_handler import ImageHandler
from ui.annotation_controller import AnnotationController

SQUARE_CONTOUR = [[0, 0], [10, 0], [10, 10], [0, 10]]


def _fake_sam_response(score=0.9, area=100):
    """Mimics local_sam_service's segment_points/segment_bbox return shape."""
    return {"predictions": [{"masks": [{"contours": [SQUARE_CONTOUR], "score": score, "area": area}]}]}


class FakeSamService:
    def segment_points(self, image_path, points):
        return _fake_sam_response()

    def segment_bbox(self, image_path, bbox):
        return _fake_sam_response()


@pytest.fixture
def image_handler(monkeypatch):
    monkeypatch.setattr("network.image_handler.get_shared_service", lambda: FakeSamService())
    return ImageHandler()


@pytest.fixture
def annotation_ctrl():
    # AnnotationController only touches `parent.session_ctrl` behind a hasattr guard,
    # so a bare object (no Qt, no real main window) is enough to construct it.
    return AnnotationController(SimpleNamespace())


def test_request_segmentation_sets_singular_prediction_key(image_handler):
    result = image_handler.request_segmentation("fake.png", x=5, y=5)

    assert result is not None
    assert result["prediction"] == result["predictions"][0]["masks"]


def test_request_bbox_segmentation_sets_singular_prediction_key(image_handler):
    result = image_handler.request_bbox_segmentation("fake.png", bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10})

    assert result is not None
    assert result["prediction"] == result["predictions"][0]["masks"]


def test_single_candidate_point_click_produces_an_annotation(image_handler, annotation_ctrl):
    """End-to-end regression test for the actual user-facing bug: a normal
    (single-candidate) point click must produce a real Annotation, not silently
    do nothing."""
    result = image_handler.request_segmentation("fake.png", x=5, y=5)

    annotation = annotation_ctrl._process_segmentation_result(result, "fake.png")

    assert annotation is not None
    assert annotation.confidence == pytest.approx(0.9)
    assert annotation.properties["area"] == 100
    np.testing.assert_array_equal(annotation.contour, np.array(SQUARE_CONTOUR, dtype=np.int32))


def test_process_segmentation_result_without_singular_key_returns_none(annotation_ctrl):
    """Characterizes the original bug: a response with only the plural
    'predictions' key (what request_segmentation used to return, unmodified)
    must be rejected rather than silently mis-parsed."""
    broken_result = _fake_sam_response()  # only has 'predictions', no 'prediction'

    annotation = annotation_ctrl._process_segmentation_result(broken_result, "fake.png")

    assert annotation is None
