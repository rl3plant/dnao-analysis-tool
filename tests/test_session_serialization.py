"""Regression test for a real, high-impact bug found while testing the
classifier-retraining path: Annotation.update_center_of_mass() assumed a flat
(N, 2) contour, but every real contour in this app - from SAM, from
cv2.findContours, anywhere - is OpenCV's (N, 1, 2) point format. Averaging
over axis=0 on that shape produces a 1-element array, not two floats, which
Session.to_dict() then passes straight through unconverted into json.dump().

Net effect: saving a session containing any real annotation raised
`TypeError: Object of type ndarray is not JSON serializable`, caught deep
inside session_controller.py and surfaced to the user only as a generic
"Save Failed" dialog - i.e. the tool's actual core feature (saving your
annotations) didn't work for real data.
"""
import json

import numpy as np

from data_handling.data_models import Annotation, Session

# OpenCV's real contour shape: (N, 1, 2), not (N, 2).
REAL_SHAPED_CONTOUR = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], dtype=np.int32)


def test_center_of_mass_is_a_plain_float_pair_not_an_ndarray():
    ann = Annotation(id="a1", source_image="img.png", contour=REAL_SHAPED_CONTOUR)

    assert ann.center_of_mass == (10.0, 10.0)
    assert all(isinstance(v, float) for v in ann.center_of_mass)


def test_session_with_real_contour_annotation_is_actually_json_serializable():
    session = Session(name="serialization_test")
    session.annotations["img.png"] = [
        Annotation(id="a1", source_image="img.png", contour=REAL_SHAPED_CONTOUR, class_label="intact"),
    ]

    # This is exactly what session_controller.py's save_session() does; must
    # not raise.
    serialized = json.dumps(session.to_dict())

    round_tripped = Session.from_dict(json.loads(serialized))
    assert round_tripped.annotations["img.png"][0].center_of_mass == [10.0, 10.0]
