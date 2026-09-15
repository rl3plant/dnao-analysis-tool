"""Confirms the GUI's batch segmentation entry point (`run_pipeline`) always
delegates to neo-toolkit's *current* `segment_afm_file`/`segment_image` with no
parameter overrides of its own.

Why this matters: neo-toolkit's segmentation pipeline is a live sibling
checkout (see `utils/neo_toolkit_bootstrap.py`), not a vendored/pinned copy -
so whenever neo-toolkit's own tuned defaults change (e.g. the Bayesian-tuned
morph_kernel/area_ratio/nms_iou from its "hardening" pass), this GUI tool
picks them up automatically on next run, with no vendoring step to remember.
This test guards the other half of that claim: that `run_pipeline` itself
never hardcodes/overrides any of those tuned parameters, which would silently
pin this tool to stale values even as neo-toolkit's defaults keep improving.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.instance_segmentation import pipeline as gui_pipeline


@pytest.fixture
def fake_segment_afm_file(monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return np.zeros((10, 10), dtype=np.uint8), []

    # `run_pipeline` does `from neo_toolkit.segmentation.pipeline import
    # segment_afm_file` *inside* the function body, so it re-resolves this
    # attribute from the real neo_toolkit module at call time - patch it there.
    monkeypatch.setattr("neo_toolkit.segmentation.pipeline.segment_afm_file", fake)
    monkeypatch.setattr(gui_pipeline, "get_shared_service", lambda: SimpleNamespace(sam="fake-sam-instance"))
    return calls


def test_run_pipeline_forwards_to_neo_toolkit_with_no_overrides(fake_segment_afm_file):
    gui_pipeline.run_pipeline("fake_image.png")

    assert len(fake_segment_afm_file) == 1
    args, kwargs = fake_segment_afm_file[0]

    assert args == ("fake_image.png", "fake-sam-instance")
    assert kwargs == {}, (
        f"run_pipeline is overriding neo-toolkit's tuned segmentation defaults with {kwargs!r} - "
        "this should stay empty so this tool always inherits neo-toolkit's current tuning."
    )


def test_run_pipeline_ignores_legacy_visualization_kwargs(fake_segment_afm_file):
    """The old (pre-neo_toolkit) pipeline accepted enable_visualization/etc.
    call-site kwargs; run_pipeline still accepts them for compatibility but
    they must be no-ops now, not errors."""
    result = gui_pipeline.run_pipeline(
        "fake_image.png", enable_visualization=True, visualization_output_dir="/tmp/x",
        save_individual_stages=True)

    assert result == []
    assert len(fake_segment_afm_file) == 1
