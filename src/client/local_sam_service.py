"""In-process replacement for the old Flask SAM2 server (formerly src/server/).

The old server wrapped a `models.sam2_segmentation.SAM2Segmentation` class
that was gitignored and never committed to this repo -- so this tool's
segmentation was non-functional as checked into git, not just outdated.
Point-prompt segmentation now goes through
neo_toolkit.segmentation.sam_backend.SamSegmenter (see ../../../neo-toolkit).
Box-prompt segmentation reaches into that same SamSegmenter's underlying
SAM2ImagePredictor directly (`_predictor`, a "private" attribute) since
neo_toolkit's wrapper only exposes point prompts and adding box-prompt
support there is out of scope for this tool's cleanup -- narrow, documented
use of one extra method on an object we already own, not a stable public API
we're depending on.

Response shapes below intentionally mirror the old server's JSON responses
(see old_stuff/DNAO-Analysis-Tool/src/server/app.py) so the GUI/annotation
code that already parses those shapes needed no changes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.neo_toolkit_bootstrap import ensure_neo_toolkit_on_path

ensure_neo_toolkit_on_path()

from neo_toolkit.segmentation.sam_backend import SamSegmenter, largest_contour  # noqa: E402

logger = logging.getLogger(__name__)


def _mask_entry(contour: np.ndarray, area: float, score: float, index: int) -> Dict:
    return {"contours": [contour.tolist()], "area": float(area), "score": float(score), "index": index}


class LocalSamService:
    """One shared SamSegmenter; caches which image is currently embedded."""

    def __init__(self, checkpoint_path: str, model_cfg: str, device: Optional[str] = None):
        logger.info(f"Loading SAM2 ({checkpoint_path}, {model_cfg})...")
        self._sam = SamSegmenter(checkpoint_path, model_cfg=model_cfg, device=device)
        self._current_path: Optional[str] = None
        logger.info("SAM2 loaded")

    @property
    def sam(self) -> SamSegmenter:
        """The underlying SamSegmenter, for callers that want segment_afm_file() etc."""
        return self._sam

    def embed(self, image_path: str) -> bool:
        if self._current_path == image_path:
            return True
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._sam.set_image(image_rgb)
            self._current_path = image_path
            return True
        except Exception:
            logger.exception(f"Error embedding {image_path}")
            return False

    def segment_points(self, image_path: str, points: Sequence[Tuple[float, float]]) -> Optional[Dict]:
        if not self.embed(image_path):
            return None
        predictions = []
        for x, y in points:
            candidates = self._sam.predict_point((float(x), float(y)), multimask=True)
            masks = [_mask_entry(c.contour, c.area, c.sam_score, i) for i, c in enumerate(candidates)]
            predictions.append({"point": [float(x), float(y)], "masks": masks})
        return {"points": [[float(x), float(y)] for x, y in points], "predictions": predictions}

    def segment_bbox(self, image_path: str, bbox: Dict[str, int]) -> Optional[Dict]:
        if not self.embed(image_path):
            return None
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
        try:
            masks_np, scores_np, _ = self._sam._predictor.predict(box=box, multimask_output=True)
        except Exception:
            logger.exception("Box-prompt SAM2 prediction failed")
            return None

        mask_results = []
        for i, (mask, score) in enumerate(zip(masks_np, scores_np)):
            binary = mask > 0.5
            contour = largest_contour(binary)
            if contour is None:
                continue
            mask_results.append(_mask_entry(contour, float(binary.sum()), float(score), i))
        if not mask_results:
            return None
        predictions = [{"point": [float(v) for v in box], "masks": mask_results}]
        return {"points": [[float(v) for v in box]], "predictions": predictions}

    def segment_contour(self, image_path: str, contour: np.ndarray) -> Optional[Dict]:
        """Best-effort refinement: derive a bbox from the contour and re-segment.

        The old tool called a `/segment-mask` server endpoint that never
        existed in src/server/app.py. Routing it through box-prompt
        segmentation instead makes the "fuse selected annotations" refine
        step actually work.
        """
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        return self.segment_bbox(image_path, {"x1": x, "y1": y, "x2": x + w, "y2": y + h})


_shared_service: Optional[LocalSamService] = None


def get_shared_service() -> LocalSamService:
    """One SAM2 model per process, shared by ImageHandler and run_pipeline()."""
    global _shared_service
    if _shared_service is None:
        from utils.config import get_settings

        settings = get_settings()
        _shared_service = LocalSamService(settings["sam2_checkpoint"], settings["sam2_model_cfg"])
    return _shared_service
