import logging
from typing import Dict, List, Optional

from data_handling.data_models import Session
from local_sam_service import get_shared_service

logger = logging.getLogger(__name__)


def run_pipeline(
    image_path: str,
    session: Optional[Session] = None,
    **_unused_visualization_kwargs,
) -> List[Dict]:
    """
    Run the instance segmentation pipeline, backed by neo_toolkit.segmentation.
    Also handles raw AFM formats (.spm/.ibw/.jpk/.mi), not just rendered images.

    Args:
        image_path: Path to the input image (or raw AFM scan file)
        session: Unused; kept for call-site compatibility

    Returns:
        List of mask dictionaries ({"contours": [[x, y], ...], "area", "score"})
    """
    from neo_toolkit.segmentation.pipeline import segment_afm_file

    try:
        logger.info(f"Starting pipeline for image: {image_path}")
        sam = get_shared_service().sam
        _preprocessed_image, instances = segment_afm_file(image_path, sam)
        masks = [
            {
                "contours": [instance.contour.tolist()],
                "area": float(instance.area),
                "score": float(instance.sam_score),
            }
            for instance in instances
        ]
        logger.info(f"Pipeline completed. Generated {len(masks)} masks")
        return masks

    except Exception as e:
        logger.exception(f"Error in pipeline: {e}")
        return []
