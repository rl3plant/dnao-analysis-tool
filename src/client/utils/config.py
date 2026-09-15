"""
Configuration module for the DNAO Analysis Tool client
"""

import logging
import os

logger = logging.getLogger(__name__)

# Add new setting for using precomputed contours
USE_PRECOMPUTED_CONTOURS = False  # When True, load contours from local files instead of server processing

# Mask filtering settings
MASK_AREA_THRESHOLD = 1.0  # Threshold factor for area-based filtering

def get_settings():
    """
    Return application settings

    Returns:
        Dictionary containing application settings
    """
    return {
        # SAM2 checkpoint used by local_sam_service.py. Get one with
        # neo-toolkit's scripts/download_sam2_checkpoint.py; defaults to
        # this repo's own checkpoints/ dir (gitignored, same convention).
        "sam2_checkpoint": os.environ.get("DNAO_SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_tiny.pt"),
        "sam2_model_cfg": os.environ.get("DNAO_SAM2_MODEL_CFG", "sam2.1/sam2.1_hiera_t.yaml"),

        # Image file extensions to recognize
        "image_extensions": ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
        
        # Default annotation colors
        "colors": {
            "intact": (0, 255, 0),    # Green for intact DNAOs
            "damaged": (255, 0, 0),   # Red for damaged DNAOs
            "unknown": (0, 0, 255)    # Blue for unknown/other
        },
        
        # Request timeouts (seconds)
        "timeouts": {
            "default": 30,
            "mask_generation": 60
        },
        
        # Add new setting for using precomputed contours
        'USE_PRECOMPUTED_CONTOURS': USE_PRECOMPUTED_CONTOURS,
        
        # Mask filtering settings
        'MASK_AREA_THRESHOLD': MASK_AREA_THRESHOLD,
    }

def save_settings(settings_dict):
    """Save settings to the module variables"""
    global USE_PRECOMPUTED_CONTOURS, MASK_AREA_THRESHOLD

    if 'USE_PRECOMPUTED_CONTOURS' in settings_dict:
        USE_PRECOMPUTED_CONTOURS = settings_dict['USE_PRECOMPUTED_CONTOURS']
    
    if 'MASK_AREA_THRESHOLD' in settings_dict:
        MASK_AREA_THRESHOLD = settings_dict['MASK_AREA_THRESHOLD']
    
    # You might want to save to a config file here
    logger.info("Settings updated") 