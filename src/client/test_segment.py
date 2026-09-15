import os
import sys
import logging
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Get the src directory
project_root = os.path.dirname(src_dir)  # Get the project root
sys.path.insert(0, project_root)

from network.image_handler import ImageHandler
from utils.logger import setup_logger

def main():
    # Set up logging
    logger = setup_logger(log_level=logging.DEBUG)
    logger.info("Starting test segmentation script")

    # Initialize the application (needed for QPixmap)
    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        logger.error(f"usage: {sys.argv[0]} <image_path>")
        return
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return

    # Segmentation runs in-process now (see local_sam_service.py); server_url
    # is accepted for compatibility but unused.
    image_handler = ImageHandler()

    # Generate masks
    logger.info(f"Sending image {image_path} to server for mask generation")
    response = image_handler.generate_masks(image_path)

    if not response:
        logger.error("Failed to generate masks")
        return

    # Process masks into annotations
    annotations = image_handler.process_masks(response, image_path)
    logger.info(f"Generated {len(annotations)} annotations")

    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    # Convert to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # Create QImage and QPixmap for drawing
    qimage = QImage(image_rgb.data, width, height, image_rgb.strides[0], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)

    # Draw annotations on the image
    result_pixmap = image_handler.draw_annotations(pixmap, annotations)

    # Save the result
    output_path = os.path.splitext(image_path)[0] + "_annotated.png"
    result_pixmap.save(output_path)
    logger.info(f"Saved annotated image to: {output_path}")

if __name__ == "__main__":
    main()