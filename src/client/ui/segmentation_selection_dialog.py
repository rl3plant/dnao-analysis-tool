from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QWidget, QGridLayout, QSizePolicy)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QShortcut, QKeySequence
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(200, 200)
        self.setMaximumSize(300, 300)
        # Add size policy to allow the widget to grow/shrink
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            # Calculate the scaled size while maintaining aspect ratio
            scaled_size = self.pixmap.size().scaled(
                self.size(),
                Qt.KeepAspectRatio
            )
            # Calculate the position to center the pixmap
            x = (self.width() - scaled_size.width()) // 2
            y = (self.height() - scaled_size.height()) // 2
            # Draw the pixmap centered in the widget
            painter.drawPixmap(x, y, scaled_size.width(), scaled_size.height(), self.pixmap)
            painter.end()

class SegmentationSelectionDialog(QDialog):
    def __init__(self, parent=None, image_path: str = None, predictions: List[Dict] = None, prompt_point = None):
        super().__init__(parent)
        self.setWindowTitle("Select Segmentation")
        self.setModal(True)
        
        # Store the predictions
        self.predictions = predictions
        self.prompt_point = prompt_point
        self.selected_index = -1
        
        # Create the main layout
        main_layout = QVBoxLayout()
        
        # Create a grid layout for the segmentation previews
        grid_layout = QGridLayout()
        
        # Create preview widgets for each prediction
        self.preview_widgets = []
        for i, pred in enumerate(predictions):
            # Create a widget to hold the preview
            preview_widget = PreviewWidget()
            
            # Store the prediction index in the widget
            preview_widget.prediction_index = i
            
            # Add to grid layout
            grid_layout.addWidget(preview_widget, 0, i)
            self.preview_widgets.append(preview_widget)
            
            # Create a label for the preview number
            number_label = QLabel(f"{i+1}")
            number_label.setAlignment(Qt.AlignCenter)
            grid_layout.addWidget(number_label, 1, i)
        
        main_layout.addLayout(grid_layout)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create buttons for each prediction
        self.buttons = []
        for i in range(len(predictions)):
            button = QPushButton(f"Select {i+1}")
            button.clicked.connect(lambda checked, idx=i: self.select_prediction(idx))
            button_layout.addWidget(button)
            self.buttons.append(button)
        
        main_layout.addLayout(button_layout)
        
        # Set the main layout
        self.setLayout(main_layout)
        
        # Load and process the image
        if image_path:
            self.load_image(image_path)
            
        # Set up keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        """Set up keyboard shortcuts for selection"""
        for i in range(len(self.predictions)):
            # Create shortcut for number keys 1-3
            shortcut = QShortcut(QKeySequence(str(i+1)), self)
            shortcut.activated.connect(lambda idx=i: self.select_prediction(idx))
    
    def load_image(self, image_path: str):
        """Load and process the image for preview"""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return
                
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate zoom region if we have a prompt point
            if self.prompt_point is not None:
                # Get image dimensions
                height, width = image.shape[:2]
                
                # Calculate zoom region (half of the original size)
                zoom_width = width // 2
                zoom_height = height // 2
                
                # Calculate the region around the prompt point
                # Start by centering the prompt point in the zoomed region
                x = self.prompt_point.x() - zoom_width // 2
                y = self.prompt_point.y() - zoom_height // 2
                
                # Then adjust if we're going out of bounds
                if x < 0:
                    x = 0
                elif x + zoom_width > width:
                    x = width - zoom_width
                    
                if y < 0:
                    y = 0
                elif y + zoom_height > height:
                    y = height - zoom_height
                
                # Extract the zoomed region
                zoomed_image = image[y:y+zoom_height, x:x+zoom_width]
                
                # Calculate the center point in the zoomed region
                center_x = self.prompt_point.x() - x
                center_y = self.prompt_point.y() - y
            else:
                zoomed_image = image
                center_x = image.shape[1] // 2
                center_y = image.shape[0] // 2
            
            # For each prediction, create a preview
            for i, pred in enumerate(self.predictions):
                if 'contours' in pred:
                    # Get the contour
                    contour = pred['contours'][0] if isinstance(pred['contours'], list) else pred['contours']
                    
                    # Create a copy of the zoomed image for this preview
                    preview_image = zoomed_image.copy()
                    
                    # If we're using a zoomed region, adjust the contour coordinates
                    if self.prompt_point is not None:
                        # Adjust contour coordinates to the zoomed region
                        adjusted_contour = contour - np.array([x, y])
                        # Draw the adjusted contour
                        cv2.drawContours(preview_image, [adjusted_contour], -1, (0, 255, 0), 2)
                    else:
                        # Draw the original contour
                        cv2.drawContours(preview_image, [np.array(contour)], -1, (0, 255, 0), 2)
                    
                    # Draw the center point as a red cross
                    cross_size = 10
                    cv2.line(preview_image, 
                            (center_x - cross_size, center_y),
                            (center_x + cross_size, center_y),
                            (255, 0, 0), 2)  # Red horizontal line
                    cv2.line(preview_image,
                            (center_x, center_y - cross_size),
                            (center_x, center_y + cross_size),
                            (255, 0, 0), 2)  # Red vertical line
                    
                    # Convert to QPixmap
                    height, width = preview_image.shape[:2]
                    bytes_per_line = 3 * width
                    preview_data = preview_image.copy()
                    q_image = QImage(preview_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    
                    # Set the pixmap in the widget
                    preview_widget = self.preview_widgets[i]
                    preview_widget.setPixmap(pixmap)
                    
        except Exception as e:
            print(f"Error loading image: {str(e)}")
    
    def select_prediction(self, index: int):
        """Select a prediction and close the dialog"""
        self.selected_index = index
        self.accept()
    
    def get_selected_prediction(self) -> Optional[Dict]:
        """Get the selected prediction"""
        if 0 <= self.selected_index < len(self.predictions):
            return self.predictions[self.selected_index]
        return None 