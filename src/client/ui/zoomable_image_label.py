from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QPainter
from PySide6.QtCore import Qt

class ZoomableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self._pixmap = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_display()

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        
        # Get mouse position relative to the label
        mouse_pos = event.position()
        
        # Calculate zoom factor
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1.0 / 1.1
        old_zoom = self._zoom
        self._zoom *= zoom_factor
        self._zoom = max(0.1, min(self._zoom, 10.0))
        
        # Calculate the zoom center point
        label_size = self.size()
        pixmap_size = self._pixmap.size()
        
        # Calculate the visible pixmap area
        visible_width = min(label_size.width(), pixmap_size.width() * self._zoom)
        visible_height = min(label_size.height(), pixmap_size.height() * self._zoom)
        
        # Calculate the mouse position relative to the pixmap
        mouse_x_ratio = mouse_pos.x() / label_size.width()
        mouse_y_ratio = mouse_pos.y() / label_size.height()
        
        # Adjust pan to keep mouse position fixed
        if self._zoom > old_zoom:  # Zooming in
            self._pan_x += (mouse_x_ratio - 0.5) * (1 - zoom_factor) * visible_width
            self._pan_y += (mouse_y_ratio - 0.5) * (1 - zoom_factor) * visible_height
        
        self._update_display()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            # Reset zoom and pan on middle mouse button
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._update_display()
        else:
            super().mousePressEvent(event)

    def _update_display(self):
        if self._pixmap:
            # Calculate the scaled pixmap size
            scaled_size = self._pixmap.size() * self._zoom
            scaled_pixmap = self._pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Create a pixmap the size of the label
            label_size = self.size()
            if label_size.width() > 0 and label_size.height() > 0:
                display_pixmap = QPixmap(label_size)
                display_pixmap.fill(Qt.transparent)
                
                # Calculate position to center the scaled pixmap
                x = (label_size.width() - scaled_pixmap.width()) // 2 + self._pan_x
                y = (label_size.height() - scaled_pixmap.height()) // 2 + self._pan_y
                
                # Draw the scaled pixmap onto the display pixmap
                painter = QPainter(display_pixmap)
                painter.drawPixmap(x, y, scaled_pixmap)
                painter.end()
                
                super().setPixmap(display_pixmap)
            else:
                super().setPixmap(scaled_pixmap)