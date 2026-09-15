import sys
import os
import logging
from PySide6.QtWidgets import QApplication

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Get the src directory
project_root = os.path.dirname(src_dir)  # Get the project root
sys.path.insert(0, project_root)

# We need to import directly from files in the same directory structure
from ui.main_window import DNAOApplication
from utils.logger import setup_logger

def main():
    # Set up logging
    logger = setup_logger(log_level=logging.DEBUG)  # Change to DEBUG level to catch more issues
    logger.info("Starting DNAO Analysis Tool")

    # Initialize the application
    app = QApplication(sys.argv)
    window = DNAOApplication()
    window.setWindowTitle("DNAO Analysis Tool")

    # Set application style sheet for modern look
    app.setStyle("Fusion")
    
    # Show window and run the application
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
