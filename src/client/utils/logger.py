import os
import logging
import sys
from datetime import datetime

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Set up and configure a logger
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, logs to console only.
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("DNAO")
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # If log file specified, add file handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # If no log file is specified, use a default location
        log_dir = os.path.join(os.path.expanduser("~"), ".dnao_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d")
        default_log_file = os.path.join(log_dir, f"dnao_{date_str}.log")
        
        file_handler = logging.FileHandler(default_log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
