import logging
import os
from datetime import datetime


def setup_logger(name=None, log_dir="logs", log_level="INFO"):
    """
    Sets up a logger that writes to both a file and the console.

    Args:
        name (str): Name of the logger. Use None for the root logger.
        log_dir (str): Directory to save log files.
        log_level (str): Logging level (e.g., "INFO", "DEBUG").

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    log_filepath = os.path.join(log_dir, log_filename)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent duplicate handlers if logger is already configured
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
