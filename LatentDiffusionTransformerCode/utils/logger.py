"""
Logging utility.
"""
import logging
import sys

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Sets up a unified logger.

    Args:
        name (str): The name of the logger.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding multiple handlers if the logger was already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

