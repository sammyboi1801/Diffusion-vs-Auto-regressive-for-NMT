import logging
import sys

def setup_logger(name, level=logging.INFO):
    """
    Sets up a simple logger that prints to console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler if not already added
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger