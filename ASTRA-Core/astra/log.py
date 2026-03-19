import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured professional logger for ASTRA-Core.
    
    Args:
        name: The name of the module (e.g., __name__)
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if get_logger is called multiple times
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Professional standard format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Default level is INFO
        logger.setLevel(logging.INFO)
        
        # Prevent propagation to the root logger to avoid double-printing
        logger.propagate = False
        
    return logger
