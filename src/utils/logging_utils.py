import logging
import sys
from datetime import datetime
import os


def setup_logger(name: str, 
                log_file: str = None,
                level: int = logging.INFO,
                console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
