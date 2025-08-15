# bot/logger_config.py
import logging
import sys
from .config import LOG_LEVEL, LOG_FILE

def setup_logger():
    """Configures the logger based on settings in config.py."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE)
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger when this module is imported
logger = setup_logger()