# logger.py
# Centralized logging configuration for the entire application.

import logging

def setup_logger():
    """Configures and returns a logger for the game engine."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("QuantumVoxelEngine")

logger = setup_logger()
