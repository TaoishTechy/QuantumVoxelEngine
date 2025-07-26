# logger.py
# Centralized logging configuration for the entire application.

import logging
import queue
from logging.handlers import QueueHandler, QueueListener

def setup_logger() -> logging.Logger:
    """Configures and returns a thread-safe logger for the game engine."""
    log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    listener = QueueListener(log_queue, handler)
    listener.start()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    return root_logger

logger = setup_logger()
