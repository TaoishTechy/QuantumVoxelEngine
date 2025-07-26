# logger.py
# Centralized, thread-safe, and production-grade logging configuration.

import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any

# --- Constants ---
LOG_FILE = "game.log"
MAX_LOG_SIZE_MB = 5
LOG_BACKUP_COUNT = 3

class EnhancedFormatter(logging.Formatter):
    """A custom formatter to add more context to log messages."""
    def format(self, record: logging.LogRecord) -> str:
        # You can add more context here, like module or function name if available
        record.threadName = record.threadName or 'MainThread'
        return super().format(record)

def handle_exception(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
    """
    Global exception hook to log any unhandled exceptions before the program exits.
    This is critical for debugging crashes.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log KeyboardInterrupt, let it exit cleanly
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Unhandled exception caught:", exc_info=(exc_type, exc_value, exc_traceback))

def setup_logger() -> logging.Logger:
    """
    Configures and returns a thread-safe logger for the game engine.
    
    This setup uses a queue to handle log records from multiple threads
    without blocking, writing to both the console and a rotating log file.
    """
    log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)
    
    # --- Handlers ---
    # Console Handler for immediate feedback
    console_handler = logging.StreamHandler()
    
    # Rotating File Handler for persistent logs
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024, 
        backupCount=LOG_BACKUP_COUNT
    )
    
    # --- Formatter ---
    formatter = EnhancedFormatter(
        '%(asctime)s [%(levelname)-8s] (%(threadName)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # --- Listener ---
    # The listener pulls logs from the queue and sends them to the actual handlers
    listener = QueueListener(log_queue, console_handler, file_handler)
    listener.start()

    # --- Root Logger Configuration ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler) # All logs go through the queue

    # --- Set the global exception hook ---
    sys.excepthook = handle_exception
    
    return root_logger

# --- Global Logger Instance ---
logger = setup_logger()
