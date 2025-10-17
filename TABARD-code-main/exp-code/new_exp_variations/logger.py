# ── custom_logging.py ──────────────────────────────────────────────────────

import logging
import os

def setup_custom_logger(
    logfile_name: str = "model-batch_prediction_job.log",
    level: int = logging.INFO,
    log_dir: str = None
) -> logging.Logger:
    """
    Configure and return a root logger that writes to `logfile_name` with the format:
        %(asctime)s - %(levelname)s - %(message)s
    
    If `log_dir` is given, it will be created (if necessary) and the log file
    will be placed under that directory. Otherwise, the file will be created
    in the current working directory.
    
    Usage:
        import custom_logging
        logger = custom_logging.setup_custom_logger()
        logger.info("…")
    
    This function ensures that multiple calls to setup_custom_logger()
    do not add duplicate handlers.
    """
    # Determine the absolute path for the log file
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logfile_path = os.path.join(log_dir, logfile_name)
    else:
        logfile_path = logfile_name

    # Get the root logger (or create it)
    logger = logging.getLogger()
    logger.setLevel(level)

    # If there are already handlers attached, do not add another FileHandler
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(logfile_path) 
               for h in logger.handlers):
        # Create a FileHandler
        file_handler = logging.FileHandler(logfile_path)
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
