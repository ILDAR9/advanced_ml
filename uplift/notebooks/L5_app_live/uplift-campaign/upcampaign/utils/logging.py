import logging


def create_logger(logger_name: str, filename: str = None, logging_level: int = logging.DEBUG) -> logging.Logger:
    """Creates and returns logger with given filename."""
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%F %T"
    )
    console_logging_handler = logging.StreamHandler()
    console_logging_handler.setFormatter(formatter)
    if filename:
        file_logging_handler = logging.FileHandler(filename)
        file_logging_handler.setFormatter(formatter)
        logger.addHandler(file_logging_handler)
    
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(console_logging_handler)
    logger.setLevel(logging_level)
    return logger


class SilentLogger:
    def info(self, *args, **kwargs):
        pass

    
SILENT_LOGGER = SilentLogger()