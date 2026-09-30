import logging

class Logger:
    """
    Provides a simple interface for application logging.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    @staticmethod
    def configure(level: str):
        log_level = getattr(logging, level.upper(), None)

        if log_level is None:
            raise ValueError(f"Unknown log level: {level}")

        logging.basicConfig(
            level=log_level,
            format=(
                "[%(asctime)s] | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    def debug(self, message, *args):
        self.logger.debug(message, *args)

    def info(self, message, *args):
        self.logger.info(message, *args)

    def warning(self, message, *args):
        self.logger.warning(message, *args)

    def error(self, message, *args):
        self.logger.error(message, *args)

    def critical(self, message, *args):
        self.logger.critical(message, *args)