import logging

class Logger:

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    @staticmethod
    def configure():
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "[%(asctime)s] | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S"
        )

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