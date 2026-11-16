import logging
import os
import sys
from pathlib import Path


class _TerminalLogFormatter(logging.Formatter):
    """Compact console formatting that matches the live dashboard."""

    _STYLES = {
        logging.DEBUG: ("·", "38;5;245"),
        logging.INFO: ("›", "38;5;208"),
        logging.WARNING: ("!", "38;5;220"),
        logging.ERROR: ("×", "38;5;203"),
        logging.CRITICAL: ("‼", "1;38;5;203"),
    }

    def __init__(self, color_enabled: bool):
        super().__init__()
        self.color_enabled = color_enabled

    def format(self, record):
        marker, color = self._STYLES.get(
            record.levelno,
            ("›", "38;5;208"),
        )
        prefix = f"{marker} "
        if self.color_enabled:
            prefix = f"\x1b[{color}m{prefix}\x1b[0m"

        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"
        return prefix + message


class Logger:
    """
    Provides a simple interface for application logging.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    @staticmethod
    def configure(
        level: str,
        filename: str = "adaline_debug.log",
        console_enabled: bool = True,
    ):
        log_level = getattr(logging, level.upper(), None)

        if log_level is None:
            raise ValueError(f"Unknown log level: {level}")

        log_path = Path(filename)
        if not log_path.is_absolute():
            log_path = Path("logs") / log_path.name
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(
            fmt=(
                "[%(asctime)s] | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

        handlers = [file_handler]
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(_TerminalLogFormatter(
                color_enabled=(
                    getattr(sys.stdout, "isatty", lambda: False)()
                    and "NO_COLOR" not in os.environ
                    and os.environ.get("TERM", "") != "dumb"
                ),
            ))
            handlers.append(console_handler)

        logging.basicConfig(
            handlers=handlers,
            level=log_level,
            force=True,
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

    def exception(self, message, *args):
        self.logger.exception(message, *args)

    def critical(self, message, *args):
        self.logger.critical(message, *args)
