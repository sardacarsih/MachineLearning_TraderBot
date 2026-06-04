"""
============================================
Structured Logging System
============================================
Centralized logging with file rotation, console output, and trade logging.
Uses loguru for enhanced logging capabilities.
"""

import os
import sys
from datetime import datetime
from loguru import logger


class TradingLogger:
    """
    Professional logging system for the ML trading bot.
    Provides separate log files for different concerns:
    - main.log: General application logs
    - trades.log: Trade execution logs
    - signals.log: ML signal generation logs
    - errors.log: Error-only logs
    """

    _initialized = False
    _console_handler_id = None
    _file_handler_ids = []

    @classmethod
    def setup(
        cls,
        log_dir: str = None,
        debug: bool = False,
        force_reinit: bool = False,
        quiet_console: bool = False,
    ):
        """
        Initialize the logging system.

        Args:
            log_dir: Directory for log files. Defaults to project logs/ dir.
            debug: Enable debug-level logging.
            force_reinit: Force re-initialization (e.g. after symbol-specific paths are set).
            quiet_console: Disable stdout logging so a Rich dashboard can own the terminal.
        """
        if cls._initialized and not force_reinit:
            console_is_quiet = cls._console_handler_id is None
            if quiet_console == console_is_quiet:
                return
            force_reinit = True

        if log_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base, "logs")

        os.makedirs(log_dir, exist_ok=True)

        # Remove default handler
        logger.remove()
        cls._console_handler_id = None
        cls._file_handler_ids = []

        # Date stamp for log files
        date_stamp = datetime.now().strftime("%Y%m%d")

        # Console handler - colorized output
        log_level = "DEBUG" if debug else "INFO"
        if not quiet_console:
            cls._console_handler_id = logger.add(
                sys.stdout,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                ),
                level=log_level,
                colorize=True,
            )

        # Main application log
        cls._file_handler_ids.append(logger.add(
            os.path.join(log_dir, f"main_{date_stamp}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG" if debug else "INFO",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,  # Thread-safe
        ))

        # Trade execution log
        cls._file_handler_ids.append(logger.add(
            os.path.join(log_dir, f"trades_{date_stamp}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            level="INFO",
            rotation="5 MB",
            retention="90 days",
            filter=lambda record: "trade" in record["extra"].get("category", ""),
            enqueue=True,
        ))

        # Signal generation log
        cls._file_handler_ids.append(logger.add(
            os.path.join(log_dir, f"signals_{date_stamp}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            level="INFO",
            rotation="5 MB",
            retention="30 days",
            filter=lambda record: "signal" in record["extra"].get("category", ""),
            enqueue=True,
        ))

        # Error-only log
        cls._file_handler_ids.append(logger.add(
            os.path.join(log_dir, f"errors_{date_stamp}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            level="ERROR",
            rotation="5 MB",
            retention="90 days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        ))

        cls._initialized = True
        logger.info("Logging system initialized")

    @staticmethod
    def get_logger():
        """Get the logger instance."""
        return logger

    @staticmethod
    def trade_log(message: str, **kwargs):
        """Log a trade event."""
        logger.bind(category="trade").info(message, **kwargs)

    @staticmethod
    def signal_log(message: str, **kwargs):
        """Log a signal event."""
        logger.bind(category="signal").info(message, **kwargs)


def setup_logging(log_dir: str = None, debug: bool = False, quiet_console: bool = False):
    """Convenience function to initialize logging."""
    # If log_dir is explicitly provided, force re-init to use new directory
    force = log_dir is not None
    TradingLogger.setup(log_dir=log_dir, debug=debug, force_reinit=force, quiet_console=quiet_console)
    return logger


def get_logger():
    """Get the configured logger instance."""
    if not TradingLogger._initialized:
        TradingLogger.setup()
    return logger
