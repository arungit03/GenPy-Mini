"""Logging setup with idempotent console and optional file handlers."""

import logging as _logging
from pathlib import Path


def setup_logger(
    name: str = "genpy",
    level: int = _logging.INFO,
    log_file: str | Path | None = None,
) -> _logging.Logger:
    """Configure and return a logger without adding duplicate handlers."""
    logger = _logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = _logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not any(getattr(handler, "_genpy_console", False) for handler in logger.handlers):
        console = _logging.StreamHandler()
        console.setFormatter(formatter)
        console._genpy_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_file is not None:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = any(
            getattr(handler, "_genpy_log_file", None) == str(target.resolve())
            for handler in logger.handlers
        )
        if not existing:
            file_handler = _logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler._genpy_log_file = str(target.resolve())  # type: ignore[attr-defined]
            logger.addHandler(file_handler)
    return logger
