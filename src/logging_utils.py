"""logging_utils.py

Shared logging configuration for the sustainability-stylist pipeline.

Every pipeline script should call ``get_logger(__name__)`` instead of using
``print()`` directly. This gives consistent timestamps, log levels, and (once
enabled) a persistent run log for the dissertation appendix / reproducibility
evidence.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger(log_dir: Path | None = None) -> None:
    """Configure the root logger once per process."""
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline_run.log")
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    _configured = True


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Return a module-level logger with consistent formatting.

    Parameters:
    name:
        Pass ``__name__`` from the calling module.
    log_dir:
        Optional directory to also write a persistent ``pipeline_run.log``
        file. Pass ``OUTPUTS`` from ``config.py`` if you want every pipeline
        run logged to disk for reproducibility evidence.
    """
    _configure_root_logger(log_dir)
    return logging.getLogger(name)