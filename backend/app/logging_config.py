"""Centralized logging setup. Level via LOG_LEVEL (default INFO); JSON-ish single
line format so logs are greppable in CloudWatch/Loki. Call setup_logging() once at
startup (main.py). Modules then use `logging.getLogger(__name__)`."""
import logging
import os

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    # Uvicorn manages its own access logger; keep it but align the app loggers.
    logging.getLogger("uvicorn.error").setLevel(getattr(logging, level, logging.INFO))
    _CONFIGURED = True
