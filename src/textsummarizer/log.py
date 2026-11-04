import logging
import sys
from pathlib import Path

LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(message)s]"
_MANAGED = "_textsummarizer_managed"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Attach handlers to the "textsummarizer" logger. Call from entrypoints only.

    Handlers installed by an earlier call are replaced, so repeated calls never
    duplicate output. Nothing is written to disk unless `log_file` is given.
    """
    logger = logging.getLogger("textsummarizer")
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED, False):
            logger.removeHandler(handler)
            handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    formatter = logging.Formatter(LOG_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _MANAGED, True)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    # Libraries such as transformers can install a handler on the root logger as a side
    # effect; without this every line would be printed a second time in their format.
    logger.propagate = False
