import logging

import pytest


@pytest.fixture(autouse=True)
def reset_package_logging():
    """Remove handlers that configure_logging installed, so tests don't leak them."""
    yield
    package_logger = logging.getLogger("textsummarizer")
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
        handler.close()
