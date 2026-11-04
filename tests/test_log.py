import logging

from textsummarizer.log import configure_logging

PACKAGE_LOGGER = logging.getLogger("textsummarizer")


def test_repeated_calls_do_not_duplicate_handlers():
    configure_logging()
    configure_logging()
    assert len(PACKAGE_LOGGER.handlers) == 1


def test_log_file_receives_messages(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    configure_logging(log_file=log_file)
    logging.getLogger("textsummarizer.test").info("hello pipeline")
    assert "hello pipeline" in log_file.read_text()
    assert len(PACKAGE_LOGGER.handlers) == 2


def test_nothing_is_written_without_a_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configure_logging()
    logging.getLogger("textsummarizer.test").info("stdout only")
    assert list(tmp_path.iterdir()) == []
