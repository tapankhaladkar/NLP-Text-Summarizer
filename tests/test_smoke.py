import importlib
import pkgutil

import textsummarizer
from textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


def test_every_module_imports():
    failures = {}
    for module in pkgutil.walk_packages(textsummarizer.__path__, prefix="textsummarizer."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:
            failures[module.name] = repr(exc)
    assert not failures, failures


def test_config_paths_point_at_repo_files():
    assert CONFIG_FILE_PATH.is_file()
    assert PARAMS_FILE_PATH.is_file()
