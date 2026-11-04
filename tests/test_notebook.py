import json
import re
from pathlib import Path

from typer.testing import CliRunner

from textsummarizer.cli import app

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "colab_train.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def test_notebook_is_clean_and_holds_no_secrets():
    text = NOTEBOOK.read_text()
    assert json.loads(text)["nbformat"] == 4
    assert all(c.get("outputs", []) == [] for c in cells() if c["cell_type"] == "code")
    assert not re.search(r"hf_[A-Za-z0-9]{10,}", text)


def test_notebook_calls_only_commands_the_cli_has():
    listed = CliRunner().invoke(app, ["--help"]).output
    used = set()
    for cell in cells():
        for line in "".join(cell["source"]).splitlines():
            match = re.match(r"!textsummarizer\s+(?:--\S+\s+\S+\s+)*([a-z]+)", line)
            if match:
                used.add(match.group(1))
    assert {"all", "train", "evaluate", "publish"} <= used
    assert all(command in listed for command in used)
