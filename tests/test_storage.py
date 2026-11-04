from pathlib import Path

import pytest

from textsummarizer.storage import atomic_directory


def old_target(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.txt").write_text("old")
    return target


def write_new(tmp):
    tmp.mkdir()
    (tmp / "new.txt").write_text("new")


def test_atomic_directory_publishes_on_success(tmp_path):
    target = tmp_path / "out"
    with atomic_directory(target) as tmp:
        write_new(tmp)
    assert (target / "new.txt").read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_atomic_directory_replaces_an_existing_target_without_leftovers(tmp_path):
    target = old_target(tmp_path)
    with atomic_directory(target) as tmp:
        write_new(tmp)
    assert [p.name for p in target.iterdir()] == ["new.txt"]
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_failure_inside_the_block_keeps_the_old_target(tmp_path):
    target = old_target(tmp_path)
    with pytest.raises(RuntimeError):
        with atomic_directory(target) as tmp:
            write_new(tmp)
            raise RuntimeError("boom")
    assert (target / "old.txt").read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_failed_swap_restores_the_old_target(tmp_path, monkeypatch):
    target = old_target(tmp_path)
    real_rename = Path.rename

    def flaky(self, destination):
        if self.name.endswith(".tmp"):
            raise OSError("rename failed")
        return real_rename(self, destination)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(OSError):
        with atomic_directory(target) as tmp:
            write_new(tmp)
    assert (target / "old.txt").read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]
