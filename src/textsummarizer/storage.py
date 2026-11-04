import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict


@contextmanager
def atomic_directory(target: Path) -> Iterator[Path]:
    """Yield a scratch directory that replaces `target` only if the block succeeds.

    The block writes into `<name>.tmp` next to `target`. On clean exit an existing
    `target` is renamed to `<name>.old`, the scratch directory is renamed into its place
    and `.old` is removed; if that swap fails the old `target` is restored. If the block
    raises, the scratch directory is removed and `target` is untouched. An interrupted
    run therefore never leaves a half-written `target` for a later run to trust.
    """
    tmp = target.with_name(target.name + ".tmp")
    old = target.with_name(target.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    if old.exists() and target.exists():
        shutil.rmtree(old)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    had_target = target.exists()
    try:
        if had_target:
            target.rename(old)
        tmp.rename(target)
    except BaseException:
        if had_target and old.exists() and not target.exists():
            old.rename(target)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def save_dataset_dict(dataset_dict: "DatasetDict", target: Path) -> None:
    """Save `dataset_dict` to `target` so that `target` is either complete or absent."""
    with atomic_directory(target) as tmp:
        dataset_dict.save_to_disk(str(tmp))
