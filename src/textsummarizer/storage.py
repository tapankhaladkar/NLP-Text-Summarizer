import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict


def save_dataset_dict(dataset_dict: "DatasetDict", target: Path) -> None:
    """Save `dataset_dict` to `target` so that `target` is either complete or absent.

    Writes to a sibling `<name>.tmp` directory first and renames it into place, so
    an interrupted save never leaves a half-written `target` for a later run to
    trust. An existing `target` is replaced only after the new data is fully written.
    """
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        dataset_dict.save_to_disk(str(tmp))
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
