import os
from typing import *  # pylint: disable=W0401,W0614
from typing import Pattern

from utils.progress import Progress
from utils.strings import truncate
from utils.lists import split_by


def walk_files(
    root_path: str,
    pattern: Optional[Pattern],
    progress: bool = False,
    batch: int = 0,
) -> Generator[Tuple[str, List[str]], None, None]:
    if not progress:
        for path, _, files in os.walk(root_path):
            if pattern is not None:
                files = list(filter(pattern.match, files))
            if len(files) == 0:
                continue
            if batch < 1:
                yield path, files
            for files_batch in split_by(files, batch):
                yield path, files_batch
        return

    num_files = sum(len(files) for _, files in walk_files(root_path, pattern))
    with Progress(total=num_files) as pbar:
        for path, files in walk_files(root_path, pattern=pattern, batch=batch):
            pbar.set_description(truncate(path, -32, "…").rjust(32))
            yield path, files
            pbar.update(len(files))


def rebase_path(
    input_base: str, output_base: str, path: str,
) -> Tuple[str, str]:
    out_path = output_base
    rel_path = os.path.relpath(path, input_base)
    if rel_path != ".":
        out_path = os.path.join(out_path, rel_path)
    return out_path
