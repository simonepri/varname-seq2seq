import os
from typing import *
from typing import Pattern

from tqdm import tqdm

from utils.strings import truncate


def walk_files(
    path: str, pattern: Optional[Pattern], progress=False
) -> Generator[Tuple[str, List[str]], None, None]:
    if not progress:
        for path, _, files in os.walk(path):
            if pattern is not None:
                files = list(filter(lambda f: pattern.match(f), files))
            if len(files) > 0:
                yield path, files
        return

    num_files = sum(len(files) for _, files in walk_files(path, pattern))
    with tqdm(total=num_files) as pbar:
        for path, files in walk_files(path, pattern):
            pbar.set_description(truncate(path, -32, "…").rjust(32))
            yield path, files
            pbar.update(len(files))


def split_file_path(path: str) -> Tuple[str, str]:
    dir = os.path.dirname(path)
    return dir, os.path.relpath(path, dir)
