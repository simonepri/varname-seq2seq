import itertools
from typing import *


def split_by(
    sequence: List[Any], length: int
) -> Generator[List[Any], None, None]:
    iterable = iter(sequence)

    while True:
        res = list(itertools.islice(iterable, length))
        if len(res) == 0:
            break
        yield res
