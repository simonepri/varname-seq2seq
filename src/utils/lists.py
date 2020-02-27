import itertools
from typing import *  # pylint: disable=W0401,W0614


def split_by(
    sequence: List[Any], length: int
) -> Generator[List[Any], None, None]:
    iterable = iter(sequence)

    while True:
        res = list(itertools.islice(iterable, length))
        if len(res) == 0:
            break
        yield res
