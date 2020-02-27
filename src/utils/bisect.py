from typing import *  # pylint: disable=W0401,W0614

# pylint: disable=C0103
def bisect_right(
    a: List[Any], x: Any, key: Callable[[Any], Any], lo: int = 0, hi: int = None
) -> int:
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < key(a[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


# pylint: disable=C0103
def bisect_left(
    a: List[Any], x: Any, key: Callable[[Any], Any], lo: int = 0, hi: int = None
) -> int:
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(a[mid]) < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# pylint: disable=C0103
def index_lt(
    a: List[Any], x: Any, key: Callable[[Any], Any], lo: int = 0, hi: int = None
) -> int:
    """Locate the index of the rightmost value less than x"""
    i = bisect_left(a, x, key, lo, hi)
    if i:
        return i - 1
    raise ValueError


# pylint: disable=C0103
def index_le(
    a: List[Any], x: Any, key: Callable[[Any], Any], lo: int = 0, hi: int = None
) -> int:
    """Locate the index of the rightmost value less than or equal to x"""
    i = bisect_right(a, x, key, lo, hi)
    if i:
        return i - 1
    raise ValueError


# pylint: disable=C0103
def index_gt(
    a: List[Any], x: Any, key: Callable[[Any], Any], lo: int = 0, hi: int = None
) -> int:
    """Locate the index of the leftmost value greater than x"""
    i = bisect_right(a, x, key, lo, hi)
    if i != len(a):
        return i
    raise ValueError


# pylint: disable=C0103
def index_ge(
    a: List[Any], x: Any, key: Callable[[Any], Any], lo: int = 0, hi: int = None
) -> int:
    """Locate the index of the leftmost item greater than or equal to x"""
    i = bisect_left(a, x, key, lo, hi)
    if i != len(a):
        return i
    raise ValueError
