import sys
from typing import *

from tqdm import tqdm


class Progress(tqdm):
    def __init__(self, *args: Any, **kwds: Dict[str, Any]) -> None:
        l_bar = "{desc}: {n_fmt}/{total_fmt} ({percentage:.0f}%)"
        r_bar = "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        no_bar_format = f"{l_bar} {r_bar}"

        kwds["bar_format"] = kwds.get("bar_format", no_bar_format)
        kwds["file"] = kwds.get("file", sys.stdout)
        super(Progress, self).__init__(*args, **kwds)


class ByteProgress(Progress):
    def __init__(self, *args: Any, **kwds: Dict[str, Any]) -> None:
        kwds["unit"] = "B"
        kwds["unit_scale"] = True
        kwds["miniters"] = kwds.get("miniters", 1)
        super(ByteProgress, self).__init__(*args, **kwds)

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
