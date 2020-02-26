from typing import *

import torch


def find_first(
    tensor: torch.Tensor, value: torch.dtype, axis: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    nonz = tensor == value
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)
