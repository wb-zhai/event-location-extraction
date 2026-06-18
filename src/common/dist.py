import os
from typing import Any, List, Optional, Sequence, TypeVar, cast

import torch
import torch.distributed as torch_dist
from torch._C._distributed_c10d import ProcessGroup

from src.common.log import get_logger

logger = get_logger(__name__)

TObj = TypeVar("TObj")


def get_rank(group: ProcessGroup | None = None) -> int:
    """Get the rank of the current process.

    .. seealso:: :func:`torch.distributed.get_rank`

    Returns:
        int: The rank of the current process.
    """
    if torch_dist.is_available() and torch_dist.is_initialized():
        return torch_dist.get_rank(group)
    return int(os.getenv("LOCAL_RANK", "0"))
