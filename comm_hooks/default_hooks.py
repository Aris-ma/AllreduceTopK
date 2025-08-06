# mypy: allow-untyped-defs
from typing import Any, Callable, cast, Tuple

import torch
import torch.distributed as dist

from comm_hooks.utils import HookState, dtype_bits, tensor_bits

__all__ = [     
    "allreduce_hook",
    "my_allreduce_hook",
]


def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor, hook_state: None 
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size() 

    # Apply the division first to avoid overflow, especially for FP16.（先除以进程数再进行 all_reduce 可以避免一些精度问题，特别是 float16（容易溢出））
    tensor.div_(group_to_use.size())

    if hook_state is not None:
        
        # AllReduce 通信量计算
        comm_bits = 2 * (world_size - 1) * tensor_bits(tensor)
        
        # 累加到全局统计量
        hook_state.comm_bits_this_round += comm_bits

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0]) # fut.value() 返回的是一个长度为 1 的张量列表（即 [tensor]），所以取下标 0
    )


def my_allreduce_hook(state: HookState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    state.maybe_accumulate_momentum_on_bucket(bucket)
    state.maybe_increase_iter(bucket)
    process_group = state.process_group
    return _allreduce_fut(process_group, bucket.buffer(),state)


def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    return _allreduce_fut(process_group, bucket.buffer())

