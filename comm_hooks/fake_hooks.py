# mypy: allow-untyped-defs
import logging
from functools import partial
from typing import Dict

import torch
import torch.distributed as dist

import comm_hooks.default_hooks as default_hooks

from comm_hooks.utils import HookState, _get_allgather_out_list, dtype_bits, tensor_bits

logger = logging.getLogger(__name__)


class SparseState(HookState):
    
    def __init__(self,
        process_group: dist.ProcessGroup,
        compress_ratio: float = 0.01,
        start_compress_iter: int = 2,  
        sparse_type: str = "row",
        random: bool = False,
        use_error_feedback: str = "noef",  # 默认是noef
        random_seed: int = 0,
    ):
        super().__init__(process_group)
        self.total_bit_before_compression = 0
        self.total_bit_after_compression = 0
        self.compress_ratio = compress_ratio

        self.iter = 0
        self.start_compress_iter = start_compress_iter
        self.error_decay = 1.0
        self.large_batch_init = False   # Whether using large batch initialization in EF21
        self.sparse_type = sparse_type
        self.compressor_name = f"{sparse_type}-wise sparsification"

        # random state
        import numpy as np
        self.random = random
        self.rng = torch.Generator()
        self.rng.manual_seed(random_seed)
        
        # error feedback
        self.use_error_feedback = use_error_feedback # EF开关
        self.error_dict: Dict[int, torch.Tensor] = {}
        self.global_error_dict: Dict[int, torch.Tensor] = {}
        self.random_seed = random_seed




def sparse_hook_sync(
    state: SparseState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer


    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    rank = dist.get_rank(group=group_to_use) 

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer() 
    tensors = bucket.gradients() 

    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor, state)
    

    # Apply sparse after `start_compress_iter` iterations.
    device = tensors[0].device  
    dtype = tensors[0].dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index() 
    total_length = input_tensor.shape[0]
    if state.use_error_feedback == "ef14":
        if bucket_index in state.error_dict:  # 误差缓存已存在，即之前已经累积过误差，所以现在把上轮没发送出去的部分 $E_{i-1}$ 加回
            # input_tensor = \nabla F_i + E_{i-1}
            input_tensor.add_(state.error_dict[bucket_index], alpha=1.0)
        else: # 还没有误差缓存（第一次通信该 bucket），create a zero tensor.
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
    elif state.use_error_feedback == "ef21":
        if bucket_index in state.error_dict:
            # input_tensor = \nabla F_i - E_{i-1}
            input_tensor.add_(state.error_dict[bucket_index], alpha=-1.0)
        else:
            # E_0 = \nabla F_0 
            logger.info("A tensor of length %s that represents local/global error is created.", total_length)
            state.error_dict[bucket_index] = torch.clone(input_tensor).detach()
            # allreduce \nabla F_0
            dist.all_reduce(input_tensor, group=group_to_use, async_op=False) # async_op=False 表示会阻塞程序执行，直到 all_reduce 完全完成。
            input_tensor.div_(world_size)
            # \overline{E_0} = \overline{\nabla F_0}
            state.global_error_dict[bucket_index] = torch.clone(input_tensor).detach()
            # reset the full input tensor
            state.maybe_increase_iter(bucket) # 判断是否需要将当前迭代轮数 state.iter 加 1（有时一个迭代（state.iter）可能会对应多个 bucket，而我们只在所有 bucket 都完成一次同步后再加一次迭代数）
            fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
            fut.set_result(input_tensor)
            return fut

    # 压缩:
    # seed 
    if state.random:    # 在每个通信 hook 调用前重新设定随机种子，以便保证使用 RandK 时，在不同 GPU 上压缩出的索引（即indices = torch.randperm(tensor.numel())[:k]这步每个GPU所得的结果一样）是一样的
        seed = torch.randint(0, 1_000_000_000, (1,), generator=state.rng).item() 
                        # 用固定的随机种子让所有 GPU 同步采样！！
                        # state.rng	是构造时固定的 RNG 实例（torch.Generator()），通常在 rank 0 （即第一个GPU）上
                        # torch.randint(..., generator=state.rng)	（在第一个GPU上）用可控 RNG 采样出一个 seed
        torch.manual_seed(seed) # 把这个种子设置成 PyTorch 的全局种子，使得后续 randperm 一样
    
    # 构建压缩器、计算 k 值
    sparsify_func = {
        "row": sparsify_by_row, 
        "column": sparsify_by_column, 
        "tensor": sparsify}[state.sparse_type] 
    sparsify_func = partial(sparsify_func, random=state.random) # 用 partial 固定 random 参数

    cal_k_func = {
        "row": cal_k_by_row, 
        "column": cal_k_by_column, 
        "tensor": cal_k}[state.sparse_type] 
    k_list = [cal_k_func(tensor, state.compress_ratio) for tensor in tensors]

    sum_k = sum(k_list) # k_list中元素求和即为values_memory和indices_memory的长度。
    values_memory = torch.empty(sum_k, dtype=dtype, device=device) # 初始化为0
    indices_memory = torch.empty(sum_k, dtype=torch.int, device=device)
    _, _, bits_sum = compress_tensor_to_memory(tensors, k_list, values_memory, indices_memory, sparsify_func, state.compress_ratio, state.use_error_feedback)


    # 更新一下 state.error_dict
    if state.use_error_feedback == "ef14":
        state.error_dict[bucket_index].copy_(input_tensor)              # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
    elif state.use_error_feedback == "ef21":
        # if bucket.is_last():
        #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], Error Norm: {state.global_error_dict[bucket_index].norm().item()}")
        # if bucket.is_last():
        #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], global error: {state.global_error_dict[bucket_index][-5:]}")
        #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], compressed diff: {input_tensor[-5:]}, local error: {state.error_dict[bucket_index][-5:]}")
        state.error_dict[bucket_index].add_(input_tensor, alpha=state.error_decay)    # E_i = E_{i-1} + C[\nabla F_i - E_{i-1}]
        # if bucket.is_last():
        #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], updated local error: {state.error_dict[bucket_index][-5:]}")

    # Allreduce 或 Allgather ，并解压回原梯度张量（聚合或直接替换）
    if state.random: 
        # Allreduce the values
        state.comm_bits_this_round += 2 *(world_size -1) * bits_sum
        dist.all_reduce(values_memory, group=group_to_use, async_op=False)
        values_memory.div_(world_size)

        # Zero the input tensor.
        input_tensor.zero_()     
        decompress_memory_to_tensor_and_aggregate(tensors, k_list, values_memory, indices_memory, aggregate=False)
    else:    
        # Allgather the values and indices
        values_memory_allgather = _get_allgather_out_list(values_memory, world_size)
        indices_memory_allgather = _get_allgather_out_list(indices_memory, world_size)
        
        state.comm_bits_this_round += (world_size -1) * world_size * bits_sum
        dist.all_gather(values_memory_allgather, values_memory, group=group_to_use, async_op=False)
        dist.all_gather(indices_memory_allgather, indices_memory, group=group_to_use, async_op=False)
        
        # Zero the input tensor.
        input_tensor.zero_()     
        for values_memory, indices_memory in zip(values_memory_allgather, indices_memory_allgather):
            decompress_memory_to_tensor_and_aggregate(tensors, k_list, values_memory, indices_memory, aggregate=True)
        input_tensor.div_(world_size)
    
    # 更新 global_error_dict
    if state.use_error_feedback == "ef21":
        state.global_error_dict[bucket_index].add_(input_tensor, alpha=state.error_decay) # \overline{E_i} = \overline{E_{i-1}} + \overline{C[\nabla F_i - E_{i-1}]}
        input_tensor.copy_(state.global_error_dict[bucket_index])

    state.maybe_increase_iter(bucket)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    # fut.set_result(input_tensor / world_size)  原代码有typo
    fut.set_result(input_tensor)
    return fut



class GroupTopKState(HookState):
    def __init__(self, 
        process_group: dist.ProcessGroup, 
        r: int = 4, 
        compress_ratio: float = 0.08, 
        start_compress_iter: int = 2, 
        use_error_feedback="noef", 
        seed=0, 
        error_decay=1.0,
        gradual_compression=True,  # 新增：渐进式压缩
        warmup_iters=100          # 新增：压缩预热步数
    ):
        super().__init__(process_group)
        self.r = r
        self.base_compress_ratio = compress_ratio  # 保存原始压缩比
        self.compress_ratio = compress_ratio
        self.gradual_compression = gradual_compression
        self.warmup_iters = start_compress_iter + warmup_iters

        self.iter = 0
        self.start_compress_iter = start_compress_iter

        self.seed = seed

        # error feedback
        self.use_error_feedback = use_error_feedback # EF开关
        self.error_dict: Dict[int, torch.Tensor] = {}
        self.global_error_dict: Dict[int, torch.Tensor] = {}
        self.error_decay = error_decay  

        # 新增：压缩切换的平滑处理
        self.compression_started = False

        # 设置统一的 RNG，用于 sample 每一轮的投影矩阵 seed
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
    
    def get_current_compress_ratio(self):
        """计算当前的压缩比，支持渐进式压缩"""
        if not self.gradual_compression or not self.compression_started:
            return self.base_compress_ratio
            
        # 渐进式压缩：从较高的压缩比逐渐降低到目标压缩比
        compress_progress = self.iter - self.start_compress_iter
        if compress_progress < self.warmup_iters:
            # 从 0.8 渐进到 base_compress_ratio
            start_ratio = 0.8
            progress = compress_progress / self.warmup_iters
            current_ratio = start_ratio - (start_ratio - self.base_compress_ratio) * progress
            return max(current_ratio, self.base_compress_ratio)
        else:
            return self.base_compress_ratio



def fake_group_topk_hook(
    state: GroupTopKState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer

    # check if the key is ready in precond adam
    state.maybe_accumulate_momentum_on_bucket(bucket)

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    rank = dist.get_rank(group=group_to_use) # 获取 当前进程 在 group_to_use 这个通信组中的 rank。（process_group 就是指的只有一个通信组）

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()  
    tensors = bucket.gradients() 
 
    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor, state)
    
    # 新增：压缩切换的平滑处理
    if not state.compression_started:
        state.compression_started = True
        logger.info(f"Starting compression at iteration {state.iter} with gradual compression enabled: {state.gradual_compression}")
        if state.use_error_feedback in ["ef14", "ef21"]:
            logger.info("Initializing error feedback mechanism for smooth compression transition")
    
    # group_topk 压缩器压缩 -----------------------------------------------------------------------------------
    device = tensors[0].device  
    dtype = tensors[0].dtype

    # 获取当前的压缩比（支持渐进式压缩）
    current_compress_ratio = state.get_current_compress_ratio()

    # allreduce
    dist.all_reduce(input_tensor, group=group_to_use, async_op=False) # async_op=False 表示会阻塞程序执行，直到 all_reduce 完全完成。
    
    bucket_index = bucket.index() 
    total_length = input_tensor.shape[0]
    if state.use_error_feedback == "ef14":
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index], alpha=1.0) # input_tensor = \nabla F_i + E_{i-1}
        else:
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
    
    state.error_dict[bucket_index].copy_(input_tensor)  # E_i = \nabla F_i + E_{i-1}

    for tensor in tensors:
        if len(tensor.shape) == 2:
            m, n = tensor.shape # [m, n]
            k = max(1, int(m * current_compress_ratio))
            V = torch.empty(n, state.r, dtype=dtype, device=device) # [n, r]
            V.normal_(generator=state.rng)
            sigma = torch.norm(tensor @ V, dim=1).abs() # [m, 1]
            zero_indices = torch.argsort(sigma, descending=True)[k:]
            tensor[zero_indices] = 0.0  # 将非top k的元素置为0
        elif len(tensor.shape) > 2:
            raise NotImplementedError("Fake Group TopK compression only supports 2D tensors.")
    
    state.error_dict[bucket_index].add_(input_tensor, alpha=-1.0)  # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]

    state.maybe_increase_iter(bucket)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)

    return fut




