# mypy: allow-untyped-defs
import logging
from functools import partial
from typing import Dict

import torch
import torch.distributed as dist

import comm_hooks.default_hooks as default_hooks

from comm_hooks.utils import HookState, _get_allgather_out_list, dtype_bits, tensor_bits

logger = logging.getLogger(__name__)


def sparsify(tensor, compress_ratio, random=False): 
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    if random: # random=True即使用randk
        indices = torch.randperm(tensor.numel(), device=tensor.device)[:k] # indices 形状(k,)
        indices=indices.to(torch.int32)
        values = tensor[indices].clone()
        # values = torch.gather(tensor, 0, indices) 
        comm_bits = tensor_bits(values)
    else:
        _, indices = torch.topk(tensor.abs(), k, sorted=False)  # 使用topk, indices 形状(k,)
        indices=indices.to(torch.int32)
        values = tensor[indices].clone()
        # values = torch.gather(tensor, 0, indices) 
        comm_bits = tensor_bits(values) + tensor_bits(indices)
        # print(f"indices.dtype={indices.dtype}")
        # print(f"values.dtype={values.dtype}")

    return values, indices, comm_bits 

def sparsify_by_row(tensor, compress_ratio, random=False): # 每一行独立做 TopK 或 RandK

    if len(tensor.shape) == 1: 
        return sparsify(tensor, compress_ratio)
    else:
        num_rows, num_cols = tensor.shape
        k = max(1, int(num_cols * compress_ratio))

        if random:
            # hope that the random perm for each row is different, shape [num_rows, k]
            indices = torch.stack([torch.randperm(num_cols, device=tensor.device)[:k] for _ in range(num_rows)], dim=0) # 每行的indices不一样。
        else:
            # get the topk values and indices for each row, shape [num_rows, k]
            _, indices = torch.topk(tensor.abs(), k, dim=1, sorted=False) 
        # adjust indices to account for the row offset
        indices = indices + torch.arange(num_rows, device=tensor.device).unsqueeze(1) * num_cols
        values = torch.gather(tensor.flatten(), 0, indices.flatten())

        return values, torch.flatten(indices)
    
def sparsify_by_column(tensor, compress_ratio, random=False):
    
    if len(tensor.shape) == 1:
        return sparsify(tensor, compress_ratio)
    else:
        num_rows, num_cols = tensor.shape
        k = max(1, int(num_rows * compress_ratio))

        if random:
            # hope that the random perm for each column is different, shape [k, num_cols]
            indices = torch.stack([torch.randperm(num_rows, device=tensor.device)[:k] for _ in range(num_cols)], dim=1)
        else:
            # get the topk values and indices for each row, shape [k, num_cols]
            _, indices = torch.topk(tensor.abs(), k, dim=0, sorted=False)

        # adjust indices to account for the row offset
        indices = indices * num_cols + torch.arange(num_cols, device=tensor.device).unsqueeze(0)
        values = torch.gather(tensor.flatten(), 0, indices.flatten())

        return torch.flatten(values), torch.flatten(indices)

def cal_k(tensor, compress_ratio): 
    return max(1, int(tensor.numel() * compress_ratio))

def cal_k_by_row(tensor, compress_ratio):
    if len(tensor.shape) == 1:
        return cal_k(tensor, compress_ratio)
    num_rows, num_cols = tensor.shape
    return max(1, int(num_cols * compress_ratio)) * num_rows

def cal_k_by_column(tensor, compress_ratio):
    if len(tensor.shape) == 1:
        return cal_k(tensor, compress_ratio)
    num_rows, num_cols = tensor.shape
    return max(1, int(num_rows * compress_ratio)) * num_cols
    
def compress_tensor_to_memory(tensors, k_list, values_memory, indices_memory, sparsify_func, compress_ratio, use_error_feedback):
    offset = 0
    bits_sum = 0
    for tensor, k in zip(tensors, k_list): 
        values, indices, comm_bits = sparsify_func(tensor, compress_ratio)
        values_memory[offset:offset+k] = values
        indices_memory[offset:offset+k] = indices.to(torch.int) 
        offset += k
        bits_sum += comm_bits  # bits_sum是一个GPU的一个bucket的通信量。


        if use_error_feedback == "ef14":
            # input_tensor = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
            tensor.view(-1)[indices] = 0  # C内的部分设成0，处理后的 tensor = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
        elif use_error_feedback == "ef21":
            tensor.zero_()
            # input_tensor = C[\nabla F_i - E_{i-1}]
            tensor.view(-1)[indices] = values # 处理后的 tensor = C[\nabla F_i - E_{i-1}]，不会改变 tensor 本身的形状
    return values_memory, indices_memory, bits_sum

def decompress_memory_to_tensor_and_aggregate(tensors, k_list, values_memory, indices_memory, aggregate=True):
    # add the decompressed values to the tensors
    offset = 0
    for tensor, k in zip(tensors, k_list):
        values = values_memory[offset:offset+k]
        indices = indices_memory[offset:offset+k]
        # avoid creating a new tensor for the view
        flattened_tensor = tensor.view(-1)
        if aggregate:
            flattened_tensor[indices.to(torch.int64)] += values
        else:
            flattened_tensor[indices.to(torch.int64)] = values
        offset += k
    return None

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

        # 修复1: 预分配AllGather缓存，避免重复创建张量
        self.allgather_cache = {}
        
        # 修复2: 内存清理计数器
        self.memory_cleanup_interval = 100  # 每100个iteration清理一次内存
        
        # 修复3: 预分配随机种子，减少运行时开销
        if self.random:
            # 预生成足够多的随机种子
            self.precomputed_seeds = torch.randint(0, 1_000_000_000, (10000,), generator=self.rng).tolist()
            self.seed_idx = 0
    
    def get_or_create_allgather_cache(self, tensor, world_size, cache_key):
        """获取或创建AllGather缓存，避免重复分配内存"""
        if cache_key not in self.allgather_cache:
            self.allgather_cache[cache_key] = [
                torch.zeros_like(tensor) for _ in range(world_size)
            ]
        return self.allgather_cache[cache_key]
    
    def cleanup_memory_if_needed(self):
        """定期清理内存，避免内存泄漏"""
        if self.iter % self.memory_cleanup_interval == 0:
            # 清理CUDA缓存
            torch.cuda.empty_cache()
            
            # 限制错误字典的大小
            if len(self.error_dict) > 10:  # 只保留最近10个bucket的错误
                old_keys = list(self.error_dict.keys())[:-10]
                for key in old_keys:
                    del self.error_dict[key]
                    if key in self.global_error_dict:
                        del self.global_error_dict[key]
    
    def get_next_seed(self):
        """获取下一个预计算的随机种子"""
        if not self.random:
            return None
        
        seed = self.precomputed_seeds[self.seed_idx % len(self.precomputed_seeds)]
        self.seed_idx += 1
        return seed


def sparse_hook_sync(
    state: SparseState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer

    if state.use_error_feedback == "ef21" and state.large_batch_init:
        if state.iter < state.start_compress_iter:
            logger.info("Using large batch initialization in EF21!!")
        return sparse_hook_sync_large_batch_ef21(state, bucket)

    # check if the key is ready in precond adam
    state.maybe_accumulate_momentum_on_bucket(bucket)

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
        
        # 修复2: 定期清理内存，避免内存泄漏  
        state.cleanup_memory_if_needed()
        
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
            
            # 修复2: 定期清理内存，避免内存泄漏
            state.cleanup_memory_if_needed()
            
            fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
            fut.set_result(input_tensor)
            return fut

    # 压缩:
    # seed 
    if state.random:    # 在每个通信 hook 调用前重新设定随机种子，以便保证使用 RandK 时，在不同 GPU 上压缩出的索引（即indices = torch.randperm(tensor.numel())[:k]这步每个GPU所得的结果一样）是一样的
        # 修复3: 使用预计算的种子，减少运行时开销
        seed = state.get_next_seed()
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
        # 修复1: 使用缓存的tensor列表，避免重复分配内存
        values_cache_key = f"values_{values_memory.shape}_{values_memory.dtype}"
        indices_cache_key = f"indices_{indices_memory.shape}_{indices_memory.dtype}"
        
        values_memory_allgather = state.get_or_create_allgather_cache(values_memory, world_size, values_cache_key)
        indices_memory_allgather = state.get_or_create_allgather_cache(indices_memory, world_size, indices_cache_key)
        
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
    
    # 修复2: 定期清理内存，避免内存泄漏
    state.cleanup_memory_if_needed()

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    # fut.set_result(input_tensor / world_size)  原代码有typo
    fut.set_result(input_tensor)
    return fut


def sparse_hook_sync_large_batch_ef21(
    state: SparseState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer

    assert state.use_error_feedback == "ef21", "This hook is only for EF21"
    assert state.large_batch_init, "This hook is only for large batch initialization"

    # check if the key is ready in precond adam
    state.maybe_accumulate_momentum_on_bucket(bucket)

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    rank = dist.get_rank(group=group_to_use)

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()
    tensors = bucket.gradients()

    device = tensors[0].device
    dtype = tensors[0].dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index()
    total_length = input_tensor.shape[0]

    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < 1:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor)
    elif state.iter < state.start_compress_iter:
        if bucket_index not in state.error_dict:
            logger.info("A tensor of length %s that represents local/global error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
            state.global_error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)

        # E_0 += \nabla F
        state.error_dict[bucket_index].add_(input_tensor, alpha=1.0)
        # allreduce \nabla F
        dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
        input_tensor.div_(world_size)
        # \overline{E_0} += \overline{\nabla F_0}
        state.global_error_dict[bucket_index].add_(input_tensor, alpha=1.0)
        # reset the full input tensor
        state.maybe_increase_iter(bucket)
        
        # 修复2: 定期清理内存，避免内存泄漏
        state.cleanup_memory_if_needed()
        
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut
    elif state.iter == state.start_compress_iter:
        state.error_dict[bucket_index].div_(state.start_compress_iter - 1)
        state.global_error_dict[bucket_index].div_(state.start_compress_iter - 1)

    # input_tensor = \nabla F_i - E_{i-1}
    input_tensor.add_(state.error_dict[bucket_index], alpha=-1.0)
    diff_cp = torch.clone(input_tensor).detach()

    if state.random:    
        # 修复3: 使用预计算的种子，减少运行时开销
        seed = state.get_next_seed()
        torch.manual_seed(seed)

    sparsify_func = {"row": sparsify_by_row, "column": sparsify_by_column, "tensor": sparsify}[state.sparse_type]
    sparsify_func = partial(sparsify_func, random=state.random)
    cal_k_func = {"row": cal_k_by_row, "column": cal_k_by_column, "tensor": cal_k}[state.sparse_type]
    k_list = [cal_k_func(tensor, state.compress_ratio) for tensor in tensors]

    # Compress the tensors to memory
    sum_k = sum(k_list)
    values_memory = torch.empty(sum_k, dtype=dtype, device=device)
    indices_memory = torch.empty(sum_k, dtype=torch.int, device=device)
    compress_tensor_to_memory(tensors, k_list, values_memory, indices_memory, sparsify_func, state.compress_ratio, state.use_error_feedback)

    # E_i = E_{i-1} + C[\nabla F_i - E_{i-1}]
    if bucket.is_last() and dist.get_rank() == 0:
        logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], Diff error{(diff_cp - input_tensor).norm().item()}")
    state.error_dict[bucket_index].add_(input_tensor, alpha=1.0)

    if state.random:
        # Allreduce the values
        dist.all_reduce(values_memory, group=group_to_use, async_op=False)
        values_memory.div_(world_size)
        # Zero the input tensor.
        input_tensor.zero_()
        decompress_memory_to_tensor_and_aggregate(tensors, k_list, values_memory, indices_memory, aggregate=False)
    else:
        # Allgather the values and indices
        # 修复1: 使用缓存的tensor列表，避免重复分配内存
        values_cache_key = f"values_{values_memory.shape}_{values_memory.dtype}"
        indices_cache_key = f"indices_{indices_memory.shape}_{indices_memory.dtype}"
        
        values_memory_allgather = state.get_or_create_allgather_cache(values_memory, world_size, values_cache_key)
        indices_memory_allgather = state.get_or_create_allgather_cache(indices_memory, world_size, indices_cache_key)

        dist.all_gather(values_memory_allgather, values_memory, group=group_to_use, async_op=False)
        dist.all_gather(indices_memory_allgather, indices_memory, group=group_to_use, async_op=False)
        
        # Zero the input tensor.
        input_tensor.zero_()
        for values_memory, indices_memory in zip(values_memory_allgather, indices_memory_allgather):
            decompress_memory_to_tensor_and_aggregate(tensors, k_list, values_memory, indices_memory)
        input_tensor.div_(world_size)

    # \overline{E_i} = \overline{E_{i-1}} + \overline{C[\nabla F_i - E_{i-1}]}
    state.global_error_dict[bucket_index].add_(input_tensor, alpha=state.error_decay)
    input_tensor.copy_(state.global_error_dict[bucket_index])

    state.maybe_increase_iter(bucket)
    
    # 修复2: 定期清理内存，避免内存泄漏
    state.cleanup_memory_if_needed()

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)
    return fut


def fake_sparse_hook(
    state: SparseState, bucket: dist.GradBucket
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

    
    bucket_index = bucket.index() 
    total_length = input_tensor.shape[0]
    if state.use_error_feedback == "ef14":
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index], alpha=1.0) # input_tensor = \nabla F_i + E_{i-1}
        else:
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
    
    state.error_dict[bucket_index].copy_(input_tensor)  # E_i = \nabla F_i + E_{i-1}

    if bucket_index not in state.generator:
        state.generator[bucket_index] = torch.Generator(device = device)
        state.generator[bucket_index].manual_seed(torch.randint(0, 100000, (1,), generator=state.rng).item())

    for tensor in tensors:
        if len(tensor.shape) == 2:
            m, n = tensor.shape # [m, n]
            k = max(1, int(m * current_compress_ratio))
            if not state.random:    # topk rows
                sigma = torch.norm(tensor, dim=1).abs() # [m, 1]
                zero_indices = torch.argsort(sigma, descending=True)[k:]
            else:                   # randk rows
                zero_indices = torch.randperm(m, device=device, generator=state.generator[bucket_index])[k:]
            tensor[zero_indices] = 0.0  # 将非top k的元素置为0

        elif len(tensor.shape) > 2:
            raise NotImplementedError("Fake Group TopK compression only supports 2D tensors.")
    
    state.error_dict[bucket_index].add_(input_tensor, alpha=-1.0)  # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
    # allreduce
    dist.all_reduce(input_tensor, group=group_to_use, async_op=False) # async_op=False 表示会阻塞程序执行，直到 all_reduce 完全完成。

    state.maybe_increase_iter(bucket)
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)

    return fut


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 11, 3], [7, 4, 5, 6], [14, 7, 8, 9]])
    original_numel = x.numel()
    original_shape = x.shape

    print(f"Sparsify by row")
    values, indices = sparsify_by_row(x, 0.5)
    print(values)
    print(indices)
    # decompressed_tensor = desparsify((values, indices), original_numel).view(original_shape)
    # print(decompressed_tensor)

    print(f"Sparsify by column")
    values, indices = sparsify_by_column(x, 0.7)
    print(values)
    print(indices)
    # decompressed_tensor = desparsify((values, indices), original_numel).view(original_shape)
    # print(decompressed_tensor)

    print(f"Sparsify by whole tensor")
    values, indices = sparsify(x, 0.5)
    print(values)
    print(indices)
    # decompressed_tensor = desparsify((values, indices), original_numel).view(original_shape)
    # print(decompressed_tensor)