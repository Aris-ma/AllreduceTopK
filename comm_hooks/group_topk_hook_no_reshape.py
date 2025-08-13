# mypy: allow-untyped-defs
import logging
from functools import partial
from typing import Dict

import torch
import torch.distributed as dist

import comm_hooks.default_hooks as default_hooks

from comm_hooks.utils import HookState, _get_allgather_out_list, dtype_bits, tensor_bits

logger = logging.getLogger(__name__)


def group_topk_project_and_select(tensor, r, compress_ratio, group, use_deterministic_projection=False, projection_seed=None):

    d = tensor.numel()
    if tensor.dim() == 1:  
        # 1D tensor: no compression, pass through directly
        # 优化：避免不必要的flatten()和clone()
        P_bits = 0
        comm_bits = tensor_bits(tensor)
        indices = torch.arange(d, device=tensor.device, dtype=torch.int64)  # 使用int64保持一致性
        return tensor, indices, P_bits, comm_bits

    elif tensor.dim() == 2:
        n, m = tensor.shape
        k = max(1, int(n * compress_ratio))
    
        # 改进：支持确定性投影矩阵
        V = torch.empty(m, r, device=tensor.device, dtype=tensor.dtype)
        if use_deterministic_projection and projection_seed is not None:
            # 使用固定的seed生成投影矩阵，提高稳定性
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(projection_seed)
            V.normal_(generator=generator)
        else:
            V.normal_()  # 原地生成正态分布随机数，比torch.randn更高效
        
        # 优化：直接使用tensor @ V的结果，避免额外的clone
        P_local = tensor @ V     # shape: [n, r]
        P_bits = tensor_bits(P_local)
        
        # AllReduce operation - 这里必须clone因为all_reduce是原地操作
        dist.all_reduce(P_local, group=group)
        P_local.div_(dist.get_world_size(group))  # 原地除法

        # 优化：直接计算平方和，避免中间张量
        norms = torch.sum(P_local.square(), dim=1)  # square()比**2稍快
        _, topk_indices = torch.topk(norms, k=k, largest=True, sorted=False)
        
        # 优化：使用向量化操作计算indices，避免循环
        row_offset = topk_indices.unsqueeze(1) * m  # [k, 1]
        col_indices = torch.arange(m, device=tensor.device, dtype=torch.int64).unsqueeze(0)  # [1, m]
        indices = (row_offset + col_indices).flatten()  # shape: [k * m]
        
        selected_rows = tensor[topk_indices]
        comm_bits = tensor_bits(selected_rows)
        return selected_rows.flatten(), indices, P_bits, comm_bits
        
    else:
        # Higher dimensional tensors
        t = tensor.shape[-1]
        m = 2 * t * t
        n = d // m
        tensor_2D = tensor.view(n, m)  # 优化：使用view代替reshape，避免不必要的内存复制
        k = max(1, int(n * compress_ratio))
    
        # 改进：支持确定性投影矩阵
        V = torch.empty(m, r, device=tensor.device, dtype=tensor.dtype)
        if use_deterministic_projection and projection_seed is not None:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(projection_seed)
            V.normal_(generator=generator)
        else:
            V.normal_()
        
        P_local = tensor_2D @ V     # shape: [n, r]
        P_bits = tensor_bits(P_local)
        
        # AllReduce operation
        dist.all_reduce(P_local, group=group)
        P_local.div_(dist.get_world_size(group))

        # 优化：直接计算平方和
        norms = torch.sum(P_local.square(), dim=1)
        _, topk_indices = torch.topk(norms, k=k, largest=True, sorted=False)
        
        # 优化：使用向量化操作计算indices
        row_offset = topk_indices.unsqueeze(1) * m  # [k, 1]
        col_indices = torch.arange(m, device=tensor.device, dtype=torch.int64).unsqueeze(0)  # [1, m]
        indices = (row_offset + col_indices).flatten()  # shape: [k * m]
        
        selected_rows = tensor_2D[topk_indices]
        comm_bits = tensor_bits(selected_rows)
        return selected_rows.flatten(), indices, P_bits, comm_bits







    
def compress_tensor_to_memory(tensors, k_list, values_memory, indices_memory, compress_ratio, r, group, use_error_feedback, use_deterministic_projection=False, projection_seed=None):
    offset = 0
    bits_sum = 0
    for tensor, k in zip(tensors, k_list):  
        values, indices, P_bits, comm_bits= group_topk_project_and_select(
            tensor=tensor, 
            r=r, 
            compress_ratio=compress_ratio, 
            group=group,
            use_deterministic_projection=use_deterministic_projection,
            projection_seed=projection_seed
        )
        values_memory[offset:offset+k] = values
        indices_memory[offset:offset+k] = indices  # 优化：移除不必要的类型转换，indices已经是int64
        offset += k
        bits_sum += P_bits + comm_bits # bits_sum是一个GPU的一个bucket的通信量。


        if use_error_feedback == "ef14":
            # input_tensor = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
            tensor.view(-1)[indices] = 0  # C内的部分设成0，处理后的 tensor = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
        elif use_error_feedback == "ef21":
            tensor.zero_()
            # input_tensor = C[\nabla F_i - E_{i-1}]
            tensor.view(-1)[indices] = values # 处理后的 tensor = C[\nabla F_i - E_{i-1}]，不会改变 tensor 本身的形状
    return values_memory, indices_memory, bits_sum

def decompress_memory_to_tensor_and_aggregate(tensors, k_list, values_memory, indices_memory):
    # add the decompressed values to the tensors
    offset = 0
    for tensor, k in zip(tensors, k_list):
        values = values_memory[offset:offset+k]
        indices = indices_memory[offset:offset+k]
        # avoid creating a new tensor for the view
        flattened_tensor = tensor.view(-1)
        flattened_tensor[indices] = values  # 优化：移除不必要的类型转换，indices已经是正确类型
        offset += k
    return None

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

def cal_k(state, tensor):
    # 使用动态压缩比
    current_compress_ratio = state.get_current_compress_ratio()
    
    if tensor.dim() == 1:
        k = tensor.numel()
        # d = tensor.numel()
        # k = max(1, int(d * current_compress_ratio))
    elif tensor.dim()==2:
        k = max(1, int(tensor.shape[0] * current_compress_ratio)) * tensor.shape[1]
    else:
        t = tensor.shape[-1]
        m = 2 * t * t
        d = tensor.numel()
        n = d // m
        k = max(1, int(n * current_compress_ratio)) * m
        
    return k


def group_topk_hook(
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
            # logger.info(f"\nabla F_i - E_i-1:{input_tensor}")
        else:
            # E_0 = \nabla F_0 
            logger.info("A tensor of length %s that represents local/global error is created.", total_length)
            state.error_dict[bucket_index] = torch.clone(input_tensor).detach()
            # allreduce \nabla F_0
            state.comm_bits_this_round += tensor_bits(input_tensor)
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
    # sample 本轮投影 seed，并生成 V
    seed = torch.randint(0, 1_000_000_000, (1,), generator=state.rng).item()
    torch.manual_seed(seed)
  
    # 使用确定性投影矩阵来提高稳定性（在压缩的早期阶段）
    use_deterministic = state.compression_started and (state.iter - state.start_compress_iter < state.warmup_iters)
    projection_seed = seed if use_deterministic else None

    k_list = [cal_k(state, tensor) for tensor in tensors]

    sum_k = sum(k_list) # k_list中元素求和即为values_memory和indices_memory的长度。

    values_memory = torch.empty(sum_k, dtype=dtype, device=device) # 初始化为0
    indices_memory = torch.empty(sum_k, dtype=torch.int64, device=device)  # 优化：使用int64保持一致性
    _, _, bits_sum = compress_tensor_to_memory(
        tensors=tensors, 
        k_list=k_list, 
        values_memory=values_memory, 
        indices_memory=indices_memory, 
        compress_ratio=current_compress_ratio, 
        r=state.r, 
        group=group_to_use, 
        use_error_feedback=state.use_error_feedback,
        use_deterministic_projection=use_deterministic,
        projection_seed=projection_seed
    )

   

    # 更新一下 state.error_dict
    if state.use_error_feedback == "ef14":
        # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
        state.error_dict[bucket_index].copy_(input_tensor)              
    elif state.use_error_feedback == "ef21":
        # E_i = E_{i-1} + C[\nabla F_i - E_{i-1}]
        state.error_dict[bucket_index].add_(input_tensor)    
        
    # Allreduce，并解压回原梯度张量
    state.comm_bits_this_round += 2 *(world_size -1) * bits_sum # 全局通信bits数
    # print(f"bucket_index{bucket_index}rank{rank}已加进state.comm_bits_this_round{state.comm_bits_this_round}, bits_sum={bits_sum}")
    dist.all_reduce(values_memory, group=group_to_use, async_op=False)
    values_memory.div_(world_size)

    # Zero the input tensor.
    input_tensor.zero_() 
    decompress_memory_to_tensor_and_aggregate(tensors=tensors, k_list=k_list, values_memory=values_memory, indices_memory=indices_memory)

    # 更新 global_error_dict
    if state.use_error_feedback == "ef21":
        state.global_error_dict[bucket_index].add_(input_tensor) # \overline{E_i} = \overline{E_{i-1}} + \overline{C[\nabla F_i - E_{i-1}]}
        input_tensor.copy_(state.global_error_dict[bucket_index])

    state.maybe_increase_iter(bucket)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)

    return fut




