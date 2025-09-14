import torch
import torch.distributed as dist

import logging
#from optimizer import PrecondAdam

logger = logging.getLogger(__name__)

def _get_allgather_out_list(all_gather_in_list, world_size): # 为了 dist.all_gather 准备输出用的空列表（会返回一个长度为 world_size 的列表 out_list，每个元素是一个和输入张量一样形状的数据容器）
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list

 
class HookState:
    
    def __init__(self, process_group: dist.ProcessGroup): 
        self.process_group = process_group 
        self.start_compress_iter = 0 # start_compress_iter	表示从哪一轮迭代开始启用压缩（之前是warmup）
        self.iter = 0 

        self.total_bit_before_compression = 0 
        self.total_bit_after_compression = 0 
        self.compressor_name="none compressor" 

        self.compress_momentum = False 
        self.param_state = None 
        self.param_to_name = None 
        self.beta1 = None 
        self.adam_freeze_key = False 

        self.comm_bits_this_round = 0

    def init_momentum_field(self, param_state, beta1): # 初始化动量信息
        self.param_state = param_state
        self.beta1 = beta1
        self.compress_momentum = True # 开启动量压缩

    def maybe_accumulate_momentum_on_bucket(self, bucket: dist.GradBucket):
        if not self.compress_momentum: 
            return
        if self.iter >= self.start_compress_iter and not self.adam_freeze_key:
            self.adam_freeze_key = True # 如果启用了动量压缩且达到 start_compress_iter 后，就冻结二阶动量，只压缩一阶动量。
            logger.info(f"Freeze the second momentum of Adam optimizer after {self.iter}(included) steps")
        if self.adam_freeze_key:
            self.accumulate_momentum_on_bucket(bucket) # 调用下面的动量累积函数

    def accumulate_momentum_on_bucket(self, bucket: dist.GradBucket):
        if not self.compress_momentum:
            raise RuntimeError("Momentum compression is not enabled!")
        if self.param_state is None:
            raise RuntimeError("Parameter state is not initialized!")
        # accumulate momentum
        parameters, gradients = bucket.parameters(), bucket.gradients()
        assert len(parameters) == len(gradients), "The number of parameters and gradients should be the same."
        for tensor, grad in zip(parameters, gradients):
            state = self.param_state[tensor] # param_state[tensor]是一个字典，保存动量等信息，类似于 {"exp_avg": ..., "exp_avg_sq": ...}
            grad.mul_(1 - self.beta1).add_(state['exp_avg'], alpha=self.beta1) # state['exp_avg']是一阶动量，也就是 Adam 优化器中的 m_t
            # 这相当于grad = (1 - beta1) * grad + beta1 * exp_avg，但是使用了 mul_ 和 add_，是原地操作，减少了内存开销。add_(state['exp_avg'], alpha=self.beta1)：将动量 exp_avg 按权重 β_1 加到 grad 上。

    def maybe_increase_iter(self, bucket):
        """Track iterations and trigger log message at start of local SGD."""
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last(): 
            self.iter += 1

            if self.iter == self.start_compress_iter:
                logger.info(f"Start to apply {self.compressor_name} hook after {self.start_compress_iter} iterations.")

    def compression_bits_stats(self):

        compress_rate = ( 
            self.total_bit_before_compression / self.total_bit_after_compression
            if self.total_bit_after_compression > 0
            else 0
        )
        return (
            compress_rate,
            self.total_bit_before_compression,
            self.total_bit_after_compression,
        )


def register_comm_hook_for_ddp_model(model, process_group, args, optimizer=None):
    hook_state = None
    if args.compressor == "topk_sync" or args.compressor == "randk_sync": 
        from comm_hooks.sparse_hook_c4 import SparseState, sparse_hook_sync
        random = 'randk' in args.compressor
        hook_state = SparseState(
            process_group=process_group,
            compress_ratio=args.compress_ratio, 
            sparse_type=args.sparse_type,
            use_error_feedback=args.use_error_feedback,
            random=random,
            start_compress_iter=args.start_compress_iter,
            random_seed=args.seed,
        )
        model.register_comm_hook(hook_state, sparse_hook_sync) # 注册通信hook
  

    elif args.compressor == "group_topk_no_reshape" :
        #from comm_hooks.group_topk_hook_no_reshape_c4 import group_topk_hook, GroupTopKState
        from comm_hooks.group_topk_hook_no_reshape import group_topk_hook, GroupTopKState
        hook_state = GroupTopKState(
            process_group=process_group, 
            r=args.r, 
            use_error_feedback=args.use_error_feedback, 
            seed=args.seed,
            start_compress_iter=args.start_compress_iter,
            compress_ratio=args.compress_ratio,
        )
        model.register_comm_hook(hook_state, group_topk_hook) # 注册通信hook

    elif args.compressor == 'noop': 
        from comm_hooks.debugging_hooks import noop_hook
        model.register_comm_hook(None, noop_hook) # 注册通信hook

    elif args.compressor == 'none' : # 不压缩（即默认的 AllReduce + HookState 支持）
        from comm_hooks.default_hooks import my_allreduce_hook
        hook_state = HookState(process_group) 
        hook_state.start_compress_iter = args.start_compress_iter
        model.register_comm_hook(hook_state, my_allreduce_hook) # 注册通信hook
    else:
        raise ValueError(f"Compressor {args.compressor} not supported.")
    
    # For selective compression
    if hasattr(hook_state, 'param_to_name'):
        hook_state.param_to_name = {param: name for name, param in model.named_parameters()}
    # for param, name in hook_state.param_to_name.items():
    #     if dist.get_rank() == 0:
    #         logger.info(f"Parameter name: {name}, shape: {param.shape}")
    
    return hook_state

def add_comm_hook_args(parser):
    ### Commpressor arguments
    parser.add_argument(
        "--compressor",
        type=str,
        default="none",
        help="Set the compressor to use.",
    )
    parser.add_argument(
        "--start_compress_iter",  
        type=int,
        default=10,
        help="Set the iteration to start compression.",
    )
    parser.add_argument( 
        "--use_error_feedback",
        type=str,
        default="noef", 
        choices=["noef", "ef14", "ef21"],
        help="Set the error feedback to use.",
    )

    # 稀疏压缩参数
    parser.add_argument(
        "--sparse_type",
        type=str,
        default='tensor',
        choices=['row', 'column', 'tensor'], 
        help="Set the type of top-k sparsification to use.",
    )
    parser.add_argument(
        "--compress_ratio", 
        type=float,
        default=0.08,
        help="Set the ratio of the top-k elements to keep.",
    )
    
    parser.add_argument(
        "--r", 
        type=int,
        default=4,
        help="num of cols after projection.",
    )
    

    ### check whether the gradients are identical across all processes（梯度一致性检查）
    parser.add_argument(
        "--check_grad",
        action="store_true",
        default=False,
        help="Whether to check the identity of the gradients.",
    )
 

def dtype_bits(tensor): # 返回 tensor 的 单个元素占用的位数（bit）
    dtype = tensor.dtype
    if dtype.is_floating_point: # 浮点类型（float32、float64）使用 torch.finfo 获取其位数
        return torch.finfo(dtype).bits
    elif dtype.is_complex: # 复数（如 complex64、complex128），每个复数由两个浮点组成
        return torch.finfo(dtype).bits * 2  # Complex numbers have twice the bits
    elif dtype == torch.bool:
        return 1
    elif "int" in str(dtype): # 整型使用 torch.iinfo 获取其位数（如 int8, int32, int64）
        return torch.iinfo(dtype).bits
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
def tensor_bits(tensor):
    return tensor.numel() * dtype_bits(tensor)


def name_func_glue(args):
    if args.optimizer=="adamw":
        if args.weight_decay==0:
            if args.compress_ratio==0.08:
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_r{args.r}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}"
                        
            else:   # compress_ratio=0.2
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name = f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_r{args.r}_ratio{args.compress_ratio}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name = f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_ratio{args.compress_ratio}"
                    
        else: # weight_decay
            if args.compress_ratio==0.08:
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name = f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}"
                    
            else:   # compress_ratio=0.2
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}_ratio{args.compress_ratio}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
                    

    else:  # sgd
        if args.compress_ratio==0.08:
            if args.compressor=="group_topk_no_reshape":
                program_name = f"msgd_glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}_r{args.r}"
                
            else:
                program_name = f"msgd_glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}"
                    
        else: 
            if args.compressor=="group_topk_no_reshape":
                program_name = f"msgd_glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}_r{args.r}_ratio{args.compress_ratio}"
                
            else:
                program_name = f"msgd_glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}_ratio{args.compress_ratio}"
                
    return program_name, run_name


