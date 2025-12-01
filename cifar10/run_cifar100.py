'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

###
from tqdm import tqdm
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
# from utils import progress_bar

###
import time
import wandb
from wandb import Html
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

### 设置 WandB 项目名与 API key
os.environ["WANDB_API_KEY"] = "0f2bf1daed22671b7865ab947c9cbc917de7f80e"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

###
from comm_hooks.utils import register_comm_hook_for_ddp_model, add_comm_hook_args


# 初始化分布式环境
def init_distributed_mode(args):
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(seconds=30)
    )
    print(f"Initialized distributed training (rank {args.rank})")
    # 确认local_rank与GPU的对应关系
    # print(f"Rank {args.rank} running on CUDA device {args.local_rank} (visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')})")
    # 确认后端
    # print(f"Rank {args.rank} uses backend {dist.get_backend()}")



# 输入参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', action='store_true',help='resume from checkpoint')

###
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")

parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer')
parser.add_argument("--momentum", type=float, default=0.9, help="Set momentum for msgd.")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay for optimizer.")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.",)

parser.add_argument('--use_wandb', default=0, type=int, help='use wandb or not')
parser.add_argument('--col_rank', default=0, type=int, help=' "--r" is ambiguous while use "torchrun" instead of "accelerate", so use "--col_rank" instead of "--r" ')

parser.add_argument(
        "--gradient_accumulation_steps",  # 每多少个 batch 才执行一次 optimizer.step()梯度更新，用于显存不够时的梯度累积。实际上相当于扩大 batch size
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
###
add_comm_hook_args(parser) 
args = parser.parse_args()
args.r=args.col_rank
init_distributed_mode(args)

supported_optimizers = ['adamw', 'sgd']
assert args.optimizer in supported_optimizers, "`optimizer` should be one of the following: " + ', '.join(supported_optimizers)


###
if args.rank == 0 and args.use_wandb:
    if args.optimizer=="adamw":
        if args.compressor=="group_topk_no_reshape":
            wandb.init(
                project=f"cifar100_resnet18_group_topk_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}_ratio{args.compress_ratio}"
            )
        else:
            wandb.init(
                project=f"cifar100_resnet18_{args.compressor}_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
            )
    elif args.optimizer=="sgd":
        if args.compressor=="group_topk_no_reshape":
            wandb.init(
                project=f"msgd_cifar100_resnet18_group_topk_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}_ratio{args.compress_ratio}"
            )
        else:
            wandb.init(
                project=f"msgd_cifar100_resnet18_{args.compressor}_{args.use_error_feedback}", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
            )




device = torch.device(f"cuda:{args.local_rank}") # 使用local_rank指定GPU
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data(数据增强（训练集）)
print('Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 使用DistributedSampler
trainset = torchvision.datasets.CIFAR100(root='/home/mcy/data', train=True, download=True, transform=transform_train)
train_sampler = DistributedSampler(trainset) # 多 GPU 分布式训练 中，为了让每个 GPU 处理不同的数据子集，必须使用 DistributedSampler，它会根据当前进程的 rank 和总进程数，把数据划分给不同进程。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.per_device_train_batch_size, sampler=train_sampler, num_workers=2, pin_memory=True) # num_workers=2 是指在 当前 GPU/进程下，开两个子线程 用来加载数据（取决于CPU而不是GPU）
                                                                                                                                                        # pin_memory=True：加快 GPU 拷贝速度（一般训练时推荐开启）

testset = torchvision.datasets.CIFAR100(root='/home/mcy/data', train=False, download=True, transform=transform_test)
test_sampler = DistributedSampler(testset, shuffle=False) # 每轮测试集数据固定，结果才具有可比性和稳定性，方便观察模型随训练进展的性能变化
testloader = torch.utils.data.DataLoader(testset, batch_size=args.per_device_train_batch_size, sampler=test_sampler, num_workers=2, pin_memory=True)

# Model
print(f"[Rank {args.rank} | Local Rank {args.local_rank}] Building model..")
net = ResNet18(num_classes=100).to(device)
# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
#     checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=map_location)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

net = DDP(net, device_ids=[args.local_rank])


criterion = nn.CrossEntropyLoss()

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
if args.optimizer == "adamw":
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)
elif args.optimizer == 'sgd':  # msgd(NAG)
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

# Compressor
process_group = dist.distributed_c10d._get_default_group()
register_comm_hook_for_ddp_model(net, process_group, args, optimizer=optimizer)

# lr_scheduler
if args.optimizer == "adamw":
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)
    milestone=0.1 * args.num_train_epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=milestone) 
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_train_epochs-milestone) # T_max 表示余弦周期是多少个epoch
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[milestone]) # milestone 表示前milestone个epoch用warmup_scheduler，之后用cosine_scheduler

elif args.optimizer == 'sgd':  # msgd(NAG)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)




# Training
def train(epoch):
    if args.rank == 0:
        logger.info(f"\n[Epoch {epoch+1}/{args.num_train_epochs}] Training...")

    train_sampler.set_epoch(epoch)  # 必要
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()

        # 判断当前小 batch 是否是该累积组的最后一步
        is_last_step = ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or (batch_idx == len(trainloader) - 1)

        # 如果不是最后一步，使用 no_sync() 来避免 DDP 每次 backward 都做同步
        if hasattr(net, "no_sync") and not is_last_step:
            ctx = net.no_sync()
        else:
            # 占位上下文，便于统一写法
            class _DummyCtx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            ctx = _DummyCtx()

        with ctx: # 每个小 batch 都会执行
            outputs = net(inputs)
            batch_loss = criterion(outputs, targets)   # CrossEntropy 默认按 batch 平均（即每个样本的平均 loss）
            loss_for_backward = batch_loss / args.gradient_accumulation_steps    # 累积时对 backward 做缩放，梯度在模型参数上被累积了4次（每次1/4），总和就相当于整合了4个小 batch 的梯度。
            loss_for_backward.backward() # 梯度是累积的（loss.backward()多次调用时，梯度会加到.grad 上，一直到optimizer.zero_grad清零）

        _, predicted = outputs.max(1) 
        temloss = torch.tensor(batch_loss.item(), device=device)  # 后面会除以len(trainloader)所以用batch_loss
        temtotal = torch.tensor(targets.size(0), device=device)  # 当前这张卡上本轮的样本数
        temcorrect = torch.tensor(predicted.eq(targets).sum().item(), device=device)  # 当前这张卡上预测对的样本数

        dist.all_reduce(temloss, op=dist.ReduceOp.SUM)  
        temloss = temloss / args.world_size
        dist.all_reduce(temtotal, op=dist.ReduceOp.SUM)
        dist.all_reduce(temcorrect, op=dist.ReduceOp.SUM)

        if args.rank==0:
            train_loss += temloss.detach().item()
            total += temtotal.item()
            correct += temcorrect.item()
        
        # 累积完成时更新并清零梯度
        if is_last_step:
            optimizer.step() # 只有在这args.gradient_accumulation_steps 个小batch都执行完后（也就是第args.gradient_accumulation_steps个小 batch的反向传播结束后），才执行 optimizer.step()
            optimizer.zero_grad()
            
    if args.rank==0:
        avg_loss = train_loss / len(trainloader)
        acc = 100. * correct / total
        logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.3f} | Acc: {acc:.3f}% (correct: {correct}/total: {total})")

        if args.use_wandb:
            wandb.log({
                "train_loss": avg_loss,
                "train_accuracy": acc,
                "epoch": epoch,
                "train_correct": correct,
                "train_total": total,
            },step=epoch)




def test(epoch):
    global best_acc
    test_sampler.set_epoch(epoch)  # 可选：确保多 epoch 时也能保持稳定性
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            temloss=torch.tensor(loss.item(),device=device)
            temtotal=torch.tensor(targets.size(0),device=device)
            temcorrect=torch.tensor(predicted.eq(targets).sum().item(),device=device)

            dist.all_reduce(temloss, op=dist.ReduceOp.SUM)
            temloss = temloss / args.world_size
            dist.all_reduce(temtotal, op=dist.ReduceOp.SUM)
            dist.all_reduce(temcorrect, op=dist.ReduceOp.SUM)

            if args.rank==0:
                test_loss += temloss.detach().item()  
                total += temtotal.item()
                correct += temcorrect.item()

 
    if args.rank==0:
        acc = 100.*correct/total
        avg_loss = test_loss / len(testloader)
        logger.info(f"[Epoch {epoch+1}] Test  Loss: {avg_loss:.3f} | Acc: {acc:.3f}% (correct: {correct}/total: {total})")

        if acc > best_acc:
            best_acc = acc


        if args.use_wandb:
            wandb.log({
                "test_accuracy": acc,
                "test_loss": avg_loss,
                "best_acc": best_acc,
                "test_correct": correct,
                "test_total": total,
                "epoch": epoch,
            }, step=epoch)   

                

    

if __name__ == '__main__':
    # start_time = time.time()  # 开始计时
    print('训练开始！！！！！')
    

    if args.rank == 0:
        epoch_bar = tqdm(range(start_epoch, start_epoch + args.num_train_epochs), desc="Training Epochs")
    else:
        epoch_bar = range(start_epoch, start_epoch + args.num_train_epochs)

    for epoch in epoch_bar:
        train(epoch)
        test(epoch)
        scheduler.step()

    # end_time = time.time()  # 结束计时
    print('训练结束！！！！！')
    # total_seconds = end_time - start_time
    # minutes, seconds = divmod(int(total_seconds), 60)

    # logger.info(f"Total training time: {minutes:02} :{seconds:02} ") 

    # if args.rank == 0:
    #     wandb.log({
    #         "total_training_time": Html(f"<p>{minutes:02}:{seconds:02}</p>"),
    #         "total_training_time_minutes": minutes + seconds / 60,  
    #     })
    dist.destroy_process_group()