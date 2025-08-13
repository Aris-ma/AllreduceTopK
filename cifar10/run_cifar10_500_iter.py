'''Train CIFAR10 with PyTorch.'''
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
from comm_hooks.utils import register_comm_hook_for_ddp_model, add_comm_hook_args, dtype_bits, tensor_bits


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
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
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
parser.add_argument('--col_rank', default=0, type=int, help=' "--col_rank" is ambiguous while use "torchrun" instead of "accelerate", so use "--col_rank" instead of "--col_rank" ')

###
add_comm_hook_args(parser)
args = parser.parse_args()
args.col_rank=args.col_rank
init_distributed_mode(args)

supported_optimizers = ['adamw', 'sgd']
assert args.optimizer in supported_optimizers, "`optimizer` should be one of the following: " + ', '.join(supported_optimizers)


###
if args.rank == 0 and args.use_wandb:
    if args.optimizer=="adamw":
        if args.compressor=="group_topk_no_reshape":
            wandb.init(
                #project=f"cifar10_resnet18_group_topk_{args.use_error_feedback}", 
                project=f"cifar10_resnet18_small_bandwidth", 
                #project=f"cifar10_resnet18", 
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.col_rank}_ratio{args.compress_ratio}"
            )
        else:
            wandb.init(
                #project=f"cifar10_resnet18_{args.compressor}_{args.use_error_feedback}", 
                project=f"cifar10_resnet18_small_bandwidth", 
                #project=f"cifar10_resnet18",
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
            )
    elif args.optimizer=="sgd":
        if args.compressor=="group_topk_no_reshape":
            wandb.init(
                #project=f"msgd_cifar10_resnet18_group_topk_{args.use_error_feedback}", 
                project=f"cifar10_resnet18_small_bandwidth", 
                #project=f"cifar10_resnet18",
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.col_rank}_ratio{args.compress_ratio}"
            )
        else:
            wandb.init(
                #project=f"msgd_cifar10_resnet18_{args.compressor}_{args.use_error_feedback}", 
                project=f"cifar10_resnet18_small_bandwidth", 
                #project=f"cifar10_resnet18",
                name=f"atomo_lr{args.lr}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
            )




device = torch.device(f"cuda:{args.local_rank}") 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data(数据增强（训练集）)
print('Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 使用DistributedSampler
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_sampler = DistributedSampler(trainset) # 多 GPU 分布式训练 中，为了让每个 GPU 处理不同的数据子集，必须使用 DistributedSampler，它会根据当前进程的 rank 和总进程数，把数据划分给不同进程。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.per_device_train_batch_size, sampler=train_sampler, num_workers=2, pin_memory=True) # num_workers=2 是指在 当前 GPU/进程下，开两个子线程 用来加载数据（取决于CPU而不是GPU）
                                                                                                                                                        # pin_memory=True：加快 GPU 拷贝速度（一般训练时推荐开启）

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_sampler = DistributedSampler(testset, shuffle=False) # shuffle=False：每轮测试集数据固定，结果才具有可比性和稳定性，方便观察模型随训练进展的性能变化
testloader = torch.utils.data.DataLoader(testset, batch_size=args.per_device_train_batch_size, sampler=test_sampler, num_workers=2, pin_memory=True)

# Model
print(f"[Rank {args.rank} | Local Rank {args.local_rank}] Building model..")
net = ResNet18().to(device)
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
hook_state = register_comm_hook_for_ddp_model(net, process_group, args, optimizer=optimizer)

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
def train(epoch, hook_state, bits_iters = 500, profile_iters = 100):
    if args.rank == 0:
        logger.info(f"\n[Epoch {epoch+1}/{args.num_train_epochs}] Training...")

    train_sampler.set_epoch(epoch)  # 必要
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    

    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if hook_state.iter >= bits_iters:  # 达到指定迭代次数后退出
            break

        if hook_state.iter == 0:
            print('计时开始！！！！！')
            start_profile_time = time.time()
            

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)  
        temloss = torch.tensor(loss.item(), device="cuda")  # 当前这卡的 loss 标量，转换成张量放到 GPU 上
        temtotal = torch.tensor(targets.size(0), device="cuda")  # 当前这卡上本轮的样本数
        temcorrect = torch.tensor(predicted.eq(targets).sum().item(), device="cuda")  # 当前这卡上预测对的样本数

        dist.all_reduce(temloss, op=dist.ReduceOp.SUM)  
        dist.all_reduce(temtotal, op=dist.ReduceOp.SUM)
        dist.all_reduce(temcorrect, op=dist.ReduceOp.SUM)

        if args.rank==0:
            # train_loss += temloss.detach().item()
            # total += temtotal.item()
            # correct += temcorrect.item()
            
            #### 时间：
            if hook_state.iter == profile_iters - 1:
                end_profile_time = time.time()
                avg_iter_time = (end_profile_time - start_profile_time)/profile_iters
                logger.info(f"[Profile]Average iteration time over {profile_iters} iters:{avg_iter_time:.6f} seconds")
                if args.use_wandb:
                    wandb.log({"avg_iter_time_first_100)": wandb.Html(f"<p>Time: {avg_iter_time}</p>")})
                print('计时结束！！！！！')
                    
            #### bits：
            # 访问通信比特数
            comm_bits_this_round = getattr(hook_state, 'comm_bits_this_round', 0)

            temacc = 100. * temcorrect / temtotal
            logger.info(f"[iter_step {hook_state.iter}] Train Iteration Loss: {temloss:.3f} | TemAcc: {temacc:.3f}% (temcorrect: {temcorrect}/temtotal: {temtotal})")

            # if args.use_wandb:
            #     wandb.log({
            #         "train_loss_per_iteration": float(temloss),
            #         "train_accuracy_per_iteration": float(temacc),
            #         "iter_step": int(hook_state.iter),
            #         "train_comm_bits":int(comm_bits_this_round),
            #     },step=hook_state.iter)
       
                

            
    # if args.rank==0:
    #     avg_loss = train_loss / len(trainloader) # 一个epoch的平均loss
    #     acc = 100. * correct / total
    #     logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.3f} | Acc: {acc:.3f}% (correct: {correct}/total: {total})")

    #     if args.use_wandb:
    #         wandb.log({
    #             "train_loss": avg_loss,
    #             "train_accuracy": acc,
    #             "epoch": epoch,
    #             "train_correct": correct,
    #             "train_total": total,
    #         },step=epoch)




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
            temloss=torch.tensor(loss.item(),device="cuda")
            temtotal=torch.tensor(targets.size(0),device="cuda")
            temcorrect=torch.tensor(predicted.eq(targets).sum().item(),device="cuda")

            dist.all_reduce(temloss, op=dist.ReduceOp.SUM)
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
    
    # if args.rank == 0:
    #     epoch_bar = tqdm(range(500), desc="Training iterations")
    # else:
    #     epoch_bar = range(500)

    for epoch in range(start_epoch, start_epoch + args.num_train_epochs):
        train(epoch, hook_state)
        # test(epoch)
        # scheduler.step()
    



    # end_time = time.time()  # 结束计时
    
    print('训练结束！！！！！')
    
    # total_seconds = end_time - start_time
    # minutes, seconds = divmod(int(total_seconds), 60)

    # logger.info(f"Total training time: {minutes:02} :{seconds:02} ")

    # if args.rank == 0:
    #     wandb.log({
    #         "total_training_time": wandb.Html(f"<p>Time: {minutes:02}:{seconds:02}</p>"),
    #     })
    
    dist.destroy_process_group()