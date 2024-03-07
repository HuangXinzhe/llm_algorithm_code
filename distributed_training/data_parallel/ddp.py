"""
数据并行训练（Data Parallel）
"""
import torch.nn as nn
import torch
import argparse
from torch import distributed as dist
from torch.optim import AdamW
from torch import nn
from torch.nn import functional as F

# 创建一个解析器
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default="data",
                    help="使用的数据集")
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="batch size")
parser.add_argument("--local_rank",
                    type=int,
                    default=0,
                    help="当前gpu编号")
parser.add_argument("--model_name",
                    type=str,
                    default="bert-base-uncased",
                    help="模型名称")
parser.add_argument("--model_save_path",
                    type=str,
                    default="model.pth",
                    help="模型保存路径")


# 解析参数
args = parser.parse_args()

local_rank = dist.get_rank()  # 获取当前进程的rank
print(torch.cuda.device_count())  # 打印gpu数量
dist.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
print('world_size', torch.distributed.get_world_size())  # 打印当前进程数
torch.cuda.set_device(local_rank)  # 设置当前gpu编号为local_rank;此句也可能看出local_rank的值是什么

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # 所有gpu上的tensor相加
    rt /= nprocs
    return rt

"""
加载模型
"""
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建模型实例
My_model = MyModel()

'''
多卡训练加载数据:
# Dataset的设计上与单gpu一致，但是DataLoader上不一样。首先解释下原因：多gpu训练是，我们希望
# 同一时刻在每个gpu上的数据是不一样的，这样相当于batch size扩大了N倍，因此起到了加速训练的作用。
# 在DataLoader时，如何做到每个gpu上的数据是不一样的，且gpu1上训练过的数据如何确保接下来不被别
# 的gou再次训练。这时候就得需要DistributedSampler。
# dataloader设置方式如下，注意shuffle与sampler是冲突的，并行训练需要设置sampler，此时务必
# 要把shuffle设为False。但是这里shuffle=False并不意味着数据就不会乱序了，而是乱序的方式交给
# sampler来控制，实质上数据仍是乱序的。
'''
train_sampler = torch.utils.data.distributed.DistributedSampler(args.dataset)  # 数据采样器，保证每个gpu上的数据不一样，且不会重复，且数据是乱序的
dataloader = torch.utils.data.DataLoader(ds=args.dataset,  # ds是你的数据集
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=16,
                                         pin_memory=True,
                                         drop_last=True,
                                         sampler=train_sampler)  # 将数据分批次的喂给模型


'''
多卡训练的模型设置：
# 最主要的是find_unused_parameters和broadcast_buffers参数；
# find_unused_parameters：如果模型的输出有不需要进行反传的(比如部分参数被冻结/或者网络前传是动态的)，设置此参数为True;如果你的代码运行
# 后卡住某个地方不动，基本上就是该参数的问题。
# broadcast_buffers：设置为True时，在模型执行forward之前，gpu0会把buffer中的参数值全部覆盖
# 到别的gpu上。注意这和同步BN并不一样，同步BN应该使用SyncBatchNorm。
'''
My_model = My_model.cuda(args.local_rank)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
My_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(My_model)  # 设置多个gpu的BN同步
My_model = torch.nn.parallel.DistributedDataParallel(My_model,
                                                     device_ids=[args.local_rank],
                                                     output_device=args.local_rank,
                                                     find_unused_parameters=False,  # 如果模型的输出有不需要进行反传的(比如部分参数被冻结/或者网络前传是动态的)，设置此参数为True
                                                     broadcast_buffers=False)  # 设置为True时，在模型执行forward之前，gpu0会把buffer中的参数值全部覆盖到别的gpu上。
# 优化器
opt = AdamW(My_model.parameters(), lr=1e-5)  # 优化器

# 交叉熵损失函数
My_loss = nn.CrossEntropyLoss()  # 损失函数
# def My_loss(output, targets):
#     # 自定义损失函数，这里只是一个例子，你需要根据你的需求来实现
#     loss = (output - targets).abs().mean()
#     return loss

# 评估函数
def My_eval(model, dataloader):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy
    

'''开始多卡训练：'''
for epoch in range(200):
    train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
    My_model.train()
    for idx, sample in enumerate(dataloader):
        inputs, targets = sample[0].cuda(local_rank, non_blocking=True), sample[1].cuda(local_rank, non_blocking=True)
        opt.zero_grad()  
        output = My_model(inputs)
        loss = My_loss(output, targets)
        loss.backward()
        opt.step()
        loss = reduce_mean(loss, dist.get_world_size())  # 多gpu的loss进行平均。


'''多卡测试(evaluation)：'''
if local_rank == 0:
    My_model.eval()
    with torch.no_grad():
        acc = My_eval(My_model, dataloader)
    torch.save(My_model.module.state_dict(), args.model_save_path)
dist.barrier()  # 这一句作用是：所有进程(gpu)上的代码都执行到这，才会执行该句下面的代码
