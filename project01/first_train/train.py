# 一次完整的训练过程
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
import torch
import torch.nn as nn

# 想要用gpu，就要对网络模型、loss function、数据都调用.cuda()
# 或者调用.to(torch.device("cuda"))
# 对网络模型、loss function调用函数后可以不用重新赋值

# 读取数据集
train_data = torchvision.datasets.CIFAR10(root="../CIFAR10_dataset",train=True,
                                          transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../CIFAR10_dataset",train=False,
                                          transform=torchvision.transforms.ToTensor(),download=True)
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

# lenth
print("训练数据集大小为:{}".format(len(train_data)))
print("测试数据集大小为:{}".format(len(test_data)))

# 定义训练的设备
device = torch.device("cuda")

# 创建网络模型
snn = SimpleNN()
snn = snn.to(device)
# if(torch.cuda.is_available()):
#     snn = snn.cuda()

# 损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)
# if(torch.cuda.is_available()):
#     loss_func = loss_func.cuda()

# 优化器
learnning_rate = 1e-2 # 0.01
optimizer = torch.optim.SGD(snn.parameters(),learnning_rate)

# 设置参数
total_train_step = 0
total_test_step = 0
epoch = 10

# 使用tensorboard
writer = SummaryWriter(log_dir="../logs_train")

# 开始训练
for i in range(epoch):
    print("-------------第{}轮测试开始--------------".format(i+1))
    snn.train()
    for data in train_loader:
        imgs,targets = data
        # if (torch.cuda.is_available()):
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = snn(imgs)
        loss = loss_func(outputs,targets)
        # 优化参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if(total_train_step % 100 == 0):
            print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar(tag="train_loss",scalar_value=loss,global_step=total_train_step)

    # 测试步骤开始
    snn.eval()
    total_test_loss = 0.0
    total_accuracy = 0.0 # 对于分类问题，正确率是更好的衡量指标
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # if (torch.cuda.is_available()):
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = snn(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
            total_test_step += 1
    print("测试次数:{},总的Loss:{}".format(total_test_step, total_test_loss))
    print("测试次数:{},正确率:{}".format(total_test_step, total_accuracy/len(test_data)))
    writer.add_scalar(tag="total_test_loss", scalar_value=total_test_loss, global_step=total_test_step)
    writer.add_scalar(tag="total_accuracy", scalar_value=total_accuracy/len(test_data), global_step=total_test_step)

    # 保存模型
    torch.save(snn,"snn_{}.pth".format(i))
    print("模型已保存")

writer.close()