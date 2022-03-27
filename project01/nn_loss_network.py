# 把Loss Function加入nn进行反向传播，并优化

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, input):
        return self.model1(input)

loss = nn.CrossEntropyLoss()
snn = SimpleNN()
optim = torch.optim.SGD(snn.parameters(),lr=0.01) # 随机梯度下降
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        optim.zero_grad() # 上一轮梯度清零
        imgs,targets = data
        output = snn(imgs)
        ret_loss = loss(output,targets)
        running_loss += ret_loss # 叠加每轮的loss
        ret_loss.backward() # 反向传播
        optim.step() # 优化
    print(running_loss)

