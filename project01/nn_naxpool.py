# 最大池化：减少数据量

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1,],
#                       [2,1,0,1,1]],dtype=torch.float32)
#
# input = torch.reshape(input,(-1,1,5,5))
# print(input)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=torchvision.transforms.ToTensor())
dateloader = DataLoader(dataset,batch_size=64)

class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False) #加一个最大池化层

    def forward(self,input):
        return self.maxpool1(input)

snn = SimpleNN()

writer = SummaryWriter(log_dir="logs_maxpool")
step = 0
for data in dateloader:
    imgs,targets = data
    output = snn(imgs)
    writer.add_images(tag="input",img_tensor=imgs,global_step=step)
    writer.add_images(tag="output",img_tensor=output,global_step=step)
    step += 1

writer.close()