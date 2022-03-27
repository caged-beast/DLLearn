# 操作数据集
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,
                            stride=1,padding=0) # 加一个卷积层
    def forward(self,input):
        x = self.conv1(input)
        return x

snn = SimpleNN()

writer = SummaryWriter("logs_nn")
step = 0
for data in dataloader:
    imgs,targets = data
    output = snn(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images(tag="input",img_tensor=imgs,global_step=step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images(tag="output",img_tensor=output,global_step=step)
    step += 1

writer.close()