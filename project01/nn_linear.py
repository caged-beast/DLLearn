# 线性激活
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=torchvision.transforms.ToTensor())
dateloader = DataLoader(dataset,batch_size=64,drop_last=True)

class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Linear(in_features=196608,out_features=10) # 加一个非线性激活层
    def forward(self,x):
        return self.conv1(x)

snn = SimpleNN()

writer = SummaryWriter(log_dir="logs_activation")
step = 0
for data in dateloader:
    imgs,targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs) # 展平
    print(output.shape)
    output = snn(output)
    print(output.shape)
    # output = snn(imgs)
    # writer.add_images(tag="input",img_tensor=imgs,global_step=step)
    # writer.add_images(tag="output",img_tensor=output,global_step=step)
    step += 1

writer.close()