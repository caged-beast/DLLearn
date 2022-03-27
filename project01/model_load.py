# 加载模型

import torch
import torchvision

# 加载方式1
# model = torch.load("vgg16_method1.pth")
# print(model)

# 加载方式2
from torch import nn

vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 加载方式1有一个陷阱：要把模型加入当前文件
class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)
    def forward(self,x):
        return self.conv1(x)

model = torch.load("snn.pth")
print(model)