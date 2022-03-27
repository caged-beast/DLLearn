# 保存模型
import torch
import torchvision

# 保存方式1，保存模型+参数
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# torch.save(vgg16,"vgg16_method1.pth")

# 保存方式2，只保存参数
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

# 加载方式1有个陷阱
class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)
    def forward(self,x):
        return self.conv1(x)

snn = SimpleNN()
torch.save(snn,"snn.pth")