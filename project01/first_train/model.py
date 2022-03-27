# 搭建神经网络
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

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


if __name__ == '__main__':  # 测试Model正确性
    snn = SimpleNN()
    input = torch.ones((64, 3, 32, 32))
    output = snn(input)
    print(output.shape)
