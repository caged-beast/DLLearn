import torch
from torch import nn
# nn示例

class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,input):
        return input+1

snn = SimpleNN()
x = torch.tensor(1.0)
output = snn(x)
print(output)