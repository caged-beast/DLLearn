# 损失函数
import torch
from torch import nn

input = torch.tensor([1,2,3],dtype=torch.float32)
output = torch.tensor([1,2,5],dtype=torch.float32)

# loss = nn.L1Loss(reduction='sum')
loss = nn.MSELoss()
ret = loss(input,output)
print(ret)

# 交叉熵
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss1 = nn.CrossEntropyLoss()
ret1 = loss1(x,y)
print(ret1)
