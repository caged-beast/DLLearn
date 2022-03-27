import torch
import torch.nn.functional as F
# 卷积操作

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1,],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

print(F.conv2d(input,kernel,stride=1));
print(F.conv2d(input,kernel,stride=2));
print(F.conv2d(input,kernel,stride=1,padding=1));