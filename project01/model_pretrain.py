# 使用别人的Model

import torch
import torch.nn as nn
import torchvision

vgg16_false = torchvision.models.vgg16(pretrained=False)
# pretrained=True 就使用别人训练好的参数
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true) # vgg16模型用于ImageNet数据集，最后一个线性层是4096到1000，可以分1000个类
# 要想用到CIFAR-10数据集上，要自己加一个1000到10的线性层，或者把最后一层改成4096到10
# vgg16_true.classifier[6] = nn.Linear(4096,10)
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)
