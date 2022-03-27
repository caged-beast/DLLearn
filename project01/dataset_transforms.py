# dataset和transforms结合

import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 下载官方数据集 可以复制下载链接到迅雷下载
train_set = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset",train=True,transform=dataset_trans)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset",train=False,download=True,transform=dataset_trans)

# print(test_set[0])
# print(test_set.classes)
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("CIFAR10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()

