import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(root="./CIFAR10_dataset",train=False,transform=dataset_trans,download=True)

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

##测试数据集中第一张图片
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("Dataloader_logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        #注意这里是add_images，加s
        step += 1

writer.close()