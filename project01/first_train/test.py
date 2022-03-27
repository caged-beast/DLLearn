# 用训练好的模型进行测试
import torch
import torchvision
from PIL import Image
from model import *

img_path = "./horse.png"
image = Image.open(img_path)
print(image)
# 把图片转换成32*32
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

# 在gpu上训练的model测试也要用gpu
device = torch.device("cuda")

# 加载模型
model = torch.load("snn_9.pth")
model.to(device)
print(model)

# 测试
image = torch.reshape(image,(1,3,32,32)) # 先把测试数据转换成正确格式
image = image.to(device)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))