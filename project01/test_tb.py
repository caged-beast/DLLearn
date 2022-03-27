import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_path = "dataset/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=3x",3*i,i)

writer.close()

# 在命令行输入 tensorboard --logdir=logs --port=8888 设置端口，默认端口为6006