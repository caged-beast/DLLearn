from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# transforms把PIL图片转换成tensor格式

img_path = "dataset/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img_PIL = Image.open(img_path)
writer = SummaryWriter("logs")

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img_PIL)
writer.add_image("Tensor_img",img_tensor)

# Normalize归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Norm_img",img_norm)

# Resize
print(img_PIL.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img_PIL)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize_img",img_resize)

# Compose

trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img_PIL)
writer.add_image("Resize_img_2",img_resize_2)

# RandomCrop 随机裁剪

trans_random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img_PIL)
    writer.add_image("RandomCrop_img", img_crop,i)

writer.close()

