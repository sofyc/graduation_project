from metrics import psnr, ssim
import torchvision.transforms as transforms
import cv2
import torch

def png2tensor(image_path):
	img = cv2.imread(image_path)
	# b, g, r = cv2.split(img)
	# img = cv2.merge([r, g, b])
	transf = transforms.ToTensor()
	img_tensor = transf(img)
	img_tensor = img_tensor[:, 10:-10, 20:-20]
	# print(img_tensor)
	return img_tensor

x, y = png2tensor("DCP/synthetic/00004584.png"), png2tensor("original/synthetic/00004584.png")
meanx = torch.mean(x)
meany = torch.mean(y)
x = x * meany / meanx
# print(meanx, meany)
# exit()

x = x.unsqueeze(0)
y = y.unsqueeze(0)
ssim1 = ssim(x, y).item()
psnr1 = psnr(x, y)
print(ssim1)
print(psnr1)