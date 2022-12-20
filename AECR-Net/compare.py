from metrics import psnr, ssim
import torchvision.transforms as transforms
import cv2

def png2tensor(image_path):
	img = cv2.imread(image_path)
	# b, g, r = cv2.split(img)
	# img = cv2.merge([r, g, b])
	transf = transforms.ToTensor()
	img_tensor = transf(img)
	return img_tensor

x, y = png2tensor("data/00004584_clear.png"), png2tensor("data/result.png")
x = x.unsqueeze(0)
y = y.unsqueeze(0)
ssim1 = ssim(x, y).item()
psnr1 = psnr(x, y)
print(ssim1)
print(psnr1)