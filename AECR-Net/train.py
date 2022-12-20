import torch, os, sys, torchvision, argparse
import torchvision.transforms as transforms
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
# from option import opt, log_dir
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from metrics import psnr, ssim
from models.AECRNet import *
from models.CR import *
import cv2


import json
from tqdm import trange

warnings.filterwarnings('ignore')
device='cuda' if torch.cuda.is_available() else 'cpu'

def png2tensor(image_path):
	img = cv2.imread(image_path)
	# b, g, r = cv2.split(img)
	# img = cv2.merge([r, g, b])
	transf = transforms.ToTensor()
	img_tensor = transf(img)
	return img_tensor

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--model_name", type=str, default="test_train")
	parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--w_loss_l1', type=float, default=1)
	parser.add_argument('--w_loss_vgg7', type=float, default=0)
	parser.add_argument('--model_dir', type=str,default='./trained_models/')

	args = parser.parse_args()
	return args

args = parse_args()

def lr_schedule_cosdecay(t,T,init_lr=args.lr):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr


def train(net, clear_train_paths, haze_train_paths, clear_test_paths, haze_test_paths, optim, criterion):
	losses = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	ssims = []
	psnrs = []
	total_steps = args.epochs * args.steps
	print('train from scratch *** ')
	for epoch in range(args.epochs):
		for step in trange(start_step, args.steps):
			overall_step = epoch*steps + step
			net.train()
			lr = lr_schedule_cosdecay(overall_step, total_steps)
			for param_group in optim.param_groups:
				param_group["lr"] = lr

			x, y = png2tensor(haze_train_paths[step]), png2tensor(clear_train_paths[step])
			# x, y = next(iter(loader_train)) # [x, y] 10:10
			x = x.unsqueeze(0).to(device)
			y = y.unsqueeze(0).to(device)

			out = net(x)

			loss_vgg7, all_ap, all_an, loss_rec = 0, 0, 0, 0
			if args.w_loss_l1 > 0:
				loss_rec = criterion[0](out, y)
			if args.w_loss_vgg7 > 0:
				loss_vgg7, all_ap, all_an = criterion[1](out, y, x)

			loss = args.w_loss_l1*loss_rec + args.w_loss_vgg7*loss_vgg7
			loss.backward()
			
			optim.step()
			optim.zero_grad()
			losses.append(loss.item())

			print(f'\rloss:{loss.item():.5f} l1:{args.w_loss_l1*loss_rec:.5f} contrast: {args.w_loss_vgg7*loss_vgg7:.5f} all_ap:{all_ap:.5f} all_an:{all_an:.5f}| step :{step}/{steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',end='', flush=False)

		# save_model_dir = f'{args.model_dir}/{args.model_name}/{epoch}.txt'
		with torch.no_grad():
			ssim_eval, psnr_eval = test(net, clear_test_paths, haze_test_paths, args, epoch)

		log = f'\nstep :{overall_step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'

		print(log)
		with open(f'./logs_train/{args.model_name}.txt', 'a') as f:
			f.write(log + '\n')

		ssims.append(ssim_eval)
		psnrs.append(psnr_eval)

		if psnr_eval > max_psnr:
			max_ssim = max(max_ssim, ssim_eval)
			max_psnr = max(max_psnr, psnr_eval)
			save_model_dir = f'{args.model_dir}/{args.model_name}/model.best'
			print(f'\n model saved at step :{overall_step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')

			torch.save({
				'epoch': epoch,
				'step': overall_step,
				'max_psnr': max_psnr,
				'max_ssim': max_ssim,
				'ssims': ssims,
				'psnrs': psnrs,
				'losses': losses,
				'model': net.state_dict(),
				'optimizer': optim.state_dict()
			}, save_model_dir)

	# np.save(f'./numpy_files/{model_name}_{steps}_losses.npy', losses)
	# np.save(f'./numpy_files/{model_name}_{steps}_ssims.npy', ssims)
	# np.save(f'./numpy_files/{model_name}_{steps}_psnrs.npy', psnrs)

def test(net, clear_test_paths, haze_test_paths, args, epoch):
	net.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []

	for i in range(len(clear_test_paths)):
		x, y = png2tensor(haze_test_paths[i]), png2tensor(clear_test_paths[i])

		x = x.unsqueeze(0).to(device)
		y = y.unsqueeze(0).to(device)
		with torch.no_grad():
			pred = net(x)

		ssim1 = ssim(pred, y).item()
		psnr1 = psnr(pred, y)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		pred[pred>1] = 1
		pred[pred<0] = 0
		pred = (pred.squeeze(0).permute(1 ,2 ,0)*255).int()
		cv2.imwrite(f'{args.model_dir}/{args.model_name}/result_{epoch}.png', pred.cpu().numpy())

	return np.mean(ssims), np.mean(psnrs)

def set_seed_torch(seed=2018):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
	start_time = time.time()
	print(args)
	if not os.path.exists(args.model_dir):
		os.mkdir(args.model_dir)
	if not os.path.exists(args.model_dir + args.model_name):
		os.mkdir(args.model_dir + args.model_name)

	set_seed_torch(666)
	basedir = args.data_dir
	clear_train_paths = [os.path.join(basedir, 'clear/train', f) for f in sorted(os.listdir(basedir + '/clear/train')) \
		if f.endswith('png')]
	clear_test_paths = [os.path.join(basedir, 'clear/test', f) for f in sorted(os.listdir(basedir + '/clear/test')) \
		if f.endswith('png')]
	haze_train_paths = [os.path.join(basedir, 'haze/train', f) for f in sorted(os.listdir(basedir + '/haze/train')) \
		if f.endswith('png')]
	haze_test_paths = [os.path.join(basedir, 'haze/test', f) for f in sorted(os.listdir(basedir + '/haze/test')) \
		if f.endswith('png')]

	# if not opt.resume and os.path.exists(f'./logs_train/{opt.model_name}.txt'):
	# 	print(f'./logs_train/{opt.model_name}.txt 已存在，请删除该文件……')
	# 	exit()
	if not os.path.exists(f'./logs_train'):
		os.mkdir(f'./logs_train')

	with open(f'./logs_train/{args.model_name}.txt', 'w+') as f:
		json.dump(args.__dict__, f, indent=2)

	# loader_train = loaders_[opt.trainset]
	# loader_test = loaders_[opt.testset]
	net = Dehaze(3, 3)
	net = net.to(device)
	steps = len(clear_train_paths)
	args.steps = steps

	print("epoch_size: ", steps)
	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True

	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total_params: ==> {}".format(pytorch_total_params))

	criterion = []
	criterion.append(nn.L1Loss().to(device))
	criterion.append(ContrastLoss(ablation=False))



	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net, clear_train_paths, haze_train_paths, clear_test_paths, haze_test_paths, optimizer, criterion)

	# train(net, loader_train, loader_test, optimizer, criterion)
	

