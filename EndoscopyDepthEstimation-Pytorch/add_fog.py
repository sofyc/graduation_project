import torch
import torchvision.transforms as transforms
import cv2
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import glob
import PIL.Image as pil
from  matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import random
import imageio

# def read_pfm(fpath, expected_identifier="Pf"):
#     # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    
#     def _get_next_line(f):
#         next_line = f.readline().decode('utf-8').rstrip()
#         # ignore comments
#         while next_line.startswith('#'):
#             next_line = f.readline().rstrip()
#         return next_line
    
#     with open(fpath, 'rb') as f:
#         #  header
#         identifier = _get_next_line(f)
#         if identifier != expected_identifier:
#             raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

#         try:
#             line_dimensions = _get_next_line(f)
#             dimensions = line_dimensions.split(' ')
#             width = int(dimensions[0].strip())
#             height = int(dimensions[1].strip())
#         except:
#             raise Exception('Could not parse dimensions: "%s". '
#                             'Expected "width height", e.g. "512 512".' % line_dimensions)

#         try:
#             line_scale = _get_next_line(f)
#             scale = float(line_scale)
#             assert scale != 0
#             if scale < 0:
#                 endianness = "<"
#             else:
#                 endianness = ">"
#         except:
#             raise Exception('Could not parse max value / endianess information: "%s". '
#                             'Should be a non-zero number.' % line_scale)

#         try:
#             data = np.fromfile(f, "%sf" % endianness)
#             data = np.reshape(data, (height, width))
#             #data = np.flipud(data)
#             with np.errstate(invalid="ignore"):
#                 data *= abs(scale)
#         except:
#             raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

#         return np.ascontiguousarray(data)

def png2tensor(image_path):
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    transf = transforms.ToTensor()
    img_tensor = transf(img)
    return img_tensor

def pfm2tensor(image_path):
    img_np = read_pfm(image_path)
    img_tensor = torch.from_numpy(img_np)
    return img_tensor

def png2numpy(image_path):
    img_np = imageio.imread(image_path, ignoregamma=True)
    return img_np

def pfm2numpy(image_path):
    img_np = read_pfm(image_path)
    return img_np

def add_haze(image_path, depth_path, output_path, env_light):
    '''
    Note: env_light is a list representing [opacity, Tensor[r, g, b]] of haze.
    '''
    img_tensor = png2tensor(image_path)
    dep_tensor = torch.from_numpy(np.load(depth_path))

    # this is a medium output to examine depth map
    toPIL = transforms.ToPILImage()
    medium = toPIL(dep_tensor[None,...,:].expand([3,-1,-1])/10)

    haze_opacity = env_light[0]
    haze_color = env_light[1]

    hazy_mask_tensor = torch.exp(-haze_opacity*dep_tensor)[None,...,:].expand(3,-1,-1)

    hazy_image_tensor = img_tensor * hazy_mask_tensor + haze_color[...,None,None] * (1-hazy_mask_tensor)

    toPIL = transforms.ToPILImage()
    output = toPIL(hazy_image_tensor)
    output.save(output_path)

    return hazy_image_tensor

def choose_env():
    return [0.05, torch.Tensor([0.5774,0.5774,0.5774])]

if __name__ == '__main__':
    basedir = "/home/fuyc/EndoscopyDepthEstimation-Pytorch/image"
    image_paths = [os.path.join(basedir, 'color', f) for f in sorted(os.listdir(basedir + '/color')) \
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depth_paths = [os.path.join(basedir, 'depth', f) for f in sorted(os.listdir(basedir + '/depth')) \
        if f.endswith('npy')]
    # image_paths = [os.path.join(basedir, f) for f in sorted(os.listdir(basedir)) \
    #     if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # depth_paths = [os.path.join(basedir, f) for f in sorted(os.listdir(basedir)) \
    #     if f.endswith('pfm')]
    env_light = choose_env()

    for i in trange(len(image_paths)):
        image_path = image_paths[i]
        depth_path = depth_paths[i]
        output_path = os.path.join(basedir, 'result_haze', os.path.split(image_path)[1])
        add_haze(image_path, depth_path, output_path, env_light)
