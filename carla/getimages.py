from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils512 import *
import numpy as np
import os
import open3d as o3d
from open3d import open3d
from tqdm import trange



parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',       type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--folder',     type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--debug', action='store_true')
# args = parser.parse_args(args=['atlas_baseline=0, autoencoder=1,panos_baseline=0'])





# Encoder must be trained with all types of frames,dynmaic, static all


args = parser.parse_args()


original = np.load(args.data)


original = np.load(args.data)[:,:,:,::2][:]
if original.shape[1]==2:
	original = from_polar_np(original)


if not os.path.exists('samplesVid/images'):
	os.makedirs('samplesVid/images') 


i=0
for frame_num in trange(  original.shape[0]):
	if(i<2048):
		i+=1
		frame = (torch.Tensor(original[frame_num:frame_num+1]).cuda())
		frame = frame.detach().cpu().numpy()
		# frame=from_polar(torch.Tensor(original[frame_num:frame_num+1,:,:,:]).cuda()).detach().cpu()
		plt.figure()
		plt.xlim([-0.15, 0.15])
		plt.ylim([-0.20, 0.20])
		plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
		plt.savefig('samplesVid/images/'+str(frame_num)+'.jpg') 
	else:
		break
                                                          
