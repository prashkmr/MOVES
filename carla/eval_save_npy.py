#This is the new eval that saves to disk the files are recosntructing them
#instead of using my_Eval.py now use this for evaluating a recosntruction

from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm,trange 
from utils512 import * 
from torch.utils.data import DataLoader, Dataset
from models512 import *

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=2048,            help='size of minibatch used during training')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--data',               type=str,   default='',             required=True, help='Location of the data to train')
parser.add_argument('--ae_weight',          type=str,   default='',             required=True, help='Location of the data to train')
# parser.add_argument('--savename',           type=str,   default='',             required=True, help='Name of the reconstructed npy to save')

parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

FILE_PATH = args.data


def save_on_disk(static,dynamic,reconstructedStaticWhole):
	np.savez('SaveRecons/static',static.cpu())
	np.savez('SaveRecons/dynamic',dynamic)
	np.savez('SaveRecons/recon_st',reconstructedStaticWhole)

class Attention_loader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self,lidar):
        super(Attention_loader, self).__init__()

        self.lidar = lidar
        # self.mask = mask

    def __len__(self):
        return self.lidar.shape[0]

    def __getitem__(self, index):
        
        return index, self.lidar[index]#, self.mask[index]






model = VAE(args).cuda()


network = torch.load(args.ae_weight)


model.load_state_dict(network['gen_dict'])
model.eval()



npy = [8]

process_input = from_polar if args.no_polar else lambda x : x
# print 






recons=[]
original=[]

for file in npy:

    lidar_train = (np.load(args.data + "lidar/d{}.npy".format(str(file)))[:,:,5:45,:])
    # print(lidar_train.shape)  #2048,2,40,512
    
    data_val = Attention_loader(lidar_train)

    val_loader  = torch.utils.data.DataLoader(data_val, batch_size=args.batch_size,
                shuffle=False, num_workers=4, drop_last=False)
    
    output =[]
    for i, img in enumerate(val_loader):
      
        image = img[1].cuda()
        recon, kl_cost,z = model(process_input(image))
        output.append(recon[:,:,:40].detach().cpu().numpy())
        print(recon.shape)
   
    
    np.save(str(file) +'moves.npy',np.concatenate(output, axis = 0))
    print('Saved', file)
