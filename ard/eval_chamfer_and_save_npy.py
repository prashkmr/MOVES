
from __future__ import print_function

import argparse
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
import torch
import sys
from torchsummary import summary
import numpy as np
import os
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models512 import *
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

import sklearn.metrics as metrics
from utils512 import *


parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--ae_weight',          type=str,   default='',             help='Location of the weights')
parser.add_argument('--data',               type=str,   default='',             help='Loction of the dataset')
parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')



#---------------------------------------------------------------
#Helper Function and classes
class Pairdata(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, lidar):
        super(Pairdata, self).__init__()
        
        self.lidar = lidar

    def __len__(self):
        return self.lidar.shape[0]

    def __getitem__(self, index):
        
        return index, self.lidar[index]

#---------------------------------------------------------------
args = parser.parse_args()

class Attention_loader_dytost(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dynamic, static):
        super(Attention_loader_dytost, self).__init__()

        self.dynamic = dynamic
        self.static = static

    def __len__(self):
        return self.dynamic.shape[0]

    def __getitem__(self, index):
        
        return index, self.dynamic[index], self.static[index]





def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """

    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist






# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
# out_dir = os.path.join(sys.argv[1], 'final_samples')
# maybe_create_dir(out_dir)
save_test_dataset = False

fast = True


loss = get_chamfer_dist

size = 8

npydata = [3]
# npydata = [9 ,14]
orig = []
pred = []

out = np.ndarray(shape=(3072,3,12,512))

totalcd = 0
totalhd = 0

with torch.no_grad():
  
  for i in npydata:
    ii = 0 
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = VAE(args).to(device)
    

    model = model.cuda()
    
    weight = torch.load(args.ae_weight)
    model.load_state_dict(weight['gen_dict'])
    
    
    model.eval() 

    
    
    lidar_static    = np.load(args.data + "lidar/d{}.npy".format(str(i)))[:,:,2:14,::2].astype('float32')
    lidar_dynamic   = np.load(args.data + "lidar/s{}.npy".format(str(i)))[:,:,2:14,::2].astype('float32')
    

    test_loader    = Attention_loader_dytost(lidar_dynamic, lidar_static)
    
    loader = (torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
                        shuffle=False, num_workers=1, drop_last=True)) #False))

    loss_fn = loss()
    # process_input = (lambda x : x) if model.args.no_polar else to_polar
    process_input = from_polar if args.no_polar else lambda x : x
    
    # noisy reconstruction
    for noise in [0]:
        losses, losses1 = [], []
        # losses1 = []
        for batch in loader:
            lidar_dynamic = batch[1].cuda()
            lidar_static = batch[2].cuda()
            # print(lidar_dynamic.shape)
            recon, _,_  = model( lidar_dynamic )
            recon = recon[:,:,:12,:]
            # print(recon.shape)
           
            # print(recon.shape, lidar_static.shape)
            losses += [loss_fn(from_polar(recon), from_polar(lidar_static))]
            
            # print(recon.shape)  [8,3,10240]
            out[ii*args.batch_size:(ii+1)*args.batch_size]   =    from_polar(recon).detach().cpu().numpy().reshape(-1, 3, 12, 512)
            ii+=1
        np.save( str(i) + '.npy', out)    
        # print('Saved ', i)

        losses = torch.stack(losses).mean().item()

        totalcd += losses

        print('Chamfer Loss for {}: {:.4f}'.format(i, losses))


        del recon, losses

