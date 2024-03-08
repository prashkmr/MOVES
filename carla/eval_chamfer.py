#This is the code where we evaluate our method/model on CARLA on EMD/Chamfer / Result is 210 ans 1.

import argparse
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
import torch
import sys

import numpy as np
import os

from utils512 import * 

from models512 import *

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=128,            help='size of minibatch used during training')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--ae_weight',          type=str,   default='',             help='Location of the weights')
parser.add_argument('--data',               type=str,   default='',             help='Loction of the dataset')
parser.add_argument('--emb_dims', type=int,          default=1024, metavar='N', help='Dimension of embeddings')

parser.add_argument('--debug', action='store_true')




#---------------------------------------------------------------
#Helper Function and classes
class Pairdata(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self,pairDynamic, pairStatic):
        super(Pairdata, self).__init__()
        
        self.pairDynamic       = pairDynamic
        self.pairStatic      = pairStatic

    def __len__(self):
        return self.pairDynamic.shape[0]

    def __getitem__(self, index):
        
        return index, self.pairDynamic[index], self.pairStatic[index]

#-------------------------------------------------------------------------------
args = parser.parse_args()



# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200

save_test_dataset = False

fast = True



model = VAE(args).cuda()



model = model.cuda()
network=torch.load(args.ae_weight)

model.load_state_dict(network['gen_dict'])

model.eval() 

loss = get_chamfer_dist

# npydata = ['8','9','10','11','12','13','14']
npydata = ['15']
total = 0
with torch.no_grad():
  for i in npydata:


    print('test set reconstruction')

    print('loading Testing data')

    
    lidar    = (np.load(args.data + "lidar/d{}.npy".format(str(i)))[:1024,:,5:45,:])#.astype('float32').reshape(-1, 3, 1024)
    lidarSt  = (np.load(args.data + "lidar/s{}.npy".format(str(i)))[:1024,:,5:45,:])#.astype('float32').reshape(-1, 3, 1024)
    
    
    # args.batch_size = 160
    test_loader    = Pairdata(lidar, lidarSt)
    loader = (torch.utils.data.DataLoader(test_loader, batch_size= args.batch_size,
                        shuffle=False, num_workers=1, drop_last=True)) #False))

    loss_fn = loss()

    process_input = from_polar if args.no_polar else lambda x : x
    

    for noise in [0]:
        losses = []
        # losses1 = []
        ind = 0 
        for batch in loader:
            lidar = batch[1].cuda()
            # mask  = batch[2].cuda()
            lidarStat=batch[2].cuda()
           
            recon,_,_  = model(process_input(lidar))
            
            recon = recon[:,:,:40,:]
            
           
            losses += [loss_fn(from_polar(recon.reshape(-1,2,40,512)), from_polar(lidarStat.reshape(-1,2,40,512)))]

        losses = torch.stack(losses).mean().item()
        
        print('Chamfer Loss for {}: {:.4f}'.format(i, losses))
        total +=losses
        

        del recon, losses

