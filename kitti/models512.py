import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import numpy as np
import os
import pdb
from utils512 import *


# --------------------------------------------------------------------------
# Core Models
# --------------------------------------------------------------------------
class netG(nn.Module):  #decoder
    def __init__(self, args, nz=100, ngf=64, nc=3, base=4, ff=(1,8)):
        super(netG, self).__init__()
        self.args = args
        conv = nn.ConvTranspose2d

        layers  = []
        layers += [nn.ConvTranspose2d(nz, ngf * 8, ff, 1, 0, bias=False)]
        layers += [nn.BatchNorm2d(ngf * 8)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 8, ngf * 4, (2,4), stride=2, padding=(0,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 4)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 4, ngf * 4, (4,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 4)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 4, ngf * 2, (3,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 2)]
        layers += [nn.ReLU(True)]


        layers += [nn.ConvTranspose2d(ngf * 2, ngf * 2, (2,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 2)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 2, ngf, (1,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)]
        layers += [nn.Tanh()]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(-1).unsqueeze(-1)

        return self.main(input)


class scene_discriminator(nn.Module):
    def __init__(self, pose_dim, nf=256):
        super(scene_discriminator, self).__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
                # nn.Dropout(p=0.5),
                nn.Linear(pose_dim*2, int(pose_dim)),
                nn.Sigmoid(),
                # nn.Dropout(p=0.5),
                nn.Linear(int(pose_dim), int(pose_dim/2)),
                nn.Sigmoid(),
                # nn.Dropout(p=0.5),
                nn.Linear(int(pose_dim/2), int(pose_dim/4)),
                nn.Sigmoid(),
                nn.Linear(int(pose_dim/4),int(pose_dim/8)),
                nn.Sigmoid(),
                nn.Linear(int(pose_dim/8),1),
                nn.Sigmoid()
                )


    def forward(self, input1,input2):
        output = self.main(torch.cat((input1, input2),1).view(-1, self.pose_dim*2))
        return output







class netD(nn.Module): #encoder
    def __init__(self, args, ndf=64, nc=2, nz=1, lf=(1,8)):
        super(netD, self).__init__()
        self.encoder = True if nz > 1 else False

        layers  = []
        layers += [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)]

        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf*2, ndf * 2, 3, 2, 1, bias=False)]

        layers += [nn.BatchNorm2d(ndf * 2)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)]

        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False)]

        layers += [nn.BatchNorm2d(ndf * 4)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf * 4, ndf * 8, (2,4), 2, (0,1), bias=False)]

        layers += [nn.BatchNorm2d(ndf * 8)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        self.main = nn.Sequential(*layers)
        self.out  = nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False)

    def forward(self, input, return_hidden=False):
        if input.size(-1) == 3:
            input = input.transpose(1, 3)

        output_tmp = self.main(input)
        output = self.out(output_tmp)

        if return_hidden:
            return output, output_tmp

        return output if self.encoder else output.view(-1, 1).squeeze(1)




class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args

        
        mult = 1 if args.autoencoder else 2
        self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
        self.decode = netG(args, nz=args.z_dim, nc=2)

    def forward(self, x):
        z = self.encode(x)
        while z.dim() != 2:
            z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None, z
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)




