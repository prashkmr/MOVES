#In this version, the generator is only trained to recosntruct the kitti images back not the carla dynamic images
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX

from utils512 import * 
from models512 import * 


parser = argparse.ArgumentParser(description='GAN training of LiDAR')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--loss', type=int, default=1, help='0 == LSGAN, 1 == RaLSGAN')

parser.add_argument('--base_dir', type=str, default='runs/test')
parser.add_argument('--dis_iters', type=int, default=1, help='disc iterations per 1 gen iter')
parser.add_argument('--no_polar', type=int, default=0)
parser.add_argument('--optim',  type=str, default='rmsprop')
parser.add_argument('--pose_dim',           type=int,   default=160,            help='size of the pose vector')
parser.add_argument('--data',  type=str, required= True, default='', help='Location of the dataset')
parser.add_argument('--log',  type=str, required= True, default='', help='Name of the log folder')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--ae_weight',          type=str,   default='',             required=True, help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')



#parser.add_argument('--atlas_baseline', type=int,   default=0)
#parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--gen_lr', type=float, default=1e-4)
parser.add_argument('--dis_lr', type=float, default=1e-4)
args = parser.parse_args()
DATA = args.data
RUN_SAVE_PATH = args.log
maybe_create_dir(args.base_dir+RUN_SAVE_PATH)
print_and_save_args(args, args.base_dir)

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



static  = []
dynamic = []
dynamick= []

def loadNP(index):
    global static, dynamic, stMask, dyMask
    print('Loading npys...')
    for i in trange(index):
        static.append( np.load( DATA  + "lidar/s{}.npy".format(i) ) [:,:,5:45,:].astype('float32'))
        dynamic.append( np.load( DATA + "lidar/d{}.npy".format(i)  )[:,:,5:45,:].astype('float32'))
        dynamick.append( np.load( DATA + "kitti/lidar/k{}.npy".format(i)  )[:,:,5:45,:].astype('float32'))
        

loadNP(4)


# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
writes = 0


dis = scene_discriminator(args.pose_dim).cuda()
gen = VAE(args).cuda()
genk= VAE(args).cuda()

# print(gen)
# print(dis)

gen.apply(weights_init)
dis.apply(weights_init)
genk.apply(weights_init)

class Attention_loader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """
    def __init__(self,static1, static2, dynamic1, dynamick):
        super(Attention_loader, self).__init__()
        self.static1   = static1
        self.static2   = static2
        self.dynamic1  = dynamic1
        self.dynamick  = dynamick
        # self.dynamick1 = dynamick1
 
    def __len__(self):
        return min(self.static1.shape[0]-1, self.static2.shape[0])-1

    def __getitem__(self, index):
        
        return index, self.static1[index], self.static2[index], self.dynamic1[index], self.dynamick[index], self.dynamick[index+1]


def load(npyList):
    retList=[]
    for i in npyList:
        print(i)
        s1 = static[i] 
        s2 = static[(i+1)%len(npyList)]
        d1 = dynamic[i]
        d2 = dynamick[(i+1)%len(npyList)]
     

        data_train = Attention_loader(s1, s2, d1, d2)

        train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
        #del data_train
        retList.append(train_loader)
    print(retList)
    return retList
import math
npyList = [i for i in range(4)]
npyList1 =load(npyList)


if args.optim.lower() == 'adam': 
    gen_optim = optim.Adam(list(gen.parameters()) + list(genk.parameters()), lr=args.gen_lr, betas=(0.5, 0.999), weight_decay=0)
    dis_optim = optim.Adam(dis.parameters(), lr=args.dis_lr, betas=(0.5, 0.999), weight_decay=0)
elif args.optim.lower() == 'rmsprop': 
    gen_optim = optim.RMSprop(list(gen.parameters()) + list(genk.parameters()), lr=args.gen_lr)
    dis_optim = optim.RMSprop(dis.parameters(), lr=args.dis_lr)


loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1)
# gan training


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False



def calc_mmd_loss(x, y, alpha=0.001):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    B = x.shape[0]

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    K = torch.exp(-alpha * (rx.t() + rx - 2 * xx))
    L = torch.exp(-alpha * (ry.t() + ry - 2 * yy))
    P = torch.exp(-alpha * (rx.t() + ry - 2 * zz))

    beta = 1.0 / (B * (B - 1))
    gamma = 2.0 / (B * B)

    return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)

triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
ck = torch.load(args.ae_weight)


gen.load_state_dict(ck['gen_dict'])
genk.load_state_dict(ck['gen_dict'])
dis.load_state_dict(ck['dis_dict'])
freeze(gen)
freeze(dis)

import math

# ------------------------------------------------------------------------------------------------
for epoch in range(600):
    # lmbda = 2

    lmbda = 2 / (1 + math.exp(-10 * (epoch) / 1000)) - 1
    print(lmbda)
    print('epochs: ',epoch)
    for i in npyList1:
        data_iter = iter(i)
        iters = 0
        real_d, fake_d, fake_g, losses_g, losses_d, delta_d, fake_d_carla, fake_d_kitti = [[] for _ in range(8)]
        process_input = from_polar if args.no_polar else lambda x : x

        while iters < len(i):
            j = 0
            # if iters > 10 : break
            # print(iters)
        
            """ Update Discriminator Network """
            for p in dis.parameters():
                p.requires_grad = True

            while j < args.dis_iters and iters < len(i):
                j += 1; iters += 1

                inputs = data_iter.next()
            
                # train with real data
                recon0, kl_cost0, z_real0 = gen(process_input(inputs[1].cuda()))
                recon1, kl_cost1, z_real1 = gen(process_input(inputs[2].cuda()))
               

                real_out = dis(z_real0, z_real1)
            
                real_d += [real_out.mean().detach()]
                
                # train with fake data 
                # noise = torch.cuda.FloatTensor(args.batch_size, 100).normal_()
                recon2, _  , z_fake2 = gen(process_input(inputs[3].cuda()))
                reconk, _  , z_fakek = genk(process_input(inputs[4].cuda()))

                

                fake_out_carla = dis(z_fake2 , z_real1)
                fake_out_kitti = dis(z_fakek , z_real1)
                

                fake_d_carla += [fake_out_carla.mean().detach()]
                fake_d_kitti += [fake_out_kitti.mean().detach()]
                
                if args.loss == 0 : 
                    dis_loss_carla = (((real_out - fake_out_carla.mean() - 1) ** 2).mean() + \
                                ((fake_out_carla - real_out.mean() + 1) ** 2).mean()) / 2

                    dis_loss_kitti = (((real_out - fake_out_kitti.mean() - 1) ** 2).mean() + \
                                ((fake_out_kitti - real_out.mean() + 1) ** 2).mean()) / 2
                else:
                    dis_loss_carla = (torch.mean((real_out - 1) ** 2) + torch.mean((fake_out_carla - 0) ** 2)) / 2
                    dis_loss_kitti = (torch.mean((real_out - 1) ** 2) + torch.mean((fake_out_kitti - 0) ** 2)) / 2

                dis_loss = (dis_loss_carla + dis_loss_kitti) / 2.0

                loss_contrastive = triplet_loss(z_real0, z_real1, z_fake2)
                dis_loss += loss_contrastive


               
                losses_d += [dis_loss.mean().detach()]
                # delta_d  += [(real_out.mean() - fake.mean()).detach()]

            
                dis_optim.zero_grad()
                dis_loss.backward()
                dis_optim.step()

            """ Update Generator network """
            for p in dis.parameters():
                p.requires_grad = False

            # noise = torch.cuda.FloatTensor(args.batch_size, 100).normal_()
            recon2, kl_cost2, z_fake2 = gen(process_input(inputs[3].cuda()))
            recon1, kl_cost1, z_real1 = gen(process_input(inputs[2].cuda()))
            recon0, kl_cost0, z_real0 = gen(process_input(inputs[1].cuda()))

            reconk, kl_costk, z_fakek = genk(process_input(inputs[4].cuda()))
            reconk1, _      , z_fakek1= genk(process_input(inputs[5].cuda()))
            

            fake_out_carla = dis(z_fake2, z_real1)
            fake_out_kitti = dis(z_fakek, z_real1) 
            fake_out = (fake_out_carla + fake_out_kitti) / 2.0

            fake_g += [fake_out.mean().detach()]        
            
            if args.loss == 0: 
                iters += 1
                inputs = inputs
            
                # raise SystemError
                real_out = dis(z_real0, z_real1)
                gen_loss_carla = (((real_out - fake_out_carla.mean() + 1) ** 2).mean() + \
                            ((fake_out_carla - real_out.mean() - 1) ** 2).mean()) / 2
                
                gen_loss_kitti = (((real_out - fake_out_kitti.mean() + 1) ** 2).mean() + \
                            ((fake_out_kitti - real_out.mean() - 1) ** 2).mean()) / 2

            else:
                gen_loss_carla = torch.mean((fake_out_carla - 1.) ** 2)
                gen_loss_kitti = torch.mean((fake_out_kitti - 1.) ** 2)

            gen_loss = (gen_loss_carla + gen_loss_kitti) / 2.0

           

            
            recloss_kitti = loss_fn(reconk[:,:,0:40,:], inputs[4].cuda()).mean(dim=0)
            
            mmd     = calc_mmd_loss(z_real0.view(args.batch_size, -1), z_fakek.view(args.batch_size, -1))
            mmd1 = 0

            
            gen_loss +=  lmbda*mmd1 + recloss_kitti

            losses_g += [gen_loss.detach()]
        
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

        
        print_and_log_scalar(writer, 'recloss-kitti', recloss_kitti.detach(), writes)
        print_and_log_scalar(writer, 'real_out', real_d, writes)
        print_and_log_scalar(writer, 'fake_out', fake_d, writes)
        print_and_log_scalar(writer, 'fake_out_g', fake_g, writes)
        print_and_log_scalar(writer, 'delta_d', delta_d, writes)
        print_and_log_scalar(writer, 'losses_gen', losses_g, writes)
        print_and_log_scalar(writer, 'losses_dis', losses_d, writes)
        writes += 1

        # save some training reconstructions
 

        if (epoch) % 15 == 0 :

            
            state = {
            'epoch': epoch + 1, 
            'genk_dict': genk.state_dict(),
            'gen_dict': gen.state_dict()
            # 'gen_optim': gen_optim.state_dict(),
            # 'dis_optim': dis_optim.state_dict()
            }
            torch.save(state, os.path.join(args.base_dir + RUN_SAVE_PATH, 'gen_{}.pth'.format(epoch)))
            print('saved models')