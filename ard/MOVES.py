import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
from tqdm import trange
from utils512 import * 
from models512 import * 
import tensorboardX

parser = argparse.ArgumentParser(description='GAN training of LiDAR')
parser.add_argument('--batch_size', type=int, default=256)
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
stMask  = []
dyMask  = []
def loadNP(index):
    global static, dynamic, stMask, dyMask
    print('Loading npys...')
    for i in trange(index):
        static.append( np.load( DATA  + "lidar/s{}.npy".format(i) ) [:,:,2:14,::2].astype('float32'))
        dynamic.append( np.load( DATA + "lidar/d{}.npy".format(i)  )[:,:,2:14,::2].astype('float32'))


loadNP(3)


# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
writes = 0


# construct model and ship to GPU
dis = scene_discriminator(args.pose_dim).cuda()
gen = VAE(args).cuda()

print(gen)
print(dis)

gen.apply(weights_init)
dis.apply(weights_init)



triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
#  output = triplet_loss(anchor, positive, negative)





class Attention_loader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """
    def __init__(self,static1, static2, dynamic1):
        super(Attention_loader, self).__init__()

        self.static1 = static1
        self.static2 = static2
        self.dynamic1= dynamic1
 
    def __len__(self):
        return min(self.static1.shape[0], self.static2.shape[0])

    def __getitem__(self, index):
        
        return index, self.static1[index],self.static2[index], self.dynamic1[index] 


def load(npyList):
    retList=[]
    for i in npyList:
        print(i)
        s1 = static[i] 
        s2 = static[(i+1)%len(npyList)]
        d1 = dynamic[i]

     

        data_train = Attention_loader(s1, s2, d1)

        train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
        #del data_train
        retList.append(train_loader)
    print(retList)
    return retList

npyList = [i for i in range(3)]
npyList1 =load(npyList)





if args.optim.lower() == 'adam': 
    gen_optim = optim.Adam(gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.999), weight_decay=0)
    dis_optim = optim.Adam(dis.parameters(), lr=args.dis_lr, betas=(0.5, 0.999), weight_decay=0)
elif args.optim.lower() == 'rmsprop': 
    gen_optim = optim.RMSprop(gen.parameters(), lr=args.gen_lr)
    dis_optim = optim.RMSprop(dis.parameters(), lr=args.dis_lr)


loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1)
# gan training
# ------------------------------------------------------------------------------------------------
for epoch in range(1501):
    print('epochs: ',epoch)
    for i in npyList1:
        data_iter = iter(i)
        iters = 0
        real_d, fake_d, fake_g, losses_g, losses_d, delta_d = [[] for _ in range(6)]
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
               

                real_out = dis(z_real0,z_real1)
            
                real_d += [real_out.mean().detach()]
                
                # train with fake data 
                # noise = torch.cuda.FloatTensor(args.batch_size, 100).normal_()
                recon2, kl_cost2, z_fake2 = gen(process_input(inputs[3].cuda()))

                fake_out = dis(z_fake2 , z_real1)

                fake_d += [fake_out.mean().detach()]
                
                if args.loss == 0 : 
                    dis_loss = (((real_out - fake_out.mean() - 1) ** 2).mean() + \
                                ((fake_out - real_out.mean() + 1) ** 2).mean()) / 2
                else:
                    dis_loss = (torch.mean((real_out - 1) ** 2) + torch.mean((fake_out - 0) ** 2)) / 2

                
            
                #---------------------------------
                #Add Contrastive loss to total discrimiantor Loss
                
                
                loss_contrastive = triplet_loss(z_real0, z_real1, z_fake2)
                dis_loss += loss_contrastive
                #---------------------------------
                
                losses_d += [dis_loss.mean().detach()]
                #delta_d  += [(real_out.mean() - fake.mean()).detach()]

            
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
            fake_out = dis(z_fake2, z_real1)

            fake_g += [fake_out.mean().detach()]        
            
            if args.loss == 0: 
                iters += 1
                inputs = inputs
            
                # raise SystemError
                real_out = dis(z_real0, z_real1)
                gen_loss = (((real_out - fake_out.mean() + 1) ** 2).mean() + \
                            ((fake_out - real_out.mean() - 1) ** 2).mean()) / 2
            else:
                gen_loss = torch.mean((fake_out - 1.) ** 2)

            
                

            recloss = loss_fn(recon2[:,:,0:12,:], inputs[1].cuda()).mean(dim=0)


            gen_loss += recloss

            losses_g += [gen_loss.detach()]
        
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

        print('recloss:',"       ", recloss.item())
        
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
            'gen_dict': gen.state_dict(),
            'dis_dict': dis.state_dict(),
            'gen_optim': gen_optim.state_dict(),
            'dis_optim': dis_optim.state_dict()
            }
            torch.save(state, os.path.join(args.base_dir + RUN_SAVE_PATH, 'gen_{}.pth'.format(epoch)))
            print('saved models')