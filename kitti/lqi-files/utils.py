import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm

####################################
def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output

##############
count = 0
# -------------------------------------------------------------------------
# Handy Utilities
# -------------------------------------------------------------------------
def to_polar_np(velo):
    if len(velo.shape) == 4:
        velo = velo.transpose(1, 2, 3, 0)

    if velo.shape[2] > 4:
        assert velo.shape[0] <= 4
        velo = velo.transpose(1, 2, 0, 3)
        switch=True
    else:
        switch=False
#     print("inside to velo")
#     print(velo[:,:,3][(velo[:,:,3]!=0)&(velo[:,:,3]!=1)])
    
    # assumes r x n/r x (3,4) velo
    dist = np.sqrt(velo[:, :, 0] ** 2 + velo[:, :, 1] ** 2)
    # theta = np.arctan2(velo[:, 1], velo[:, 0])
    out = np.stack([dist, velo[:, :, 2],velo[:,:,3]], axis=2)
    #out = np.stack([dist, velo[:, :, 2]], axis=2)
    
    if switch:
        out = out.transpose(2, 0, 1, 3)

    if len(velo.shape) == 4: 
        out = out.transpose(3, 0, 1, 2)
    
    
#     print(dist.shape)
#     print("out ki shape")
#     print(out.shape)
    
    return out

def to_polar(velo):
    if len(velo.shape) == 4:
        velo = velo.permute(1, 2, 3, 0)

    if velo.shape[2] > 4:
        assert velo.shape[0] <= 4
        velo = velo.permute(1, 2, 0, 3)
        switch=True
    else:
        switch=False
    
    # assumes r x n/r x (3,4) velo
    dist = torch.sqrt(velo[:, :, 0] ** 2 + velo[:, :, 1] ** 2)
    # theta = np.arctan2(velo[:, 1], velo[:, 0])
    
   
    
    out = torch.stack([dist, velo[:, :, 2]], dim=2)
    
    if switch:
        out = out.permute(2, 0, 1, 3)

    if len(velo.shape) == 4: 
        out = out.permute(3, 0, 1, 2)
    
    return out

def from_polar(velo):
    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[:, 0], velo[:, 1]
    x = torch.Tensor(np.cos(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    y = torch.Tensor(np.sin(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    out = torch.stack([x,y,z], dim=1)

    return out

def from_polar_np(velo):
    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[0,:], velo[1,:]
    x = np.cos(angles) * dist
    y = np.sin(angles) * dist
    out = np.stack([x,y,z], axis=0)
    return out.astype('float32')

def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)

def log_point_clouds(writer, data, name, step):
    if len(data.shape) == 3:
        data = [data]
    
    out = np.stack([from_polar(x.transpose(1, 2, 0)) for x in \
            data.cpu().data.numpy()])
    out = torch.tensor(out).float()

    for i, cloud in enumerate(out):
        cloud = cloud.view(-1, 3)
        writer.add_embedding(cloud, tag=name + '_%d' % i, global_step=step)

def print_and_save_args(args, path):
    print(args)
    # let's save the args as json to enable easy loading
    import json
    with open(os.path.join(path, 'args.json'), 'w') as f: 
        json.dump(vars(args), f)

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def remove_zeros(pc):
  
    
    xx = torch.cuda.FloatTensor(pc)

    if xx.dim() == 3: 
        xx = xx.unsqueeze(0)
    

    
    

    return xx.cpu().data.numpy() 
    iters = 0
    pad = 2
    ks = 5
    while (xx[:, 0] == 0).sum() > 0 : 

        if iters  > 100:
            raise ValueError()
            ks += 2
            pad += 1

        
        mask = (xx[:, 0] == 0).unsqueeze(1).float()
        

        
        out_a,indices = F.max_pool2d(xx[:, 0], ks, padding=pad, stride=1,return_indices=True)

        out_b = F.max_pool2d(xx[:, 1], ks, padding=pad, stride=1)
        out_c = retrieve_elements_from_indices(xx[:,2].unsqueeze(0), indices)
        

        out_b = out_b.expand_as(out_a)
        

      
        
        out = torch.stack([out_a, out_b,out_c.squeeze(0)], dim=1)

        
        mask = (xx[:, 0] == 0).unsqueeze(1)
        mask = mask.float()
        

        xx = xx * (1 - mask) + (mask) * out
        
        

        iters += 1
    
#     print("bahar aaj ja aa bhi ab")
    return xx.cpu().data.numpy()




def preprocess(dataset, lidar_range):
    # remove outliers 
    min_a, max_a = np.percentile(dataset[:, :, :, [0]], 0), np.percentile(dataset[:, :, :, [0]], 99)
    min_b, max_b = np.percentile(dataset[:, :, :, [1]], 0), np.percentile(dataset[:, :, :, [1]], 99)
    min_c, max_c = np.percentile(dataset[:, :, :, [2]], 0), np.percentile(dataset[:, :, :, [2]], 99)


    print("Finding crop masks")
    mask = np.maximum(dataset[:, :, :, 0] < min_a, dataset[:, :, :, 0] > max_a)
    mask = np.maximum(mask, np.maximum(dataset[:, :, :, 1] < min_b, dataset[:, :, :, 1] > max_b))
    mask = np.maximum(mask, np.maximum(dataset[:, :, :, 2] < min_c, dataset[:, :, :, 2] > max_c))
    dist = dataset[:, :, :, 0] ** 2 + dataset[:, :, :, 1] ** 2
    mask = np.maximum(mask, dist < 7)
    print("Masking")
    dataset = dataset * (1 - np.expand_dims(mask, -1))
#    dataset=np.array(dataset,dtype=np.float32)
    dataset[:,:,:,:3]=dataset[:,:,:,:3]/120   # Max LIDAR value
    dataset[:,:,:,:][(dataset[:,:,:,3]!=0)&(dataset[:,:,:,3]!=1)]=0    
    dataset = to_polar_np(dataset).transpose(0, 3, 1, 2)
    previous = (dataset[:, 0] == 0).sum()

    remove = []
    print("Remove zeros")
    
    for i in tqdm(range(dataset.shape[0])):

        pp = remove_zeros(dataset[i]).squeeze(0)

        dataset[i] = pp


    for i in remove:
        dataset = np.concatenate([dataset[:i-1], dataset[i+1:]], axis=0)
        print("yeh_kyun ho rha hain")
    return dataset


def show_pc(velo, save=0, save_path=None):
    import mayavi.mlab

    fig = mayavi.mlab.figure(size=(1400, 700), bgcolor=(0,0,0)) 

    if len(velo.shape) == 3:
        if velo.shape[0] == 3 : 
            velo = velo.transpose(1,2,0)

        assert velo.shape[2] == 3
        velo = velo.reshape((-1, 3))

    max_ = np.absolute(velo[:, :2]).max()
    nodes = mayavi.mlab.points3d(
        velo[:, 0],   # x
        velo[:, 1],   # y
        velo[:, 2],   # z
        scale_factor=0.008, #0.022,     # scale of the points
        figure=fig) 
    
    nodes.glyph.scale_mode = 'scale_by_vector'
    color = (velo[:, 2] - velo[:, 2].min()) / (velo[:, 2].max() - velo[:, 2].min())
    color = (velo[:, 2] - -0.069667026) / ( 0.0041348818 - -0.069667026)
    
    nodes.mlab_source.dataset.point_data.scalars = color
    print('showing pc')
    aa, bb = -95, -40 #np.random.randint(-105, -85), np.random.randint(-55, -35)
    print(aa, bb)
    mayavi.mlab.view(azimuth=-87, elevation=-40, focalpoint=(0, 0, np.median(velo[:, -1])))
    f = mayavi.mlab.gcf()
    f.scene.camera.zoom(2.7)

    if save:
        print(save)
        mayavi.mlab.savefig('../inter_images_2/{}.png'.format(i))
        mayavi.mlab.close()
    elif save_path is not None:
        mayavi.mlab.savefig(save_path)
        mayavi.mlab.close()
    else:
        mayavi.mlab.show()

def show_pc_lite(velo, ind=1, show=True):
    velo=velo.cpu()
    # print(velo.shape)
    import matplotlib.pyplot as plt
    plt.scatter(velo[:, 0], velo[:, 1], s=0.7, color='k')
    plt.show() 


def to_attr(args_dict):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    return AttrDict(args_dict)


def load_model_from_file(path, epoch, model='dis'):
    from models import netD, netG, VAE
    import json
    with open(os.path.join(path, 'args.json'), 'r') as f: 
        old_args = json.load(f)

    old_args = to_attr(old_args)
    if 'gen' in model.lower():
        try:
            z_ = old_args.z_dim
            model_ = VAE(old_args)
        except:
            z_ = 100
            model_ = netG(old_args, nz=z_, nc= 3 if old_args.no_polar else 2)
    elif 'dis' in model.lower():
        model_ = netD(old_args)
    else: 
        raise ValueError('%s is not a valid model name' % model)

    model_.load_state_dict(torch.load(os.path.join(path, 'models/%s_%d.pth' % (model, epoch))))
    print('model successfully loaded')

    return model_, epoch 


def batch_pairwise_dist(A, B):
    # pa, pb are bs x points x 3
    r_A = (A * A).sum(dim=2, keepdim=True)
    r_B = (B * B).sum(dim=2, keepdim=True)
    m = torch.bmm(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D

def chamfer_quadratic(a,b):
    D = batch_pairwise_dist(a,b)
    return D.min(dim=-1)[0], D.min(dim=-2)[0]


# Utilities for baseline
def get_chamfer_dist(get_slow=False):
    try:
        if get_slow: raise ValueError

        import sys
        sys.path.insert(0, './nndistance')
        from modules.nnd import NNDModule
        dist = NNDModule()
    except:
        dist = chamfer_quadratic

    def loss(a, b):
        if a.dim() == 4:
            if a.size(1) == 2: 
                a = from_polar(a)

            assert a.size(1) == 3
            a = a.permute(0, 2, 3, 1).contiguous().reshape(a.size(0), -1, 3)
            
        if b.dim() == 4:
            if b.size(1) == 2: 
                b = from_polar(b)

            assert b.size(1) == 3
            b = b.permute(0, 2, 3, 1).contiguous().reshape(b.size(0), -1, 3)

        assert a.dim() == b.dim() == 3
        if a.size(-1) != 3: 
            assert a.size(-2) == 3
            a = a.transpose(-2, -1).contiguous()
        
        if b.size(-1) != 3: 
            assert b.size(-2) == 3
            b = a.transpose(-2, -1).contiguous()

        dist_a, dist_b = dist(a, b)
        return dist_a.sum(dim=-1) + dist_b.sum(dim=-1)

    return loss


