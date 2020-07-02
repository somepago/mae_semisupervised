from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
# from scipy.misc import imsave
import torchvision
from scipy import stats
from torch.nn import functional as F
from src.utils import *
import src.losses as losses
import torch.nn.functional as F
import argparse
from loss import *
from torch import nn
import torch.nn.init as nninit

# random seed for reciprocity
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


########################################## Hyper Parameters ###################################
#10 - 64
#11 - 100
#12 - 32
#13 - eps + 64
#14 remove term 1
#15 remove term 2
#18 remove term 3
#16 - 11
#17 - 10
#19 change term 2 to compare in latent space
#20 detach 
#21 add noise to generator (not really)
#22 baseline
#23 baseline for real
#24 baseline w/ preprocessing
#25 remove term 2 (again) w/ preprocessing 
#26 remove term 2 (again) w/ preprocessing + eps
#27 old code w/ no term 2
#28 no term 1
#29 old code w/ no term 2 longer old encoder
#30 old code w/ no term 2 longer w/ epsold encoder
#31 old code w/ no term 2 longer w/ 100 dimo ld encoder
#32 old code w/ new encoder
#33 new code old encoder term 2 coef
#34 new code old encoder no term 2

#root directory
dataroot = "./data"
modelroot = "./model900"
imgroot = "./image900"
saveModelRoot = "./model900"


#num of workers
workers = 4
gpu = 0

#batch size
batch_size = 64

#spatial size of training images
image_size = 32

#gradient penalty lambda
LAMBDA = 10

#num channels
nc = 3

#size of latent vector(z)
nz = 128

#size of generator feature maps
ngf = image_size

#size of discriminator feature maps
ndf = image_size

#num training epochs (running through the batch once)
num_epochs = 30

#optimizer learning rate
lr = 0.0002

#beta1 hyperparam for Adam optimizers
beta1 = 0.5

#num of GPUs available
ngpu = 1

#Nummber of discriminator updates per generator updata
tfd = 5

#What datasets we are using
dataset = 'cifar10' # use 9 classes
outset = 'cifar10' ## use 1 class

# epsilons for hinge loss
eps_1 = 0
eps_2 = 0
eps_3 = 0

# re-train or use a trained model
load = True

# bins for histogram
bins = list(range(16))

# proportion of validation images to total images
proportion_valid = 0.15

# save images, histograms every __ epochs
save_rate = 1



############################## Neural Networks Architecture ###############################
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
    
def max_singular_value(w_mat, u, power_iterations):

    for _ in range(power_iterations):
        v = l2normalize(torch.mm(u, w_mat.data))

        u = l2normalize(torch.mm(v, torch.t(w_mat.data)))

    sigma = torch.sum(torch.mm(u, w_mat) * v)

    return u, sigma, v



class Linear(torch.nn.Linear):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_features), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)


    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_features, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)

            # w_bar = torch.div(w_mat, sigma)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
            # self.w_bar = w_bar.detach()
            # self.sigma = sigma.detach()
        else:
            w_bar = self.weight
        return F.linear(input, w_bar, self.bias)


class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_channels), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_channels, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.conv2d(input, w_bar, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Embedding(torch.nn.Embedding):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.num_embeddings), requires_grad=False))
        else:
            self.register_buffer("u", None)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.num_embeddings, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.embedding(
            input, w_bar, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class CategoricalBatchNorm(torch.nn.Module):

    def __init__(self, num_features, num_categories, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(CategoricalBatchNorm, self).__init__()
        self.batch_norm = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.gamma_c = torch.nn.Embedding(num_categories, num_features)
        self.beta_c = torch.nn.Embedding(num_categories, num_features)
        torch.nn.init.constant_(self.batch_norm.running_var.data, 0)
        torch.nn.init.constant_(self.gamma_c.weight.data, 1)
        torch.nn.init.constant_(self.beta_c.weight.data, 0)

    def forward(self, input, y):
        ret = self.batch_norm(input)
        gamma = self.gamma_c(y)
        beta = self.beta_c(y)
        gamma_b = gamma.unsqueeze(2).unsqueeze(3).expand_as(ret)
        beta_b = beta.unsqueeze(2).unsqueeze(3).expand_as(ret)
        return gamma_b*ret + beta_b

class Block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 kernel_size=3, stride=1, padding=1, optimized=False, spectral_norm=1):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.optimized = optimized
        self.hidden_channels = out_channels if not hidden_channels else hidden_channels

        self.conv1 = Conv2d(self.in_channels, self.hidden_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.conv2 = Conv2d(self.hidden_channels, self.out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.s_conv = None
        ## use differrnt optimizert sart
        torch.nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        if self.in_channels != self.out_channels or optimized:
            self.s_conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0,
                                 spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.s_conv.weight.data, 1.)

        self.activate = torch.nn.ReLU()

    def residual(self, input):
        x = self.conv1(input)
        x = self.activate(x)
        x = self.conv2(x)
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        if self.s_conv:
            x = self.s_conv(x)
        return x


    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r

class Gblock(Block):

    def __init__(self, in_channels, out_channels, hidden_channels=None ,
                 kernel_size=3, stride=1, padding=1, upsample=True):
        super(Gblock, self).__init__(in_channels, out_channels, hidden_channels, kernel_size, stride, padding,
                                     upsample, spectral_norm=0)
        self.upsample = upsample

        self.bn1 = self.batch_norm(self.in_channels)
        self.bn2 = self.batch_norm(self.hidden_channels)
        if upsample:
            # self.up = torch.nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
            self.up = lambda a: torch.nn.functional.interpolate(a, scale_factor=2)
        else:
            self.up = lambda a: None

        if upsample:
            self.s_conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0,
                                 spectral_norm_pi=0)

    def batch_norm(self, num_features):
        return torch.nn.BatchNorm2d(num_features)

    def residual(self, input):
        x = input
        x = self.bn1(x)
        x = self.activate(x)
        if self.upsample:
            x = self.up(x)
            # output_size = list(input.size())
            # output_size[-1] = output_size[-1] * 2
            # output_size[-2] = output_size[-2] * 2
            # x = self.up(x, output_size=output_size)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x

    def shortcut(self, input):
        x = input
        if self.upsample:
            x = self.up(x)
        if self.s_conv:
            x = self.s_conv(x)
        return x

    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r


## 改成spectral_norm
class Dblock(Block):

    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, stride=1, padding=1,
                 downsample=False, spectral_norm=1):
        super(Dblock, self).__init__(in_channels, out_channels, hidden_channels, kernel_size, stride, padding,
                                     downsample, spectral_norm)
        self.downsample = downsample

        if(self.downsample):
            self.s_conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0,
                                 spectral_norm_pi=0)
    def residual(self, input):
        x = self.activate(input)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.s_conv:
            x = self.s_conv(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r


class BaseGenerator(torch.nn.Module):

    def __init__(self, z_dim, ch, d_ch=None, bottom_width=4):
        super(BaseGenerator, self).__init__()
        self.z_dim = z_dim
        self.ch = ch
        self.d_ch = d_ch if d_ch else ch
        self.bottom_width = bottom_width
        self.dense = torch.nn.Linear(self.z_dim, self.bottom_width * self.bottom_width * self.d_ch)
        torch.nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        self.blocks = torch.nn.ModuleList()
        self.final = self.final_block()

    def final_block(self):
        conv = torch.nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(conv.weight.data, 1.)
        final_ = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.ch),
            torch.nn.ReLU(),
            conv,
            torch.nn.Tanh()
        )
        return final_


    def forward(self, input):
        x = self.dense(input)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return x


class ResnetGenerator32(BaseGenerator):

    def __init__(self, ch=256, z_dim=128,bottom_width=4):
        super(ResnetGenerator32, self).__init__(z_dim, ch, ch, bottom_width)
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True))
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True))
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True))



class BaseDiscriminator(torch.nn.Module):

    def __init__(self, in_ch, out_ch=None, l_bias=True, spectral_norm=1,stack = 3):
        super(BaseDiscriminator, self).__init__()
        self.activate = torch.nn.ReLU()
        self.ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.blocks = torch.nn.ModuleList([Block(stack, self.ch, optimized=True, spectral_norm=spectral_norm)])
        self.l = Linear(self.out_ch, 1, l_bias, spectral_norm_pi=spectral_norm)
        torch.nn.init.xavier_uniform_(self.l.weight.data, 1.)

    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        x = torch.sum(x, (2, 3))
        output = self.l(x)
        return output

    
    def feature(self, input):
        batch_size = input.shape[0]
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        x = torch.sum(x, (2, 3))
        feature = x.view(batch_size,-1)
        return feature



class ResnetDiscriminator32(BaseDiscriminator):

    def __init__(self, ch=128,  spectral_norm=1,stack = 3):
        super(ResnetDiscriminator32, self).__init__(ch, ch, l_bias=False, spectral_norm=spectral_norm,stack = stack)
        self.blocks.append(Dblock(self.ch, self.ch, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))









def BasicDisEncNeuro(input,output,filter_size,stride,padding,bias): 
    return nn.Sequential(nn.Conv2d(input, output, filter_size , stride, padding, bias = bias),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, inplace=True))


def BasicGenNeuro(input,output,filter_size,stride,padding,bias):    
    return nn.Sequential(nn.ConvTranspose2d(input, output, filter_size , stride, padding, bias = bias),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True))


def BasicDisEncNeuro(input,output,filter_size,stride,padding,bias): 
    return nn.Sequential(nn.Conv2d(input, output, filter_size , stride, padding, bias = bias),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, inplace=True))





# create encoder
class Encoder(nn.Module):
    def __init__(self,ngpu,nz=64,nc = 3):
        super(Encoder,self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            BasicDisEncNeuro(nc,ndf*2,4,2,1,bias=False),
            BasicDisEncNeuro(ndf*2,ndf*4,4,2,1,bias = False),
            BasicDisEncNeuro(ndf*4,ndf*8,4,2,1,bias = False),
            nn.Conv2d(ndf*8,self.nz,4,1,0,bias=False),
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nz)
        return x




DIM = 128

class Discriminator(nn.Module):
    def __init__(self,channel):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(channel, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d()
        )   

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

    def feature(self,input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        return output






  
'''
GAN model definitions.
'''

def avg_pool2d(x):
    '''Twice differentiable implementation of 2x2 average pooling.'''
    return (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4

class GeneratorBlock(nn.Module):
    '''ResNet-style block for the generator model.'''

    def __init__(self, in_chans, out_chans, upsample=False):
        super().__init__()

        self.upsample = upsample

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.upsample:
            shortcut = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=False)
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)

        return x + shortcut

class Generator(nn.Module):
    '''The generator model.'''

    def __init__(self):
        super().__init__()

        feats = 128
        self.input_linear = nn.Linear(128, 4*4*feats)
        self.block1 = GeneratorBlock(feats, feats, upsample=True)
        self.block2 = GeneratorBlock(feats, feats, upsample=True)
        self.block3 = GeneratorBlock(feats, feats, upsample=True)
        self.output_bn = nn.BatchNorm2d(feats)
        self.output_conv = nn.Conv2d(feats, 3, kernel_size=3, padding=1)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.input_linear else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

        self.last_output = None

    def forward(self, *inputs):
        x = inputs[0]

        x = self.input_linear(x)
        x = x.view(-1, 128, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_bn(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.output_conv(x)
        x = nn.functional.tanh(x)

        self.last_output = x

        return x

class DiscriminatorBlock(nn.Module):
    '''ResNet-style block for the discriminator model.'''

    def __init__(self, in_chans, out_chans, downsample=False, first=False):
        super().__init__()

        self.downsample = downsample
        self.first = first

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if not self.first:
            x = nn.functional.relu(x, inplace=False)
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut

class Res_Discriminator(nn.Module):
    '''The discriminator (aka critic) model.'''

    def __init__(self,channel):
        super().__init__()

        feats = 128
        self.block1 = DiscriminatorBlock(channel, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = nn.Linear(128, 1)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs):
        x = inputs[0]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)
        x = self.output_linear(x)

        return x
    def feature(self,*inputs):
        x = inputs[0]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)

        return x


class Res_Encoder(nn.Module):
    '''The discriminator (aka critic) model.'''

    def __init__(self,channel):
        super().__init__()

        feats = 128
        self.block1 = DiscriminatorBlock(channel, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block5 = nn.Conv2d(128,128,8,1,0,bias=True)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs):
        x = inputs[0]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x)
        x = self.block5(x)
        x = x.view(-1, 128)
        return x




# FC(120, 60, tanh)-FC(60, 30, tanh)-FC(30, 10, tanh)-FC(10, 1, none)
def BasicEnckd99Neuro(input,output,bias): 
    return nn.Sequential(nn.Linear(input,output,bias),
            nn.LeakyReLU(0.2, inplace=True))

def BasicGenkd99Neuro(input,output,bias): 
    return nn.Sequential(nn.Linear(input,output,bias),
            nn.ReLU(inplace=True))

## finish
class KD99Encoder(nn.Module):
    def __init__(self,nz=32,nc = 121,bias = True):
        super(KD99Encoder,self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            BasicEnckd99Neuro(self.nc,64,bias),
            BasicEnckd99Neuro(64,self.nz,bias),
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nz)
        return x

class KD99Generator(nn.Module):
    def __init__(self,nz=32,nc = 121,bias = True):
        super(KD99Generator,self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            BasicGenkd99Neuro(self.nz,64,bias),
            BasicGenkd99Neuro(64,128,bias),
            nn.Linear(128,self.nc,bias)
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nc)
        return x



class KD99Drec(nn.Module):
    def __init__(self,nz=1,nc = 121,bias = True):
        super(KD99Drec,self).__init__()
        self.main = nn.Sequential(
            Linear(nc*2,128,spectral_norm_pi=1),
            nn.LeakyReLU(0.2),
            Linear(128,25,spectral_norm_pi=1),
            nn.LeakyReLU(0.2)
            )
        self.pred = Linear(25,1,spectral_norm_pi=1)
    def forward(self,input):
        x = self.pred(self.main(input))
        x = x.view(-1,1)
        return x
    def feature(self,input):
        b = input.shape[0]
        x = self.main(input)
        x = x.view(b,-1)
        return x

class KD99inter(nn.Module):
    def __init__(self,nz=1,nc = 121,bias = True):
        super(KD99inter,self).__init__()
        self.main = nn.Sequential(
            Linear(nc*2,128,bias),
            nn.LeakyReLU(0.2),
            Linear(128,64,bias),
            nn.LeakyReLU(0.2)
            )
        self.pred = Linear(64,1,bias)
    def forward(self,input):
        x = self.pred(self.main(input))
        x = x.view(-1,1)
        return x
    def feature(self,input):
        b = input.shape[0]
        x = self.main(input)
        x = x.view(b,-1)
        return x



class ARRHEncoder(nn.Module):
    def __init__(self,nz=64,nc = 274 ,bias = True):
        super(ARRHEncoder,self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            BasicEnckd99Neuro(self.nc,256,bias),
            BasicEnckd99Neuro(256,128,bias),
            BasicEnckd99Neuro(128,self.nz,bias),
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nz)
        return x

class ARRHGenerator(nn.Module):
    def __init__(self,nz=64,nc = 274,bias = True):
        super(ARRHGenerator,self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            BasicGenkd99Neuro(self.nz,128,bias),
            BasicGenkd99Neuro(128,256,bias),
            nn.Linear(256,self.nc,bias)
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nc)
        return x

class ARRHDrec(nn.Module):
    def __init__(self,nz=1,nc = 274,bias = True):
        super(ARRHDrec,self).__init__()
        self.main = nn.Sequential(
            Linear(nc*2,256,spectral_norm_pi=1),
            nn.LeakyReLU(0.2),
            Linear(256,64,spectral_norm_pi=1),
            nn.LeakyReLU(0.2),
            )
        self.pred = Linear(64,1,spectral_norm_pi=1)
    def forward(self,input):
        x = self.pred(self.main(input))
        x = x.view(-1,1)
        return x
    def feature(self,input):
        b = input.shape[0]
        x = self.main(input)
        x = x.view(b,-1)
        return x


##
class ARRHDinter(nn.Module):
    def __init__(self,nz=1,nc = 274,bias = True):
        super(ARRHDinter,self).__init__()
        self.main = nn.Sequential(
            Linear(nc*2,256,spectral_norm_pi=1),
            nn.ReLU(inplace=True),
            Linear(256,128,spectral_norm_pi=1),
            nn.ReLU(inplace=True),
            Linear(128,64,spectral_norm_pi=1),
            nn.ReLU(inplace=True),
            )
        self.pred = Linear(64,1,spectral_norm_pi=1)
    def forward(self,input):
        x = self.pred(self.main(input))
        x = x.view(-1,1)
        return x
    def feature(self,input):
        b = input.shape[0]
        x = self.main(input)
        x = x.view(b,-1)
        return x





class ThEncoder(nn.Module):
    def __init__(self,nz=4,nc = 6 ,bias = True):
        super(ThEncoder,self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            BasicEnckd99Neuro(self.nc,12,bias),
            BasicEnckd99Neuro(12,self.nz,bias),
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nz)
        return x

class ThGenerator(nn.Module):
    def __init__(self,nz=4,nc = 6,bias = True):
        super(ThGenerator,self).__init__()
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            BasicGenkd99Neuro(self.nz,13,bias),
            nn.Linear(13,self.nc,bias)
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nc)
        return x

class ThDrec(nn.Module):
    def __init__(self,nz=1,nc = 6 ,bias = True):
        super(ThDrec,self).__init__()
        self.main = nn.Sequential(
            Linear(nc*2, 4,spectral_norm_pi=1),
            nn.LeakyReLU(0.2),
            )
        self.pred = Linear(4,1,spectral_norm_pi=1)
    def forward(self,input):
        x = self.pred(self.main(input))
        x = x.view(-1,1)
        return x
    def feature(self,input):
        b = input.shape[0]
        x = self.main(input)
        x = x.view(b,-1)
        return x


##
class ThDinter(nn.Module):
    def __init__(self,nz=1,nc = 6 ,bias = True):
        super(ThDinter,self).__init__()
        self.main = nn.Sequential(
            Linear(nc*2,6,spectral_norm_pi=1),
            nn.ReLU(inplace=True),
            )
        self.pred = Linear(6,1,spectral_norm_pi=1)
    def forward(self,input):
        x = self.pred(self.main(input))
        x = x.view(-1,1)
        return x
    def feature(self,input):
        b = input.shape[0]
        x = self.main(input)
        x = x.view(b,-1)
        return x



def BasicEncNeuro(input,output,filter_size,stride,padding,activation=True): 
    conv = nn.Conv2d(input, output, kernel_size=filter_size , stride=stride, padding = padding)
    torch.nn.init.kaiming_normal_(conv.weight.data, a=0.2)
    if(activation):
        return nn.Sequential(conv,nn.LeakyReLU(0.2, inplace=True))
    else:
        return conv
# create encoder
class Encoder256(nn.Module):
    def __init__(self,nc = 3,depth=64,scales = 3, latent = 16):
        super(Encoder256,self).__init__()
        self.nc = nc
        self.depth = depth
        self.scales = scales
        self.latent = latent
        self.main = nn.Sequential(
            BasicEncNeuro(nc,64,1,1,0),
            BasicEncNeuro(64,64,3,1,1),
            BasicEncNeuro(64,64,3,1,1),
            nn.AvgPool2d(2, stride=2),
            BasicEncNeuro(64,128,3,1,1),
            BasicEncNeuro(128,128,3,1,1),
            nn.AvgPool2d(2, stride=2),
            BasicEncNeuro(128,256,3,1,1),
            BasicEncNeuro(256,256,3,1,1),
            nn.AvgPool2d(2, stride=2),
            BasicEncNeuro(256,512,3,1,1),
            BasicEncNeuro(512,16,3,1,1,activation=False),
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1,256)
        return x



class cft(nn.Module):
    def __init__(self,nc = 256):
        super(cft,self).__init__()
        self.main = nn.Linear(nc,10,bias= True) 


    def forward(self,input):
        return self.main(input)


# parser.add_argument('--criterion', default='param',
#                     help='param|nonparam, How to estimate KL')
# parser.add_argument('--KL', default='qp', help='pq|qp')
# parser.add_argument('--noise', default='sphere', help='normal|sphere')

# opt = parser.parse_args()

# if opt.criterion == 'param':
#     print('Using parametric criterion KL_%s' % opt.KL)
#     KL_minimizer = losses.KLN01Loss(direction=opt.KL, minimize=True)
#     KL_maximizer = losses.KLN01Loss(direction=opt.KL, minimize=False)
# elif opt.criterion == 'nonparam':
#     print('Using NON-parametric criterion KL_%s' % opt.KL)
#     KL_minimizer = losses.SampleKLN01Loss(direction=opt.KL, minimize=True)
#     KL_maximizer = losses.SampleKLN01Loss(direction=opt.KL, minimize=False)
# else:
#     assert False, 'criterion?'




