from __future__ import print_function
#%matplotlib inline
import argparse
import os
import time
import datetime
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from anom_utils import post_process, generate_image, reconstruction_loss, latent_reconstruction_loss
from anom_utils import l1_latent_reconstruction_loss, anomaly_score, score_and_auc
from model import Encoder,ResnetDiscriminator32,ResnetGenerator32,ResnetDiscriminator32,Res_Discriminator,Res_Encoder

from data import load_data

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument('--manualSeed', type = int, default=999, help='set the seed for the model manually')
    parser.add_argument('--dataroot',  default = "./CIFAR10",
                    help='Location of the data')
    parser.add_argument('--dataset', default='cifar10', help='name of the dataset we are working with')
    parser.add_argument('--batchsize', type = int, default=64, help='train/val batchsize')
    parser.add_argument('--image_size', type = int, default= 32, help='size of training images')
    parser.add_argument('--num_channels', type = int, default=3, help='number of channels')

    parser.add_argument('--ngpu', type = int, default=1, help='number of GPUs available')
    parser.add_argument('--workers', type = int, default=4, help='number of worker CPU nodes')
    parser.add_argument('--isize', type = int, default=32, help='pq|qp')
    parser.add_argument('--abnormal_class', default='cat', help='name of the abnormal class, changes based on the dataset')
    #for version argument, if we are training with anomalies use train_anom, otherwise use train_no_anom
    parser.add_argument('--version',default = 'train_no_anom',choices = ['train_anom', 'train_no_anom'] ,help='training data includes anomalies or not')
    parser.add_argument('--anom_pc',type = float ,default = 0, help='percentage of each anomaly class to use in training')
    parser.add_argument('--lr',type = float ,default = 2e-4)
    parser.add_argument('--nz',type = int ,default = 256)
    parser.add_argument('--weights',type = float, default = 0.5, help='hyperparameter in AE loss')
    parser.add_argument('--cuda',type= int, default = 0)
    parser.add_argument('--use_mode',default = 'last')
    parser.add_argument('--interpolate_points',type = int, default = 2, help = 'number of interpolated in training data')
    parser.add_argument('--use_decay_learning',type=str, default = 'False')
    parser.add_argument('--use_linearly_decay',type=str, default = 'False')
    parser.add_argument('--ch', type=int,default = 128)
    parser.add_argument('--model_load_path',default = '', help='path to trained model, otherwise the model will train from scratch')

    
    
    
    opt = parser.parse_args()
    
    #loading the model
    netE = Encoder(ngpu=1,nz=nz, nc = 3,ndf = ndf)
    netE.load_state_dict(torch.load('%s/netbestE_%s.pth' % (opt.model_load_path,epoch)))
    netE.eval()
    netG = ResnetGenerator32(z_dim = nz)
    netG.load_state_dict(torch.load('%s/netbestG_%s.pth' % (opt.model_load_path,epoch)))
    netG.eval()
    netD2 = ResnetDiscriminator32(stack = 6 , ch= opt.ch)
    netD2.load_state_dict(torch.load('%s/netbestD2_%s.pth' % (opt.model_load_path,epoch)))
    netD2.eval()
    
    
    #loading the dataset
    b_size = opt.batchsize
    dataloader = load_data(opt)
    dataloaderTest =  dataloader['test']
    
    test_auc, test_anom_score = score_and_auc(dataloaderTest, netG, netE, netD2,device ,ngpu, break_iters = 1000)
    print(('test_auc:%f') % test_auc)
    print(('test_anom_score:%f') % test_anom_score)
    
    
    
    