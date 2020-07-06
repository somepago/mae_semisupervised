from __future__ import print_function
#%matplotlib inline
import argparse
import os
import time
import datetime
import random
from itertools import chain

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
# import matplotlib.animation as animation
import matplotlib
import torchvision
from src.utils import *
import src.losses as losses
import torch.nn.functional as F
from anom_utils import post_process, generate_image, reconstruction_loss, latent_reconstruction_loss
from anom_utils import l1_latent_reconstruction_loss, anomaly_score, score_and_auc
from torchvision.utils import save_image


from loss import threeDCritic, twoDCritic, DCritic, second_discriminator_loss, threeEGcritic, twoEGcritic, EGcritic
# from data_old import load_data
import data
from newevaluate import evaluate

from model import Encoder,ResnetDiscriminator32,ResnetGenerator32,ResnetDiscriminator32,Res_Discriminator,Res_Encoder

matplotlib.rc("text", usetex=False)

import wandb



#gowthami - check if thhese are used anywhere, otherwise remove
#num of workers
# workers = 4
# gpu = 0

#gradient penalty lambda
# LAMBDA = 10

#What datasets we are using
# dataset = 'cifar10' # use 9 classes
# outset = 'cifar10' ## use 1 class
# eps_1 = 0
# eps_2 = 0

#this one is used in the regularization term, need to check the actual formula
eps_3 = 0


parser = argparse.ArgumentParser()

parser.add_argument('--manualSeed', type = int, default=999, help='set the seed for the model manually')
parser.add_argument('--dataroot',  default = "./CIFAR10/",
                help='Location of the data')
parser.add_argument('--dataset', default='cifar10', help='name of the dataset we are working with')
#for version argument, if we are training with anomalies use train_anom, otherwise use train_no_anom
parser.add_argument('--version',default = 'train_no_anom',choices = ['train_anom', 'train_no_anom'] ,help='training data includes anomalies or not')
parser.add_argument('--anom_pc',type = float ,default = 0, help='percentage of each anomaly class to use in training')
parser.add_argument('--batchsize', type = int, default=64, help='train/val batchsize')
parser.add_argument('--val_split', type = float, default=0.01, help='%of train data to split into val data')

parser.add_argument('--image_size', type = int, default= 32, help='size of training images')
parser.add_argument('--num_channels', type = int, default=3, help='number of channels')

parser.add_argument('--ngpu', type = int, default=1, help='number of GPUs available')
parser.add_argument('--workers', type = int, default=32, help='number of worker CPU nodes')
parser.add_argument('--isize', type = int, default=32, help='pq|qp')
parser.add_argument('--abnormal_classes', default='cat', help='name of the abnormal class, changes based on the dataset')

parser.add_argument('--lr',type = float ,default = 2e-4)
parser.add_argument('--nz',type = int ,default = 256)
parser.add_argument('--weights',type = float, default = 0.5, help='hyperparameter in AE loss')
parser.add_argument('--model_load_path',default = '', help='path to trained model, otherwise the model will train from scratch')
parser.add_argument('--cuda',type= int, default = 0)
parser.add_argument('--use_mode',default = 'last')
parser.add_argument('--interpolate_points',type = int, default = 2, help = 'number of interpolated in training data')
parser.add_argument('--use_decay_learning',type=str, default = 'False')
parser.add_argument('--use_linearly_decay',type=str, default = 'False')

parser.add_argument('--num_epochs',type = int, default = 100, help = 'number of training epochs')
parser.add_argument('--start', type=int, default = 0)
parser.add_argument('--use_penalty', action = 'store_true')
parser.add_argument('--ch', type=int,default = 128)
parser.add_argument('--update_ratio', type=int,default = 5, help='number of discriminator updates to generator update')
parser.add_argument('--save_model_epochs',type = int, default = 10, help='save the model after this many epochs')
parser.add_argument('--save_logs_epochs',type = int, default = 1, help='save the model after this many epochs')
parser.add_argument('--save_model_root',  default = "logs",
                help='Location to save the model wrt train file')


##version c means people use interpolate inside

opt = parser.parse_args()
runname = str(opt.abnormal_classes) + '_' + str(opt.version) + '_' + str(opt.anom_pc) + 'pc'

wandb.init(project="mae-trial", name=runname)
wandb.config.update(opt)

print("Chosen Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

#num channels
nc = opt.num_channels

#size of generator feature maps
ngf = opt.image_size

#size of discriminator feature maps
ndf = opt.image_size

#num of GPUs available
ngpu = opt.ngpu

#Number of discriminator updates per generator update
tfd = opt.update_ratio

# save images, histograms every __ epochs
save_rate_logs = opt.save_logs_epochs
save_rate_model = opt.save_model_epochs

# if we want to decay learning rate
use_decay_learning = eval(opt.use_decay_learning)
use_linearly_decay = eval(opt.use_linearly_decay)

#gowthami - to check
opt.use_second_feature_loss = False
opt.use_spectural_norm = False

print(opt)



#loading the data

b_size = opt.batchsize
anom_classes = eval(opt.abnormal_classes)

transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

if opt.dataset ==  'cifar10':
	dataset_train = data.CIFAR10Anom(root=opt.dataroot, stage='train', transform = transform,
								anom_classes=anom_classes, valid_split=opt.val_split,
								anom_ratio=opt.anom_pc, seed = opt.manualSeed)
	dataset_valid = data.CIFAR10Anom(root=opt.dataroot, stage='valid', transform = transform,
								anom_classes=anom_classes, valid_split=opt.val_split,
								anom_ratio=opt.anom_pc, seed = opt.manualSeed)
	dataset_test =  data.CIFAR10Anom(root=opt.dataroot, stage='test', transform = transform,
								anom_classes=anom_classes, valid_split=0,
								anom_ratio=opt.anom_pc, seed = opt.manualSeed)
	
	trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=b_size, 
											  drop_last = True,
                                           shuffle = True, num_workers=opt.workers)
	validloader = torch.utils.data.DataLoader(dataset_valid, batch_size=b_size, 
											  drop_last = True,
                                           shuffle = True, num_workers=opt.workers)
	testloader = torch.utils.data.DataLoader(dataset_test, batch_size=b_size, 
											 drop_last = True,
                                           shuffle = False, num_workers=opt.workers)
	

# dataloader = load_data(opt)
# dataloaderTrain = dataloader['train']
# dataloaderTest =  dataloader['test']

lr = opt.lr
nz = opt.nz
num_epochs = opt.num_epochs

##creating folders to save the models and logs
date = str(datetime.datetime.now())
date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")

log_dir = os.path.join(os.getcwd(),"logs" ,"log_" + date + '_'+opt.abnormal_classes)

imgroot = os.path.join(log_dir, opt.version + '_' + str(opt.interpolate_points), 'image' )
saveModelRoot = os.path.join(log_dir, opt.version + '_' + str(opt.interpolate_points), 'model' )
tboardroot = os.path.join(log_dir, opt.version + '_' + str(opt.interpolate_points), 'tboard' )
print(imgroot)
print(saveModelRoot)
print(tboardroot)

try:
	os.makedirs(saveModelRoot)
except OSError: 
	print("Model folder already exists")

try:
	os.makedirs(imgroot)
except OSError: 
	print("Image folder already exists")

#set the summary writer
# writer = SummaryWriter(tboardroot)

#set the device
device = torch.device("cuda:%s" % (opt.cuda) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

#define the models
# netE is the encoder, netG is the Generator,netD is the interpolation discriminator and netD2 is the reconstruction discriminator

netE = Encoder(ngpu=1,nz=nz, nc = 3,ndf = ndf)
netD = Res_Discriminator(channel=6,ch= opt.ch)
netG = ResnetGenerator32(z_dim = nz)
netD2 = ResnetDiscriminator32(stack = 6 , ch= opt.ch)
	
#if you are using more than 1GPU, let's parallelize the code. gowthami to check if this is working
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  netE = nn.DataParallel(netE)
  netD = nn.DataParallel(netD)
  netG = nn.DataParallel(netG)
  netD2 = nn.DataParallel(netD2)


netE.to(device)
netD2.to(device)
netG.to(device)
netD.to(device)

optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (0, .9))
optimizerD2 = optim.Adam(netD2.parameters(), lr = lr, betas = (0, .9))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (0, .9))
optimizerE = optim.Adam(netE.parameters(), lr = lr, betas = (.5, .9))

#gowthami- check what's the point of this?
start = opt.start
print("Starting Training")

# gowthami - what is this best score? and what is this train loss small?
# best_score = 0.5
## small == 3000 ## smallest value is 3000
# train_loss_small = 3000

# gowthami - did they use it in the final run? chek with Yexin
print(use_decay_learning, use_linearly_decay)
print('start from %s ' % start)
n_iter = 0

starttime = time.time()

if(opt.model_load_path==''):
	test_score = []
	valid_score = []
	for epoch in range(start,num_epochs):

		for i, data in enumerate(trainloader, 0):
			n_iter = n_iter + 1
			# optimize discriminator tfd times
			for t in range(tfd):
				netD.zero_grad()
				netE.zero_grad()
				netG.zero_grad()
				
				#Train with all real data
				real = data[0].to(device)
				real = real.reshape(b_size, nc, opt.image_size, opt.image_size)
				noise = netE(real).detach()
				fake = netG(noise).detach()

				
				if(opt.interpolate_points == 3):
					alpha1 = torch.FloatTensor(b_size, 1).uniform_(0, 0.5)
					alpha2 = torch.FloatTensor(b_size, 1).uniform_(0, 0.5)
				
					errD_real = threeDCritic(netE,netG,netD,real,alpha1,alpha2,reg=0.2,n_iter=n_iter,t=t,writer=None)
				elif(opt.interpolate_points == 2):
					alpha = torch.FloatTensor(b_size, 1).uniform_(0, 0.5)
					errD_real = twoDCritic(netE,netG,netD,real,alpha,reg=0.2)
				
				elif(opt.interpolate_points == 1):
					alpha = torch.FloatTensor(b_size, 1).uniform_(0, 0.5)
					errD_real = DCritic(netE,netG,netD,real,alpha,reg=0.2)
				

				## 5 means no interpolate
				if(opt.interpolate_points != 5):
					errD_real.backward()

				optimizerD.step()
				errD_recon = second_discriminator_loss(real, netD2, netE, netG,fake,writer=None,use_penalty=opt.use_penalty)
				errD_recon.backward()
				optimizerD2.step()

            
                
			if(True):
# 				noise2 = torch.randn(b_size, nz, device = device)
				
				term_3 = (torch.clamp((torch.norm(netE(real), dim = (1))) - math.sqrt(nz) - eps_3, min= 0)).mean()				
				
				rec_real = netG( netE(real) )
				real_rec = torch.cat((real,rec_real),dim=1)				
				outputD2 = netD2(real_rec).view(-1)

				if(opt.interpolate_points == 3):
					output = threeEGcritic(netE,netG,netD,real,alpha1,alpha2,weights = opt.weights)
					
				elif(opt.interpolate_points == 2):
					output = twoEGcritic(netE,netG,netD,real,alpha,weights = opt.weights)
				
				elif(opt.interpolate_points == 1):
					output = EGcritic(netE,netG,netD,real,alpha,weights = opt.weights)
                    
                #is it 0 or the above formula. I commented out 0 part. gowthami - check

# 				elif(opt.interpolate_points == 1):
# 					output = 0
	
				errG =  - outputD2.mean()  +  term_3  + output


				netG.zero_grad()
				netE.zero_grad()
				errG.backward()
				optimizerG.step()
				optimizerE.step()
                
				wb_iter = len(trainloader)*epoch + n_iter
				wandb.log({'epoch': epoch,'iteration':wb_iter ,'loss_Dinter': errD_real, 
                          'loss_Drecon':errD_recon, 'loss_AE': errG})
			if i % 50 == 0:
				print('[%d/%d][%d/%d]'
					%(epoch, num_epochs, i, len(trainloader)))
		
### validating and saving area ###

		if epoch % save_rate_logs == 0:
			with torch.no_grad():
                
                # calculcating auroc on test dataset
				test_auc, test_anom_score = score_and_auc(testloader, netG, netE, netD2,device ,ngpu, break_iters = 70)
				print(('test_auc:%f') % test_auc)
                
				_, val_anom_score = score_and_auc(validloader, netG, netE, netD2, device,ngpu, break_iters = 50)
				print(('train_score:%f') % val_anom_score)
                
				wandb.log({'test_auc': test_auc,  
                          'test_anom_mean':test_anom_score, 'train_anom_mean': val_anom_score})
                
                # printing epoch losses
				with open(os.path.join(imgroot, "epoch_losses.txt"), "a") as f:
					currenttime = time.time()
					elapsed = currenttime - starttime
					f.write("{} \t {:.2f}\t {:.5f}\t {:.5f}\t {:.5f}".format(epoch, elapsed, test_auc, test_anom_score, val_anom_score) + "\n")
				starttime = time.time()
                
               # save images
				output = None
				for i,data in enumerate(testloader, 0):
					test = data[0][:32,:,:,:].to(device)
					row = torch.cat((test,netG(netE(test))), dim=2)
					if output is None:
						output = row
					else:
						output = torch.cat((output, row), dim=1)
					save_image(output, os.path.join(imgroot, "img-{}.png".format(epoch)))
					break
                
				if epoch % save_rate_model == 0:
					torch.save(netG.state_dict(),'%s/netbestG_%s.pth' % (saveModelRoot, epoch))
					torch.save(netD.state_dict(),'%s/netbestD_%s.pth' % (saveModelRoot,epoch))
					torch.save(netE.state_dict(),'%s/netbestE_%s.pth' % (saveModelRoot,epoch))
					torch.save(netD2.state_dict(),'%s/netbestD2_%s.pth' % (saveModelRoot,epoch))


                    
                    
                    
# 		if(epoch == 45):
# 			if(use_decay_learning and not use_linearly_decay):
# 				print('decrease learning rate to half',flush = True)
# 				half_adjust_learning_rate(optimizerD, epoch, num_epochs,lr)
# 				half_adjust_learning_rate(optimizerD2,epoch,num_epochs,lr)
# 				half_adjust_learning_rate(optimizerG,epoch,num_epochs,lr)
# 				half_adjust_learning_rate(optimizerE,epoch,num_epochs,lr)
# 			else:
# 				print('still use 2e-4 for trainning')