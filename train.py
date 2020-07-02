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
import matplotlib
# import matplotlib.animation as animation
# from scipy.misc import imsave
import torchvision
# from scipy import stats
from torch.nn import functional as F
from src.utils import *
import src.losses as losses
import torch.nn.functional as F
import argparse

from torch.utils.tensorboard import SummaryWriter


from loss import *
from data import load_data
from newevaluate import evaluate

from model import Encoder,ResnetDiscriminator32,ResnetGenerator32,ResnetDiscriminator32,Res_Discriminator,Res_Encoder
from loss  import *

matplotlib.rc("text", usetex=False)


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
parser.add_argument('--dataroot',  default = "./CIFAR10",
                help='Location of the data')
parser.add_argument('--dataset', default='cifar10', help='pq|qp')
parser.add_argument('--batchsize', type = int, default=64, help='pq|qp')
parser.add_argument('--image_size', type = int, default= 32, help='size of training images')
parser.add_argument('--num_channels', type = int, default=3, help='number of channels')

parser.add_argument('--workers', type = int, default=4, help='pq|qp')
parser.add_argument('--manualseed', type = int, default=999, help='pq|qp')
parser.add_argument('--isize', type = int, default=32, help='pq|qp')
parser.add_argument('--abnormal_class', default='cat', help='pq|qp')
parser.add_argument('--version',default = 'idiotest256')
parser.add_argument('--lr',type = float ,default = 2e-4)
parser.add_argument('--nz',type = int ,default = 256)
parser.add_argument('--weights',type = float, default = 0.5, help='hyperparameter in AE loss')
parser.add_argument('--load_path',default = '', help='path to trained model, otherwise the model will train from scratch')
parser.add_argument('--cuda',type= int, default = 0)
parser.add_argument('--use_mode',default = 'last')
parser.add_argument('--interpolate_points',type = int, default = 1)
parser.add_argument('--use_decay_learning',type=str, default = 'False')
parser.add_argument('--use_linearly_decay',type=str, default = 'False')

parser.add_argument('--num_epochs',type = int, default = 100)
parser.add_argument('--start', type=int, default = 0)
parser.add_argument('--use_penalty', action = 'store_true')
parser.add_argument('--ch', type=int,default = 128)
parser.add_argument('--update_ratio', type=int,default = 5, help='number of discriminator updates to generator update')
parser.add_argument('--save_rate_epochs',type = int, default = 1, help='save the model after this many epochs')


##version c means people use interpolate inside

opt = parser.parse_args()

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
ngpu = 1

#Number of discriminator updates per generator update
tfd = opt.update_ratio

# save images, histograms every __ epochs
save_rate = opt.save_rate_epochs

# if we want to decay learning rate
use_decay_learning = eval(opt.use_decay_learning)
use_linearly_decay = eval(opt.use_linearly_decay)

#gowthami - to check
opt.use_second_feature_loss = False
opt.use_spectural_norm = False

print(opt)

batch_size = opt.batchsize
dataloader = load_data(opt)
dataloaderTrain = dataloader['train']
dataloaderTest =  dataloader['test']

nz = opt.nz
num_epochs = opt.num_epochs


modelroot = "model" + opt.abnormal_class + opt.version + str(opt.interpolate_points)
imgroot = "./image" + opt.abnormal_class + opt.version + str(opt.interpolate_points)
saveModelRoot = "./model" + opt.abnormal_class + opt.version + str(opt.interpolate_points)

print(modelroot)
print(imgroot)
print(saveModelRoot)


try:
	os.mkdir(saveModelRoot)
except OSError: 
	print("Model folder already exists")
try:
	os.mkdir(modelroot)
except OSError: 
	print("Model folder already exists")
try:
	os.mkdir(imgroot)
except OSError: 
	print("Image folder already exists")
	


device = torch.device("cuda:%s" % (opt.cuda) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

writer = SummaryWriter('tboard'+opt.abnormal_class + opt.version + str(opt.interpolate_points) )


#define the models

netE = Encoder(ngpu=1,nz=nz, nc = 3)
netD = Res_Discriminator(channel=6)
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

optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (0, .9))
optimizerD2 = optim.Adam(netD2.parameters(), lr = opt.lr, betas = (0, .9))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (0, .9))
optimizerE = optim.Adam(netE.parameters(), lr = opt.lr, betas = (.5, .9))

#gowthami- check what's the point of this?
start = opt.start
print("Starting Training")

# gowthami - what is this best score? and what is this train loss small?
best_score = 0.5
## small == 3000 ## smallest value is 3000
train_loss_small = 3000

# gowthami - did they use it in the final run? chek with Yexin
print(use_decay_learning, use_linearly_decay)
print('start from %s ' % start)
n_iter = 0

if(opt.load_path==''):
	test_score = []
	valid_score = []
	for epoch in range(start,num_epochs):
		if(epoch == 45):
			if(use_decay_learning and not use_linearly_decay):
				print('decrease learning rate to half',flush = True)
				half_adjust_learning_rate(optimizerD, epoch, num_epochs)
				half_adjust_learning_rate(optimizerD2,epoch,num_epochs)
				half_adjust_learning_rate(optimizerG,epoch,num_epochs)
				half_adjust_learning_rate(optimizerE,epoch,num_epochs)
			else:
				print('still use 2e-4 for trainning')
			
			torch.save(netG.state_dict(),'%s/net45G.pth' % (saveModelRoot))
			torch.save(netD.state_dict(),'%s/net45D.pth' % (saveModelRoot))
			torch.save(netE.state_dict(),'%s/net45E.pth' % (saveModelRoot))
			torch.save(netD2.state_dict(),'%s/net45D2.pth' % (saveModelRoot) )

		if(epoch == 60):
			print('save 60 epochs models ')


			torch.save(netG.state_dict(),'%s/net60G.pth' % (saveModelRoot))
			torch.save(netD.state_dict(),'%s/net60D.pth' % (saveModelRoot))
			torch.save(netE.state_dict(),'%s/net60E.pth' % (saveModelRoot))
			torch.save(netD2.state_dict(),'%s/net60D2.pth' % (saveModelRoot) )

		if(epoch == 70):
			print('save 70 epochs models ')
			

			torch.save(netG.state_dict(),'%s/net70G.pth' % (saveModelRoot))
			torch.save(netD.state_dict(),'%s/net70D.pth' % (saveModelRoot))
			torch.save(netE.state_dict(),'%s/net70E.pth' % (saveModelRoot))
			torch.save(netD2.state_dict(),'%s/net70D2.pth' % (saveModelRoot) )

		

		for i, data in enumerate(dataloaderTrain, 0):
			n_iter = n_iter + 1;
			# optimize discriminator tfd times
			for t in range(tfd):
				netD.zero_grad()
				netE.zero_grad()
				netG.zero_grad()
				
				#Train with all real data
				
				real = data[0].to(device)
				print(real.shape)
				b_size = opt.batchsize
				real = real.reshape(b_size, nc, opt.image_size, opt.image_size)

				noise = netE(real).detach()
				fake = netG(noise).detach()

				
				if(opt.interpolate_points == 3):
					alpha1 = torch.FloatTensor(b_size, 1).uniform_(0, 0.5)
					alpha2 = torch.FloatTensor(b_size, 1).uniform_(0, 0.5)
				
					errD_real = threeDCritic(netE,netG,netD,real,alpha1,alpha2,reg=0.2,n_iter=n_iter,t=t,writer=writer)
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
				second_discriminator_loss(real, netD2, netE, netG, optimizerD2,fake,writer,use_penalty=opt.use_penalty)				


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
			if i % 50 == 0:
				print('[%d/%d][%d/%d]'
					%(epoch, num_epochs, i, len(dataloaderTrain)))
		


		if epoch % save_rate == 0:
			score_list = []
			score_label = []
			score_list_train_feature = []
			with torch.no_grad():
				for i,data in enumerate(dataloaderTest, 0):
					test = data[0].to(device)
					if (torch.cuda.device_count() > 1):
						a1 = netD2.module.feature(torch.cat((test,test),dim =1))
						a2 = netD2.module.feature(torch.cat((test,netG(netE(test)).detach()),dim=1))
					else:
						a1 = netD2.feature(torch.cat((test,test),dim =1))
						a2 = netD2.feature(torch.cat((test,netG(netE(test)).detach()),dim=1))
					
					score_list.append(l1_latent_reconstruction_loss(a1,a2 ))
					score_label.append(data[1].cpu().tolist())

				score_list = list([loss for lst in score_list for loss in lst])
				score_label = list([loss for lst in score_label for loss in lst])
				score = evaluate(score_label,score_list)

				test_score.append(score)
				
				print(('test:%f') % score)

				for i,data in enumerate(dataloaderTrain,0):
					test = data[0].to(device)

					if (torch.cuda.device_count() > 1):
						a1 = netD2.module.feature(torch.cat((test,test),dim =1))
						a2 = netD2.module.feature(torch.cat((test,netG(netE(test)).detach()),dim=1))
						score_list_train_feature.append(l1_latent_reconstruction_loss(a1,a2 ))
					else:
						a1 = netD2.feature(torch.cat((test,test),dim =1))
						a2 = netD2.feature(torch.cat((test,netG(netE(test)).detach()),dim=1))
						score_list_train_feature.append(l1_latent_reconstruction_loss(a1,a2 ))
				

				train_loss = np.array(list([loss for lst in score_list_train_feature for loss in lst])).mean()
					
				print(('train:%f') % train_loss)

				if(epoch >= 60):
					if(train_loss < train_loss_small):
						train_loss_small = train_loss
						print(train_loss_small)
						print( ('best model test score %f') % score )
						torch.save(netG.state_dict(),'%s/netbestG.pth' % (saveModelRoot))
						torch.save(netD.state_dict(),'%s/netbestD.pth' % (saveModelRoot))
						torch.save(netE.state_dict(),'%s/netbestE.pth' % (saveModelRoot))
						torch.save(netD2.state_dict(),'%s/netbestD2.pth' % (saveModelRoot))



torch.save(netG.state_dict(),'%s/netlastG.pth' % (saveModelRoot))
torch.save(netD.state_dict(),'%s/netlastD.pth' % (saveModelRoot))
torch.save(netE.state_dict(),'%s/netlastE.pth' % (saveModelRoot))
torch.save(netD2.state_dict(),'%s/netlastD2.pth' % (saveModelRoot) )