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
import torch

import torch.nn.functional as F


LAMBDA = 10
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def interpolated_adversarial_loss(insample_batch, out_of_sample,a,b,netE):
	if(out_of_sample.shape[0] == 1):
		out_of_sample = out_of_sample.expand_as(insample_batch.shape)

	batch_size = min(insample_batch.shape[0],out_of_sample.shape[0])

	insample_batch = insample_batch[:batch_size]
	out_of_sample = out_of_sample[:batch_size]


	batch_size = insample_batch.shape[0]
	m = torch.distributions.beta.Beta(a,b)
	l = torch.unsqueeze(torch.unsqueeze(m.sample(sample_shape = torch.Size([batch_size,1])).to(device),-1),-1)

	interpolate_image = l * insample_batch + (1 - l)*out_of_sample

	##(batch_size, pro of classifying as in sample)
	result = netE.classify(interpolate_image)

	labels = torch.ones((batch_size,1)).to(device)
	loss1 = l * F.binary_cross_entropy(result,labels,reduce = False)
	
	loss2 = (1-l)*F.binary_cross_entropy(result,1-labels,reduce = False)

	result2 = netE.classify(insample_batch)
	result3 = netE.classify(out_of_sample)

	loss3 = F.binary_cross_entropy(result2,labels,reduce = False)
	loss4 = F.binary_cross_entropy(result3,1-labels,reduce = False)
	return (loss1 + loss2 + loss3 + loss4).mean()


def second_discriminator_loss( input1, netD2, netE, netG , fake = None ,writer=None,use_penalty = True):
	netD2.zero_grad()
	netE.zero_grad()
	netG.zero_grad()

	rec_real = netG( netE(input1) ).detach()
	real_real = torch.cat((input1,input1),dim =1)
	real_recreal = torch.cat((input1,rec_real),dim =1)
	
	output = netD2(real_real).view(-1)
	output = F.relu(1 - output)
	errD2_real = output.mean()


	output = netD2(real_recreal).view(-1)
	output = F.relu(1+output)
	errD2_fake = output.mean()

	loss = errD2_real + errD2_fake

	if(use_penalty):
		lossD2 = calc_gradient_penalty_secondD(netD2, input1, fake)
		loss+= lossD2
        
	return loss




def gminusgegloss(noise,netG,netE):
	batch_size = noise.shape[0]
	difference = (netG(noise) - netG(netE(netG(noise)))).view(batch_size,-1)
	loss = torch.sum(torch.abs(difference),dim=1).mean()
	return loss




def calc_gradient_penalty(netD, real_data, fake_data):
	b_size = real_data.shape[0]
	alpha = torch.unsqueeze(torch.unsqueeze((torch.rand(b_size, 1)), -1), -1)
	alpha = alpha.to(device)

	#this finds stuff on the line between real and fake 
	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	interpolates = interpolates.to(device)
	interpolates = torch.tensor(interpolates, requires_grad = True)

	#Runs the discriminator on the resulting interpolated points
	disc_interpolates = netD(interpolates)

	#Calculates the gradient
	gradients = autograd.grad(outputs = disc_interpolates, inputs=interpolates, grad_outputs = torch.ones(disc_interpolates.size()).to(device) , create_graph = True, retain_graph = True, only_inputs=True)[0]
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty


def calc_gradient_penalty_secondD(netD2, real_data, fake_data):
	b_size = real_data.shape[0]
	alpha = torch.unsqueeze(torch.unsqueeze((torch.rand(b_size, 1)), -1), -1)
	# alpha = torch.rand(b_size, 1)
	alpha = alpha.to(device)
	#this finds stuff on the line between real and fake 
	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	# interpolates = real_data
	interpolates = interpolates.to(device)
	interpolates = torch.tensor(interpolates, requires_grad = True)
	real_inter = torch.cat((real_data,interpolates),dim=1)
	#Runs the discriminator on the resulting interpolated points
	disc_interpolates = netD2(real_inter)

	#Calculates the gradient
	gradients = autograd.grad(outputs = disc_interpolates, inputs=real_inter, grad_outputs = torch.ones(disc_interpolates.size()).to(device) , create_graph = True, retain_graph = True, only_inputs=True)[0]
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty


# alpha = torch.FloatTensor(b_size, 1).uniform_(0, 0.5)

def DCritic(netE,netG,netD,x,alpha,reg=0.2):
	e1 = netE(x).detach()
	idx = [i for i in range(e1.size(0)-1, -1,-1)]
	idx = torch.LongTensor(idx).to(device)
	e2 = e1.index_select(0, idx)
	alpha = alpha.to(device)
	interpolate =  alpha * e1 + (1 - alpha)*e2
	ri = netG(interpolate).detach()
	ae = netG(e1).detach()
	fake_loss = torch.pow(netD(ri) - alpha,2).mean()
	real_loss = torch.pow(netD(ae + reg * (x - ae)),2).mean()
	
	return fake_loss + real_loss




## aloha [0,1]
## EGcritic should not use detach
def EGcritic(netE,netG,netD,x,alpha,weights = 0.5):
	e1 = netE(x)
	idx = [i for i in range(e1.size(0)-1, -1,-1)]
	idx = torch.LongTensor(idx).to(device)
	e2 = e1.index_select(0, idx)
	alpha = alpha.to(device)
	interpolate =  alpha * e1 + (1 - alpha)*e2

	ri = netG(interpolate)
	return weights * torch.pow(netD(ri),2).mean()



## test 这两个
def twoDCritic(netE,netG,netD,x,alpha,reg=0.2):
	e1 = netE(x).detach()
	idx = [i for i in range(e1.size(0)-1, -1,-1)]
	idx = torch.LongTensor(idx).to(device)
	e2 = e1.index_select(0, idx)
	

	alpha = alpha.to(device)
	interpolate =  alpha * e1 + (1 - alpha)*e2
	

	ri = netG(interpolate).detach()





	x1 = x
	x2 = x.index_select(0,idx)	

	e1_ri = torch.cat((x1,ri),dim=1).detach()
	e2_ri = torch.cat((x2,ri),dim=1).detach()


	ae = netG(e1).detach()


##。没错 ok 已经check 完成
	fake1_loss = torch.pow(netD(e1_ri) - (1-alpha),2).mean()
	fake2_loss = torch.pow(netD(e2_ri) - alpha,2).mean()
	## not quite sure if we should use regulazation or not

	standard1 = torch.cat((x1 ,ae+ reg * (x1 - ae)),dim=1)

	real1_loss = torch.pow(netD(standard1),2).mean()
	
	return (0.5*fake1_loss + 0.5*fake2_loss) + real1_loss

## test 这两个
def twoEGcritic(netE,netG,netD,x,alpha,weights = 0.5):
	e1 = netE(x)
	idx = [i for i in range(e1.size(0)-1, -1,-1)]
	idx = torch.LongTensor(idx).to(device)
	e2 = e1.index_select(0, idx)
	alpha = alpha.to(device)
	interpolate =  alpha * e1 + (1 - alpha)*e2
	ri = netG(interpolate)

	x1 = x
	x2 = x.index_select(0,idx)	

	e1_ri = torch.cat((x1,ri),dim=1)
	e2_ri = torch.cat((x2,ri),dim=1)

	return weights * ( alpha*torch.pow(netD(e1_ri),2) + (1-alpha)*torch.pow(netD(e2_ri),2)).mean()






###############   Give D(x,hatx) 0 and D(x1, G(E(xinter))) and D(x2,G(E(xinter))) 1  because they are all fakes 
###############   GE try to recover alpha and 1 - alpha from Discriminator
###############   alpha * z1 + (1-alpha)*z2
###############   
def twoDCritic2(netE,netG,netD,x,alpha,reg=0.2):
	e1 = netE(x).detach()
	idx = [i for i in range(e1.size(0)-1, -1,-1)]
	idx = torch.LongTensor(idx).to(device)
	e2 = e1.index_select(0, idx)
	

	alpha = alpha.to(device)
	interpolate =  alpha * e1 + (1 - alpha)*e2
	

	ri = netG(interpolate).detach()


	ae = netG(e1).detach()


	x1 = x
	x2 = x1.index_select(0,idx)	

	e1_ri = torch.cat((x1,ri),dim=1).detach()
	e2_ri = torch.cat((x2,ri),dim=1).detach()




##。没错 ok 已经check 完成
	fake1_loss = torch.pow(netD(e1_ri) - 1,2)
	fake2_loss = torch.pow(netD(e2_ri) - 1,2)
	## not quite sure if we should use regulazation or not

	standard1 = torch.cat((x1 ,ae+ reg * (x1 - ae)),dim=1)

	real1_loss = torch.pow(netD(standard1),2).mean()
	
	return (  (1-alpha)*fake1_loss + (alpha)*fake2_loss).mean() + real1_loss




## e1_ri and e2_ri should recover form alpha and 1 - alpha
## e1_ri should recover from 1 - alpha
## e2_ri should recover from alpha 
## because 0 is non_interpolate point

def twoEGcritic2(netE,netG,netD,x,alpha,weights = 0.5):
	e1 = netE(x)
	idx = [i for i in range(e1.size(0)-1, -1,-1)]
	idx = torch.LongTensor(idx).to(device)
	e2 = e1.index_select(0, idx)
	alpha = alpha.to(device)
	interpolate =  alpha * e1 + (1 - alpha)*e2
	ri = netG(interpolate)

	x1 = x
	x2 = x.index_select(0,idx)	

	e1_ri = torch.cat((x1,ri),dim=1)
	e2_ri = torch.cat((x2,ri),dim=1)

	return weights * ( torch.pow(netD(e1_ri)- (1 - alpha),2) + (torch.pow(netD(e2_ri)-(alpha),2) )).mean()






## alpha1 and alpha2 , alpha1 and alpha2 should be sampled from uniform distribution [0,0.5]
def threeDCritic(netE,netG,netD,x,alpha1,alpha2,writer,n_iter,t,reg=0.2):
	
	e1 = netE(x).detach()
	b_size = x.shape[0]
	p = int(b_size/3)
	
	idx1 = [i for i in range(p)]
	
	idx2 = [i for i in range(p,2*p)]
	
	idx3 = [i for i in range(2*p,b_size)]
	
	idxe2 = idx3 + idx1 + idx2
	idxe2 = torch.LongTensor(idxe2).to(device)

	idxe3 = idx2 + idx3 + idx1
	idxe3 = torch.LongTensor(idxe3).to(device)
	

	e2 = e1.index_select(0, idxe2)
	e3 = e1.index_select(0, idxe3)


	x1 = x
	x2 = x.index_select(0,idxe2)
	x3 = x.index_select(0,idxe3)


	alpha1 = alpha1.to(device)
	
	alpha2 = alpha2.to(device)


	interpolate = alpha1*e1 + alpha2*e2 + (1 - alpha1 - alpha2 )*e3
	
	
	ri = netG(interpolate).detach()

	e1_ri = torch.cat((x1,ri),dim=1).detach()
	e2_ri = torch.cat((x2,ri),dim=1).detach()
	e3_ri = torch.cat((x3,ri),dim=1).detach()

	fake1_loss = torch.pow(netD(e1_ri) - (1-alpha1),2).mean()
	fake2_loss = torch.pow(netD(e2_ri) - (1-alpha2),2).mean()
	fake3_loss = torch.pow(netD(e3_ri) - (alpha1 + alpha2),2).mean()
	## not quite sure if we should use regulazation or not

	ae = netG(e1).detach()

	standard1 = torch.cat((x1 ,ae+ reg * (x1 - ae)),dim=1)


	## non-interpolate point should be close to 1
	real1_loss = torch.pow(netD(standard1),2).mean()

	if(t == 4):
		writer.add_scalars('Df1Loss', {'f1_loss':fake1_loss.data.cpu().numpy()}, n_iter)
		writer.add_scalars('Df2Loss', {'f2_loss':(fake2_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('Df3Loss', {'f3_loss':(fake3_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('Dr1Loss', {'r1_loss':(real1_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('DtotalLoss', {'total':((fake1_loss/3 +  fake2_loss/3 + fake3_loss/3) + real1_loss).data.cpu().numpy()}, n_iter)


	return (fake1_loss/3 +  fake2_loss/3 + fake3_loss/3) + real1_loss



def threeEGcritic(netE,netG,netD,x,alpha1,alpha2,weights = 0.5):
	e1 = netE(x)
	b_size = x.shape[0]
	p = int(b_size/3)
	
	idx1 = [i for i in range(p)]
	
	idx2 = [i for i in range(p,2*p)]
	
	idx3 = [i for i in range(2*p,b_size)]


		
	idxe2 = idx3 + idx1 + idx2
	idxe2 = torch.LongTensor(idxe2).to(device)

	idxe3 = idx2 + idx3 + idx1
	idxe3 = torch.LongTensor(idxe3).to(device)
	

	e2 = e1.index_select(0, idxe2)
	e3 = e1.index_select(0, idxe3)


	x1 = x
	x2 = x.index_select(0,idxe2)
	x3 = x.index_select(0,idxe3)

	alpha1 = alpha1.to(device)
	alpha2 = alpha2.to(device)


	interpolate = alpha1*e1 + alpha2*e2 + (1 - alpha1 - alpha2 )*e3
	ri = netG(interpolate)

	e1_ri = torch.cat((x1,ri),dim=1)
	
	e2_ri = torch.cat((x2,ri),dim=1)
	
	e3_ri = torch.cat((x3,ri),dim=1)


	return weights * ( alpha1*torch.pow(netD(e1_ri),2) + (alpha2)*torch.pow(netD(e2_ri),2)  
			+ (1-alpha1-alpha2)*torch.pow(netD(e3_ri),2) ).mean()

def threeDCritic2(netE,netG,netD,x,alpha1,alpha2,writer,n_iter,t,reg=0.2):
	
	e1 = netE(x).detach()
	b_size = x.shape[0]
	p = int(b_size/3)
	
	idx1 = [i for i in range(p)]
	
	idx2 = [i for i in range(p,2*p)]
	
	idx3 = [i for i in range(2*p,b_size)]
	
	idxe2 = idx3 + idx1 + idx2
	idxe2 = torch.LongTensor(idxe2).to(device)

	idxe3 = idx2 + idx3 + idx1
	idxe3 = torch.LongTensor(idxe3).to(device)
	

	e2 = e1.index_select(0, idxe2)
	e3 = e1.index_select(0, idxe3)


	x1 = x
	x2 = x.index_select(0,idxe2)
	x3 = x.index_select(0,idxe3)


	alpha1 = alpha1.to(device)
	
	alpha2 = alpha2.to(device)


	interpolate = alpha1*e1 + alpha2*e2 + (1 - alpha1 - alpha2 )*e3

	
	ae = netG(e1).detach()
	ae2 = ae.index_select(0,idxe2)
	ae3 = ae.index_select(0,idxe3)

	standard1 = torch.cat((x1 ,ae+ reg * (x1 - ae)),dim=1)
	


	x1apex = ae + reg*(x1-ae)
	x2apex = ae2 + reg*(x2 - ae2)
	x3apex = ae3 + reg*(x3 - ae3)
	
	ri = netG(interpolate).detach()

	e1_ri = torch.cat((x1apex,ri),dim=1).detach()
	e2_ri = torch.cat((x2apex,ri),dim=1).detach()
	e3_ri = torch.cat((x3apex,ri),dim=1).detach()

	fake1_loss = torch.pow(netD(e1_ri) - (1-alpha1),2).mean()
	fake2_loss = torch.pow(netD(e2_ri) - (1-alpha2),2).mean()
	fake3_loss = torch.pow(netD(e3_ri) - (alpha1 + alpha2),2).mean()
	## not quite sure if we should use regulazation or not



	## non-interpolate point should be close to 1
	real1_loss = torch.pow(netD(standard1),2).mean()

	if(t == 4):
		writer.add_scalars('Df1Loss', {'f1_loss':fake1_loss.data.cpu().numpy()}, n_iter)
		writer.add_scalars('Df2Loss', {'f2_loss':(fake2_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('Df3Loss', {'f3_loss':(fake3_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('Dr1Loss', {'r1_loss':(real1_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('DtotalLoss', {'total':((fake1_loss/3 +  fake2_loss/3 + fake3_loss/3) + real1_loss).data.cpu().numpy()}, n_iter)


	return (fake1_loss/3 +  fake2_loss/3 + fake3_loss/3) + real1_loss
## alpha1 and alpha2 , alpha1 and alpha2 should be sampled from uniform distribution [0,0.5]

def threeEGcritic2(netE,netG,netD,x,alpha1,alpha2,weights = 0.5,reg=0.2):
	e1 = netE(x)
	b_size = x.shape[0]
	p = int(b_size/3)
	
	idx1 = [i for i in range(p)]
	
	idx2 = [i for i in range(p,2*p)]
	
	idx3 = [i for i in range(2*p,b_size)]


		
	idxe2 = idx3 + idx1 + idx2
	idxe2 = torch.LongTensor(idxe2).to(device)

	idxe3 = idx2 + idx3 + idx1
	idxe3 = torch.LongTensor(idxe3).to(device)
	

	e2 = e1.index_select(0, idxe2)
	e3 = e1.index_select(0, idxe3)


	x1 = x
	x2 = x.index_select(0,idxe2)
	x3 = x.index_select(0,idxe3)

	alpha1 = alpha1.to(device)
	alpha2 = alpha2.to(device)


	interpolate = alpha1*e1 + alpha2*e2 + (1 - alpha1 - alpha2 )*e3
	ri = netG(interpolate)



	## no need to detach here
	## maybe we need detach
	## test with detach
	ae = netG(e1)
	ae2 = ae.index_select(0,idxe2)
	ae3 = ae.index_select(0,idxe3)

	x1apex = ae + reg*(x1-ae)
	x2apex = ae2 + reg*(x2 - ae2)
	x3apex = ae3 + reg*(x3 - ae3)


	e1_ri = torch.cat((x1apex,ri),dim=1)
	
	e2_ri = torch.cat((x2apex,ri),dim=1)
	
	e3_ri = torch.cat((x3apex,ri),dim=1)


	return weights * ( alpha1*torch.pow(netD(e1_ri),2) + (alpha2)*torch.pow(netD(e2_ri),2)  
			+ (1-alpha1-alpha2)*torch.pow(netD(e3_ri),2) ).mean()



def fourDCritic(netE,netG,netD,x,alpha1,alpha2,alpha3,writer,n_iter,t,reg=0.2):
	
	e1 = netE(x).detach()
	b_size = x.shape[0]
	p = int(b_size/4)
	idx1 = [i for i in range(p)]
	
	idx2 = [i for i in range(p,2*p)]
	
	idx3 = [i for i in range(2*p,3*p)]

	idx4 = [i for i in range(3*p,4*p)]


	idxe2 = idx4 + idx1 + idx2 + idx3
	idxe2 = torch.LongTensor(idxe2).to(device)

	idxe3 = idx3 + idx4 + idx1 + idx2 
	idxe3 = torch.LongTensor(idxe3).to(device)

	idxe4 = idx2 + idx3 + idx4 + idx1
	idxe4 = torch.LongTensor(idxe4).to(device)

	

	e2 = e1.index_select(0, idxe2)
	e3 = e1.index_select(0, idxe3)
	e4 = e1.index_select(0, idxe4)


	x1 = x
	x2 = x.index_select(0,idxe2)
	x3 = x.index_select(0,idxe3)
	x4 = x.index_select(0,idxe4)


	alpha1 = alpha1.to(device)
	
	alpha2 = alpha2.to(device)

	alpha3 = alpha3.to(device)

	alpha4 = 1 - alpha1 - alpha2 - alpha3


	interpolate = alpha1*e1 + alpha2*e2 + (alpha3)*e3 + alpha4*e4
	
	
	ri = netG(interpolate).detach()

	e1_ri = torch.cat((x1,ri),dim=1).detach()
	e2_ri = torch.cat((x2,ri),dim=1).detach()
	e3_ri = torch.cat((x3,ri),dim=1).detach()
	e4_ri = torch.cat((x4,ri),dim=1).detach()

	fake1_loss = torch.pow(netD(e1_ri) - (1-alpha1),2).mean()
	fake2_loss = torch.pow(netD(e2_ri) - (1-alpha2),2).mean()
	fake3_loss = torch.pow(netD(e3_ri) - (1-alpha3),2).mean()
	fake4_loss = torch.pow(netD(e4_ri) - (1-alpha4),2).mean()
	## not quite sure if we should use regulazation or not

	ae = netG(e1).detach()

	standard1 = torch.cat((x1 ,ae+ reg * (x1 - ae)),dim=1)


	## non-interpolate point should be close to 1
	real1_loss = torch.pow(netD(standard1),2).mean()

	if(t == 4):
		writer.add_scalars('Df1Loss', {'f1_loss':fake1_loss.data.cpu().numpy()}, n_iter)
		writer.add_scalars('Df2Loss', {'f2_loss':(fake2_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('Df3Loss', {'f3_loss':(fake3_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('Dr1Loss', {'r1_loss':(real1_loss).data.cpu().numpy()}, n_iter)
		writer.add_scalars('DtotalLoss', {'total':((fake1_loss/3 +  fake2_loss/3 + fake3_loss/3) + real1_loss).data.cpu().numpy()}, n_iter)


	return (fake1_loss/4 +  fake2_loss/4 + fake3_loss/4 + fake4_loss/4)  + real1_loss



def fourEGcritic(netE,netG,netD,x,alpha1,alpha2,alpha3,weights):
	e1 = netE(x)
	
	b_size = x.shape[0]
	p = int(b_size/4)
	idx1 = [i for i in range(p)]
	
	idx2 = [i for i in range(p,2*p)]
	
	idx3 = [i for i in range(2*p,3*p)]

	idx4 = [i for i in range(3*p,4*p)]


	idxe2 = idx4 + idx1 + idx2 + idx3
	idxe2 = torch.LongTensor(idxe2).to(device)

	idxe3 = idx3 + idx4 + idx1 + idx2 
	idxe3 = torch.LongTensor(idxe3).to(device)

	idxe4 = idx2 + idx3 + idx4 + idx1
	idxe4 = torch.LongTensor(idxe4).to(device)

	

	e2 = e1.index_select(0, idxe2)
	e3 = e1.index_select(0, idxe3)
	e4 = e1.index_select(0, idxe4)


	x1 = x
	x2 = x.index_select(0,idxe2)
	x3 = x.index_select(0,idxe3)
	x4 = x.index_select(0,idxe4)


	alpha1 = alpha1.to(device)
	
	alpha2 = alpha2.to(device)

	alpha3 = alpha3.to(device)

	alpha4 = 1 - alpha1 - alpha2 - alpha3


	interpolate = alpha1*e1 + alpha2*e2 + (alpha3)*e3 + alpha4*e4
	
	
	ri = netG(interpolate)

	e1_ri = torch.cat((x1,ri),dim=1)
	e2_ri = torch.cat((x2,ri),dim=1)
	e3_ri = torch.cat((x3,ri),dim=1)
	e4_ri = torch.cat((x4,ri),dim=1)


	return weights * ( alpha1*torch.pow(netD(e1_ri),2) + (alpha2)*torch.pow(netD(e2_ri),2)  
			+ (alpha3)*torch.pow(netD(e3_ri),2) + alpha4*torch.pow(netD(e4_ri),4)).mean()


