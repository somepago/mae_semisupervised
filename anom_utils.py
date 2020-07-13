import torch
import torchvision.transforms as transforms
from newevaluate import evaluate
from itertools import chain
import numpy as np


def post_process(image):
	image = image.view(-1, 3, 32, 32)
	image = image.mul(0.5).add(0.5)
	return image

def generate_image(image, frame, name):
	image = image.cpu()
	image = post_process(image)
	image = transforms.ToPILImage()(vutils.make_grid(image, padding=2, normalize=False))


def reconstruction_loss(image1, image2):
	nc, image_size, _ = image1.shape
	image1, image2 = post_process(image1), post_process(image2)
	norm = torch.norm((image2 - image1).view(-1,nc*image_size*image_size), dim=(1))
	return norm.view(-1).data.cpu().numpy()


#Calculates the L2 loss between image1 and image2
def latent_reconstruction_loss(image1, image2):
	norm = torch.norm((image2 - image1), dim=1)
	return norm.view(-1).data.cpu().numpy()

def l1_latent_reconstruction_loss(image1, image2):
	norm = torch.sum(torch.abs(image2 - image1),dim=1)
	return norm.view(-1).data.cpu().numpy()

    

def adjust_learning_rate(optimizer, epoch, num_epochs, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate - lrate * (epoch-45)/(num_epochs - 45)
    print('use learning rate %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def half_adjust_learning_rate(optimizer, epoch, num_epochs, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate - 1e-4
    print('use learning rate %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def anomaly_score(data, netG, netE, netD2, ngpu=1):
	if (ngpu > 1):
		a1 = netD2.module.feature(torch.cat((data,data),dim =1))
		a2 = netD2.module.feature(torch.cat((data,netG(netE(data)).detach()),dim=1))
	else:
		a1 = netD2.feature(torch.cat((data,data),dim =1))
		a2 = netD2.feature(torch.cat((data,netG(netE(data)).detach()),dim=1))
	return l1_latent_reconstruction_loss(a1,a2)

def score_and_auc(dataLoader, netG, netE, netD2, device, ngpu=1, break_iters = 100):
	score_list = []
	score_label = []
	count=0
	with torch.no_grad():
		for i,data in enumerate(dataLoader, 0):
# 			targets = data[1].to(device)
# 			print(torch.unique(targets, return_counts=True))
# 			if count>=break_iters:
# 				break
			real = data[0].to(device)
			score_label.append(data[1].to(device).tolist())
			score_list.append(anomaly_score(real,netG, netE, netD2, ngpu))
			
			count+=1
# 		import ipdb;ipdb.set_trace()
		score_list = list(chain.from_iterable(score_list))
		score_label = list(chain.from_iterable(score_label))
		print(score_list[:5], score_list[-5:])
		print(score_label[:5], score_label[-5:])
		score_anom_mean = np.array(score_list).mean()
		if np.sum(score_label) == 0:
			print('only non anomolous samples are passed' + str(len(score_label)))
			score_auc = 0
		else:
			score_auc = evaluate(score_label,score_list)
		
	return score_auc, score_anom_mean