import torch
import torchvision.transforms as transforms

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