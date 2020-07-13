"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import argparse


##
def load_data(opt):
    """ Load Data
    Args:
        opt ([type]): Argument Parser
    Raises:
        IOError: Cannot Load Dataset
    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    print(opt.dataset)


    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset in ['cifar10']:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
        dataset['test'] = CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=classes[opt.abnormal_class],
            manualseed=opt.manualSeed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualSeed == -1
                                                     else lambda x: np.random.seed(opt.manualSeed)))
                      for x in splits}        
        return dataloader

    

    elif opt.dataset in ['cifarop']:
        print('use cifa10 a noramal and 9 abnormal')

        splits = ['train', 'test', 'val']
        drop_last_batch = {'train': True, 'test': False ,'val': False}
        shuffle = {'train': True, 'test': False ,'val': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
        dataset['test'] = CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
        dataset['val'] = CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets, \
        dataset['val'].data, dataset['val'].targets = get_cifar_anomaly_datasetop(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=classes[opt.abnormal_class],
            manualseed=opt.manualSeed
        )

        
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualSeed == -1
                                                     else lambda x: np.random.seed(opt.manualSeed)))
                      for x in splits}        
        return dataloader






    elif opt.dataset in ['mnist']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=opt.abnormal_class,
            manualseed=opt.manualSeed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualSeed == -1
                                                     else lambda x: np.random.seed(opt.manualSeed)))
                      for x in splits}


        return dataloader

    elif opt.dataset in ['mnist2']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist2_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            nrm_cls_idx=opt.abnormal_class,
            proportion=opt.proportion,
            manualseed=opt.manualSeed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualSeed == -1
                                                     else lambda x: np.random.seed(opt.manualSeed)))
                      for x in splits}
        return dataloader

    else:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualSeed == -1
                                                     else lambda x: np.random.seed(opt.manualSeed)))
                      for x in splits}

        
        return dataloader

##
def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]
    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != 5:
        # Random seed.
        # Concatenate the original train and test sets.
        print('Im at manual seed area in dataloader')
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
#     new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
#     new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    new_tst_img = np.concatenate((nrm_tst_img[:1000], abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl[:1000], abn_tst_lbl), axis=0)
    print(len(new_trn_lbl), len(new_tst_lbl))
    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl



# def get_cifar_anomaly_datasetop(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
#     """[summary]
#     Arguments:
#         trn_img {np.array} -- Training images
#         trn_lbl {np.array} -- Training labels
#         tst_img {np.array} -- Test     images
#         tst_lbl {np.array} -- Test     labels
#     Keyword Arguments:
#         abn_cls_idx {int} -- Anomalous class index (default: {0})
#     Returns:
#         [np.array] -- New training-test images and labels.
#     """
#     # Convert train-test labels into numpy array.
#     trn_lbl = np.array(trn_lbl)
#     tst_lbl = np.array(tst_lbl)

#     # --
#     # Find idx, img, lbl for abnormal and normal on org dataset.
#     nrm_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
#     abn_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    

#     nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
#     abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
#     nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
#     abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    
#     nrm_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
#     abn_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    
#     nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
#     abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
#     nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
#     abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

#     # --
#     # Assign labels to normal (0) and abnormals (1)
#     nrm_trn_lbl[:] = 0
#     nrm_tst_lbl[:] = 0
#     abn_trn_lbl[:] = 1
#     abn_tst_lbl[:] = 1

#     # --
#     if manualseed != -1:
#         # Random seed.
#         # Concatenate the original train and test sets.
#         nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
#         nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
#         abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
#         abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)


#         # Split the normal data into the new train and tests.
#         idx = np.arange(len(nrm_lbl))
#         np.random.seed(manualseed)
#         np.random.shuffle(idx)

#         nrm_trn_len = int(len(idx) * 0.80)

#         nrm_trn_idx = idx[:3600]
#         nrm_tst_idx = idx[nrm_trn_len:]
#         nrm_val_idx = idx[3600:nrm_trn_len]



#         nrm_trn_img = nrm_img[nrm_trn_idx]
#         nrm_trn_lbl = nrm_lbl[nrm_trn_idx]

#         nrm_tst_img = nrm_img[nrm_tst_idx]
#         nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

#         nrm_val_img = nrm_img[nrm_val_idx]
#         nrm_val_lbl = nrm_lbl[nrm_val_idx]



#         abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
#         abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

#         idx = np.arange(len(abn_lbl))

#         np.random.seed(manualseed)
#         np.random.shuffle(idx)

#         abn_trn_len = int(len(idx) * 0.80)

#         abn_trn_idx = idx[:abn_trn_len]
        
#         abn_tst_idx = idx[abn_trn_len:]

#         abn_tst_img = abn_img[abn_tst_idx]
#         abn_tst_lbl = abn_lbl[abn_tst_idx]

#     # Create new anomaly dataset based on the following data structure:
#     # - anomaly dataset
#     #   . -> train
#     #        . -> normal
#     #   . -> test
#     #        . -> normal
#     #        . -> abnormal
#     new_trn_img = np.copy(nrm_trn_img)
#     new_trn_lbl = np.copy(nrm_trn_lbl)

#     new_val_img = np.copy(nrm_val_img)
#     new_val_lbl = np.copy(nrm_val_lbl)



#     new_tst_img = np.concatenate((nrm_tst_img,abn_tst_img), axis=0)
#     new_tst_lbl = np.concatenate((nrm_tst_lbl,abn_tst_lbl), axis=0)

#     return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl,new_val_img,new_val_lbl


    






