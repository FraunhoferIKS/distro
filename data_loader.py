""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

Data loader for the experiments

This file contains the data loader for the experiments. It is based on the
data loader from the pytorch-ood repository. The data loader is used to
load the datasets and to create the dataloaders for the experiments.

"""

import torch
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN
from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    GaussianNoise,
    UniformNoise,
)
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


def prepare_data(dataset_path, batch_size, dataset):

    """ 
    Prepare the data for the experiments
    
    Parameters
    ----------
        dataset_path : str
            Path to the datasets.
        batch_size : int
            Batch size for the dataloaders.
        dataset : str
            Name of the dataset.
    
    Returns
    -------
        id_loader : torch.utils.data.DataLoader
            Dataloader for the in-distribution dataset.
        loaders : dict
            Dictionary containing the dataloaders for the out-of-distribution
            datasets.
    """

    trn = transforms.ToTensor()
    crop = transforms.Compose([transforms.CenterCrop(32), trn])
    resize = transforms.Compose([transforms.Resize((32, 32)), trn])

    cifar10 = True if dataset == 'cifar10' else False
    if cifar10:
        id_dataset = CIFAR10(dataset_path, train=False, transform=trn, download=False)
    else:
        id_dataset = CIFAR100(dataset_path, train=False, transform=trn, download=False)

    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # create all OOD datasets
    np.random.seed(seed=1)
    
    # ood_datasets = [CIFAR100, SVHN]#, LSUNCrop, TinyImageNetCrop, LSUNResize] #TinyImageNetResize, Textures
    ood_datasets = {
        'CIFAR100' : CIFAR100('/storage/project_data/robustness_certificates/data/images_classic/', train=False, transform=trn, download=True)
        if cifar10 else CIFAR10('/storage/datasets/torchvision_cache/', train=False, transform=trn, download=False),
        'SVHN' : SVHN('/storage/datasets/torchvision_cache/svhn/', split='test', transform=trn, download=False),
        'LSUNCrop' : LSUNCrop('/storage/datasets/processed/', transform=crop, download=False),
        'GaussianNoise' : GaussianNoise(length=1000, size=(32, 32, 3), transform=trn, seed=1),
        'LSUNResize' : LSUNResize('/storage/datasets/processed/', transform=resize, download=False),
        'TinyImageNetCrop' : TinyImageNetCrop('/storage/datasets/processed/', transform=crop, download=False),
        # 'TinyImageNetResize' : TinyImageNetResize('/storage/datasets/processed/', transform=resize, download=False),
        'Textures' : Textures('/storage/datasets/processed/', transform=resize, download=False),
        'UniformNoise' : UniformNoise(length=1000, size=(32, 32, 3), transform=trn, seed=1),
    }

    target_size = 1000 #400 if args.dset_in_name=='RImgNet' else
    

    loaders = {}
    for name, dataset in ood_datasets.items():
        subset = np.random.choice(len(dataset), size=min(len(dataset), target_size), replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(subset)
        # dataset_out_test = ood_dataset('/storage/datasets/processed/', transform=transform, train=False, download=False)
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=sampler)

    return id_loader, loaders


def prepare_loader_for_robustness(dataset_path, batch_size):
    """ Prepare the loader for the robustness evaluation 
    
    Args:
        dataset_path (str): path to the dataset
        batch_size (int): batch size for the loader
        
    Returns:
        loader (torch.utils.data.DataLoader): loader for the robustness evaluation
    """
    np.random.seed(seed=1)
    trn = transforms.ToTensor()
    id_dataset = CIFAR10(dataset_path, train=False, transform=trn, download=False)
    subset = np.random.choice(
        len(id_dataset), size=min(len(id_dataset), 1000), replace=False)
    sampler = torch.utils.data.SubsetRandomSampler(subset)

    return DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=sampler)

def prepare_training_data(dataset_path, batch_size):
    """ 
    Prepare the loader for the training of the model
    
    Args:
        dataset_path (str): path to the dataset
        batch_size (int): batch size for the loader

    Returns:
        loader (torch.utils.data.DataLoader): loader for the training of the model
    """

    dataset = CIFAR10(dataset_path, train=True, transform=transforms.ToTensor(), download=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader