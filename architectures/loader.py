""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

import torch
import numpy as np
from pathlib import Path
from architectures import densenet, wrn, wrn_virtual
from architectures import prood_resnet, provable_classifiers, modules_ibp
from architectures.resnet import ResNet18_32x32
import torchvision.transforms as transforms

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)



def standard_args(batch_size, model_path):
    """
    Standard arguments for the diffusion model
    
    Parameters
    ----------
        batch_size : int - Batch size for the loader
        model_path : str - Path to the diffusion model
        
    Returns
    -------
    dict
        Dictionary with the standard arguments
    """
    return dict(
        image_size=32,
        clip_denoised=True,
        num_samples=10000,
        batch_size=batch_size,
        use_ddim=False,
        model_path=model_path,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.3,
        learn_sigma=True,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=4000 if 'CIFAR10' in str(model_path) else 1000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def prepare_diffusion(args):
    """
    Prepare the diffusion model
    
    Parameters
    ----------
        args : dict
            Dictionary containing the arguments for the diffusion model.
        
    Returns
    -------
        model : torch.nn.Module - The diffusion model.
        diffusion : Diffusion - The diffusion process.

    """
    defaults = model_and_diffusion_defaults()
    defaults.update({k: args[k] for k in defaults.keys()})

    model, diffusion = create_model_and_diffusion(**defaults)
    model.load_state_dict(
            torch.load(args['model_path'], map_location="cpu")
        )
    args = dotdict(args)

    return model, diffusion



class Plain(torch.nn.Module):
    """ 
    This is a PyTorch nn.Module that wraps a given model and applies 
    a resize and normalization transformation to the input data before 
    forwarding it through the model.    
    
    Parameters
    ----------
        model : torch.nn.Module
            The model to wrap.
        resize : torchvision.transforms.Resize, optional
        normalization : torchvision.transforms.Normalize, optional
    """

    def __init__(self, model, resize=None, normalization=None) -> None:
        super().__init__()

        self.model = model
        self.normalization = normalization
        self.resize = resize

    def forward(self, x):
        if self.resize is not None:
            x = self.resize(x)
        if self.normalization is not None:
            x = self.normalization(x)
        try:
            return self.model(x).logits 
        except:
            return self.model(x)
        
class Diffusion(torch.nn.Module):
    """ 
    This is a PyTorch nn.Module that wraps a given model and applies 
    a resize and normalization transformation to the input data before 
    forwarding it through the model.    
    
    Parameters
    ----------
        model : torch.nn.Module
            The model to wrap.
        resize : torchvision.transforms.Resize, optional
        normalization : torchvision.transforms.Normalize, optional
    """

    def __init__(self, model, diffusion, dds_model,  resize=None, normalization=None) -> None:
        super().__init__()

        self.model = model
        self.diffusion = diffusion
        self.dds_model = dds_model
        self.normalization = normalization
        self.resize = resize

        if self.normalization is not None:
            self.inv_normalization = transforms.Normalize(
                mean= [-m/s for m, s in zip(normalization.mean, normalization.std)],
                std= [1/s for s in normalization.std]
                )
        
        self.t = 0

    def fix_step_size(self, sigma):
        """" Fix the step size of the diffusion process """
        # Get the timestep t corresponding to noise level sigma
        target_sigma = sigma * 2
        real_sigma = 0
        t = 0
        while real_sigma < target_sigma:
            t += 1
            a = self.diffusion.sqrt_alphas_cumprod[t]
            b = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
            real_sigma = b / a
        # for diffusion
        self.t = t
    
    def denoise(self, x_start, t):
        t_batch = torch.tensor([t] * len(x_start)).cuda()
        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(
            x_start=x_start, t=t_batch, noise=noise)
        
        # with torch.no_grad():
        return self.diffusion.p_sample(
                self.dds_model,
                x_t_start,
                t_batch,
                clip_denoised=True
            )['pred_xstart']


    def forward(self, x):

        if self.normalization is not None:
            x = self.normalization(x)

        x = self.denoise(x, self.t)

        if self.normalization is not None:
            x = self.inv_normalization(x)

        try:
            return self.model(x).logits 
        except:
            return self.model(x)

def load_predictor(args, models_path):
    """
    Load the predictor model
    
    Parameters
    ----------
        args : dict 
            Dictionary containing the arguments for the predictor model.
        models_path : str
            Path to the models.
        
    Returns
    -------
        predictor : torch.nn.Module - The predictor model.

    """

    dataset = 'CIFAR10' if args.dataset == 'cifar10' else 'CIFAR100'
    num_classes = 10 if args.dataset == 'cifar10' else 100

    if args.standardized:
        models_path = models_path/Path(f'standardized_resnet/{dataset}') # path to the standardized models root

    if args.experiment == 'plain':
        # Plain model
        if args.standardized:
            predictor_path= models_path/Path('plain.ckpt')
            predictor = ResNet18_32x32(num_classes=num_classes, normalize=True)
        else:
            predictor_path = models_path/Path('ProoD/'+dataset+'/Plain.pt')
            predictor = prood_resnet.get_ResNet(dset=dataset)
        predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))

    elif args.experiment == 'diffusion':
        # Diffusion model
        mean, std = [0.5]*3, [0.5]*3
        normalize = transforms.Normalize(mean=mean, std=std)
        diffusion_path = models_path/Path('our/'+dataset+'/denoiser.pt')
        dds_model, diffusion = prepare_diffusion(standard_args(args.batch_size, diffusion_path))

        predictor_path = models_path/Path('ProoD/'+dataset+'/OE.pt')
        base_model = prood_resnet.get_ResNet(dset=dataset)
        base_model.load_state_dict(torch.load(predictor_path, map_location='cpu'), strict=False)
        predictor = Diffusion(base_model, diffusion, dds_model, normalization=normalize)

    elif args.experiment == 'oe':
        # OE exposure
        if args.standardized:
            predictor_path= models_path/Path('oe.pt')
            predictor = ResNet18_32x32(num_classes=num_classes, normalize=True)
        else:  
            predictor = prood_resnet.get_ResNet(dset=dataset)
            predictor_path = models_path/Path('ProoD/'+dataset+'/OE.pt')
        predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))

    elif args.experiment == 'atom':
        # ATOM
        if args.standardized:
            predictor_path = models_path/Path('atom.pth')
            predictor = ResNet18_32x32(num_classes = num_classes + 1, normalize=True)
            predictor.load_state_dict(torch.load(predictor_path)['state_dict'])
        else:
            predictor_path = models_path/Path('ProoD/'+dataset+'/ATOM.pt')
            predictor, normalization = densenet.get_densenet(num_classes=num_classes+1)
            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'), strict=False)
            predictor = Plain(predictor, normalization=normalization)

    elif args.experiment == 'acet':
        # ACET
        if args.standardized:
            predictor_path = models_path/Path('acet.pth')
            predictor = ResNet18_32x32(num_classes=num_classes, normalize=True)
            predictor.load_state_dict(torch.load(predictor_path)['state_dict'])
        else:
            predictor_path = models_path/Path('ProoD/'+dataset+'/ACET.pt')
            predictor, normalization = densenet.get_densenet(num_classes=num_classes)
            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu')['state_dict'])
            predictor = Plain(predictor, normalization=normalization)

    elif args.experiment == 'good':
        # GOOD 80
        predictor_path = models_path/Path('ProoD/'+dataset+'/GOOD80.pt')
        predictor = provable_classifiers.CNN_IBP(dset_in_name=dataset, size='XL', last_bias=True)
        predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))

    elif args.experiment == 'prood':
        num = 3 if args.dataset == 'cifar10' else 1
        # ProoD
        if args.standardized:
            predictor_path = models_path/Path(f'prood_delta{num}.pt')
            base_model = ResNet18_32x32(num_classes=num_classes, normalize=True)
            detector = provable_classifiers.CNN_IBP(dset_in_name=dataset, size='S', last_bias=True, num_classes=1, last_layer_neg=True, normalize=True)
            predictor = modules_ibp.JointModel(base_model, detector, classes=num_classes)
            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        else:
            predictor_path = models_path/Path('ProoD/'+dataset+f'/ProoD {num}.pt')
            base_model = prood_resnet.get_ResNet(dset=''+dataset+'')
            detector = provable_classifiers.CNN_IBP(dset_in_name=dataset, size='S', last_bias=True, num_classes=1, last_layer_neg=True)
            predictor = modules_ibp.JointModel(base_model, detector, classes=num_classes)
            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))

    elif args.experiment == 'distro':
        # ProoD
        mean, std = [0.5]*3, [0.5]*3
        normalize = transforms.Normalize(mean=mean, std=std)
        if args.standardized:
            diffusion_path = models_path/Path('denoiser.pt')
            dds_model, diffusion = prepare_diffusion(standard_args(args.batch_size, diffusion_path))

            num = 3 if args.dataset == 'cifar10' else 1
            predictor_path = models_path/Path(f'prood_delta{num}.pt')
            base_model = ResNet18_32x32(num_classes=num_classes, normalize=True)
            detector = provable_classifiers.CNN_IBP(dset_in_name=dataset, size='S', last_bias=True, num_classes=1, last_layer_neg=True, normalize=True)
            predictor = modules_ibp.JointDiffusionModel(base_model, detector, diffusion, dds_model, normalization=normalize, classes=num_classes)

            # Load detector weights
            predictor_path = models_path/Path(f'prood_delta{num}.pt')
            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'), strict=False)
        else:
            diffusion_path = models_path/Path('our/'+dataset+'/denoiser.pt')
            dds_model, diffusion = prepare_diffusion(standard_args(args.batch_size, diffusion_path))

            base_model = prood_resnet.get_ResNet(dset=''+dataset+'')        
            detector = provable_classifiers.CNN_IBP(dset_in_name=dataset, size='S', last_bias=True, num_classes=1, last_layer_neg=True)
            predictor = modules_ibp.JointDiffusionModel(base_model, detector, diffusion, dds_model, normalization=normalize, classes=10)

            # # Load detector weights
            num = 3 if args.dataset == 'cifar10' else 1
            predictor_path = models_path/Path('ProoD/'+dataset+f'/ProoD {num}.pt')

            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'), strict=False)

    elif args.experiment == 'vos':
        mean = np.array([x / 255 for x in [125.3, 123.0, 113.9]])
        std = np.array([x / 255 for x in [63.0, 62.1, 66.7]])
        normalize = transforms.Normalize(mean, std)

        ## Load the predictor
        predictor = wrn_virtual.WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        # Virtual Outliers Synthesis
        predictor_path = models_path/Path('our/CIFAR10/vos.pt')
        predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))

        predictor = Plain(predictor, normalization=normalize)

    elif args.experiment == 'logit':
        if args.standardized:
            predictor_path = models_path/Path('logitnorm.ckpt')
            predictor = ResNet18_32x32(num_classes=num_classes, normalize=True)
            predictor.load_state_dict(torch.load(predictor_path))
        else:
            mean = np.array([x / 255 for x in [125.3, 123.0, 113.9]])
            std = np.array([x / 255 for x in [63.0, 62.1, 66.7]])
            normalize = transforms.Normalize(mean, std)

            ## Load the predictor
            predictor = wrn.WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)

            ## Logit Norm
            predictor_path = models_path/Path('our/CIFAR10/logitnorm.pt')
            predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            predictor = Plain(predictor, normalization=normalize)

    else:
        raise ValueError('Unknown experiment')

    return predictor