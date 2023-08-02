import torch
import adversarial.attacks as attacks
import adversarial.apgd as apgd
from accelerate import Accelerator
from autoattack import AutoAttack
from denoiser_diffusion import Smooth
import numpy as np

def get_conf_lb(model, loader, num_classes, epsilon, temperature=1.0, score_type='softmax', atom:bool = False):

    """ Compute the lower bound on the confidence of the model on the dataset
    
    Args:
        model: the model to evaluate
        loader: the dataset to evaluate on
        num_classes: the number of classes in the dataset
        epsilon: the radius of the L-inf ball to evaluate on
        temperature: the temperature to use for the softmax
        score_type: the type of score to use for the softmax
        atom: whether to use the atom loss or not
    
    Returns:
        the lower bound on the confidence of the model on the dataset
        
    """
    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)

    restarts = 5
    budget = 5
    from_logits = True
    iterations = 200

    try: 
        model.fix_step_size()
    except:
        diffusion = False


    if atom:
        apgd_loss = 'last_conf'
        reduction = lambda x: - torch.log_softmax(x, dim=1)[:, -1]
        loss = attacks.LastConf()
    else:
        if score_type == 'energy':
            reduction = lambda x: - temperature* torch.logsumexp(x / temperature, dim=1)
        else:
            reduction = lambda x: torch.softmax(x, dim=1).max(dim=1)[0]

        apgd_loss = 'max_conf'
        loss = attacks.MaxConf(from_logits)
    
    attack = apgd.APGDAttack(model, n_iter=100*budget, n_iter_2=22*budget, n_iter_min=6*budget, size_decr=3,
                             norm='Linf', n_restarts=restarts, eps=epsilon, show_loss=False, seed=0,
                             loss=apgd_loss, show_acc=False, eot_iter=1, save_steps=False,
                             save_dir='./results/', thr_decr=.75, check_impr=False,
                             normalize_logits=False, device=accelerator.device, apply_softmax=from_logits, classes=num_classes)
    
    stepsize = 0.1

    
    noise = attacks.DeContraster(epsilon)
    attack1 = attacks.MonotonePGD(epsilon, iterations, stepsize, num_classes, momentum=0.9, 
                                  norm='inf', loss=loss, normalize_grad=False, early_stopping=0, restarts=0,
                                  init_noise_generator=noise, model=model, save_trajectory=False)
    
    noise = attacks.UniformNoiseGenerator(min=-epsilon, max=epsilon)
    attack2 = attacks.MonotonePGD(epsilon, iterations, stepsize, num_classes, momentum=0.9, 
                                  norm='inf', loss=loss, normalize_grad=False, early_stopping=0, restarts=3,
                                  init_noise_generator=noise, model=model, save_trajectory=False)
    
    noise = attacks.NormalNoiseGenerator(sigma=1e-4)
    attack3 = attacks.MonotonePGD(epsilon, iterations, stepsize, num_classes, momentum=0.9, 
                                  norm='inf', loss=loss, normalize_grad=False, early_stopping=0, restarts=3,
                                  init_noise_generator=noise, model=model, save_trajectory=False)
    
    attack_list = [attack1, attack2, attack3]
    
    best = []
    for batch_idx, (x, y) in enumerate(loader):
        
        out = []
        output = model(x)

        with torch.no_grad():
            score = reduction(output)
        
        out = [ score.detach().cpu()]

        attacked_point = attack.perturb(x.clone(), y)[0]
        output = model(attacked_point)

        score = reduction(output)
        
        out.append(score.detach().cpu())

        for att in attack_list:
            attacked_point = att(x.clone(), y)
            output = model(attacked_point)

            score = reduction(output)
            
            out.append(score.detach().cpu())
        
        max_conf, att_idx = torch.stack(out, dim=0).max(0)
        
        best.append(max_conf)
    best = torch.cat(best, 0).cpu().numpy()
    return best.copy()


def compute_adv_accuracy(model, loader, epsilon, version='standard', output_path='./'):

    """ Compute the certified accuracy of the model on the dataset
    
    Args:
        model: the model to evaluate
        loader: the dataset to evaluate on
        epsilon: the radius of the L-inf ball to evaluate on
        version: the version of AutoAttack to use
        
    Returns:
        the certified accuracy of the model on the dataset
        
    """
    accelerator = Accelerator(split_batches=True)
    model, loader = accelerator.prepare(model, loader)

    try: 
        model.fix_step_size(epsilon*np.sqrt(32*32*3))
    except:
        diffusion = False

    adversary = AutoAttack(
        model, norm='Linf', eps=epsilon, version=version, log_path=output_path/'attack.txt'
        )

    certified_accuracy = []

    for batch_idx, (x, y) in enumerate(loader):
        x_adv = adversary.run_standard_evaluation(x, y)
        output = model(x_adv)

        certified_accuracy.append(output.argmax(dim=1) == y)


    certified_accuracy = torch.cat(certified_accuracy, 0)

    return 100*sum(certified_accuracy)/len(certified_accuracy)


def compute_certified_accuracy(model, loader, sigma, batch_size, num_classes):

    """ Compute the certified accuracy of the model on the dataset
    
    Args:
        model: the model to evaluate
        loader: the dataset to evaluate on
        sigma: the standard deviation of the Gaussian noise
        batch_size: the batch size to use for the evaluation
        num_classes: the number of classes in the dataset
        
    Returns:
        the certified accuracy of the model on the dataset
        
    """

    accelerator = Accelerator(split_batches=True)
    model, loader = accelerator.prepare(model, loader)

    verifier = Smooth(
        model, num_classes=num_classes, sigma=sigma, device=accelerator.device)
    ranges = np.arange(0, 2.1, 0.25)
    certified_accuracy = {}
    for r in ranges:
        certified_accuracy[str(r)] = 0
    torch.manual_seed(1)

    for x, y in loader.dataset:
        
        cAHat, radius = verifier.certify(
            x, n0=100, n=10000, alpha=0.001, batch_size=batch_size)
        if cAHat == y:
            for r in ranges:
                if r <= radius:
                    certified_accuracy[str(r)] += 1

    for r in ranges:
        certified_accuracy[str(r)] /= len(loader.dataset)

    return certified_accuracy
    