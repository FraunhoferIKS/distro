""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

from accelerate import Accelerator
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy
from denoiser_diffusion import Smooth
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


to_np = lambda x: x.data.cpu().numpy()

def get_measures(_pos, _neg, recall_level=0.95):
    """ Compute the AUROC, AUPR, and FPR at a given recall level. 
    
    Args:
        _pos: list of positive scores
        _neg: list of negative scores
        recall_level: recall level at which to compute the FPR
    Returns:
        auroc: AUROC
        aupr: AUPR
        fpr: FPR at the given recall level    
    """

    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    """ Compute the FPR and FDR at a given recall level.

    Args:
        y_true: true labels
        y_score: scores
        recall_level: recall level at which to compute the FPR
        pos_label: label of the positive class
    Returns:
        fpr: FPR at the given recall level
        fdr: FDR at the given recall level
    """
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def compute_score(model, loader, temperature=1.0, score_type='softmax', atom:bool=False):
    """ Compute the confidence scores of the given model on the given dataset.
    
    Args:
        model: the model to evaluate
        loader: the dataset loader
        temperature: temperature for the softmax
        score_type: type of score to compute
        atom: whether to compute the atom score
    Returns:
        scores: the scores of the model on the dataset
    """
    
    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)
    model = model.eval()

    predictions = torch.tensor([], device=accelerator.device)
    with torch.no_grad():
        for x, _ in loader:

            output = model(x)
            if atom:
                score = - torch.log_softmax(output, dim=1)[:, -1]
            elif score_type == 'energy':
                score = - temperature * torch.logsumexp(output / temperature, dim=1)
            elif score_type == 'softmax':
                score, _ = torch.nn.functional.softmax(output/temperature, dim=1).max(dim=1)
            else:
                raise ValueError('Unknown score type')

            predictions = torch.cat((predictions, score), 0)

    return np.array(to_np(predictions).copy())


def compute_certify_score(
    model, 
    loader, 
    num_classes, 
    sigma = 0.1, 
    temperature=1.0, 
    score_type='softmax', 
    ranges=[0.1],  
    batch_size = 1, 
    atom:bool=False):

    """
        Compute the certify score for a given model and loader

        Args:
            model: the model to evaluate
            loader: the dataset loader
            num_classes: number of classes
            sigma: sigma for the smoothness
            temperature: temperature for the softmax
            score_type: type of score to compute
            ranges: ranges for the certify score
            batch_size: batch size for the certify score
            atom: whether to compute the atom score
        
        Returns:
            certified_scores: the certify scores of the model on the dataset
    """

    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)
    model = model.eval()
    
    verifier = Smooth(
        model, num_classes=num_classes, sigma=sigma, device=accelerator.device)

    certified_scores = {}
    for r in ranges:
        certified_scores[str(r)] = []

    with torch.no_grad():
        for idx, (x, _) in enumerate(loader.dataset):

            cAHat, radius = verifier.certify(
                x, n0=100, n=10000, alpha=0.01, batch_size=batch_size)
            output = model(x.to(accelerator.device).unsqueeze(0))

            if cAHat == -1:
                certified_scores[str(ranges[0])].append(0)
            else:
                if atom:
                    score = - torch.log_softmax(output, dim=1)[:, -1]
                elif score_type == 'energy':
                    score = - temperature * torch.logsumexp(output / temperature, dim=1)
                elif score_type == 'softmax':
                    score = torch.nn.functional.softmax(output / temperature, dim=1)
                    score = score[0][cAHat]
                else:
                    raise ValueError('Unknown score type')
                
                score = np.sqrt(2/np.pi)*radius + score

                for r in ranges:
                    if r <= radius:
                        certified_scores[str(r)].append(to_np(score))
                    else:
                        certified_scores[str(r)].append(0)

            if idx >= 999: break

    return certified_scores.copy()

def get_conf_ibp_general(model, loader, epsilon=0.0, num_classes=10, atom=False):
    """
        Compute the guaranteed confidence of the model using IBP
    
    Args:
    
    model: the model to evaluate
    loader: the data loader
    epsilon: the epsilon value for the IBP
    
    Returns:
    
    conf: the confidence of the model
    """

    accelerator = Accelerator()
    out = []
    model, loader = accelerator.prepare(model, loader)
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)

    with torch.no_grad():
        for x, _ in loader:
            model = BoundedModule(model, x)
            input_bounds = BoundedTensor(x, ptb)
            _, ub = model.compute_bounds(x=(input_bounds,), method="IBP")
            ub_score = torch.softmax(ub, dim=1).max(dim=1)[0]

            if not torch.all(ub_score.eq(1.0)):
                print(ub_score)

            out.append(ub_score.detach().cpu())

    return torch.cat(out, dim=0).exp().numpy()


def get_conf_ibp(model, loader, epsilon=0.0):

    """ Compute the guaranteed confidence of the model using IBP 
    
    Args:
    
    model: the model to evaluate
    loader: the data loader
    epsilon: the epsilon value for the IBP
    
    Returns:
    
    conf: the confidence of the model
    """

    accelerator = Accelerator()
    out = []
    model, loader = accelerator.prepare(model, loader)
    with torch.no_grad():
        for x, _ in loader:
            l, u = torch.clamp(x-epsilon, 0, 1), torch.clamp(x+epsilon, 0, 1)
            out.append(model.ibp_forward(l, u)[1].detach().cpu())

    return torch.cat(out, dim=0).exp().numpy()


def get_conf_ibp_good(model, loader, num_classes, epsilon=0.0):

    """ Compute the guaranteed confidence of the model using IBP

    Args:

    model: the model to evaluate
    loader: the data loader
    epsilon: the epsilon value for the IBP

    Returns:

    conf: the confidence of the model
    """

    accelerator = Accelerator()
    ud_logit_out = []
    model, loader = accelerator.prepare(model, loader)
    for x, _ in loader:
        _, _, ud_logit_out_batch = model.ibp_elision_forward(
            torch.clamp(x - epsilon, 0, 1), torch.clamp(x + epsilon, 0, 1), num_classes) 
        ud_logit_out_batch = ud_logit_out_batch.detach().cpu().numpy()
        ud_logit_out.append(ud_logit_out_batch)
        
    ud_logit_out = np.concatenate(ud_logit_out, axis=0)
    
    ub_el_out_log_confidences = ub_log_confs_from_ud_logits(ud_logit_out, force_diag_0=False)
    return np.exp(np.nan_to_num(ub_el_out_log_confidences))


def ub_log_confs_from_ud_logits(ud_logits, force_diag_0=False): #upper bound differences matrix
    if force_diag_0: #with elision, this is already given
        for i in range(ud_logits.shape[-1]): 
            ud_logits[:, i, i] = 0
    ub_log_probs = -scipy.special.logsumexp(-ud_logits, axis=-1)
    ub_log_confs = np.amax(ub_log_probs, axis=-1)
    return ub_log_confs

