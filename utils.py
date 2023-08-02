""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

from accelerate import Accelerator
import torch
import pandas as pd
import argparse
import numpy as np
from transformers import get_scheduler
from tqdm.auto import tqdm
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def prepare_paths(args):
    """ Set the paths for the models and the datasets """

    models_path = Path(os.environ.get('MODELS_PATH')).resolve()
    output_path = Path(os.environ.get('OUTPUT_PATH')).resolve()

    assert output_path.exists(), f'Output path {output_path} does not exist'

    if args.all: args.clean = args.guar = args.adv = True
    
    output_path = output_path/Path(args.dataset+'/')/Path(args.experiment+'_'+args.score)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    dataset_path = Path(os.environ.get('DATASETS_PATH')).resolve()

    return models_path, output_path, dataset_path

to_prc = lambda x: np.round(x*100, 2)

def init_results():
    results = {'Dataset':[], 'AUC': [], 'AUPR': [], 'FPR': []}
    cols = ['AUC', 'AUPR', 'FPR']
    return results, cols

def init_certify_results(ranges):
    """ Initialize the results for the certification """
    results = {}
    for r in ranges:
        result, cols = init_results()
        results[str(r)] = result

    return results, cols

def store_results(dataset_name:str, results:dict, measures:list, cols: list):
    output = dataset_name
    for idx, metric in enumerate(cols):
        results[metric].append(to_prc(measures[idx]))
        output += f', {metric}: {results[metric][-1]:.2f}'
        
    return results, output

def store_average(results: dict, cols: list, output_path:str):
    output = 'Average'
    for metric in cols:
        results[metric].append(np.round(np.mean(results[metric]), 2))
        output += f', {metric}: {results[metric][-1]:.2f}'
    print(output)
    results['Dataset'].append('Average')

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f'{output_path}.csv', index=False)


def compute_accuracy(model, loader):

    """ Compute the accuracy of a model on a given dataset. 
    
    Args:
        model (nn.Module): the model to evaluate
        loader (DataLoader): the dataset to evaluate on 
    Returns:
        float: the accuracy of the model on the dataset
    
    """
    accelerator = Accelerator(split_batches=True)
    model, loader = accelerator.prepare(model, loader)
    model = model.eval()

    num_clean, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            predictions = model(x).argmax(dim=1)

            num_clean += (predictions == y).sum()
            num_samples += y.size(0)

    return to_prc(float(num_clean)/float(num_samples))


def fine_tune(model, loader, normalize=None, diffusion=None, dds_model=None):

    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)
    model.train()
    num_epochs = 3
    num_training_steps = num_epochs * len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for x, y in loader:
            optimizer.zero_grad()
            with torch.no_grad():
                if diffusion is not None:
                    # Define the time steps (One-shot denoising)
                    t = torch.ones(x.shape[0], dtype=torch.long, device=accelerator.device)

                    diffused = diffusion.p_sample(dds_model, x, t=t)
                    x = diffused['sample'].clamp(0, 1)
    
            if normalize is not None:
                output = model(normalize(x))
            else:
                output = model(x)
            
            loss = criterion(output, y)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    return model