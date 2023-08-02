""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""


import torch
import utils, metrics
from pathlib import Path
from data_loader import prepare_data, prepare_training_data
from architectures.loader import load_predictor
import adversarial.aauc as internal_methods
import numpy as np
import argparse




def main(args):

    """ Main function to run the experiments
    
    Args:
        args (argparse.Namespace): Arguments of the experiment
        
    Returns:
        None
    
    """

    torch.manual_seed(1)
    np.random.seed(1)

    parameters = {}

    models_path, output_path, dataset_path = utils.prepare_paths(args)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    if args.experiment == 'atom': num_classes += 1

    parameters['model'] = load_predictor(args, models_path)
    id_loader, loaders = prepare_data(
        dataset_path=dataset_path, batch_size=args.batch_size, dataset=args.dataset)
    parameters['loader'] = id_loader

    print(f'ID: Clean Accuracy {utils.compute_accuracy(**parameters):.2f}')

    if args.finetune:
        parameters['model'] = fine_tuning(args, parameters, dataset_path, models_path)        

    if args.experiment == 'atom': parameters['atom'] = True

    if args.adv_robustness:
        """ Compute the robustness of the model on in distribution data """

        for eps in [2/255, 8/255]:
            robustness = internal_methods.compute_adv_accuracy(
                parameters['model'], loader=parameters['loader'], epsilon=eps,
                version='rand' if args.experiment in ['distro', 'diffusion'] else 'standard',
                output_path=output_path
                )
            print(f'\nID: Adversarial Robustness {robustness:.2f}\n')

    if args.certify_robustness:
        for sigma in [0.12, 0.25]:
            robustness = internal_methods.compute_certified_accuracy(
                parameters['model'], loader=parameters['loader'], 
                sigma=sigma, batch_size=args.batch_size, num_classes=num_classes
            )
            values = 'ID: Certified Robustness '
            for r, value in robustness.items():
                values += f'radius: {r} = {100*value:.2f}, '

            print(values)
    
    parameters['score_type'] = args.score
    id_score = metrics.compute_score(**parameters)
    del parameters['score_type']

    if args.clean:
        print(" Compute the clean AUC, AUPR, and FPR@95 ")
        parameters['temperature'] = 1
        parameters['score_type'] = args.score
        compute_auroc(
            args, parameters, loaders, id_score, output_path, num_classes, name='clean'
            )
        del parameters['temperature'], parameters['score_type']

    if args.guar:
        print(" Compute the guaranteed l-infinity norm AUC, AUPR, and FPR@95 ")
        parameters['epsilon'] = 0.01
        compute_auroc(
            args, parameters, loaders, id_score, output_path, num_classes, name='guaranteed'
            )
        del parameters['epsilon']

    if args.certify:
        print(" Compute the guaranteed l-2 norm AUC, AUPR, and FPR@95 ")
        certify_path = output_path/Path('certify')
        certify_path.mkdir(parents=True, exist_ok=True)
        compute_certify_auroc(
            args, parameters, loaders, id_loader, certify_path, num_classes)

    if args.adv:
        print(" Compute the adversarial AUC, AUPR, and FPR@95 ")
        parameters['epsilon'] = 0.01
        parameters['num_classes'] = 10
        parameters['temperature'] = 1
        parameters['score_type'] = args.score
        compute_auroc(
            args, parameters, loaders, id_score, output_path, num_classes, name='adversarial'
            )
        del parameters['temperature'], parameters['num_classes'], parameters['score_type']


def fine_tuning(args, parameters, dataset_path, models_path):

    parameters['loader'] = prepare_training_data(
        dataset_path=dataset_path, batch_size=args.batch_size
        )
    if args.experiment in ['vos', 'logit'] and args.diff:
        predictor = utils.fine_tune(**parameters)
        
        print(f'ID: Clean Accuracy after fine-tuning {utils.compute_accuracy(**parameters):.2f}')

        if args.experiment == 'vos':
            torch.save(predictor.state_dict(), 
                models_path/Path('our/CIFAR10/vos/fine_tuned_vos_cifar10.pt')
            )
        elif args.experiment == 'logit':
            torch.save(predictor.state_dict(), 
                models_path/Path('our/CIFAR10/logitnorm/fine_tuned_logit_cifar10.pt')
                )
    return predictor


def compute_certify_auroc(args, parameters, loaders, id_loader, output_path, num_classes=10):
    """ Compute the AUC, AUPR, and FPR@95 for the guaranteed scores """

    for sigma in [0.12]:
        parameters['ranges'] = np.arange(0, 2.1, 0.25)
        results, columns = utils.init_certify_results(parameters['ranges'])

        parameters['sigma'] = sigma
        parameters['loader'] = id_loader
        parameters['batch_size'] = args.batch_size
        parameters['num_classes'] = num_classes
        id_certify_scores = metrics.compute_certify_score(**parameters)
    
        for dataset_name, loader in loaders.items():
            parameters['loader'] = loader

            ood_certify_scores = metrics.compute_certify_score(**parameters)

            for r, result in results.items():
                results[r]['Dataset'].append(dataset_name)

                id_score = np.array(id_certify_scores[r])
                ood_score = np.array(ood_certify_scores[r])
                if np.count_nonzero(id_score == 0) < len(id_score) or \
                    np.count_nonzero(ood_score == 0) < len(ood_score):
                    measures = metrics.get_measures(id_score, ood_score)

                    results[r], output = utils.store_results(
                        dataset_name=dataset_name, 
                        results=result, 
                        measures=measures, 
                        cols=columns)
                    print(f'Sigma: {sigma}, Radius: {r}, ' + output)

                    if r == '0.0':
                        np.save(
                            output_path/f'{sigma}_{dataset_name}', 
                            np.array([id_score, ood_score])
                            )

        for r, result in results.items():
            path = output_path/f'{str(sigma)}_{str(r)}'
            if result['AUC']:
                utils.store_average(
                    results=result, cols=columns, output_path=path)

    del parameters['ranges'], parameters['sigma'], parameters['batch_size']


def compute_auroc(args, parameters, loaders, id_score, output_path, num_classes, name:str = 'clean'):
    """ Compute the clean AUC, AUPR, and FPR@95 """
    
    results, columns = utils.init_results()
    for dataset_name, loader in loaders.items():

        parameters['loader'] = loader
        # compute AUROC
        results['Dataset'].append(dataset_name)
        if name == 'clean':
            ood_score = metrics.compute_score(**parameters)
        elif name == 'guaranteed':
            if args.experiment in ['good']:
                parameters['num_classes'] = num_classes
                ood_score = metrics.get_conf_ibp_good(**parameters)
            elif args.experiment in ['prood', 'distro']:
                ood_score = metrics.get_conf_ibp(**parameters)
            else:
                parameters['num_classes'] = num_classes
                ood_score = metrics.get_conf_ibp_general(**parameters)

        elif name == 'adversarial':
            parameters['num_classes'] = num_classes
            ood_score = internal_methods.get_conf_lb(**parameters)
        else:
            raise NotImplementedError(f'{name} is not implemented')

        measures = metrics.get_measures(id_score, ood_score)
        results, output = utils.store_results(
            dataset_name=dataset_name, results=results, measures=measures, cols=columns)
        print(output)
    utils.store_average(results=results, cols=columns, output_path=output_path/Path(name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute the AUC, AUPR, and FPR@95 for the DDS models')
    parser.add_argument('--all', type=utils.str2bool, nargs='?', 
        const=True, default=False, help='Compute clean, guaranteed and adversarial AUC, AUFPR, FPR95')
    parser.add_argument('--clean', type=utils.str2bool, nargs='?', 
        const=True, default=False, help='Compute clean AUC, AUFPR, FPR95')
    parser.add_argument('--guar', type=utils.str2bool, nargs='?', 
        const=True, default=False, help='Compute guaranteed AUC, AUFPR, FPR95')
    parser.add_argument('--adv', type=utils.str2bool, nargs='?', 
        const=True, default=False, help='Compute adversarial AUC, AUFPR, FPR95')
    parser.add_argument('--certify', type=utils.str2bool, nargs='?', 
        const=True, default=False, help='Compute certify AUC, AUFPR, FPR95')
    parser.add_argument('--adv_robustness', type=utils.str2bool, nargs='?',
        const=True, default=False, help='Compute adversarial robustness on the ID dataset')
    parser.add_argument('--certify_robustness', type=utils.str2bool, nargs='?',
        const=True, default=False, help='Compute certified robustness on the ID dataset')
    parser.add_argument('--diff', type=utils.str2bool, nargs='?', 
        const=True, default=False, help='Add diffusion model')
    parser.add_argument('--finetune', type=utils.str2bool, nargs='?', 
        const=True, default=False, help='Fine-tune the model')
    parser.add_argument('--experiment', type=str, default='vos', 
        choices=['distro', 'vos', 'logit', 'oe', 'prood', 'good', 'plain', 'acet', 'atom', 'diffusion'], 
        help='Experiment to run')
    parser.add_argument('--dataset', type=str, default='cifar10',
        choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--score', type=str, default='softmax', 
        choices=['softmax', 'energy', 'logit'], help='Score to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--standardized', type=bool, default=False, help='Do standardized tests (with all models having similar model and normalization)')


    args = parser.parse_args()
    print(args)
    main(args)
