import argparse
import collections
import csv
import os
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from logger import logger
from parse_config import ConfigParser
from trainer.trainer import Trainer
from utils import set_all_seeds

import data as module_data
import models as module_arch
import evaluation as module_metric

import pruning as module_prune
import quantization as module_quantize
from pruning import prune_model
from quantization import quantize_model

CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/checkpoints'
RUNS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/runs'
SEED = 42
set_all_seeds(SEED)


def pruning_sensitivity_analysis(trainer, sparsities, test_fn, train=True, logger=None):
    """Perform a sensitivity test for a model's weights parameters.
    The model should be trained to maximum accuracy, because we aim to understand
    the behavior of the model's performance in relation to pruning of a specific
    weights tensor.
    By default this function will test all of the model's parameters.
    The return value is a complex sensitivities dictionary: the dictionary's
    key is the name (string) of the weights tensor.  The value is another dictionary,
    where the tested sparsity-level is the key, and a (top1, top5, loss) tuple
    is the value.
    Below is an example of such a dictionary:
    .. code-block:: python
    {'features.module.6.weight':    {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.492, 79.1,   1.9161),
                                     0.10: (56.212, 78.854, 1.9315),
                                     0.15: (35.424, 60.3,   3.0866)},
     'classifier.module.1.weight':  {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.514, 79.07,  1.9159),
                                     0.10: (56.434, 79.074, 1.9138),
                                     0.15: (54.454, 77.854, 2.3127)} }
    The test_func is expected to execute the model on a test/validation dataset,
    and return the results for top1 and top5 accuracies, and the loss value.
    """

    sensitivities = OrderedDict()
    model = trainer.model
    for pruner in config['pruners']:
        for param_name in pruner['levels'].keys():
            if model.state_dict()[param_name].dim() not in [2, 4]:
                continue

            # Make a copy of the model, because when we apply the zeros mask (i.e.
            # perform pruning), the model's weights are altered
            model_cpy = deepcopy(model)
            trainable_params = filter(lambda p: p.requires_grad, model_cpy.parameters())
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

            trainer.model = model_cpy
            trainer.optimizer = optimizer
            trainer.lr_scheduler = lr_scheduler

            sensitivity = OrderedDict()
            for l in sparsities:
                sparsity = float(l)
                if logger is not None:
                    logger.info("Testing sensitivity of %s to %s [%0.1f%% sparsity]" % (
                        param_name, test_fn.__name__, sparsity * 100))
                # Create the pruner (a level pruner), the pruning policy and the
                # pruning schedule.

                # Element-wise sparsity
                levels = {param_name: sparsity}

                fn = getattr(module_prune, pruner["type"])
                iterations = 5
                for it in range(iterations):
                    test_fn(model_cpy, fn, levels, logger)

                    # Test and record the performance of the pruned model
                    if train:
                        trainer.train()
                _, acc1, acc5 = trainer._valid_epoch(-1).values()
                if logger is not None:
                    logger.info(
                        "Tested sensitivity of {} [ sparsity amount:{:.1f} ] | acc@1:{:.2f} | acc@5:{:.2f}"
                        .format(param_name,
                                sparsity,
                                acc1, acc5))
                sensitivity[sparsity] = [acc1, acc5]
            sensitivities[param_name] = sensitivity
    return sensitivities


def quantization_sensitivity_analysis(trainer, bits, test_fn, train=True, logger=None):
    """Perform a sensitivity test for a model's weights parameters.
    The model should be trained to maximum accuracy, because we aim to understand
    the behavior of the model's performance in relation to pruning of a specific
    weights tensor.
    By default this function will test all of the model's parameters.
    The return value is a complex sensitivities dictionary: the dictionary's
    key is the name (string) of the weights tensor.  The value is another dictionary,
    where the tested sparsity-level is the key, and a (top1, top5, loss) tuple
    is the value.
    Below is an example of such a dictionary:
    .. code-block:: python
    {'features.module.6.weight':    {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.492, 79.1,   1.9161),
                                     0.10: (56.212, 78.854, 1.9315),
                                     0.15: (35.424, 60.3,   3.0866)},
     'classifier.module.1.weight':  {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.514, 79.07,  1.9159),
                                     0.10: (56.434, 79.074, 1.9138),
                                     0.15: (54.454, 77.854, 2.3127)} }
    The test_func is expected to execute the model on a test/validation dataset,
    and return the results for top1 and top5 accuracies, and the loss value.
    """

    sensitivities = OrderedDict()
    model = trainer.model
    quantizer = config['quantizer']
    sensitivity = OrderedDict()

    for b in range(bits, 0, -1):

        # Make a copy of the model, because when we apply the zeros mask (i.e.
        # perform pruning), the model's weights are altered
        model_cpy = deepcopy(model)
        trainable_params = filter(lambda p: p.requires_grad, model_cpy.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer.model = model_cpy
        trainer.optimizer = optimizer
        trainer.lr_scheduler = lr_scheduler

        if logger is not None:
            logger.info("Testing sensitivity to {} [{:d} bits]".format( test_fn.__name__, b))
        # Create the pruner (a level pruner), the pruning policy and the
        # pruning schedule.

        levels = quantizer['levels']

        # Element-wise sparsity
        levels = {p: b for p in levels.keys()}

        fn = getattr(module_quantize, quantizer["type"])

        test_fn(model_cpy, fn, levels, logger)

        # Test and record the performance of the pruned model
        if train:
            trainer.train()
        _, acc1, acc5 = trainer._valid_epoch(-1).values()
        if logger is not None:
            logger.info(
                "Tested sensitivity [{:d} bits] | acc@1:{:.4f} | acc@5:{:.4f}"
                .format(b,
                        acc1, acc5))
        sensitivity[b] = [acc1, acc5]
    return sensitivity


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    model.load_state_dict(torch.load(config.resume)['state_dict'])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # get function handles of loss and metrics
    criterion = getattr(F, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # avoid model reinitialization inside trainer
    config.resume = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    bits = 8

    sensitivities = quantization_sensitivity_analysis(trainer, bits, quantize_model, train=True, logger=logger)
    fig = plot_sensitivities(sensitivities)
    fig.savefig('mnist_sensitivity_analysis_retrain.png')

    sensitivities_to_csv(sensitivities, 'mnist_sensitivity_analysis_retrain.csv')


def plot_sensitivities(sensitivities):
    """Create a mulitplot of the sensitivities.
    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    try:
        # sudo apt-get install python3-tk
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Function plot_sensitivity requires package matplotlib which"
              "is not installed in your execution environment.\n"
              "Skipping the PNG file generation")
        return

    for param_name, sensitivity in sensitivities.items():
        sense = [values[0] for sparsity, values in sensitivity.items()]
        sparsities = [sparsity for sparsity, values in sensitivity.items()]
        plt.plot(sparsities, sense, label=param_name)
    fig = plt.figure()
    plt.ylabel('top5')
    plt.xlabel('sparsity')
    plt.title('Pruning Sensitivity')
    plt.grid()
    plt.legend(loc='lower center',
               ncol=2, mode="expand", borderaxespad=0.)
    return fig


def sensitivities_to_csv(sensitivities, fname):
    """Create a CSV file listing from the sensitivities dictionary.
    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write the header
        writer.writerow(['parameter', 'sparsity', 'top1', 'top5', 'loss'])
        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=lambda prog:
                                   argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--wd', '--weight_decay'], type=float, target='optimizer;args;weight_decay'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
