import argparse
import collections
import csv
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch import nn

from parse_config import ConfigParser
from trainer.lit_model import LitModel
from trainer.trainer import get_trainer
from utils import set_all_seeds,set_deterministic

import data as module_data
import models as module_arch
from trainer import metrics as module_metric

import pruning as module_prune
import quantization as module_quantize

CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/checkpoints'
RUNS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/runs'
SEED = 42
set_all_seeds(42)
set_deterministic()

_MODULE_CONTAINERS = (LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)


def pruning_sensitivity_analysis(config, pruner, sparsities, name="weight"):
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

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = LitModel(config, config.init_obj('arch', module_arch))
    if config.resume:
        checkpoint = torch.load(config.resume)
        model.model.load_state_dict(checkpoint['state_dict'])

    current_modules = [m_name for m_name, m in model.model.named_modules() if
                       not isinstance(m, _MODULE_CONTAINERS) and hasattr(m, name)]
    trainer = Trainer(default_root_dir=config.save_dir, accelerator="gpu", deterministic=True)

    for m_name in current_modules:

        sensitivity = OrderedDict()

        for l in sparsities:
            model_cpy = deepcopy(model)

            module = getattr(model_cpy.model, m_name)

            sparsity = float(l)
            # Create the pruner (a level pruner), the pruning policy and the
            # pruning schedule.

            pruner.apply(module, name, sparsity)

            log = trainer.validate(model_cpy, valid_data_loader)
            sensitivity[sparsity] = log[0]
        sensitivities[m_name] = sensitivity
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
            logger.info("Testing sensitivity to {} [{:d} bits]".format(test_fn.__name__, b))
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
    sparsities = 1 - np.logspace(-2, 0, 20)
    pruner = module_prune.L1Unstructured

    sensitivities = pruning_sensitivity_analysis(config, pruner=pruner, sparsities=sparsities)
    fig = plot_sensitivities(sensitivities)
    fig.savefig(config.save_dir / f"{config['name'].lower()}_sensitivity_analysis.png")

    sensitivities_to_csv(sensitivities, config.save_dir / f"{config['name'].lower()}sensitivity_analysis.csv")


def plot_sensitivities(sensitivities, metric='val_accuracy'):
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
    fig = plt.figure()
    for param_name, sensitivity in sensitivities.items():
        sense = [values[metric] for sparsity, values in sensitivity.items()]
        sparsities = [sparsity for sparsity, values in sensitivity.items()]
        plt.plot(sparsities, sense, label=param_name)
    plt.ylabel(metric)
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
        writer.writerow(['parameter', 'sparsity', 'loss', 'top1', 'top5'])
        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values.values()))


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
