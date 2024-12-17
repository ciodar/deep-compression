import argparse
import collections
import csv
import logging
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pytorch_lightning
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn.utils import prune

from compression import get_pruned
from parse_config import ConfigParser
from trainer.lit_model import LitModel
from trainer.trainer import get_trainer
from utils import set_all_seeds, set_deterministic, load_compressed_checkpoint

import data as module_data
import models as module_arch
import compression as module_compression

CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/checkpoints'
RUNS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/runs'
SEED = 42
set_all_seeds(42)
# set_deterministic()

_MODULE_CONTAINERS = (LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)

logger = logging.getLogger(__name__)


def sensitivity_analysis(config, fn, amounts, name="weight"):
    """Perform a sensitivity test for a model's weights parameters.
    The model should be trained to maximum accuracy, because we aim to understand
    the behavior of the model's performance in relation to pruning of a specific
    weights tensor.
    By default this function will test all of the model's parameters.
    The return value is a sensitivities dictionary: the dictionary's
    key is the name (string) of the weights tensor.  The value is another dictionary,
    where the tested sparsity-level is the key, and a (loss, top1, top5) tuple
    is the value.
    Below is an example of such a dictionary:
    .. code-block:: python
    {'model.fc1.weight':    {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.492, 79.1,   1.9161),
                                     0.10: (56.212, 78.854, 1.9315),
                                     0.15: (35.424, 60.3,   3.0866)},
     'model.fc2.weight':  {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.514, 79.07,  1.9159),
                                     0.10: (56.434, 79.074, 1.9138),
                                     0.15: (54.454, 77.854, 2.3127)} }
    """

    sensitivities = OrderedDict()

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = LitModel(config, config.init_obj('arch', module_arch))
    if config.resume:
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['state_dict'])

    current_modules = [m_name for m_name, m in model.model.named_modules() if
                       not isinstance(m, _MODULE_CONTAINERS) and hasattr(m, name)]

    trainer = Trainer(default_root_dir=config.save_dir, accelerator="gpu", deterministic=True)

    for m_name in current_modules:

        sensitivity = OrderedDict()

        for amount in amounts:
            model_cpy = deepcopy(model)

            module = getattr(model_cpy.model, m_name)

            # Create the pruner (a level pruner), the pruning policy and the
            # pruning schedule.

            fn.apply(module, name, amount)

            log = trainer.test(model_cpy, valid_data_loader)
            sensitivity[amount] = log[0]
        sensitivities[m_name] = sensitivity
    return sensitivities


def pruning_sensitivity_analysis(config, amounts, train=True, params=["weight"]):
    sensitivity = OrderedDict()

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = LitModel(config, config.init_obj('arch', module_arch))
    if config.resume:
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['state_dict'])

    for amount in amounts:
        trainer = get_trainer(config)
        model_cpy = deepcopy(model)
        current_modules = [m for m in model_cpy.model.modules() if not isinstance(m, _MODULE_CONTAINERS)]
        parameters_to_prune = [(m, p) for p in params for m in current_modules if hasattr(m, p)]

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        if train:
            trainer.fit(model_cpy, data_loader, valid_data_loader)

        log = trainer.test(model_cpy, valid_data_loader)
        sensitivity[amount] = log[0]
    return sensitivity


def quantization_sensitivity_analysis(config, amounts, train=True, params=["weight"]):
    sensitivities = OrderedDict()

    fns = ['linear_quantization', 'forgy_quantization', 'density_quantization']

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = LitModel(config, config.init_obj('arch', module_arch))
    if config.resume:
        checkpoint = torch.load(config.resume)
        load_compressed_checkpoint(model, checkpoint)
    trainer = Trainer(accelerator="gpu", deterministic=True)
    for fn in fns:
        sensitivity = OrderedDict()
        quantization_fn = getattr(module_compression.quantization, fn)
        for amount in amounts:

            model_cpy = deepcopy(model)
            current_modules = [m for m in model_cpy.model.modules() if not isinstance(m, _MODULE_CONTAINERS) and isinstance(m, nn.Conv2d)]
            parameters_to_quantize = [(m, p) for p in params for m in current_modules if hasattr(m, p)]

            for module, name in parameters_to_quantize:
                print("Quantizing {} into {:d} bits...".format(module, amount))
                quantization_fn(module, name=name, bits=amount)

            if train:
                trainer.fit(model_cpy, data_loader, valid_data_loader)

            log = trainer.test(model_cpy, valid_data_loader)
            sensitivity[amount] = log[0]
        sensitivities[fn] = sensitivity
    return sensitivities


def main(config):
    # sparsities = 1 - np.logspace(-2, 0, 10)
    # experiment = 'global pruning w/o retrain'
    # sensitivity = pruning_sensitivity_analysis(config, train=True, amounts=sparsities)

    sensitivities = quantization_sensitivity_analysis(config,train=False,amounts=list(range(1,9)))

    # fig.savefig(config.save_dir / f"{config['name'].lower()}_sensitivity_analysis.png")
    # sensitivities = {experiment: sensitivity}
    fname = config.save_dir / f"{config['name'].lower()}_sensitivity_analysis.csv"

    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write the header
        writer.writerow(['experiment', 'sparsity', 'loss', 'top1', 'top5'])
        for experiment, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([experiment] + [sparsity] + list(values.values()))

    # sensitivities_to_csv(sensitivities, config.save_dir / f"{config['name'].lower()}_sensitivity_analysis.csv")


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
    for param_name, sensitivity in sorted(sensitivities.items()):
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
