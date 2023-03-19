"""
Train a model on a chosen dataset.

Usage:
    $ python train.py dataset model --args
Example:
    $ python train.py mnist lenet300 --lr=1e-3 --wd=1e-2 --batch_size=128 --optimizer=sgd

Models:
    https://codeberg.org/ciodar/model-compression/src/branch/master/models
"""
import collections
import os
import argparse

from parse_config import ConfigParser
from trainer.lit_model import LitModel
from trainer.trainer import get_trainer

from utils import set_all_seeds

import data as module_data
import models as module_arch

SEED = 42
set_all_seeds(SEED)


def main(config):
    logger = config.get_logger('lightning')

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = LitModel(config, config.init_obj('arch', module_arch))
    logger.info(model)

    trainer = get_trainer(config)
    trainer.fit(model, data_loader, valid_data_loader)


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
