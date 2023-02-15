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

import torch
import torch.nn.functional as F

from parse_config import ConfigParser
from trainer.trainer import Trainer

from utils import set_all_seeds

import data as module_data
import models as module_arch
import evaluation as module_metric

CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/checkpoints'
RUNS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/runs'
SEED = 42
set_all_seeds(SEED)


def train(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # get function handles of loss and metrics
    criterion = getattr(F, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    #
    # # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    #
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=lambda prog:
                                   argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    args.add_argument('-c', '--config', default='configs/config.json', type=str,
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
    train(config)
