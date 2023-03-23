"""
Train a model on a chosen dataset.
The model and dataset are defined in a JSON file as
"arch": {
        "type": "ModelClass",
        "args": {
            "model_args": value
        }
    }
"data_loader": {
        "type": "DataLoaderClass",
        "args": {
            "data_dir": "data/",
            "batch_size": 128,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 6
        }
    }

a configuration file example is config.json

Usage:
    $ python train.py -c config.json --args
Example:
    $ python train.py -c config.json --lr=1e-3 --wd=1e-2 --batch_size=128

Models:
    https://codeberg.org/ciodar/model-compression/src/branch/master/models
"""
import argparse
import collections

import torch


import data as module_data
import models as module_arch
from parse_config import ConfigParser
from trainer.lit_model import LitModel
from trainer.trainer import get_trainer
from utils import set_all_seeds

SEED = 42
set_all_seeds(SEED)


def main(config):
    logger = config.get_logger()

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = LitModel(config, config.init_obj('arch', module_arch))

    logger.info("Start training")
    if config.resume:
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['state_dict'])
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
    parser = ConfigParser.from_args(args, options)
    main(parser)
