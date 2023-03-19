import argparse
import torch
import torch.nn.functional as F

import data as module_data
import models as module_arch
import trainer.metrics as module_metric

from parse_config import ConfigParser
from pruning import compression_stats
from trainer.trainer import get_trainer
from utils import set_all_seeds

SEED = 42
set_all_seeds(SEED)

def main(config):

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    model = config.init_obj('arch', module_arch, config)
    # logger.info(model)

    compression_stats(model)

    # trainer = get_trainer(config)
    # trainer.fit(model, data_loader, valid_data_loader)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
