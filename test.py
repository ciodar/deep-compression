import argparse

import data as module_data
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn

import models as module_arch
from parse_config import ConfigParser
from trainer.lit_model import LitModel
from trainer.trainer import get_trainer
import torch.nn.utils.prune as prune

from utils import load_compressed_checkpoint

_MODULE_CONTAINERS = (LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)


def main(config):
    logger = config.get_logger()

    # Override settings
    config['data_loader']['args']['training'] = False
    config['data_loader']['args']['validation_split'] = 0.0
    config['data_loader']['args']['shuffle'] = False

    test_data_loader = config.init_obj('data_loader', module_data)
    print(len(test_data_loader))

    model = LitModel(config, config.init_obj('arch', module_arch))

    if config.resume:
        checkpoint = torch.load(config.resume)
        model = load_compressed_checkpoint(model, checkpoint)
    logger.info(model)

    trainer = Trainer(logger=None, accelerator="gpu", deterministic=True, enable_progress_bar=False,
                      enable_model_summary=False, enable_checkpointing=False)
    log = trainer.test(model, test_data_loader)
    print(log)


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
