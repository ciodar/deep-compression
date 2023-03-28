import argparse

import data as module_data
import torch

import models as module_arch
from parse_config import ConfigParser
from trainer.lit_model import LitModel
from trainer.trainer import get_trainer


def main(config):
    logger = config.get_logger()

    test_data_loader = config.init_obj('data_loader', module_data)

    model = LitModel(config, config.init_obj('arch', module_arch))

    logger.info("Start training")
    if config.resume:
        checkpoint = torch.load(config.resume)
        model.model.load_state_dict(checkpoint['state_dict'])
    logger.info(model)

    trainer = get_trainer(config)
    log = trainer.validate(model, test_data_loader)
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
