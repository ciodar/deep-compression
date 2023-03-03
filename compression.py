import argparse
import torch
import torch.nn.functional as F

import data as module_data
import models as module_arch
import evaluation as module_metric
import quantization as module_quantize

import torch.nn.utils.prune as prune
from pruning import prune_model

from parse_config import ConfigParser

from trainer.trainer import Trainer


def main(config):
    logger = config.get_logger('compression')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(F, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    logger.info('Accuracy before compression: {:.2f}'.format(checkpoint['monitor_best']))
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    pruners = config['pruners']
    for pruner in pruners:
        prune_fn = getattr(prune, pruner['type'])

        model = prune_model(model, prune_fn, pruner['levels'], logger)

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        config.resume = None

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

        if pruner['finetune_weights']:
            trainer.train()

        _, acc1, acc5 = trainer._valid_epoch(-1).values()

        if logger is not None:
            logger.info(
                "Tested model after pruning - acc@1:{:.2f} | acc@5:{:.2f}".format(acc1, acc5))

    quantizer = config['quantizer']
    quantize_fn = getattr(module_quantize, quantizer['type'])

    model = module_quantize.quantize_model(model, quantize_fn, quantizer['levels'], logger)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    if quantizer['finetune_weights']:
        trainer.train()

    _, acc1, acc5 = trainer._valid_epoch(-1).values()

    if logger is not None:
        logger.info(
            "Tested model after quantization - acc@1:{:.2f} | acc@5:{:.2f}".format(acc1, acc5))


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
