import collections
import os
import argparse

import optuna
import torch
import torch.nn.functional as F
from optuna.trial import TrialState

from parse_config import ConfigParser
from trainer.trainer import Trainer

import data as module_data
import models as module_arch
import evaluation as module_metric
from utils import set_all_seeds, MetricTracker

SEED = 42
EPOCHS = 20
set_all_seeds(SEED)


def objective(trial):
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = getattr(F, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd = trial.suggest_float("wd", 1e-5, 1e-1, log=True)
    m = trial.suggest_float("m", 0, 1)

    scheduler = trial.suggest_categorical("scheduler", [None, "StepLR"])

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, config['optimizer']['type'])(model.parameters(), lr=lr, weight_decay=wd,
                                                                  momentum=m)
    lr_scheduler = None
    if scheduler is not None:
        sz = trial.suggest_int("sz", 1, EPOCHS)
        g = trial.suggest_float("g", 1e-3, 9e-1, log=True)

        lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler)(optimizer, step_size=sz, gamma=g)

    train_metrics = MetricTracker('loss', *[m.__name__ for m in metrics], writer=None)
    valid_metrics = MetricTracker('loss', *[m.__name__ for m in metrics], writer=None)

    for epoch in range(EPOCHS):
        model.train()
        train_metrics.reset()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_metrics.update('loss', loss.item())
            for met in metrics:
                train_metrics.update(met.__name__, met(output, target))

        # Validation of the model.
        model.eval()
        valid_metrics.reset()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_data_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                for met in metrics:
                    valid_metrics.update(met.__name__, met(output, target))

        accuracy = valid_metrics.result()['accuracy']

        if lr_scheduler is not None:
            lr_scheduler.step()

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def main(config):
    logger = config.get_logger('hoa')

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info("  Number of finished trials: {:d}".format(len(study.trials)))
    logger.info("  Number of pruned trials: {:d}".format(len(pruned_trials)))
    logger.info("  Number of complete trials: {:d}".format(len(complete_trials)))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))


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
