"""
Train a model on a chosen dataset.

Usage:
    $ python train.py dataset model --args
Example:
    $ python train.py mnist lenet300 --lr=1e-3 --wd=1e-2 --batch_size=128 --optimizer=sgd

Models:
    https://codeberg.org/ciodar/model-compression/src/branch/master/models
"""

import time, os, datetime
import argparse
import pathlib as pl
import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import optuna

from utils import plot_classes_preds, weight_histograms
from dataset import get_mnist_loader
from utils import plot_classes_preds, weight_histograms, set_all_seeds
from models.lenet import LeNet5, LeNet300

CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/checkpoints'
RUNS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/runs'


class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def training_loop(epochs, model, optimizer, device, train_loader, valid_loader, loss_fn, logging_interval=100,
                  scheduler=None, checkpoint_path=None, writer=None, opt_trial=None):
    start_time = time.time()
    best_valid_acc, best_epoch = -float('inf'), 0
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc1, train_acc5 = train_epoch(model, optimizer, device, train_loader, loss_fn,
                                                         logging_interval)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Acc@1/train', train_acc1, epoch)
            writer.add_scalar('Acc@5/train', train_acc5, epoch)

        valid_loss, valid_acc1, valid_acc5 = evaluate(model, valid_loader, device, loss_fn)

        if valid_acc1 > best_valid_acc:
            best_valid_acc, best_epoch = valid_acc1, epoch + 1
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)

        if writer:
            writer.add_scalar('Loss/test', valid_loss, epoch)

            writer.add_scalar('Acc@1/test', valid_acc1, epoch)
            writer.add_scalar('Acc@5/test', valid_acc5, epoch)
            # writer.add_figure('predictions vs. actuals',
            #                  plot_classes_preds(model, inputs[:4], labels[:4], classes),
            #                  epoch)
            weight_histograms(writer, epoch, model)

        print(f'Epoch: {epoch + 1:03d}/{epochs:03d} '
              f'| Train accuracy: {train_acc1 :.2f}% '
              f'| Validation accuracy: {valid_acc1 :.2f}% '
              f'| Train loss: {train_loss :.3f}'
              f'| Validation loss: {valid_loss :.3f}'
              f'| Best Validation '
              f'(Ep. {best_epoch:03d}): {best_valid_acc :.2f}%')

        if opt_trial:
            opt_trial.report(valid_acc1, epoch)
            # Handle pruning based on the intermediate value.
            if opt_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if scheduler is not None:
            scheduler.step()

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    return best_valid_acc


def train_epoch(model, optimizer, device, loader, loss_fn, logging_interval=100):
    start_time = time.time()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.train()
    with tqdm.tqdm(total=len(loader)) as pbar:
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # update averages with batch
            losses.update(loss.item(), inputs.size(0))

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not i_batch % logging_interval:
                pbar.set_postfix(batch=i_batch, loss=losses.avg, acc1=top1.avg, acc5=top5.avg)
                pbar.update(logging_interval)

    return losses.avg, top1.avg, top5.avg


def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, labels, (1, 5))

            # calculate loss (without reduction since it is calculated in AvgMeter)
            loss = loss_fn(outputs, labels)

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

    return losses.avg, top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(args):
    # retrieve parameters

    dataset = args.dataset
    seed = args.seed
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_interval = args.log_interval
    save_path = pl.Path(args.save_path if args.save_path else CHECKPOINT_DIR)

    if save_path.is_dir():
        name = '%s-%s-%s' % (dataset, args.model, timestamp)
    else:
        name = save_path.stem
        save_path = save_path.parent

    save_path = pl.Path.joinpath(save_path, name + '.pth')

    set_all_seeds(seed)
    if args.tb:
        writer = SummaryWriter('%s/%s/base/%s/%s' % (RUNS_DIR, dataset, args.model, timestamp))
    else:
        writer = None

    num_epochs = args.epochs
    # Try to get GPU device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dataset == 'mnist':
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if args.val_split:
            assert 0 < args.val_split < 1
            train_loader, val_loader, test_loader = get_mnist_loader(args.batch_size, val_split=args.val_split
                                                                     , resize=args.model == 'lenet-5')
        else:
            train_loader, test_loader = get_mnist_loader(args.batch_size
                                                         , resize=args.model == 'lenet-5')

    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar_loader(args.batch_size)

    if args.model == 'lenet-300':
        model = LeNet300(10)
    elif args.model == 'lenet-5':
        model = LeNet5(10)

    if writer:
        images, labels = next(iter(train_loader))
        writer.add_graph(model, images)

    model = model.to(device)

    lr = args.lr
    momentum = args.momentum if args.momentum else 0
    weight_decay = args.wd if args.wd else 0

    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if args.decay_step != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    loss_fn = F.cross_entropy

    best_valid_acc = training_loop(num_epochs, model, optimizer, device, train_loader, test_loader, loss_fn,
                                   log_interval, scheduler, save_path, writer)

    # Perform pruning

    if writer:
        writer.add_hparams(
            {'dataset': dataset, 'model': args.model, 'epochs': num_epochs, 'batch_size': args.batch_size,
             'learning rate': lr, 'momentum': momentum, 'weight decay': weight_decay,
             'optimizer': args.optimizer, 'decay_step': args.decay_step, 'gamma': args.gamma},
            {'accuracy': best_valid_acc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    parser.add_argument('dataset', type=str, help='Select a dataset. Available options: (mnist,cifar,mini-imagenet)')
    parser.add_argument('model', type=str,
                        help='Select a model. Available options: (lenet-300-100,lenet-5,alexnet,vgg)')
    parser.add_argument('--val_split', type=float, help='fraction of training set to be used for validation')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of each minibatch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--decay_step', type=int, default=0, help='Decays the learning rate every decay_step epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='Decays the learning rate by gamma')
    parser.add_argument('--log_interval', type=int, default=200, help='Log results every log_interval iterations')
    parser.add_argument('--save_path', type=str, help='Save trained model in the provided path')
    parser.add_argument('--tb', default=False, action='store_true', help='Enables TensorBoard writer')

    train(parser.parse_args())
