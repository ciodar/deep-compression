import time, os
import argparse
import datetime
import pathlib as pl

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_mnist_loader
from utils import plot_classes_preds, weight_histograms, set_all_seeds
from models.lenet import LeNet5, LeNet300
from pruning import threshold_prune
from train import training_loop

CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/checkpoints'
RUNS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/runs'


def compression(args):
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

    save_path = pl.Path.joinpath(save_path, name+'.pth')

    set_all_seeds(seed)
    if args.tb:
        writer = SummaryWriter('%s/%s/%s/%s' % (RUNS_DIR, dataset, args.model, timestamp))
    else:
        writer = None

    num_epochs = args.epochs
    # Try to get GPU device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dataset == 'mnist':
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if args.val_split:
            assert 0 < args.val_split < 1
            train_loader, val_loader, test_loader = get_mnist_loader(args.batch_size, val_fraction=args.val_split
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

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if args.decay_step != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    loss_fn = F.cross_entropy

    best_valid_acc, _ = training_loop(num_epochs, model, optimizer, device, train_loader, test_loader, loss_fn,
                                      log_interval, scheduler, save_path, writer)

    # Perform pruning

    if writer:
        writer.add_hparams(
            {'dataset': dataset, 'model': args.model, 'epochs': num_epochs, 'batch_size': args.batch_size,
             'learning rate': lr, 'momentum': momentum, 'weight decay': weight_decay,
             'optimizer': args.optimizer, 'decay_step': args.decay_step, 'gamma': args.gamma},
            {'accuracy': best_valid_acc})

    threshold_prune(model, args.s)
    save_path = pl.Path.joinpath(save_path.parent, name+'-pruned'+'.pth')
    torch.save(model.state_dict(), save_path)

    # lr = 0.1*lr
    training_loop(num_epochs, model, optimizer, device, train_loader, test_loader, loss_fn, log_interval, scheduler,
                  save_path, writer)
    save_path = pl.Path.joinpath(save_path.parent, name + '-finetuned'+'.pth')
    torch.save(model.state_dict(), save_path)


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
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--decay_step', type=int, default=0, help='Decays the learning rate every decay_step epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='Decays the learning rate by gamma')
    parser.add_argument('--log_interval', type=int, default=200, help='Log results every log_interval iterations')
    parser.add_argument('--save_path', type=str, help='Save trained model in the provided path')
    parser.add_argument('--tb', default=False, action='store_true', help='Enables TensorBoard writer')
    parser.add_argument('--s', default=1, type=int, help='Pruning threshold multiplier')

    compression(parser.parse_args())
