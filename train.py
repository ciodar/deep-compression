"""
Train a model on a chosen dataset.

Usage:
    $ python train.py dataset model --args
Example:
    $ python train.py mnist lenet300 --lr=1e-3 --wd=1e-2 --batch_size=128 --optimizer=sgd

Models:
    https://codeberg.org/ciodar/model-compression/src/branch/master/models
"""


import time
import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_mnist_loader
from utils import plot_classes_preds, weight_histograms, set_all_seeds
from models.lenet import LeNet5,LeNet300


def training_loop(epochs, model, optimizer, device, train_loader, valid_loader, loss_fn, logging_interval=100,
                  scheduler=None, checkpoint_path=None, writer=None):
    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    best_valid_acc, best_epoch = -float('inf'), 0
    running_loss = 0.0
    for epoch in range(epochs):
        model.train()
        for i_batch, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # LOGGING
            running_loss += loss.item()
            minibatch_loss_list.append(loss.item())
            if not i_batch % logging_interval:
                print('***Epoch: %03d/%03d | Batch:%04d/%04d | Loss: %.3f' % (
                    epoch + 1, epochs, i_batch, len(train_loader), loss.item()))
                if writer:
                    # ...log the running loss
                    writer.add_scalar('Running Loss/train',
                                      running_loss / logging_interval,
                                      global_step=epoch * len(train_loader) + i_batch)
                    writer.add_figure('predictions vs. actuals',
                                      plot_classes_preds(model, inputs[:4], labels[:4],[0,1,2,3,4,5,6,7,8,9]),
                                      global_step=epoch * len(train_loader) + i_batch)
                    weight_histograms(writer,epoch*len(train_loader)+i_batch,model)
                running_loss = 0.0


        with torch.no_grad():
            train_acc, train_loss = evaluate(model, train_loader, device, loss_fn)
            print('***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f' % (
                epoch + 1, epochs, train_acc, train_loss))
            valid_acc, valid_loss = evaluate(model, valid_loader, device, loss_fn)
            print(f'Epoch: {epoch + 1:03d}/{epochs:03d} '
                  f'| Train accuracy: {train_acc :.2f}% '
                  f'| Validation accuracy: {valid_acc :.2f}% '
                  f'| Train loss: {train_loss :.3f}'
                  f'| Validation loss: {valid_loss :.3f}'
                  f'| Best Validation '
                  f'(Ep. {best_epoch:03d}): {best_valid_acc :.2f}%')
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc, best_epoch = valid_acc, epoch + 1
                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

        if scheduler is not None:
            scheduler.step()
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', valid_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', valid_acc, epoch)

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    return minibatch_loss_list, train_acc_list, valid_acc_list


def evaluate(model, data_loader, device, loss_fn, topk=(1,)):
    model.eval()
    with torch.no_grad():
        topk_correct, num_examples = torch.zeros(len(topk)), 0
        curr_loss = 0.
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # calculate topk predictions
            for i, k in enumerate(topk):
                _, predicted = torch.topk(outputs, k, 1)
                # update running loss and correct predictions count
                topk_correct[i] += torch.eq(labels[:, None].expand_as(predicted), predicted).any(dim=1).sum().item()

            # calculate loss
            loss = loss_fn(outputs, labels, reduction='sum')
            curr_loss += loss
            num_examples += labels.size(0)

    topk_acc = topk_correct / num_examples * 100
    if len(topk_acc == 1):
        topk_acc = topk_acc.item()
    else:
        topk_acc = topk_acc.numpy().tolist()

    curr_loss = curr_loss / num_examples

    return topk_acc, curr_loss.item()


class MyMultiStepLR(object):
    def __init__(self, lr_dict):
        self.lr_dict = lr_dict
        self.curr_lr = lr_dict[0]

    def get_lr(self, epoch):
        if (epoch) in self.lr_dict:
            self.curr_lr = self.lr_dict[epoch]
        return self.curr_lr

def main(args):
    # retrieve parameters
    dataset = args.dataset
    seed = args.seed
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_interval = args.log_interval
    checkpoint_path = args.save_path

    set_all_seeds(seed)
    if args.tb:
        writer = SummaryWriter('runs/%s/%s/%s' % (dataset, args.model, timestamp))
    else:
        writer = None

    num_epochs = args.epochs
    # Try to get GPU device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dataset == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_loader,test_loader = get_mnist_loader(transform,args.batch_size)
        
    elif dataset == 'cifar':
        train_loader,test_loader = get_cifar_loader(transform,args.batch_size)

    if args.model == 'lenet-300':
        model = LeNet300(10)
    elif args.model == 'lenet-5':
        model = LeNet5(10)
    model = model.to(device)

    lr = args.lr
    momentum = args.momentum if args.momentum else 0
    weight_decay = args.wd if args.wd else 0

    if args.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum, weight_decay=weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    scheduler = None
    if args.decay_step != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    loss_fn = F.cross_entropy

    training_loop(num_epochs,model,optimizer,device,train_loader,test_loader,loss_fn,log_interval,scheduler,checkpoint_path,writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    parser.add_argument('dataset', type=str, help='Select a dataset. Available options: (mnist,cifar,mini-imagenet)')
    parser.add_argument('model', type=str, help='Select a model. Available options: (lenet-300-100,lenet-5,alexnet,vgg)')
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

    main(parser.parse_args())

