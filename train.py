import time
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import plot_classes_preds

def training_loop(epochs, model, optimizer, device, train_loader, valid_loader, loss_fn, logging_interval=100,
                  scheduler=None, best_model_save_path=None, writer=None):
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
                running_loss = 0.0
                # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
                # tb_writer.add_figure('predictions vs. actuals',
                #                   plot_classes_preds(model, inputs, labels),
                #                   global_step=epoch * len(train_loader) + i_batch)

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
                if best_model_save_path:
                    torch.save(model.state_dict(), best_model_save_path)

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
