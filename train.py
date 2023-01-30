import time
import torch
from torch.utils.tensorboard import SummaryWriter



def training_loop(epochs, model, optimizer, device, train_loader, valid_loader, loss_fn, logging_interval=100,
                  scheduler=None, best_model_save_path=None, tb_writer=None):
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
                if tb_writer:
                    # ...log the running loss
                    tb_writer.add_scalar('training loss',
                                         running_loss / logging_interval,
                                         epoch * len(train_loader) + i_batch)
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

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    return minibatch_loss_list, train_acc_list, valid_acc_list


def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        curr_loss = 0.
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # calculate loss
            loss = loss_fn(outputs, labels, reduction='sum')
            # calculate predictions
            _, predicted = torch.max(outputs, 1)
            # update running loss and correct predictions count
            num_examples += labels.size(0)
            curr_loss += loss
            correct_pred += (predicted == labels).sum()

    curr_acc = correct_pred.float() / num_examples * 100
    curr_loss = curr_loss / num_examples

    return curr_acc.item(), curr_loss.item()

class MyMultiStepLR(object):
    def __init__(self, lr_dict):
        self.lr_dict = lr_dict
        self.curr_lr = lr_dict[0]

    def get_lr(self, epoch):
        if (epoch) in self.lr_dict:
            self.curr_lr = self.lr_dict[epoch]
        return self.curr_lr