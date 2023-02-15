import json
import pandas as pd
import os, random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import prune

import quantization as quantize


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# functions to show an image
def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_sparsity_matrix(model):
    # fig = plt.figure()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.detach().cpu()
            plt.spy(weights, color='blue', markersize=1)
            plt.title(name)
            plt.show()
        # if isinstance(param, torch.nn.Conv2d):
        elif isinstance(module, torch.nn.Conv2d):
            weights = module.weight.detach().cpu()
            num_kernels = weights.shape[0]
            for k in range(num_kernels):
                kernel_weights = weights[k].sum(dim=0)
                tag = f"{name}/kernel_{k}"
                plt.spy(kernel_weights, color='blue', markersize=1)
                plt.title(tag)
                plt.show()

                # ax = fig.add_subplot(1, num_kernels, k + 1, xticks=[], yticks=[])
                # ax.set_title("layer {0}/kernel_{1}".format(name, k))
    # return fig


def predict_with_probs(model, images):
    output = model(images)
    _, predictions_ = torch.max(output, 1)
    predictions = np.squeeze(predictions_.cpu().numpy())
    return predictions, [F.softmax(el, dim=0)[i].item() for i, el in zip(predictions, output)]


def plot_classes_preds(model, images, labels, classes=None):
    preds, probs = predict_with_probs(model, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure()
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        imshow(images[idx].cpu().detach())
        if classes:
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        else:
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                preds[idx],
                probs[idx] * 100.0,
                labels[idx]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def weight_histograms_conv2d(writer, step, weights, name):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"{name}/kernel_{k}"
        if (flattened_weights != 0).any().item():
            writer.add_histogram(tag, flattened_weights[flattened_weights != 0], global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, name):
    flattened_weights = weights.flatten()
    tag = name
    writer.add_histogram(tag, flattened_weights[flattened_weights != 0], global_step=step, bins='tensorflow')
    # print('layer %s | std: %.3f | sparsity: %.3f%%' % (
    #    name, torch.std(flattened_weights), (flattened_weights == 0.).sum() / len(flattened_weights) * 100))


def weight_histograms(writer, step, model):
    # print("Visualizing model weights...")
    # Iterate over all model layers
    for name, module in model.named_modules():
        # Compute weight histograms for appropriate layer
        if isinstance(module, nn.Conv2d):
            weights = module.weight
            weight_histograms_conv2d(writer, step, weights, name)
        elif isinstance(module, nn.Linear):
            weights = module.weight
            weight_histograms_linear(writer, step, weights, name)


def plot_weight_histograms(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data.cpu()
            plt.hist(weight[weight != 0], bins=30, density=True)
            plt.title('layer: %s' % name)
            plt.show()
        elif isinstance(module, nn.Conv2d):
            weight = module.weight.data.cpu()
            for k in range(weight.shape[0]):
                flattened_weights = weight[k].flatten()
                tag = "layer: %s/kernel_%d" % (name, k)
                plt.hist(flattened_weights[flattened_weights != 0], bins=30, density=True)
                plt.title(tag)
                plt.show()


# def save_compressed_weights(model, save_path):
#     weight_dict = OrderedDict()
#     for name,module in model.named_modules():
#         if prune.is_pruned(module) and not isinstance(module, type(model)):
#             weight_mask = getattr(module,'weight_mask')
#             if quantize.is_quantized(module):
#                 indices = getattr(module,'weight_indices')
#                 weight_mask[weight_mask==1] = indices
#             else:
#
#             sparse_weight = sparse.csr_matrix(weight) if weight.shape[0] < weight.shape[1] else sparse.csc_matrix(
#                 weight)
#         tensor = model.state_dict()[param_tensor]
#         if prune.is_pruned(tensor):
#
#             bias = module.bias.data.cpu().numpy()
#
#             weight_dict['%s.weight' % name] = sparse_weight
#             weight_dict['%s.bias' % name] = bias
#     torch.save(weight_dict, save_path)
#
#
# def load_sparse_weights(load_path):
#     dict = torch.load(load_path)
#     for name, param in dict.items():
#         print(name)
#         if sparse.issparse(param):
#             param = param.todense()
#         dict[name] = torch.from_numpy(param)
#     return dict

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
