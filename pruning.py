from collections import OrderedDict
from scipy.sparse import csr_matrix, csc_matrix
from torch.nn.utils import prune
import torch.sparse as sparse
import math
import torch
import torch.nn as nn
import copy
import argparse

import models.lenet as lenet
import models.vgg as vgg
import models.alexnet as alexnet


class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold

#TODO: alternatively prune linear and conv2d
#TODO: iteratively prune and train

def threshold_prune(model, s):
    tot_weights = 0
    retained_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data.cpu()
            threshold = torch.std(weight).item() * s
            print('*** Pruning with threshold : %.3f for layer %s' % (threshold, name))
            # calculate pruned weights
            prune.global_unstructured([(module, "weight")], pruning_method=ThresholdPruning, threshold=threshold)
            _, weight_mask = list(module.named_buffers('weight_mask'))[0]
            if weight_mask.flatten().any().item():
                print('*** Compression factor for %s: %.3f' % (
                    name, int(len(weight_mask.flatten()) / weight_mask.flatten().sum())))

                tot_weights += len(weight_mask.flatten())
                retained_weights += weight_mask.flatten().sum()
    compression = math.inf if retained_weights == 0 else tot_weights / retained_weights
    print('Total compression factor: %.1f | %% of retained weights: %.1f ' % (compression, 1. / compression * 100.))
    return compression


def perc_prune(model, p):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=p)
            print('%s - %% of retained weights: %.1f ' % (name, sum(torch.nn.utils.parameters_to_vector(
                model.buffers()) != 0) / len(torch.nn.utils.parameters_to_vector(model.buffers()))))
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=p, n=1, dim=1)
            print('%s - %% of retained weights: %.1f ' % (name, sum(torch.nn.utils.parameters_to_vector(
                model.buffers()) != 0) / len(torch.nn.utils.parameters_to_vector(model.buffers()))))
    total_weights = len(torch.nn.utils.parameters_to_vector(model.buffers()))
    retained_weights = sum(torch.nn.utils.parameters_to_vector(model.buffers()) != 0)
    compression = total_weights / retained_weights
    print('Total retained weights: %d | Total compression factor: %.1fx | %% of retained weights: %.1f ' % (retained_weights,compression, 1. / compression * 100.))
    return compression


def apply_pruning(model):
    if prune.is_pruned(model):
        for name, module in model.named_modules():
            if prune.is_pruned(module) \
                    and (isinstance(module, torch.nn.Linear)
                         or isinstance(module, torch.nn.Conv2d)):
                prune.remove(module, 'weight')


def save_sparse_weights(model, save_path):
    weight_dict = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data.cpu().numpy()
            bias = module.bias.data.cpu().numpy()
            sparse_weight = sparse.csr_matrix(weight) if weight.shape[0] < weight.shape[1] else sparse.csc_matrix(
                weight)
            weight_dict['%s.weight' % name] = sparse_weight
            weight_dict['%s.bias' % name] = bias
    torch.save(weight_dict, save_path)

def load_sparse_weights(load_path):
    dict = torch.load(load_path)
    for name, param in dict.items():
        print(name)
        if sparse.issparse(param):
            param = param.todense()
        dict[name] = torch.from_numpy(param)
    return dict


def pruning(args):
    if args.model == 'lenet-300':
        net = lenet.LeNet300(10)
    elif args.model == 'lenet-5':
        net = lenet.LeNet5(10)

    net.load_state_dict(torch.load(args.weight_path))

    threshold_prune(net, args.s)
    apply_pruning(net)
    save_sparse_weights(net, args.output_path)

    # print(torch.load(args.output_path))


def count_nonzero_weights(model):
    return sum(p.weight.numel() for n, p in model.named_modules() if hasattr(p, 'weight'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=52, width=90))
    parser.add_argument('model', type=str,
                        help='Select a model. Available options: (lenet-300-100,lenet-5,alexnet,vgg)')
    parser.add_argument('weight_path', type=str, help='Path to the model weights')
    parser.add_argument('--s', default=1, type=float, help='threshold multiplier constant')
    parser.add_argument('--output_path', type=str, help='Path where to save the pruned model\'s weights')

    pruning(parser.parse_args())
