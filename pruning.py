import collections
import argparse
import sys

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as module_prune
import operator

from parse_config import ConfigParser
from trainer.trainer import Trainer
import data as module_data
import models as module_arch
import evaluation as module_metric

class ThresholdPruning(module_prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


#TODO: alternatively prune linear and conv2d
#TODO: iteratively prune and train

def threshold_prune_module(module, param_name, s, dim=None):
    tensor = getattr(module, param_name).data.cpu()
    shape = tensor.shape
    threshold = torch.std(tensor).item() * s
    print('*** Pruning with amount : %.3f for layer %s' % (
        threshold, param_name))
    if dim is None:
        pruned_module = module_prune.l1_unstructured(module, name='weight', amount=threshold)
    else:
        pruned_module = module_prune.ln_structured(module, param_name, amount=threshold, n=1, dim=dim)
    # calculate pruned weights
    pruned_weights = torch.sum(pruned_module.weight_mask.flatten() == 0).item()
    tot_weights = len(pruned_module.weight_mask.flatten())
    print('*** Pruned %d weights (%.3f %%)' % (
        pruned_weights, pruned_weights / tot_weights * 100,))
    return pruned_weights, tot_weights


def sparsity_prune_module(module, param_name, sparsity, dim=None):
    tensor = getattr(module, param_name).data.cpu()
    shape = tensor.shape

    if dim is None:
        pruned_module = module_prune.l1_unstructured(module, name='weight', amount=sparsity)
    else:
        pruned_module = module_prune.ln_structured(module, param_name, amount=sparsity, n=1, dim=dim)
    # calculate pruned weights
    pruned_weights = torch.sum(pruned_module.weight_mask.flatten() == 0).item()
    tot_weights = len(pruned_module.weight_mask.flatten())
    print('*** Pruned %d weights (%.3f %%)' % (
        pruned_weights, pruned_weights / tot_weights * 100,))
    return pruned_weights, tot_weights


def count_nonzero_weights(model):
    return sum(p.weight.numel() for n, p in model.named_modules() if hasattr(p, 'weight'))


def prune_model(model, prune_fn, levels, logger=None):
    for p in model.parameters():
        p.requires_grad = False
    for param, amount in levels.items():
        if logger is not None:
            logger.info('Pruning {} with amount {:.2f}'.format(param, amount))
        # get param name separated from module
        m, param = param.split('.')[0:-1], param.split('.')[-1]
        module = operator.attrgetter('.'.join(m))(model)
        prune_fn(module, param, amount)
        # calculate compression stats
        param_vector = torch.nn.utils.parameters_to_vector(module._buffers[param + '_mask'])
        if logger is not None:
            logger.info('Pruned {} weights ({:.2%} retained)'.format(sum(param_vector == 0).item(),
                                                                     sum(param_vector == 1).item() / len(param_vector)))
        # Always retrain all parameters (eg. bias) even if not pruned
        for p in module.parameters():
            p.requires_grad = True
    return model