import numpy as np
from torch.nn.utils import prune
import torch
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold

def threshold_prune(model,s):
    for name, module in model.named_modules():
        if isinstance(module,torch.nn.Linear) or isinstance(module,torch.nn.Conv2d):
            threshold = np.std(module.weight.data.cpu().numpy()) * s
            print('Pruning with threshold : %.3f for layer %s'%(threshold,name))
            # calculate pruned weights
            prune.global_unstructured([(module,"weight")],pruning_method=ThresholdPruning,threshold=threshold)
            # apply the pruning
            # prune.remove(module, 'weight')