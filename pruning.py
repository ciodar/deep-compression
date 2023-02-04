from torch.nn.utils import prune
import torch.sparse as sparse
import math
import torch
import copy

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


def threshold_prune(model, s):
    tot_weights = 0
    retained_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight = module.weight.data.cpu()
            threshold = torch.std(weight).item() * s
            print('Pruning with threshold : %.3f for layer %s' % (threshold, name))
            # calculate pruned weights
            prune.global_unstructured([(module, "weight")], pruning_method=ThresholdPruning, threshold=threshold)
            _, weight_mask = list(module.named_buffers('weight_mask'))[0]
            if weight_mask.flatten().any().item():
                print('Compression factor for %s: %.3f' % (name,int(len(weight_mask.flatten())/weight_mask.flatten().sum())))

                tot_weights += len(weight_mask.flatten())
                retained_weights += weight_mask.flatten().sum()
    compression = math.inf if retained_weights == 0 else tot_weights/retained_weights
    print ('Total compression factor: %.1f' % compression)
    return compression

def apply_prune(model):
    for name,module in model.named_modules():
        if prune.is_pruned(module) \
                and (isinstance(module,torch.nn.Linear)
                     or isinstance(module,torch.nn.Conv2d)):
            prune.remove(module, 'weight')


def save_sparse_weights(model,path):
    sparse_model = copy.deepcopy(model.cpu())
    for name,module in sparse_model.named_modules():
        if isinstance(module,torch.nn.Linear) or isinstance(module,torch.nn.Conv2d):
            weight = module.weight.data.cpu()
            if weight.shape[0] > weight.shape[1]:
                sparse_weight = weight.to_sparse()
            else:
                sparse_weight = weight.to_sparse()
            module.weight = torch.nn.Parameter(sparse_weight)
        torch.save(sparse_model.state_dict(), path)

def count_nonzero_weights(model):
    return sum(p.weight.numel() for n,p in model.named_modules() if hasattr(p,'weight'))
