import torch
from torch.nn.utils.prune import BasePruningMethod, L1Unstructured, is_pruned
import operator


class ThresholdPruning(BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        self.s = amount

    def compute_mask(self, tensor, default_mask):
        threshold = torch.std(tensor).item() * self.s
        return torch.abs(tensor) > threshold


# TODO: alternatively prune linear and conv2d
# TODO: iteratively prune and train

def l1_threshold(module, name, amount):
    ThresholdPruning.apply(module, name, amount=amount
                           )
    return module


def l1_unstructured(module, name, amount, importance_scores=None):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified `amount` of (currently unpruned) units with the
    lowest L1-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.l1_unstructured(nn.Linear(2, 3), 'weight', amount=0.2)
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    L1Unstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module