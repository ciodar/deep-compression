from typing import List, Union, Optional, Callable, Dict, Any

import torch.nn.utils.prune as pytorch_prune
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

import compression
from compression.pruning import sparsity_stats

class IterativePruning(ModelPruning):
    LAYER_TYPES = ("Linear", "Conv2d")


    def __init__(self, pruning_fn: Union[Callable, str], pruning_schedule: Dict,
                 amount: Union[int, float, List[int]] = None,
                 filter_layers: Optional[List[str]] = None, use_global_unstructured: bool = True,
                 **kwargs):
        self._use_global_unstructured = use_global_unstructured
        # custom pruning function
        if isinstance(pruning_fn, str) and pruning_fn.lower() == "l1_threshold":
            pruning_fn = compression.ThresholdPruning

        super().__init__(amount=self._compute_amount, apply_pruning=self._check_epoch, pruning_fn=pruning_fn,
                         use_global_unstructured=use_global_unstructured,
                         **kwargs)

        self._pruning_schedule = pruning_schedule
        self._filter_layers = filter_layers or self.LAYER_TYPES
        self._amount = amount

        if use_global_unstructured and isinstance(amount, list):
            raise MisconfigurationException(
                "`amount` should be either an int or a float when `use_global_unstructured`=True"
            )

    def _check_epoch(self, epoch):
        if 'target_sparsity' in self._pruning_schedule:
            total_params = sum(p.numel() for layer, _ in self._parameters_to_prune for p in layer.parameters())
            stats = [self._get_pruned_stats(m, n) for m, n in self._parameters_to_prune]
            zeros = sum(zeros for zeros, _ in stats)
            if zeros / total_params > self._pruning_schedule['target_sparsity']:
                return False
        if 'epochs' in self._pruning_schedule:
            return epoch in self._pruning_schedule['epochs']
        if 'start_epoch' in self._pruning_schedule and epoch >= self._pruning_schedule['start_epoch']:
            prune_every = self._pruning_schedule.get('prune_every', 1)
            return (epoch - self._pruning_schedule['start_epoch']) % prune_every == 0

    def _compute_amount(self, epoch):
        return self._amount

    def filter_parameters_to_prune(self, parameters_to_prune=()):
        # filter modules based on type (Linear or Conv2d)
        return list(filter(lambda p: p[0].__class__.__name__ in self._filter_layers, parameters_to_prune))

    def _apply_local_pruning(self, amount: Union[int, float, List[float]]):
        for i, (module, name) in enumerate(self._parameters_to_prune):
            self.pruning_fn(module, name=name, amount=self._amount[i])

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: LightningModule) -> None:
        if self._prune_on_train_epoch_end:
            rank_zero_debug("`ModelPruning.on_train_epoch_end`. Applying pruning")
            self._run_pruning(pl_module.current_epoch)

            if self._check_epoch(pl_module.current_epoch):
                tot_retained, tot_pruned = sparsity_stats(pl_module)
                pl_module.log("sparsity", (tot_retained / (tot_pruned + tot_retained)))
                pl_module.log("compression", ((tot_pruned + tot_retained) / tot_retained))
