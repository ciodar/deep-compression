import operator
from typing import List, Union, Dict

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

from pruning import sparsity_stats


class IterativePruning(ModelPruning):
    def __init__(self, pruning_epochs: List[int], amount: Union[int, float, List[int]] = None,
                 filter_layers: List[str] = None, use_global_unstructured: bool = True, parameters_to_prune=(),
                 **kwargs):

        self.pruning_epochs = pruning_epochs
        self._filter_layers = filter_layers
        self._amount = amount

        if use_global_unstructured and isinstance(amount, list):
            raise MisconfigurationException(
                "`amount` should be either an int or a float when `use_global_unstructured`=True"
            )

        super().__init__(amount=self._compute_amount, apply_pruning=self._check_epoch,
                         use_global_unstructured=use_global_unstructured, parameters_to_prune=parameters_to_prune,
                         **kwargs)

    def _check_epoch(self, epoch):
        return epoch in self.pruning_epochs

    def _compute_amount(self, epoch):
        return self._amount

    def filter_parameters_to_prune(self, parameters_to_prune=()):
        parameters_to_prune = list(filter(lambda p: p[0].__class__.__name__ in self._filter_layers, parameters_to_prune))
        return parameters_to_prune

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
