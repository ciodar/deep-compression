import logging
from functools import partial
from typing import Union, Callable, List, Optional, Tuple, Sequence

import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_debug
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.callbacks import Callback

import torch.nn as nn
import quantization

log = logging.getLogger(__name__)

_QUANTIZATION_FUNCTIONS = {
    "density_quantization": quantization.density_quantization,
    "forgy_quantization": quantization.forgy_quantization,
    "linear_quantization": quantization.linear_quantization
}

_QUANTIZATION_METHODS = {
    "density_quantization": quantization.DensityQuantizationMethod,
    "forgy_quantization": quantization.ForgyQuantizationMethod,
    "linear_quantization": quantization.LinearQuantizationMethod
}

_PARAM_TUPLE = Tuple[nn.Module, str]
_PARAM_LIST = Sequence[_PARAM_TUPLE]
_MODULE_CONTAINERS = (LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)


class Quantization(Callback):
    PARAMETER_NAMES = ("weight", "bias")
    def __init__(self, epoch, quantization_fn, parameters_to_quantize = None, parameter_names = None, bits: Union[int, List[int]]= None, apply_quantization: Union[bool, Callable[[int], bool]] = True, verbose: int=0):
        super().__init__()

        self._parameters_to_prune = parameters_to_quantize
        self._parameter_names = parameter_names or self.PARAMETER_NAMES
        self._global_kwargs = {}
        self._original_layers = None
        self._pruning_fn_name = None

        quantization_fn = self._create_quantization_fn(quantization_fn)

        self.quantization_fn = quantization_fn
        self._apply_quantization = apply_quantization
        self.bits = bits
        self._quantization_epoch = epoch
        self._quantize_on_train_epoch_end = False

        if verbose not in (0, 1, 2):
            raise MisconfigurationException("`verbose` must be any of (0, 1, 2)")

        self._verbose = verbose

    def _create_quantization_fn(self, quantization_fn: str, **kwargs) -> Union[Callable, quantization.BaseQuantizationMethod]:
        """
        This function takes `pruning_fn`, a function name.
        IF use_global_unstructured, pruning_fn will be resolved into its associated ``PyTorch BasePruningMethod``
        ELSE, pruning_fn will be resolved into its function counterpart from `torch.nn.utils.prune`.
        """

        quantization_fn = _QUANTIZATION_FUNCTIONS[quantization_fn]
        # save the function __name__ now because partial does not include it
        # and there are issues setting the attribute manually in ddp.
        self._quantization_fn_name = quantization_fn.__name__
        return Quantization._wrap_quantization_fn(quantization_fn, **kwargs)

    @staticmethod
    def _wrap_quantization_fn(pruning_fn, **kwargs):
        return partial(pruning_fn, **kwargs)

    def apply_quantization(self, bits: Union[int, float]) -> None:
        """Applies quantization to ``parameters_to_quantize``."""
        for module, name in self._parameters_to_quantize:
            log.debug("Quantizing {} into {:d} bits...".format(module, bits))
            self.quantization_fn(module, name=name, bits=bits)

    def setup(self, trainer: "pl.Trainer", pl_module: LightningModule, stage: str) -> None:
        parameters_to_quantize = self.sanitize_parameters_to_prune(
            pl_module, self._parameters_to_prune, parameter_names=self._parameter_names
        )

        self._parameters_to_quantize = parameters_to_quantize

    def _run_quantization(self, current_epoch: int) -> None:
        self._apply_quantization = current_epoch == self._quantization_epoch
        if self._apply_quantization:
            self.apply_quantization(self.bits)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: LightningModule) -> None:
        if not trainer.sanity_checking and not self._quantize_on_train_epoch_end:
            rank_zero_debug("`Quantization.on_validation_epoch_end`. Applying quantization")
            self._run_quantization(pl_module.current_epoch)

            if self._apply_quantization:
                compression = quantization.compression_stats(pl_module)

    @staticmethod
    def sanitize_parameters_to_prune(
            pl_module: LightningModule,
            parameters_to_quantize: Optional[_PARAM_LIST] = None,
            parameter_names: Optional[List[str]] = None,
    ) -> _PARAM_LIST:
        """
        This function is responsible of sanitizing ``parameters_to_prune`` and ``parameter_names``.
        If ``parameters_to_prune is None``, it will be generated with all parameters of the model.
        Raises:
            MisconfigurationException:
                If ``parameters_to_prune`` doesn't exist in the model, or
                if ``parameters_to_prune`` is neither a list of tuple nor ``None``.
        """
        parameters = parameter_names or Quantization.PARAMETER_NAMES

        current_modules = [m for m in pl_module.modules() if not isinstance(m, _MODULE_CONTAINERS)]

        if parameters_to_quantize is None:
            parameters_to_quantize = [(m, p) for p in parameters for m in current_modules if hasattr(m, p)]
        elif (
                isinstance(parameters_to_quantize, (list, tuple))
                and len(parameters_to_quantize) > 0
                and all(len(p) == 2 for p in parameters_to_quantize)
                and all(isinstance(a, nn.Module) and isinstance(b, str) for a, b in parameters_to_quantize)
        ):
            missing_modules, missing_parameters = [], []
            for module, name in parameters_to_quantize:
                if module not in current_modules:
                    missing_modules.append(module)
                    continue
                if not hasattr(module, name):
                    missing_parameters.append(name)

            if missing_modules or missing_parameters:
                raise MisconfigurationException(
                    "Some provided `parameters_to_tune` don't exist in the model."
                    f" Found missing modules: {missing_modules} and missing parameters: {missing_parameters}"
                )
        else:
            raise MisconfigurationException(
                "The provided `parameters_to_prune` should either be list of tuple"
                " with 2 elements: (nn.Module, parameter_name_to_prune) or None"
            )

        return parameters_to_quantize
