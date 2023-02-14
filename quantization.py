import os
from abc import ABC, abstractmethod
from typing import Tuple

# suppress Kmeans warning of memory leak in Windows
os.environ['OMP_NUM_THREADS'] = "1"

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.cluster import KMeans


# Quantization base class inspired on torch.nn.utils.BasePruningMethod
class BaseQuantizationMethod(ABC):
    _tensor_name: str
    _shape: Tuple

    def __init__(self):
        pass

    def __call__(self, module, inputs):
        r"""Looks up the weights (stored in ``module[name + '_indices']``)
        from indices (stored in ``module[name + '_centers']``)
        and stores the result into ``module[name]`` by using
        :meth:`lookup_weights`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.lookup_weights(module))

    def lookup_weights(self, module):
        assert self._tensor_name is not None, "Module {} has to be quantized".format(
            module
        )  # this gets set in apply()
        indices = getattr(module, self._tensor_name + '_indices')
        centers = getattr(module, self._tensor_name + '_centers')
        weights = F.embedding(indices, centers)
        ## debugging
        # weights.register_hook(print)
        if hasattr(module, self._tensor_name + '_mask'):
            mask = getattr(module, self._tensor_name + '_mask')
            mat = mask.clone()
            mat[mat == 1] = weights
        else:
            mat = weights.view(self._shape)
        return mat

    @abstractmethod
    def initialize_clusters(self, mat, n_points):
        pass

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        param = getattr(module, name).data.cpu()
        shape = tuple(param.shape)
        if len(shape) <= 2:
            mat = csr_matrix(param) if shape[0] < shape[1] else csc_matrix(param)
            mat = mat.data
        else:
            mat = param.numpy(force=True)

        space = cls(*args, **kwargs).initialize_clusters(mat, 2 ** bits)

        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                        algorithm="full")
        kmeans.fit(mat.reshape(-1, 1))

        method = cls(*args, **kwargs)
        # Have the quantization method remember what tensor it's been applied to
        method._tensor_name = name
        method._shape = param.shape

        centers, indices = kmeans.cluster_centers_, kmeans.labels_
        centers = torch.nn.Parameter(torch.from_numpy(centers).float())
        indices = torch.from_numpy(indices)
        # If no reparameterization was done before (pruning), delete parameter
        if name in module._parameters:
            del module._parameters[name]
        # reparametrize by saving centroids and indices to `module[name + '_centers']`
        # and `module[name + '_indices']`...
        module.register_parameter(name + "_centers", centers)
        module.register_buffer(name + "_indices", indices)
        # ... and the new quantized tensor to `module[name]`
        setattr(module, name, method.lookup_weights(module))
        # associate the quantization method to the module via a hook to
        # compute the function before every forward() (compile by run)
        module.register_forward_pre_hook(method)
        # print("Compression rate for layer %s: %.1f" % compression_rate(module,name,bits))


class LinearQuantizationMethod(BaseQuantizationMethod):
    def initialize_clusters(self, mat, n_points):
        min_ = mat.min()
        max_ = mat.max()
        space = np.linspace(min_, max_, num=2 ** n_points)
        return space

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        return super(LinearQuantizationMethod, cls).apply(module, name, bits)


class ForgyQuantizationMethod(BaseQuantizationMethod):
    def initialize_clusters(self, mat, n_points):
        samples = np.random.choice(mat, size=n_points, replace=False)
        return samples

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        return super(ForgyQuantizationMethod, cls).apply(module, name, bits)


class DensityQuantizationMethod(BaseQuantizationMethod):
    def initialize_clusters(self, mat, n_points):
        raise NotImplementedError
        return None

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        return super(DensityQuantizationMethod, cls).apply(module, name, bits)


def linear_quantize(module, name, bits):
    LinearQuantizationMethod.apply(module, name, bits)
    return module


def forgy_quantize(module, name, bits):
    ForgyQuantizationMethod.apply(module, name, bits)
    return module


def compression_rate(module, name, bits, weight_bits=32):
    param = getattr(module, name).data.cpu()
    orig = getattr(module, name + '_orig').data.cpu()
    if prune.module.is_pruned():
        n_weights = param.getnnz()
    else:
        n_weights = param.numel()
    cr = orig.numel() * 32 / n_weights * bits + 2 ** bits * weight_bits
    return cr


def is_quantized(module):
    for _, submodule in module.named_modules():
        for _, hook in submodule._forward_pre_hooks.items():
            if isinstance(hook, BaseQuantizationMethod):
                return True
    return False
