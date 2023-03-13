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
from scipy.sparse import csr_matrix
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
        weights = F.embedding(indices, centers).squeeze()
        ## debugging
        # weights.register_hook(print)
        if hasattr(module, self._tensor_name + '_mask'):
            mask = getattr(module, self._tensor_name + '_mask')
            mat = mask.detach().flatten()
            mat[torch.argwhere(mat)] = weights.view(-1, 1)
        else:
            mat = weights
        return mat.view(self._shape)

    @abstractmethod
    def initialize_clusters(self, mat, n_points):
        pass

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        param = getattr(module, name).detach()
        # get device on which the parameter is, then move to cpu
        device = param.device
        shape = param.shape
        # flatten weights to accommodate conv and fc layers
        mat = param.cpu().view(-1, 1)

        mat = csr_matrix(mat)
        mat = mat.data
        if mat.shape[0] < 2 ** bits:
            bits = int(np.log2(mat.shape[0]))
            print("Warning: number of elements {} is less than number of clusters. using {} bits for quantization.".format(mat.shape[0],bits))
        space = cls(*args, **kwargs).initialize_clusters(mat, 2 ** bits)

        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                        algorithm="full")
        kmeans.fit(mat.reshape(-1, 1))

        method = cls(*args, **kwargs)
        # Have the quantization method remember what tensor it's been applied to and weights shape
        method._tensor_name = name
        method._shape = shape

        centers, indices = kmeans.cluster_centers_, kmeans.labels_
        centers = torch.nn.Parameter(torch.from_numpy(centers).float().to(device))
        indices = torch.from_numpy(indices).to(device)
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
        space = np.linspace(min_, max_, num=n_points)
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
        x, cdf_counts = np.unique(mat, return_counts=True)
        y = np.cumsum(cdf_counts) / np.sum(cdf_counts)

        eps = 1e-2

        space_y = np.linspace(y.min() + eps, y.max() - eps, n_points)

        idxs = []
        # TODO find numpy operator to eliminate for
        for i in space_y:
            idx = np.argwhere(np.diff(np.sign(y - i)))[0]
            idxs.append(idx)
        idxs = np.stack(idxs)
        return x[idxs]

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        return super(DensityQuantizationMethod, cls).apply(module, name, bits)


def linear_quantization(module, name, bits):
    LinearQuantizationMethod.apply(module, name, bits)
    return module


def forgy_quantization(module, name, bits):
    ForgyQuantizationMethod.apply(module, name, bits)
    return module


def density_quantization(module, name, bits):
    DensityQuantizationMethod.apply(module, name, bits)
    return module


def is_quantized(module):
    for _, submodule in module.named_modules():
        for _, hook in submodule._forward_pre_hooks.items():
            if isinstance(hook, BaseQuantizationMethod):
                return True
    return False


def compression_rate(module, name, bits, weight_bits=32):
    param = getattr(module, name).data.cpu()
    orig = getattr(module, name + '_orig').data.cpu()
    if prune.module.is_pruned():
        n_weights = param.getnnz()
    else:
        n_weights = param.numel()
    cr = orig.numel() * 32 / n_weights * bits + 2 ** bits * weight_bits
    return cr



