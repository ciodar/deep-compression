import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Tuple

from compression.huffman_encoding import HuffmanEncode

# suppress Kmeans warning of memory leak in Windows
os.environ['OMP_NUM_THREADS'] = "1"

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import numpy as np
from scipy.sparse import csr_matrix, csr_array
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)


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
        if prune.is_pruned(module):
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

        # assume it is a sparse matrix, avoid to encode zeros since they are handled by pruning reparameterization
        mat = csr_matrix(mat)
        mat = mat.data
        if mat.shape[0] < 2 ** bits:
            bits = int(np.log2(mat.shape[0]))
            log.warning("Number of elements in weight matrix ({}) is less than number of clusters ({:d}). \
                        using {:d} bits for quantization."
                        .format(mat.shape[0], 2 ** bits, bits))
        space = cls(*args, **kwargs).initialize_clusters(mat, 2 ** bits)

        # could do more than one initialization for better results
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                        algorithm="lloyd")
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

    def remove(self, module):
        r"""Removes the quantization reparameterization from a module. The pruned
        parameter named ``name`` remains permanently quantized, and the parameter
        named  and ``name+'_centers'`` is removed from the parameter list. Similarly,
        the buffer named ``name+'_indices'`` is removed from the buffers.
        """
        # before removing quantization from a tensor, it has to have been applied
        assert (
                self._tensor_name is not None
        ), "Module {} has to be quantized\
                    before quantization can be removed".format(
            module
        )  # this gets set in apply()

        # to update module[name] to latest trained weights
        weight = self.lookup_weights(module)  # masked weights

        # delete and reset
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        del module._parameters[self._tensor_name + "_centers"]
        del module._buffers[self._tensor_name + "_indices"]
        module.register_parameter(self._tensor_name, weight.data)


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


def get_compression(module, name, idx_bits, huffman_encoding=False):
    # bits encoding weights
    float32_bits = 32

    all_weights = getattr(module, name).numel()
    n_weights = all_weights
    p_idx, q_idx, idx_size = 0, 0, 0

    if prune.is_pruned(module):
        attr = f"{name}_mask"
        mask = csr_array(getattr(module, attr).cpu().view(-1))
        n_weights = mask.getnnz()
        if huffman_encoding:
            # use index difference of csr matrix
            idx_diff = np.diff(mask.indices, prepend=mask.indices[0].astype(np.int8))
            # store overhead of adding placeholder zeros, then consider only indices below 2**idx_bits
            overhead = sum(map(lambda x: x // 2 ** idx_bits, idx_diff[idx_diff > 2 ** idx_bits]))
            idx_diff = idx_diff[idx_diff < 2 ** idx_bits]
            p_idx, avg_bits = HuffmanEncode.encode(idx_diff, bits=idx_bits)
            p_idx += overhead
            log.info(f" before Huffman coding: {n_weights*idx_bits:.0f} | after: {p_idx + overhead} | overhead: {overhead:.0f} | average bits: {avg_bits:.0f}")
        else:
            p_idx = n_weights * idx_bits
    if is_quantized(module):
        attr = f"{name}_centers"
        n_weights = getattr(module, attr).numel()
        attr = f"{name}_indices"
        idx = getattr(module, attr).view(-1)
        weight_bits = math.log2(n_weights)
        q_idx = idx.numel() * weight_bits
        if huffman_encoding:
            # use index difference of csr matrix
            q_idx, _ = HuffmanEncode.encode(idx.detach().cpu().numpy(), bits=weight_bits)
    # Note: compression formula in paper does not include the mask
    return all_weights * float32_bits, n_weights * float32_bits + p_idx + q_idx


def compression_stats(model, name="weight", idx_bits=5, huffman_encoding=False):
    log.info(f"Compression stats of `{model.__class__.__name__}` - `{name}`:")
    compression_dict = {
        n: get_compression(m, name, idx_bits=idx_bits, huffman_encoding=huffman_encoding) for
        n, m in model.named_modules() if
        getattr(m, name, None) is not None}

    for name, (n, d) in compression_dict.items():
        cr = n / d
        log.info(f"  Layer {name}: compression rate {1 / cr:.2%} ({cr:.1f}X) ")
    n, d = zip(*compression_dict.values())
    total_params = sum(n)
    total_d = sum(d)
    cr = total_params / total_d
    log.info(f"Total compression rate: {1 / cr:.2%} ({cr:.1f}X) ")
    return cr
