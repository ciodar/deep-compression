from typing import Tuple

import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.cluster import KMeans


# Quantization base class inspired on torch.nn.utils.BasePruningMethod
class BaseQuantizationMethod:
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
        if hasattr(module, self._tensor_name + '_mask'):
            mask = getattr(module, self._tensor_name + '_mask')
            mat = mask.clone()
            mat[mat == 1] = weights
        else:
            mat = weights.view(self._shape)
        return mat

    @classmethod
    def apply(cls, module, name, bits, *args, **kwargs):
        orig = getattr(module, name).data.cpu()
        shape = orig.shape
        mat = csr_matrix(orig) if shape[0] < shape[1] else csc_matrix(orig)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2 ** bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                        algorithm="full")
        kmeans.fit(mat.data.reshape(-1, 1))

        method = cls(*args, **kwargs)
        # Have the quantization method remember what tensor it's been applied to
        method._tensor_name = name
        method._shape = orig.shape

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


def cluster_quantize(module, name, bits):
    BaseQuantizationMethod.apply(module, name, bits)
    return module
