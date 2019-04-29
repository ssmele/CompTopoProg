import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.tensor import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import warnings

from glob import glob


"""
This file contains only the needed logic, and classes from the Hofer et al paper that we will be needing.
"""

class UpperDiagonalThresholdedLogTransform:
    def __init__(self, nu):
        self.b_1 = (torch.Tensor([1, 1]) / np.sqrt(2))
        self.b_2 = (torch.Tensor([-1, 1]) / np.sqrt(2))
        self.nu = nu

    def __call__(self, dgm):
        if len(dgm) == 0:
            return dgm
        
        self.b_1 = self.b_1.to(dgm.device)
        self.b_2 = self.b_2.to(dgm.device)

        x = torch.mul(dgm, self.b_1.repeat(dgm.size(0), 1))
        x = torch.sum(x, 1).squeeze()
        y = torch.mul(dgm, self.b_2.repeat( dgm.size(0), 1))
        y = torch.sum(y, 1).squeeze()
        i = (y <= self.nu)
        y[i] = torch.log(y[i] / self.nu)*self.nu + self.nu
        ret = torch.stack([x, y], 1)
        return ret

    
def prepare_batch(batch: [Tensor], point_dim: int=None)->tuple:
    """
    This method 'vectorizes' the multiset in order to take advances of gpu processing.
    The policy is to embed all multisets in batch to the highest dimensionality
    occurring in batch, i.e., max(t.size()[0] for t in batch).
    :param batch:
    :param point_dim:
    :return: Tensor with size batch_size x n_max_points x point_dim
    """
    if point_dim is None:
        point_dim = batch[0].size(1)
    assert (all(x.size(1) == point_dim for x in batch if len(x) != 0))

    batch_size = len(batch)
    batch_max_points = max([t.size(0) for t in batch])
    input_device = batch[0].device

    if batch_max_points == 0:
        # if we are here, batch consists only of empty diagrams.
        batch_max_points = 1

    # This will later be used to set the dummy points to zero in the output.
    not_dummy_points = torch.zeros(batch_size, batch_max_points, device=input_device)

    prepared_batch = []

    for i, multi_set in enumerate(batch):
        n_points = multi_set.size(0)

        prepared_dgm = torch.zeros(batch_max_points, point_dim, device=input_device)

        if n_points > 0:
            index_selection = torch.tensor(range(n_points), device=input_device)

            prepared_dgm.index_add_(0, index_selection, multi_set)

            not_dummy_points[i, :n_points] = 1

        prepared_batch.append(prepared_dgm)

    prepared_batch = torch.stack(prepared_batch)

    return prepared_batch, not_dummy_points, batch_max_points, batch_size


def is_prepared_batch(input):
    if not (isinstance(input, tuple) and len(input) == 4):
        return False
    else:
        batch, not_dummy_points, max_points, batch_size = input
        return isinstance(batch, Tensor) and isinstance(not_dummy_points, Tensor) and max_points > 0 and batch_size > 0


def is_list_of_tensors(input):
    try:
        return all([isinstance(x, Tensor) for x in input])

    except TypeError:
        return False


def prepare_batch_if_necessary(input, point_dimension=None):
    batch, not_dummy_points, max_points, batch_size = None, None, None, None

    if is_prepared_batch(input):
        batch, not_dummy_points, max_points, batch_size = input
    elif is_list_of_tensors(input):
        if point_dimension is None:
            point_dimension = input[0].size(1)

        batch, not_dummy_points, max_points, batch_size = prepare_batch(input, point_dimension)

    else:
        raise ValueError(
            'SLayer does not recognize input format! Expecting [Tensor] or prepared batch. Not {}'.format(input))

    return batch, not_dummy_points, max_points, batch_size

def parameter_init_from_arg(arg, size, default, scalar_is_valid=False):
    if isinstance(arg, (int, float)):
        if not scalar_is_valid:
            raise ValueError("Scalar initialization values are not valid. Got {} expected Tensor of size {}."
                             .format(arg, size))
        return torch.Tensor(*size).fill_(arg)
    elif isinstance(arg, torch.Tensor):
        assert(arg.size() == size)
        return arg
    elif arg is None:
        if default in [torch.rand, torch.randn, torch.ones, torch.ones_like]:
            return default(*size)
        else:
            return default(size)
    else:
        raise ValueError('Cannot handle parameter initialization. Got "{}" '.format(arg))
        
        

    
class SLayerExponential(torch.nn.Module):
    """
    proposed input layer for multisets [1].
    """
    def __init__(self, n_elements: int,
                 point_dimension: int=2,
                 centers_init: Tensor=None,
                 sharpness_init: Tensor=None):
        """
        :param n_elements: number of structure elements used
        :param point_dimension: dimensionality of the points of which the input multi set consists of
        :param centers_init: the initialization for the centers of the structure elements
        :param sharpness_init: initialization for the sharpness of the structure elements
        """
        super().__init__()

        self.n_elements = n_elements
        self.point_dimension = point_dimension

        expected_init_size = (self.n_elements, self.point_dimension)

        centers_init = parameter_init_from_arg(centers_init, expected_init_size, torch.rand, scalar_is_valid=False)
        sharpness_init = parameter_init_from_arg(sharpness_init, expected_init_size, lambda size: torch.ones(*size)*3)

        self.centers = Parameter(centers_init)
        self.sharpness = Parameter(sharpness_init)

    def forward(self, input)->Tensor:
        batch, not_dummy_points, max_points, batch_size = prepare_batch_if_necessary(input,
                                                                                     point_dimension=self.point_dimension)


        batch = torch.cat([batch] * self.n_elements, 1)

        not_dummy_points = torch.cat([not_dummy_points] * self.n_elements, 1)

        centers = torch.cat([self.centers] * max_points, 1)
        centers = centers.view(-1, self.point_dimension)
        centers = torch.stack([centers] * batch_size, 0)

        sharpness = torch.pow(self.sharpness, 2)
        sharpness = torch.cat([sharpness] * max_points, 1)
        sharpness = sharpness.view(-1, self.point_dimension)
        sharpness = torch.stack([sharpness] * batch_size, 0)

        x = centers - batch
        x = x.pow(2)
        x = torch.mul(x, sharpness)
        x = torch.sum(x, 2)
        x = torch.exp(-x)
        x = torch.mul(x, not_dummy_points)
        x = x.view(batch_size, self.n_elements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x

    def __repr__(self):
        return 'SLayerExponential (... -> {} )'.format(self.n_elements)

class SLayer(SLayerExponential):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn("Renaming in progress. In future use SLayerExponential.", FutureWarning)