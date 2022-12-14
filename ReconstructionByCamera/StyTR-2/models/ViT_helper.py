import torch
from torch import nn

def drop_path(need_drop, drop_probability: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_probability == 0. or not training:
        return need_drop
    keep_probability = 1 - drop_probability
    shape = (need_drop.shape[0],) + (1,) * (need_drop.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_probability + torch.rand(shape, dtype=need_drop.dtype, device=need_drop.device)
    random_tensor.floor_()  # binarize
    output = need_drop.div(keep_probability) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_probability=None):
        super(DropPath, self).__init__()
        self.drop_probability = drop_probability

    def forward(self, x):
        return drop_path(x, self.drop_probability, self.training)

from itertools import repeat
from torch._six import container_abcs


# From PyTorch internals
def _number_of_tuples(number):
    def parse(tuple_value):
        if isinstance(tuple_value, container_abcs.Iterable):
            return tuple_value
        return tuple(repeat(tuple_value, number))
    return parse


to_1tuple = _number_of_tuples(1)
to_2tuple = _number_of_tuples(2)
to_3tuple = _number_of_tuples(3)
to_4tuple = _number_of_tuples(4)



import torch
import math
import warnings


def _no_gradient_truncated_normal_(tensor, mean, standard, minimum_cutoff_value, maximum_cutoff_value):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def normal_cutoff(distribution):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(distribution / math.sqrt(2.))) / 2.

    if (mean < minimum_cutoff_value - 2 * standard) or (mean > maximum_cutoff_value + 2 * standard):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        min_cutoff_distribution = normal_cutoff((minimum_cutoff_value - mean) / standard)
        max_cutoff_distribution = normal_cutoff((maximum_cutoff_value - mean) / standard)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * min_cutoff_distribution - 1, 2 * max_cutoff_distribution - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(standard * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=minimum_cutoff_value, max=maximum_cutoff_value)
        return tensor


def truncated_normal_(tensor, mean=0., standard=1., minimum_cutoff_value=-2., maximum_cutoff_value=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        standard: the standard deviation of the normal distribution
        minimum_cutoff_value: the minimum cutoff value
        maximum_cutoff_value: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.truncated_normal_(w)
    """
    return _no_gradient_truncated_normal_(tensor, mean, standard, minimum_cutoff_value, maximum_cutoff_value)