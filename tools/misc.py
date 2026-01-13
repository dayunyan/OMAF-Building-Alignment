import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import to_2tuple
except ImportError:

    def to_2tuple(x):
        """将输入转换为 2-元组"""
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                return (x[0], x[0])
            elif len(x) == 2:
                return tuple(x)
            else:
                raise ValueError(
                    "Input must be an int, or a list/tuple of length 1 or 2."
                )
        elif isinstance(x, int):
            return (x, x)
        else:
            raise ValueError("Input must be an int, or a list/tuple.")


try:
    from timm.models.layers import trunc_normal_
except ImportError:

    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        """
        PyTorch-native trunc_normal_ a la timm
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py
        """

        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn(
                "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                "The distribution of values may be incorrect.",
                stacklevel=2,
            )
        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        """PyTorch-native trunc_normal_ a la timm"""
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)


try:
    from timm.models.layers import DropPath
except ImportError:

    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

        def __init__(self, drop_prob=0.0, scale_by_keep=True):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # B, 1, 1, 1
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and self.scale_by_keep:
                random_tensor.div_(keep_prob)
            return x * random_tensor

        def extra_repr(self):
            return f"drop_prob={round(self.drop_prob,3)}"


def constant_init(module, val, bias=0.0):
    """初始化
    weights 为 val, bias 为 0"""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module, std=0.02, bias=0.0):
    """截断正态分布初始化"""
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(module.weight, mean=0.0, std=std, a=-2.0, b=2.0)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)
