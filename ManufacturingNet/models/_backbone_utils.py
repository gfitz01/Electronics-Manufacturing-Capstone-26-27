"""Helpers for adapting torchvision backbones to non-RGB inputs.

The original ManufacturingNet wrappers replaced the first convolution of every
pretrained backbone with a freshly initialised ``nn.Conv2d``.  That had two
costs:

1. The pretrained weights of the first layer were thrown away, even when the
   input already had three channels (the common case, because the image
   transform pipeline calls ``transforms.Grayscale(num_output_channels=3)``).
2. For AlexNet the replacement also changed the geometry -- an 11x11 stride-4
   stem became 3x3 stride-1 -- which removed the 4x downsample and made every
   downstream layer operate on ~16x the spatial area.

``adapt_first_conv`` keeps the backbone's own kernel size, stride, padding and
bias, and only rebuilds the layer when the channel count genuinely differs.
When it does rebuild, pretrained weights are carried over by averaging across
the input-channel dimension, which is the standard way to move an RGB stem to
a different channel count.
"""

import torch
import torch.nn as nn


def adapt_first_conv(conv, in_channels):
    """Return a conv layer accepting ``in_channels``, preserving geometry.

    If ``conv`` already accepts ``in_channels`` it is returned untouched, so
    pretrained weights survive. Otherwise a new ``nn.Conv2d`` with identical
    kernel/stride/padding/dilation/groups/bias is created and seeded from the
    original weights, averaged over the input-channel axis.
    """
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(
            "adapt_first_conv expected an nn.Conv2d, got "
            f"{type(conv).__name__}"
        )

    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )

    with torch.no_grad():
        # mean over the original input channels -> one "grey" filter, then
        # repeat it across however many channels the user actually has.
        mean_weight = conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


def adapt_basic_conv(basic_conv, in_channels):
    """Same as :func:`adapt_first_conv` for torchvision's ``BasicConv2d``.

    ``BasicConv2d`` (used by GoogLeNet) wraps a conv plus batch-norm; only the
    inner conv needs adapting, and the batch-norm statistics stay valid.
    """
    basic_conv.conv = adapt_first_conv(basic_conv.conv, in_channels)
    return basic_conv
