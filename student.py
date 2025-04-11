'''
Project         : Rich Teacher Features for Efficient Single-Image Haze Removal
Lab             : Medical Image Computing and Artificial Intelligence Lab, National Institute of Technology, Trichy
Contributors    : Sai Mitheran, Anushri Suresh, Varun P. Gopi
'''

import torch
from torch import nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """
    It creates a convolutional layer with padding equal to half the kernel size

    :param in_channels: The number of channels in the input image
    :param out_channels: The number of filters in the convolutional layer
    :param kernel_size: The size of the convolutional kernel
    :param bias: If True, adds a learnable bias to the output. Default: True, defaults to True
    (optional)
    :return: A convolutional layer with the given parameters.
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


# It takes in a channel number and returns a layer that takes in an image and returns the image multiplied by
# a pixel attention mask


class PALayer(nn.Module):
    def __init__(self, channel):
        """
        :param channel: the number of channels in the input feature map
        """
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        `forward` takes in a tensor `x` and returns `x` multiplied by the output of `self.pa(x)`

        :param x: input tensor
        :return: The output of the network.
        """
        y = self.pa(x)
        return x * y


# It takes in a channel number and returns a class that takes in an image and returns the image multiplied by
# a channel-wise attention map
class CALayer(nn.Module):
    def __init__(self, channel):
        """
        :param channel: the number of channels in the input feature map
        """
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        `forward` takes in a tensor `x` and returns `x` multiplied by the output of pooled `self.ca(x)`

        :param x: input tensor
        :return: The output of the network.
        """
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# It's a block that takes in an input, applies a convolution, applies a ReLU, applies another
# convolution, applies a CALayer, applies a PALayer, and then adds the input back to the output.


class Inner(nn.Module):
    def __init__(
        self,
        conv,
        dim,
        kernel_size,
    ):
        """
        The function takes in the convolutional layer, the dimension of the input, and the kernel size, and
        returns a block of the network

        :param conv: convolution layer
        :param dim: the number of channels in the input and output
        :param kernel_size: the size of the convolutional kernel
        """
        super(Inner, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        """
        > We take the input, pass it through a convolutional layer, add the input back to the output, pass
        it through another convolutional layer, pass it through a channel attention layer, pass it through a
        spatial attention layer, add the input back to the output, and return the result

        :param x: input
        :return: The residual block is returning the residual of the input and the output of the
        convolutional layers.
        """
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


# > The Outer class is a sequential module that contains a number of Inner modules, and a
# convolutional layer
class Outer(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        """
        :param conv: the convolutional layer to use
        :param dim: the number of channels in the input and output
        :param kernel_size: the size of the convolutional kernel
        :param blocks: number of inner blocks
        """
        super(Outer, self).__init__()
        modules = [Inner(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        """
        `forward` takes in a tensor `x` and returns a tensor `res` which is the result of applying a
        `nn.Sequential` object `gp` to `x` and then adding `x` to the result.

        :param x: the input to the residual block
        :return: The residual block is returning the sum of the input and the output of the gp.
        """
        res = self.gp(x)
        res += x
        return res


# It's a network that takes in an image and outputs a dehazed image
class Student(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        """
        :param gps: number of groups
        :param blocks: number of residual blocks in each group
        :param conv: the convolutional layer used in the network
        """
        super(Student, self).__init__()
        self.gps = gps
        self.dim = 16
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        self.g1 = Outer(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Outer(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Outer(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(
            *[
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
                nn.Sigmoid(),
            ]
        )
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size),
        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        """
        > The function takes in an image, passes it through a series of convolutional layers, and then uses
        the output of those layers to create a weighted sum of the input image and the output of the
        convolutional layers

        :param x1: input image
        :return: The output of the network.
        """
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1
