'''
Project         : Rich Teacher Features for Efficient Single-Image Haze Removal
Lab             : Medical Image Computing and Artificial Intelligence Lab, National Institute of Technology, Trichy
Contributors    : Sai Mitheran, Anushri Suresh, Varun P. Gopi
'''

from torch import nn

# The class takes in the number of input and output channels, kernel size, stride and padding and
# creates a depthwise convolution layer with the given parameters and a pointwise convolution layer
# with a kernel size of 1


class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):

        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# It's a network that takes in an image, and outputs a higher resolution version
# of that image


class Teacher(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        """
        The function takes in the scale factor, number of channels, d, s, and m as inputs. It then creates a
        sequential model with a convolutional layer, a PReLU layer, a DSC layer, a PReLU layer, and a
        convolutional transpose layer.

        The convolutional layer has a kernel size of 5, padding of 5//2, and the number of channels is equal
        to d. The PReLU layer has d channels. The DSC layer has a kernel size of 1, and the number of
        channels is equal to s. The PReLU layer has s channels. The DSC layer has a kernel size of 3,
        padding of 3//2, and the number of channels is equal to s. The PReLU layer has s channels. The DSC
        layer has a kernel size of 1, and the number of channels is equal to d. The PReLU layer has d
        channels. The

        :param scale_factor: the scale factor of the image. For example, if you want to upscale the image by
        a factor of 2, then the scale factor is 2
        :param num_channels: number of channels in the input image, defaults to 1 (optional)
        :param d: number of feature maps in the first layer, defaults to 56 (optional)
        :param s: number of filters in the first layer of the mid-part, defaults to 12 (optional)
        :param m: number of layers in the middle part, defaults to 4 (optional)
        """
        super(Teacher, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2), nn.PReLU(d)
        )
        self.mid_part = [DSC(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend(
                [DSC(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)]
            )

        self.mid_part.extend([DSC(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        It initializes the weights of the last part of the network (the fully connected layer) with a normal
        distribution with mean 0 and standard deviation 0.001
        """
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        """
        > The function takes in an input `x` and passes it through the first part of the network, then
        passes the output of the first part through the middle part of the network, and finally passes the
        output of the middle part through the last part of the network

        :param x: input tensor
        :return: The first, mid, and last parts of the network.
        """
        first = self.first_part(x)
        mid = self.mid_part(first)
        last = self.last_part(mid)
        return first, mid, last
