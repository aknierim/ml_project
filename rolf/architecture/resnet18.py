"""Main architecture of this project, following the ResNet18 architecture."""

import torch.nn as nn


class ResNet18(nn.Module):
    """Main network architecture class. This follows
    the ResNet18 architecture.
    """

    def __init__(self) -> None:
        """Initializes the architecture. The main building
        blocks are applied sequentially after a pre block
        and are followed by a final block.
        """
        super().__init__()

        self.pre_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
        )

        self.blocks = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(128, 128, stride=2),
            ConvBlock(128, 128),
            ConvBlock(256, 256, stride=2),
            ConvBlock(256, 256),
            ConvBlock(512, 512, stride=2),
            ConvBlock(512, 512),
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(4),
        )

    def forward(self, x):
        """Skip connection if needed."""
        # WIP
        pass


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        padding_mode: str = "reflect",
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        """Initializes the convolutional block class.

        Parameters
        ----------
        n_in : int
            Number of channels in the input image.
        n_out : int
            Number of channels produced by the convolution.
        kernel_size : int or tuple, optional
            Size of the convolving kernel. Default: 3
        stride : int or tuple, optional
            Stride of the convolution. Default: 1
        padding : int or tuple, optional
            Size of the padding added to all four image sides.
            Default: 0
        padding_mode : str, optional
            `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`.
            Default: `'reflect'`
        dilation : int or tuple, optional
            Spacing between kernel elements. Default: 1
        groups : int, optional
            Number of blocked connections from input channels to
            output channels. Default: 1
        bias : bool, optional
            If `True`, adds a learnable bias to the output.
            Default: `False`
        """
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.conv = self._convolution_block()
        self.pool = self._pool()

    def _convolution_block(self):
        """Returns the main building block consisting
        of a Conv2d layer followed by BatchNorm2d, activation
        function (PReLU), a second Conv2d layer, and another
        BatchNorm2d.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_in,
                out_channels=self.n_out,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                padding_mode=self.padding_mode,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            ),
            nn.BatchNorm2d(self.n_out),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.n_in,
                out_channels=self.n_out,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                padding_mode=self.padding_mode,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            ),
            nn.BatchNorm2d(self.n_out),
        )

    def _pool(self):
        """Max pooling layer that is applied after the (main)
        convolution building block.
        """
        return nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def forward(self, x):
        """Skip connection that adds the input of
        the ConvBlock to it's output.
        """
        return x + self.conv(x)
