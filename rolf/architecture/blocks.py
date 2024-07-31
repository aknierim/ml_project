"""Implementation of the two available ResNet blocks:
Residual block and pre-activation block
"""

from collections.abc import Callable

import torch.nn as nn
from numpy.typing import ArrayLike


class ResBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        act_fn: Callable,
        subsample: bool = False,
        c_out: int = -1,
        dropout: float = 0.0,
    ) -> None:
        """Implementation of a ResBlock.

        Parameters
        ----------
        c_in : int
            Number of input features
        act_fn : callable
            Activation class constructor (e.g. nn.ReLU)
        subsample : bool
            If True, apply a stride inside the block and
            reduce the output shape by 2 in height and width
        c_out : int
            Number of output features. Only applies if
            subsample is True.
        dropout: float
            Dropout percentage.
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )

        self.downsample = (
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        )
        self.act_fn = act_fn()

    def forward(self, x: ArrayLike) -> ArrayLike:
        """Skip connection for the ResBlock.

        Parameters
        ----------
        x : array_like
            Input of the ResBlock.

        Returns
        -------
        array_like
            Output of the ResBlock.
        """
        temp = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return self.act_fn(temp + x)


class PreActBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        act_fn: Callable,
        subsample: bool = False,
        c_out: int = -1,
        dropout: float = 0.0,
    ) -> None:
        """Implementation of a PreActBlock.

        Parameters
        ----------
        c_in : int
            Number of input features
        act_fn : callable
            Activation class constructor (e.g. nn.ReLU)
        subsample : bool, optional
            If True, apply a stride inside the block and
            reduce the output shape by 2 in height and width
        c_out : int, optional
            Number of output features. Only applies if
            subsample is True.
        dropout: float, optional
            Dropout percentage.
        """
        super().__init__()

        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            act_fn(),
            nn.Dropout(dropout),
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(
                c_out,
                c_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        self.downsample = (
            nn.Sequential(
                nn.BatchNorm2d(c_in),
                act_fn(),
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=1,
                    stride=2,
                    bias=False,
                ),
            )
            if subsample
            else None
        )

    def forward(self, x: ArrayLike) -> ArrayLike:
        """Skip connection for the ResBlock.

        Parameters
        ----------
        x : array_like
            Input of the ResBlock.

        Returns
        -------
        temp + x: array_like
            Output of the ResBlock.
        """
        temp = self.net(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return temp + x
