import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c_in, activation, subsample=False, c_out=-1):
        super().__init__()

        if not subsample:
            c_out = c_in

        self.block = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            activation(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        self.downsample = (
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        )
        self.act_fn = activation()

        def forward(self, x):
            temp = self.block()

            if self.downsample is not None:
                x = self.downsample(x)

            return temp + x


class PreActBlock(nn.Module):
    def __init__(self, c_in, activation, subsample=False, c_out=-1) -> None:
        """ """
        super().__init__()

        if not subsample:
            c_out = c_in

        self.block = nn.Sequential(
            nn.BatchNorm2d(c_in),
            activation(),
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            activation(),
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
                activation(),
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

        def forward(self, x):
            temp = self.block()

            if self.downsample is not None:
                x = self.downsample(x)

            return temp + x
