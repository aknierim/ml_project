"""Main architecture of this project, following the ResNet architecture."""

import torch.nn as nn

from .blocks import PreActBlock, ResBlock

BLOCKS = {"ResBlock": ResBlock, "PreActBlock": PreActBlock}
ACTIVATION = {"prelu": nn.PReLU, "relu": nn.ReLU, "mish": nn.Mish}


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        block_groups: list[int] = [2, 2, 2, 2],
        hidden_channels: list[int] = [16, 32, 64, 128],
        activation_name: str = "prelu",
        block_name: str = "ResBlock",
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Residual net architecture.

        Parameters
        ----------
        num_classes : int, optional
            Number of classification outputs. Default: 4.
        block_groups : list[int], optional
            List of block numbers per group. Every first block
            of a group will downsample the input, except the very
            first block in the net.
        hidden_channels : list[int], optional
            List of hidden channels per block.
        activation_name : str, optional
            Name of the activation function to use.
        available_activation : dict, optional
            Dictionary of available activation functions.
            Format: 'activation_name': nn.modules.activation
        block_name : str, optional
            Name of the ResNet Block.
        """
        super().__init__()

        if block_name not in BLOCKS.keys():
            raise ValueError(f"No block '{block_name}' in BLOCKS dict!")

        if activation_name not in ACTIVATION.keys():
            raise ValueError(
                f"No activation function named '{activation_name}' "
                "in ACTIVATION dict!"
            )

        self.hyperparams = {
            "num_classes": num_classes,
            "hidden_channels": hidden_channels,
            "block_groups": block_groups,
            "activation_name": activation_name,
            "activation": ACTIVATION[activation_name],
            "block_type": BLOCKS[block_name],
            "dropout": dropout,
        }

        self._create_net()
        self._init_params()

    def _create_net(self):
        hidden_channels = self.hyperparams["hidden_channels"]

        if self.hyperparams["block_type"] == PreActBlock:
            self.input = nn.Sequential(
                nn.Conv2d(1, hidden_channels[0], kernel_size=3, padding=1, bias=False),
            )
        else:
            self.input = nn.Sequential(
                nn.Conv2d(1, hidden_channels[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels[0]),
                self.hyperparams["activation"](),
            )

        blocks = []
        for idx, group in enumerate(self.hyperparams["block_groups"]):
            for block in range(group):
                subsample = block == 0 and idx > 0
                blocks.append(
                    self.hyperparams["block_type"](
                        c_in=hidden_channels[idx if not subsample else (idx - 1)],
                        act_fn=self.hyperparams["activation"],
                        subsample=subsample,
                        c_out=hidden_channels[idx],
                        dropout=self.hyperparams["dropout"],
                    )
                )

        self.blocks = nn.Sequential(*blocks)

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels[-1], self.hyperparams["num_classes"]),
        )

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.output(x)
        return x
