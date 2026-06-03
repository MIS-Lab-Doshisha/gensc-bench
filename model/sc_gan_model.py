import numpy as np
import torch.nn as nn

"""
blocks = 2, latent_dim = 8のとき
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
DecoderOpt                               [1, 6670]                 --
├─Sequential: 1-1                        [1, 6670]                 --
│    └─Linear: 2-1                       [1, 64]                   576
│    └─BatchNorm1d: 2-2                  [1, 64]                   128
│    └─ReLU: 2-3                         [1, 64]                   --
│    └─Linear: 2-4                       [1, 6670]                 433,550
├─Softplus: 1-2                          [1, 6670]                 --
==========================================================================================
Total params: 434,254
Trainable params: 434,254
Non-trainable params: 0
Total mult-adds (M): 0.43
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 1.74
Estimated Total Size (MB): 1.79
======================================

blocks = 2
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CriticMLPOpt                             [1, 1]                    --
├─Sequential: 1-1                        [1, 1]                    --
│    └─Linear: 2-1                       [1, 6670]                 44,495,570
│    └─ReLU: 2-2                         [1, 6670]                 --
│    └─Linear: 2-3                       [1, 1]                    6,671
==========================================================================================
Total params: 44,502,241
Trainable params: 44,502,241
Non-trainable params: 0
Total mult-adds (M): 44.50
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 0.05
Params size (MB): 178.01
Estimated Total Size (MB): 178.09
==========================================================================================
"""


class GeneratorOpt(nn.Module):
    def __init__(
        self,
        latent_dim,
        generator_n_blocks,
        output_dim,
        first_hidden_dim=512,
        last_hidden_dim=3335,
    ):
        super().__init__()
        if generator_n_blocks < 2:
            raise ValueError(
                "n_blocks must be at least 2 to form a decoder with hidden layers."
            )
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers = []

        layers.append(nn.Linear(latent_dim, first_hidden_dim))
        layers.append(nn.BatchNorm1d(first_hidden_dim))
        layers.append(nn.ReLU())

        current_dim = first_hidden_dim

        remaining_blocks = generator_n_blocks - 2

        if remaining_blocks > 0:
            dims = np.geomspace(
                first_hidden_dim, last_hidden_dim, num=remaining_blocks + 1, dtype=int
            )

            for i in range(1, remaining_blocks + 1):
                layers.append(nn.Linear(current_dim, dims[i]))
                layers.append(nn.BatchNorm1d(dims[i]))
                layers.append(nn.ReLU())
                current_dim = dims[i]

        layers.append(nn.Linear(current_dim, output_dim))
        self.hidden_layers = nn.Sequential(*layers)
        self.last_layer = nn.Softplus()

    def forward(self, x):
        h = self.hidden_layers(x)
        h = self.last_layer(h)

        return h


class CriticMLPOpt(nn.Module):
    def __init__(
        self, input_dim, critic_n_blocks, first_hidden_dim=3335, last_hidden_dim=512
    ):
        """
        Initialize the MLP-based critic with a configurable number of blocks.

        Args:
            input_size (int): Size of the input vector.
            n_blocks (int): Number of blocks in the MLP.
            base_hidden_size (int): Base hidden size for the first block, subsequent blocks will increase in size.
        """
        super().__init__()
        if critic_n_blocks < 2:
            raise ValueError(
                "n_blocks must be at least 2 to form a decoder with hidden layers."
            )
        self.input_dim = input_dim

        layers = []

        layers.append(nn.Linear(input_dim, first_hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.4))

        current_dim = first_hidden_dim

        remaining_blocks = critic_n_blocks - 2

        if remaining_blocks > 0:
            dims = np.geomspace(
                first_hidden_dim, last_hidden_dim, num=remaining_blocks + 1, dtype=int
            )

            for i in range(1, remaining_blocks + 1):
                layers.append(nn.Linear(current_dim, dims[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.4))
                current_dim = dims[i]

        layers.append(nn.Linear(current_dim, 1))
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP-based critic.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.hidden_layers(x)
