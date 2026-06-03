import torch
import torch.nn as nn

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
VaeModel                                 [64, 2016]                --
├─VaeEncoder: 1-1                        [64, 16]                  --
│    └─Sequential: 2-1                   [64, 512]                 --
│    │    └─Linear: 3-1                  [64, 512]                 1,032,704
│    │    └─BatchNorm1d: 3-2             [64, 512]                 1,024
│    │    └─LeakyReLU: 3-3               [64, 512]                 --
│    └─Linear: 2-2                       [64, 16]                  8,208
│    └─Linear: 2-3                       [64, 16]                  8,208
├─VaeDecoder: 1-2                        [64, 2016]                --
│    └─Sequential: 2-4                   [64, 2016]                --
│    │    └─Linear: 3-4                  [64, 512]                 8,704
│    │    └─BatchNorm1d: 3-5             [64, 512]                 1,024
│    │    └─LeakyReLU: 3-6               [64, 512]                 --
│    │    └─Dropout: 3-7                 [64, 512]                 --
│    │    └─Linear: 3-8                  [64, 2016]                1,034,208
│    └─Softplus: 2-5                     [64, 2016]                --
==========================================================================================
Total params: 2,094,080
Trainable params: 2,094,080
Non-trainable params: 0
Total mult-adds (M): 134.02
==========================================================================================
Input size (MB): 0.52
Forward/backward pass size (MB): 2.10
Params size (MB): 8.38
Estimated Total Size (MB): 10.99
================
"""

class VaeEncoder(nn.Module):
    def __init__(self, input_dim: int = 2016, hidden_dim_1: int = 512, hidden_dim_2: int = 512, latent_dim: int = 32):
        super(VaeEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.LeakyReLU()
        )

        self.out_mu = nn.Linear(hidden_dim_1, latent_dim)
        self.out_logvar = nn.Linear(hidden_dim_1, latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, input_dim)
        x = self.layers(x)
        mu = self.out_mu(x)
        logvar = self.out_logvar(x)
        std = torch.exp(0.5 * logvar)

        return mu, std


class VaeDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim_2: int = 512, hidden_dim_1: int = 512, output_dim: int = 2016, dropout_rate: float = 0.5):
        super(VaeDecoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, output_dim),
        )

        self.lastlayer = nn.Softplus()

    def forward(self, z: torch.Tensor):
        # z: (batch_size, latent_dim)
        x = self.layers(z)
        x = self.lastlayer(x)

        return x


class VaeModel(nn.Module):
    def __init__(self, input_dim: int = 2016, hidden_dim_1: int = 512, hidden_dim_2: int = 512, latent_dim: int = 32):
        super(VaeModel, self).__init__()
        self.encoder = VaeEncoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim)
        self.decoder = VaeDecoder(latent_dim, hidden_dim_2, hidden_dim_1, input_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        mu, std = self.encoder(x)
        z = mu + std * torch.randn_like(std)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, std
