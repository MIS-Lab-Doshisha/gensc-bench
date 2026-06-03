import torch
import torch.nn as nn
import numpy as np
import networkx as nx

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GanGenerator                             [64, 2016]                --
├─Sequential: 1-1                        [64, 2016]                --
│    └─Linear: 2-1                       [64, 512]                 8,704
│    └─BatchNorm1d: 2-2                  [64, 512]                 1,024
│    └─LeakyReLU: 2-3                    [64, 512]                 --
│    └─Linear: 2-4                       [64, 2016]                1,034,208
├─Softplus: 1-2                          [64, 2016]                --
==========================================================================================
Total params: 1,043,936
Trainable params: 1,043,936
Non-trainable params: 0
Total mult-adds (M): 66.81
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 1.56
Params size (MB): 4.18
Estimated Total Size (MB): 5.74
==========================================================================================
"""

class GanGenerator(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 512, output_dim: int = 2016):
        super(GanGenerator, self).__init__()

        self.latent_dim = latent_dim

        self.layers = nn.Sequential(        
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.lastlayer = nn.Softplus()

    def forward(self, z: torch.Tensor):
        # z: (batch_size, latent_dim)
        x = self.layers(z)
        x = self.lastlayer(x)

        return x
    
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GanCritic                                [64, 1]                   --
├─Sequential: 1-1                        [64, 1]                   --
│    └─Linear: 2-1                       [64, 512]                 1,032,704
│    └─LeakyReLU: 2-2                    [64, 512]                 --
│    └─Dropout: 2-3                      [64, 512]                 --
│    └─Linear: 2-4                       [64, 1]                   513
==========================================================================================
Total params: 1,033,217
Trainable params: 1,033,217
Non-trainable params: 0
Total mult-adds (M): 66.13
==========================================================================================
Input size (MB): 0.52
Forward/backward pass size (MB): 0.26
Params size (MB): 4.13
Estimated Total Size (MB): 4.91
==========================================================================================
"""

class GanCritic(nn.Module):
    def __init__(self, input_dim: int = 2016, hidden_dim: int = 512):
        super(GanCritic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, input_dim)
        return self.layers(x)
    

