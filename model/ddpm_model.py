import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common.graph_util import (
    get_matrices_from_vecs,
    get_vecs_from_matrices,
    vec_to_adj_matrix,
)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    From improved DDPM
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class Diffuser:
    def __init__(
        self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1

        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t)

        model.train()
        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        return mu + noise * std

    def sample_vec(self, model, x_shape=(1, 64, 64)):
        model.eval()
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        x = x.detach().cpu().numpy()
        images = [vec_to_adj_matrix(vec) for vec in x]

        return images

    def sample(self, model, x_shape=(1, 1, 64, 64), copy_tril=True):
        model.eval()
        batch_size = x_shape[0]
        x = torch.randn(x_shape).to(self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        x = torch.clamp(x, min=0)
        # x = x.squeeze().detach().cpu().numpy()

        if copy_tril:
            vec = get_vecs_from_matrices(x.squeeze())
            x = get_matrices_from_vecs(vec)

        return x

    """ For MLP type model. deprecated """
    # def sample(self, model, x_shape=(1, 2016)):
    #     model.eval()
    #     batch_size = x_shape[0]
    #     x = torch.randn(x_shape, device=self.device)

    #     for i in tqdm(range(self.num_timesteps, 0, -1)):
    #         t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
    #         x = self.denoise(model, x, t)

    #     x = x.detach().cpu().numpy()
    #     images = [vec_to_adj_matrix(vec) for vec in x]

    #     return images

    # def sample_image(self, model, x_shape=(1, 28, 28)):
    #     model.eval()
    #     batch_size = x_shape[0]
    #     vec_shape = x_shape[1] * x_shape[2]
    #     x = torch.randn((batch_size, vec_shape), device=self.device)

    #     for i in tqdm(range(self.num_timesteps, 0, -1)):
    #         t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
    #         x = self.denoise(model, x, t)

    #     x = x.detach().cpu().numpy()
    #     images = x.reshape(batch_size, x_shape[1], x_shape[2])

    #     return images


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    from Improved DDPM
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_cosine_beta_schedule(num_diffusion_timesteps):
    return betas_for_alpha_bar(
        num_diffusion_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


class LinearModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = timestep_embedding(t, dim=self.embedding_dim)
        t = self.time_mlp(t)

        x = torch.cat([x, t], dim=1)  # Concatenate time embedding
        x = self.layers(x)

        return x


class DiffusionModelLinear(nn.Module):
    def __init__(
        self, input_dim: int = 2016, hidden_dim: int = 512, latent_dim: int = 32
    ):
        super().__init__()

        self.mlp1 = LinearModule(input_dim, hidden_dim, 128)
        self.mlp2 = LinearModule(hidden_dim, latent_dim, 16)
        self.mlp3 = LinearModule(latent_dim, hidden_dim, 4)
        self.mlp4 = LinearModule(hidden_dim, input_dim, 16)

        self.lastlayers = nn.Sequential(nn.Softplus())

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x: (batch_size, input_dim)
        h1 = self.mlp1(x, t)
        h2 = self.mlp2(h1, t)
        h3 = self.mlp3(h2, t)
        h3 = h1 + h3
        h4 = self.mlp4(h3, t)
        h4 = self.lastlayers(h4)

        return h4


class ConvBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, embedding_dim: int, kernel_size=3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.in_layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        )

        self.mid_layers = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = timestep_embedding(t, dim=self.embedding_dim)
        t = self.time_mlp(t)

        x = self.in_layers(x)
        N = x.size(0)
        t = t.view(N, self.output_dim, 1, 1)

        x = x + t
        x = self.mid_layers(x)
        x = self.out_layers(x)

        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.down(x)

        return x


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)

        return x


"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
DiffusionModel                           [64, 1, 64, 64]           --
├─ConvBlock: 1-1                         [64, 24, 64, 64]          --
│    └─Sequential: 2-1                   [64, 24]                  --
│    │    └─Linear: 3-1                  [64, 24]                  6,168
│    │    └─LeakyReLU: 3-2               [64, 24]                  --
│    │    └─Linear: 3-3                  [64, 24]                  600
│    └─Sequential: 2-2                   [64, 24, 64, 64]          --
│    │    └─Conv2d: 3-4                  [64, 24, 64, 64]          240
│    │    └─BatchNorm2d: 3-5             [64, 24, 64, 64]          48
│    │    └─LeakyReLU: 3-6               [64, 24, 64, 64]          --
│    └─Sequential: 2-3                   [64, 24, 64, 64]          --
│    │    └─Conv2d: 3-7                  [64, 24, 64, 64]          5,208
│    │    └─BatchNorm2d: 3-8             [64, 24, 64, 64]          48
│    │    └─LeakyReLU: 3-9               [64, 24, 64, 64]          --
│    └─Sequential: 2-4                   [64, 24, 64, 64]          --
│    │    └─Conv2d: 3-10                 [64, 24, 64, 64]          5,208
│    │    └─BatchNorm2d: 3-11            [64, 24, 64, 64]          48
│    │    └─LeakyReLU: 3-12              [64, 24, 64, 64]          --
├─DownSample: 1-2                        [64, 24, 32, 32]          --
│    └─Conv2d: 2-5                       [64, 24, 32, 32]          5,208
├─ConvBlock: 1-3                         [64, 48, 32, 32]          --
│    └─Sequential: 2-6                   [64, 48]                  --
│    │    └─Linear: 3-13                 [64, 48]                  12,336
│    │    └─LeakyReLU: 3-14              [64, 48]                  --
│    │    └─Linear: 3-15                 [64, 48]                  2,352
│    └─Sequential: 2-7                   [64, 48, 32, 32]          --
│    │    └─Conv2d: 3-16                 [64, 48, 32, 32]          10,416
│    │    └─BatchNorm2d: 3-17            [64, 48, 32, 32]          96
│    │    └─LeakyReLU: 3-18              [64, 48, 32, 32]          --
│    └─Sequential: 2-8                   [64, 48, 32, 32]          --
│    │    └─Conv2d: 3-19                 [64, 48, 32, 32]          20,784
│    │    └─BatchNorm2d: 3-20            [64, 48, 32, 32]          96
│    │    └─LeakyReLU: 3-21              [64, 48, 32, 32]          --
│    └─Sequential: 2-9                   [64, 48, 32, 32]          --
│    │    └─Conv2d: 3-22                 [64, 48, 32, 32]          20,784
│    │    └─BatchNorm2d: 3-23            [64, 48, 32, 32]          96
│    │    └─LeakyReLU: 3-24              [64, 48, 32, 32]          --
├─DownSample: 1-4                        [64, 48, 16, 16]          --
│    └─Conv2d: 2-10                      [64, 48, 16, 16]          20,784
├─ConvBlock: 1-5                         [64, 96, 16, 16]          --
│    └─Sequential: 2-11                  [64, 96]                  --
│    │    └─Linear: 3-25                 [64, 96]                  24,672
│    │    └─LeakyReLU: 3-26              [64, 96]                  --
│    │    └─Linear: 3-27                 [64, 96]                  9,312
│    └─Sequential: 2-12                  [64, 96, 16, 16]          --
│    │    └─Conv2d: 3-28                 [64, 96, 16, 16]          41,568
│    │    └─BatchNorm2d: 3-29            [64, 96, 16, 16]          192
│    │    └─LeakyReLU: 3-30              [64, 96, 16, 16]          --
│    └─Sequential: 2-13                  [64, 96, 16, 16]          --
│    │    └─Conv2d: 3-31                 [64, 96, 16, 16]          83,040
│    │    └─BatchNorm2d: 3-32            [64, 96, 16, 16]          192
│    │    └─LeakyReLU: 3-33              [64, 96, 16, 16]          --
│    └─Sequential: 2-14                  [64, 96, 16, 16]          --
│    │    └─Conv2d: 3-34                 [64, 96, 16, 16]          83,040
│    │    └─BatchNorm2d: 3-35            [64, 96, 16, 16]          192
│    │    └─LeakyReLU: 3-36              [64, 96, 16, 16]          --
├─DownSample: 1-6                        [64, 96, 8, 8]            --
│    └─Conv2d: 2-15                      [64, 96, 8, 8]            83,040
├─ConvBlock: 1-7                         [64, 192, 8, 8]           --
│    └─Sequential: 2-16                  [64, 192]                 --
│    │    └─Linear: 3-37                 [64, 192]                 49,344
│    │    └─LeakyReLU: 3-38              [64, 192]                 --
│    │    └─Linear: 3-39                 [64, 192]                 37,056
│    └─Sequential: 2-17                  [64, 192, 8, 8]           --
│    │    └─Conv2d: 3-40                 [64, 192, 8, 8]           166,080
│    │    └─BatchNorm2d: 3-41            [64, 192, 8, 8]           384
│    │    └─LeakyReLU: 3-42              [64, 192, 8, 8]           --
│    └─Sequential: 2-18                  [64, 192, 8, 8]           --
│    │    └─Conv2d: 3-43                 [64, 192, 8, 8]           331,968
│    │    └─BatchNorm2d: 3-44            [64, 192, 8, 8]           384
│    │    └─LeakyReLU: 3-45              [64, 192, 8, 8]           --
│    └─Sequential: 2-19                  [64, 192, 8, 8]           --
│    │    └─Conv2d: 3-46                 [64, 192, 8, 8]           331,968
│    │    └─BatchNorm2d: 3-47            [64, 192, 8, 8]           384
│    │    └─LeakyReLU: 3-48              [64, 192, 8, 8]           --
├─UpSample: 1-8                          [64, 192, 16, 16]         --
│    └─Conv2d: 2-20                      [64, 192, 16, 16]         331,968
├─ConvBlock: 1-9                         [64, 96, 16, 16]          --
│    └─Sequential: 2-21                  [64, 96]                  --
│    │    └─Linear: 3-49                 [64, 96]                  24,672
│    │    └─LeakyReLU: 3-50              [64, 96]                  --
│    │    └─Linear: 3-51                 [64, 96]                  9,312
│    └─Sequential: 2-22                  [64, 96, 16, 16]          --
│    │    └─Conv2d: 3-52                 [64, 96, 16, 16]          248,928
│    │    └─BatchNorm2d: 3-53            [64, 96, 16, 16]          192
│    │    └─LeakyReLU: 3-54              [64, 96, 16, 16]          --
│    └─Sequential: 2-23                  [64, 96, 16, 16]          --
│    │    └─Conv2d: 3-55                 [64, 96, 16, 16]          83,040
│    │    └─BatchNorm2d: 3-56            [64, 96, 16, 16]          192
│    │    └─LeakyReLU: 3-57              [64, 96, 16, 16]          --
│    └─Sequential: 2-24                  [64, 96, 16, 16]          --
│    │    └─Conv2d: 3-58                 [64, 96, 16, 16]          83,040
│    │    └─BatchNorm2d: 3-59            [64, 96, 16, 16]          192
│    │    └─LeakyReLU: 3-60              [64, 96, 16, 16]          --
├─UpSample: 1-10                         [64, 96, 32, 32]          --
│    └─Conv2d: 2-25                      [64, 96, 32, 32]          83,040
├─ConvBlock: 1-11                        [64, 48, 32, 32]          --
│    └─Sequential: 2-26                  [64, 48]                  --
│    │    └─Linear: 3-61                 [64, 48]                  12,336
│    │    └─LeakyReLU: 3-62              [64, 48]                  --
│    │    └─Linear: 3-63                 [64, 48]                  2,352
│    └─Sequential: 2-27                  [64, 48, 32, 32]          --
│    │    └─Conv2d: 3-64                 [64, 48, 32, 32]          62,256
│    │    └─BatchNorm2d: 3-65            [64, 48, 32, 32]          96
│    │    └─LeakyReLU: 3-66              [64, 48, 32, 32]          --
│    └─Sequential: 2-28                  [64, 48, 32, 32]          --
│    │    └─Conv2d: 3-67                 [64, 48, 32, 32]          20,784
│    │    └─BatchNorm2d: 3-68            [64, 48, 32, 32]          96
│    │    └─LeakyReLU: 3-69              [64, 48, 32, 32]          --
│    └─Sequential: 2-29                  [64, 48, 32, 32]          --
│    │    └─Conv2d: 3-70                 [64, 48, 32, 32]          20,784
│    │    └─BatchNorm2d: 3-71            [64, 48, 32, 32]          96
│    │    └─LeakyReLU: 3-72              [64, 48, 32, 32]          --
├─UpSample: 1-12                         [64, 48, 64, 64]          --
│    └─Conv2d: 2-30                      [64, 48, 64, 64]          20,784
├─ConvBlock: 1-13                        [64, 24, 64, 64]          --
│    └─Sequential: 2-31                  [64, 24]                  --
│    │    └─Linear: 3-73                 [64, 24]                  6,168
│    │    └─LeakyReLU: 3-74              [64, 24]                  --
│    │    └─Linear: 3-75                 [64, 24]                  600
│    └─Sequential: 2-32                  [64, 24, 64, 64]          --
│    │    └─Conv2d: 3-76                 [64, 24, 64, 64]          15,576
│    │    └─BatchNorm2d: 3-77            [64, 24, 64, 64]          48
│    │    └─LeakyReLU: 3-78              [64, 24, 64, 64]          --
│    └─Sequential: 2-33                  [64, 24, 64, 64]          --
│    │    └─Conv2d: 3-79                 [64, 24, 64, 64]          5,208
│    │    └─BatchNorm2d: 3-80            [64, 24, 64, 64]          48
│    │    └─LeakyReLU: 3-81              [64, 24, 64, 64]          --
│    └─Sequential: 2-34                  [64, 24, 64, 64]          --
│    │    └─Conv2d: 3-82                 [64, 24, 64, 64]          5,208
│    │    └─BatchNorm2d: 3-83            [64, 24, 64, 64]          48
│    │    └─LeakyReLU: 3-84              [64, 24, 64, 64]          --
├─Conv2d: 1-14                           [64, 1, 64, 64]           217
==========================================================================================
Total params: 2,390,617
Trainable params: 2,390,617
Non-trainable params: 0
Total mult-adds (G): 50.84
==========================================================================================
Input size (MB): 1.05
Forward/backward pass size (MB): 1295.53
Params size (MB): 9.56
Estimated Total Size (MB): 1306.14
==========================================================================================
"""


class DiffusionModel(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        base_ch = 24
        self.cb1 = ConvBlock(input_dim, base_ch, embedding_dim=256)
        self.cb2 = ConvBlock(base_ch, base_ch * 2, embedding_dim=256)
        self.cb3 = ConvBlock(base_ch * 2, base_ch * 4, embedding_dim=256)
        self.cb3_1 = ConvBlock(base_ch * 4, base_ch * 8, embedding_dim=256)
        self.cb3_2 = ConvBlock(
            base_ch * 8 + base_ch * 4, base_ch * 4, embedding_dim=256
        )
        self.cb4 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2, embedding_dim=256)
        self.cb5 = ConvBlock(base_ch * 2 + base_ch, base_ch, embedding_dim=256)
        self.last_conv = nn.Conv2d(base_ch, output_dim, kernel_size=3, padding=1)

        self.down1 = DownSample(base_ch)
        self.down2 = DownSample(base_ch * 2)
        self.down3 = DownSample(base_ch * 4)
        self.up0 = UpSample(base_ch * 8)
        self.up1 = UpSample(base_ch * 4)
        self.up2 = UpSample(base_ch * 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x: (batch_size, input_dim)
        h1 = self.cb1(x, t)
        h = self.down1(h1)
        h2 = self.cb2(h, t)
        h = self.down2(h2)
        h3 = self.cb3(h, t)
        h = self.down3(h3)
        h3_1 = self.cb3_1(h, t)
        h3_1 = self.up0(h3_1)
        h3_2 = self.cb3_2(torch.cat([h3_1, h3], dim=1), t)
        h = self.up1(h3_2)
        h = torch.cat([h, h2], dim=1)  # Concatenate skip connection
        h4 = self.cb4(h, t)
        h = self.up2(h4)
        h = torch.cat([h, h1], dim=1)
        h5 = self.cb5(h, t)

        h = self.last_conv(h5)

        return h
