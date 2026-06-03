import torch
from torch import nn
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from common.graph_util import get_matrices_from_vecs, get_vecs_from_matrices

def _pos_encoding(t, output_dim, device):
    v = torch.zeros(output_dim).to(device)
    i = torch.arange(0, output_dim).to(device)
    div_term = 10000 ** (i / output_dim)

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])

    return v


def pos_encoding(ts, output_dim, device):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim).to(device)

    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)

    return v

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


"""
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
UNet                                     --
├─ConvBlock: 1-1                         --
│    └─Sequential: 2-1                   --
│    │    └─Conv2d: 3-1                  160
│    │    └─BatchNorm2d: 3-2             32
│    │    └─ReLU: 3-3                    --
│    │    └─Conv2d: 3-4                  2,320
│    │    └─BatchNorm2d: 3-5             32
│    │    └─ReLU: 3-6                    --
│    └─Sequential: 2-2                   --
│    │    └─Linear: 3-7                  101
│    │    └─ReLU: 3-8                    --
│    │    └─Linear: 3-9                  2
├─ConvBlock: 1-2                         --
│    └─Sequential: 2-3                   --
│    │    └─Conv2d: 3-10                 4,640
│    │    └─BatchNorm2d: 3-11            64
│    │    └─ReLU: 3-12                   --
│    │    └─Conv2d: 3-13                 9,248
│    │    └─BatchNorm2d: 3-14            64
│    │    └─ReLU: 3-15                   --
│    └─Sequential: 2-4                   --
│    │    └─Linear: 3-16                 1,616
│    │    └─ReLU: 3-17                   --
│    │    └─Linear: 3-18                 272
├─ConvBlock: 1-3                         --
│    └─Sequential: 2-5                   --
│    │    └─Conv2d: 3-19                 18,496
│    │    └─BatchNorm2d: 3-20            128
│    │    └─ReLU: 3-21                   --
│    │    └─Conv2d: 3-22                 36,928
│    │    └─BatchNorm2d: 3-23            128
│    │    └─ReLU: 3-24                   --
│    └─Sequential: 2-6                   --
│    │    └─Linear: 3-25                 3,232
│    │    └─ReLU: 3-26                   --
│    │    └─Linear: 3-27                 1,056
├─ConvBlock: 1-4                         --
│    └─Sequential: 2-7                   --
│    │    └─Conv2d: 3-28                 73,856
│    │    └─BatchNorm2d: 3-29            256
│    │    └─ReLU: 3-30                   --
│    │    └─Conv2d: 3-31                 147,584
│    │    └─BatchNorm2d: 3-32            256
│    │    └─ReLU: 3-33                   --
│    └─Sequential: 2-8                   --
│    │    └─Linear: 3-34                 6,464
│    │    └─ReLU: 3-35                   --
│    │    └─Linear: 3-36                 4,160
├─ConvBlock: 1-5                         --
│    └─Sequential: 2-9                   --
│    │    └─Conv2d: 3-37                 295,168
│    │    └─BatchNorm2d: 3-38            512
│    │    └─ReLU: 3-39                   --
│    │    └─Conv2d: 3-40                 590,080
│    │    └─BatchNorm2d: 3-41            512
│    │    └─ReLU: 3-42                   --
│    └─Sequential: 2-10                  --
│    │    └─Linear: 3-43                 12,928
│    │    └─ReLU: 3-44                   --
│    │    └─Linear: 3-45                 16,512
├─ConvBlock: 1-6                         --
│    └─Sequential: 2-11                  --
│    │    └─Conv2d: 3-46                 442,496
│    │    └─BatchNorm2d: 3-47            256
│    │    └─ReLU: 3-48                   --
│    │    └─Conv2d: 3-49                 147,584
│    │    └─BatchNorm2d: 3-50            256
│    │    └─ReLU: 3-51                   --
│    └─Sequential: 2-12                  --
│    │    └─Linear: 3-52                 38,784
│    │    └─ReLU: 3-53                   --
│    │    └─Linear: 3-54                 147,840
├─ConvBlock: 1-7                         --
│    └─Sequential: 2-13                  --
│    │    └─Conv2d: 3-55                 110,656
│    │    └─BatchNorm2d: 3-56            128
│    │    └─ReLU: 3-57                   --
│    │    └─Conv2d: 3-58                 36,928
│    │    └─BatchNorm2d: 3-59            128
│    │    └─ReLU: 3-60                   --
│    └─Sequential: 2-14                  --
│    │    └─Linear: 3-61                 19,392
│    │    └─ReLU: 3-62                   --
│    │    └─Linear: 3-63                 37,056
├─ConvBlock: 1-8                         --
│    └─Sequential: 2-15                  --
│    │    └─Conv2d: 3-64                 27,680
│    │    └─BatchNorm2d: 3-65            64
│    │    └─ReLU: 3-66                   --
│    │    └─Conv2d: 3-67                 9,248
│    │    └─BatchNorm2d: 3-68            64
│    │    └─ReLU: 3-69                   --
│    └─Sequential: 2-16                  --
│    │    └─Linear: 3-70                 9,696
│    │    └─ReLU: 3-71                   --
│    │    └─Linear: 3-72                 9,312
├─ConvBlock: 1-9                         --
│    └─Sequential: 2-17                  --
│    │    └─Conv2d: 3-73                 6,928
│    │    └─BatchNorm2d: 3-74            32
│    │    └─ReLU: 3-75                   --
│    │    └─Conv2d: 3-76                 2,320
│    │    └─BatchNorm2d: 3-77            32
│    │    └─ReLU: 3-78                   --
│    └─Sequential: 2-18                  --
│    │    └─Linear: 3-79                 4,848
│    │    └─ReLU: 3-80                   --
│    │    └─Linear: 3-81                 2,352
├─Conv2d: 1-10                           17
├─MaxPool2d: 1-11                        --
├─MaxPool2d: 1-12                        --
├─Upsample: 1-13                         --
├─Upsample: 1-14                         --
├─Upsample: 1-15                         --
=================================================================
Total params: 2,280,904
Trainable params: 2,280,904
Non-trainable params: 0
=================================================================
"""

class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, base_ch, time_embed_dim)
        self.down2 = ConvBlock(base_ch, base_ch * 2, time_embed_dim)
        self.down3 = ConvBlock(base_ch * 2, base_ch * 4, time_embed_dim)
        self.down4 = ConvBlock(base_ch * 4, base_ch * 8, time_embed_dim)
        self.bot1 = ConvBlock(base_ch * 8, base_ch * 16, time_embed_dim)
        self.up4 = ConvBlock(base_ch * 8 + base_ch * 16, base_ch * 8, time_embed_dim)
        self.up3 = ConvBlock(base_ch * 4 + base_ch * 8, base_ch * 4, time_embed_dim)
        self.up2 = ConvBlock(base_ch * 2 + base_ch * 4, base_ch * 2, time_embed_dim)
        self.up1 = ConvBlock(base_ch + base_ch * 2, base_ch, time_embed_dim)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.maxpool_pad1 = nn.MaxPool2d(2, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_15 = nn.Upsample(size=15, mode="bilinear")
        self.upsample_29 = nn.Upsample(size=29, mode="bilinear")

    def forward(self, x, timesteps):
        batch_size = len(x)
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        x3 = self.down3(x, v)
        x = self.maxpool_pad1(x3)
        x4 = self.down4(x, v)
        x = self.maxpool_pad1(x4)
        x = self.bot1(x, v)

        x = self.upsample_15(x)
        x = torch.cat([x, x4], dim=1)
        x = self.up4(x, v)
        x = self.upsample_29(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)

        return x


class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all()  and (t <= T).all()
        t_idx = t - 1

        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all()  and (t <= T).all()

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

    def reverse_to_img(self, x):
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(4, 1, 116, 116), chunk_size=250):
        batch_size, _, n, _ = x_shape
        all_matrices = []
        all_x = torch.randn(x_shape, device=self.device)

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            x = all_x[start_idx:end_idx]
            current_batch_size = x.shape[0]

            for i in tqdm(range(self.num_timesteps, 0, -1)):
                t = torch.tensor([i] * current_batch_size, device=self.device, dtype=torch.long)
                with torch.no_grad():
                    x = self.denoise(model, x, t)

            x = x.squeeze()
            vecs = get_vecs_from_matrices(x)
            matrices = get_matrices_from_vecs(vecs)
            matrices = matrices.unsqueeze(1).to(self.device)
            all_matrices.append(matrices.cpu())

        final_matrices = torch.cat(all_matrices, dim=0)
        final_matrices = torch.clamp(final_matrices, min=0)
        print(final_matrices.shape)

        return final_matrices

    def sample_vec(self, model, x=None, x_shape=(4, 1, 116, 116)):
        batch_size = x_shape[0]

        if x == None:
            x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        x = x.squeeze().cpu()
        vecs = get_vecs_from_matrices(x)

        # 最小値を0にする
        torch.clamp(vecs, min=0) # type: ignore

        images = vecs.numpy()
        return images