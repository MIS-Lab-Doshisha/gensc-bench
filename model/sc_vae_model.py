import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderOpt(nn.Module):
    def __init__(self, input_dim, latent_dim, n_blocks, first_hidden_dim=3335, final_body_dim=256):
        super().__init__()
        layers = []

        num_layers = n_blocks - 1
        if num_layers <= 0:
            raise ValueError("n_blocks must be at least 2 to form an encoder with hidden layers.")

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, final_body_dim))
            layers.append(nn.BatchNorm1d(final_body_dim))
            layers.append(nn.ReLU())
            current_dim_for_head = final_body_dim

        else:
            layers.append(nn.Linear(input_dim, first_hidden_dim))
            layers.append(nn.BatchNorm1d(first_hidden_dim))
            layers.append(nn.ReLU())

            residual_blocks = num_layers - 1

            dims = np.geomspace(first_hidden_dim, final_body_dim, num = residual_blocks + 1, dtype=int)

            for i in range(residual_blocks):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
            current_dim_for_head = dims[-1]

        self.hidden_layers = nn.Sequential(*layers)
        self.linear_mu = nn.Linear(current_dim_for_head, latent_dim)
        self.linear_logvar = nn.Linear(current_dim_for_head, latent_dim)


    def forward(self, x):
        h = self.hidden_layers(x)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class DecoderOpt(nn.Module):
    def __init__(self, latent_dim, output_dim, n_blocks, first_hidden_dim=512, last_hidden_dim=3335):
        super().__init__()
        if n_blocks < 2:
            raise ValueError("n_blocks must be at least 2 to form a decoder with hidden layers.")
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers = []

        layers.append(nn.Linear(latent_dim, first_hidden_dim))
        layers.append(nn.BatchNorm1d(first_hidden_dim))
        layers.append(nn.ReLU())

        current_dim = first_hidden_dim

        remaining_blocks = n_blocks - 2

        if remaining_blocks > 0:
            dims = np.geomspace(first_hidden_dim, last_hidden_dim, num=remaining_blocks + 1, dtype=int)

            for i in range(1, remaining_blocks + 1):
                layers.append(nn.Linear(current_dim, dims[i]))
                layers.append(nn.BatchNorm1d(dims[i]))
                layers.append(nn.ReLU())
                current_dim = dims[i]

        layers.append(nn.Linear(current_dim, output_dim))

        self.hidden_layers = nn.Sequential(*layers)
        self.last_layer = nn.Softplus()

    def forward(self, z):
        h = self.hidden_layers(z)
        x_hat = self.last_layer(h)
        #x_hat = torch.sigmoid(x_hat)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_n_blocks, decoder_n_blocks, device="cpu"):
        super().__init__()
        self.encoder = EncoderOpt(input_dim, latent_dim, encoder_n_blocks).to(device)
        self.decoder = DecoderOpt(latent_dim, input_dim, decoder_n_blocks).to(device)
        self.latent_dim = latent_dim

    def forward(self, x):
        mu, std = self.encoder(x)
        z = mu + std * torch.randn_like(std)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, std
