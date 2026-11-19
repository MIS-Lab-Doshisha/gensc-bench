import numpy as np
import torch
from torch import nn

from common.graph_util import vec_to_adj_matrix


def generate_graphs_with_vae(
    model: nn.Module,
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 90,
):
    """
    Generate graphs using a trained VAE model.

    Args:
        model (VaeModel): The trained VAE model.
        num_samples (int): Number of graphs to generate.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        list: List of generated adjacency matrices.
    """
    model.eval()
    model.to(device)
    latent_dim = model.latent_dim
    generated_graphs = []

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, generator=gen, device=device)
        x_hat = model.decoder(z)
        generated_graphs = [
            vec_to_adj_matrix(x_hat[i].squeeze()).cpu().numpy()
            for i in range(num_samples)
        ]

    return generated_graphs


def generate_graphs_with_gan(
    generator: nn.Module,
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int | None = 91,
    z: torch.Tensor | None = None,
) -> np.ndarray:
    generator.eval()
    generator.to(device)
    latent_dim = generator.latent_dim
    generated_graphs = []

    with torch.no_grad():
        if z is None:
            if seed is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                z = torch.randn(num_samples, latent_dim, generator=gen, device=device)
            else:
                z = torch.randn(num_samples, latent_dim).to(device)
        else:
            z = z.to(device)
        x_hat = generator(z)
        generated_graphs = [
            vec_to_adj_matrix(x_hat[i].squeeze()).cpu().numpy()
            for i in range(num_samples)
        ]

    generated_graphs = np.array(generated_graphs)
    return generated_graphs
