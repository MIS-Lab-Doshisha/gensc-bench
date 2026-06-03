from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np
from common.util import set_seed
from common.graph_util import vec_to_adj_matrix, matrix_to_vec, get_vecs_from_matrices
from common.visualize import plot_matrix_grid
from model.vae_model import VaeModel
from model.gan_model import GanGenerator, GanCritic

def generate_graphs_with_vae(
        model: nn.Module,
        num_samples: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    set_seed(90)

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        x_hat = model.decoder(z)
        generated_graphs = [vec_to_adj_matrix(x_hat[i].squeeze()).cpu().numpy() for i in range(num_samples)]

    return generated_graphs


def generate_images_with_vae(
        model: VaeModel,
        size: tuple, # Size of the generated images (height, width)
        num_samples: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate images using a trained VAE model.

    Args:
        model (VaeModel): The trained VAE model.
        num_samples (int): Number of images to generate.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        list: List of generated images as tensors.
    """
    model.eval()
    model.to(device)
    latent_dim = model.latent_dim
    generated_images = []
    set_seed(90)

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        x_hat = model.decoder(z)
        generated_images = [x_hat[i].squeeze().cpu().numpy() for i in range(num_samples)]

    generated_images = np.array(generated_images).reshape(num_samples, *size)

    return generated_images


def generate_graphs_with_gan(
        generator: nn.Module,
        num_samples: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    generator.eval()
    generator.to(device)
    latent_dim = generator.latent_dim
    generated_graphs = []
    set_seed(91)

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        x_hat = generator(z)
        generated_graphs = [vec_to_adj_matrix(x_hat[i].squeeze()).cpu().numpy() for i in range(num_samples)]

    generated_graphs = np.array(generated_graphs)
    return generated_graphs


def generate_all_with_gan_vae():
    output_path_parent = ["checkpoints/continuous/m8/", "checkpoints/binary/ba/", "checkpoints/binary/planar/", "checkpoints/binary/sbm/", "checkpoints/binary/ws/"]

    """ Train models on graphs """
    for model_name in ["vae", "gan"]:
        for output_path in output_path_parent:
            OUTPUT_DIR = Path(output_path) / model_name
            print(f"saving to {OUTPUT_DIR}")

            if model_name == "vae":
                params_path = OUTPUT_DIR / "vae_params.json"
                with open(params_path, 'r') as f:
                    params = json.load(f)

                model = VaeModel(latent_dim=params["latent_dim"]).to(device)
                model.load_state_dict(torch.load(f"{OUTPUT_DIR}/vae_model.pth"))
                generated = generate_graphs_with_vae(model, num_samples=1000)
            elif model_name == "gan":
                model = GanGenerator(latent_dim=16).to(device)
                model.load_state_dict(torch.load(f"{OUTPUT_DIR}/gan_generator.pth"))
                generated = generate_graphs_with_gan(model, num_samples=1000, device=device)

            np.savez_compressed(f"{OUTPUT_DIR}/generated_graphs.npz", generated)
            plot_matrix_grid(generated[:9], output_path_name=f"{OUTPUT_DIR}/generated")
