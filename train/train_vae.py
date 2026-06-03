import argparse
from pathlib import Path

import numpy as np
import torch
from torchinfo import summary

from common.util import plot_losses
from common.visualize import plot_matrix_grid
from dataset.graph_dataset import GraphDataset, get_dataloaders, get_sc_dataloader
from gen.generation import generate_graphs_with_vae
from model.sc_vae_model import VAE as SCVAE
from model.vae_model import VaeModel
from train.common import get_sc_vae_params, get_vae_params
from train.vae_trainer import optimize_and_train_vae


def main(data_path, output_dir):
    get_param_func = None
    print(f"Training on dataset: {data_path}, saving to {output_dir}")
    if Path(data_path).parent.name == "sc":
        train_loader, val_loader, _ = get_sc_dataloader()
        get_param_func = get_sc_vae_params
        model_class = SCVAE
    else:
        dataset = GraphDataset(data_path, is_matrix=False)
        train_loader, val_loader, _ = get_dataloaders(dataset)
        get_param_func = get_vae_params
        model_class = VaeModel

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for data in train_loader:
        input_dim = data[0].shape[0]
        break

    train_loss, val_loss, best_params, trainer = optimize_and_train_vae(
        train_loader,
        val_loader,
        ModelClass=model_class,
        get_params_func=get_param_func,
        output_dir=output_dir,
        input_dim=input_dim,
        n_trials=100,
        num_epochs_optim=500,
        num_epochs_final=1000,
    )

    plot_losses({"Train losses": train_loss, "Val losses": val_loss}, output_dir)

    model = trainer.model
    model.load_state_dict(torch.load(f"{output_dir}/model.pth"))
    generated = generate_graphs_with_vae(model, num_samples=1000, device=trainer.device)
    np.savez_compressed(f"{output_dir}/generated_graphs.npz", generated)
    plot_matrix_grid(generated[:9], output_path_name=f"{output_dir}/generated")

    with open(Path(output_dir) / "architecture.txt", "w") as f:
        f.write(
            repr(
                summary(
                    trainer.model.encoder,
                    input_data=(torch.randn(1, input_dim).to(trainer.device),),
                )
            )
        )
        f.write("\n\n")
        f.write(
            repr(
                summary(
                    trainer.model.decoder,
                    input_data=(
                        torch.randn(1, best_params["latent_dim"]).to(trainer.device),
                    ),
                )
            )
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input dataset.",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs.",
    )
    args = argparser.parse_args()
    main(args.data_path, args.output_dir)
