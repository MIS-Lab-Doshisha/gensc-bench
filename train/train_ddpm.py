import argparse
from pathlib import Path

import numpy as np
import torch
from torchinfo import summary

from common.util import plot_losses
from common.visualize import plot_matrix_grid
from dataset.graph_dataset import GraphDataset, get_dataloaders, get_sc_dataloader
from model.ddpm_model import Diffuser, DiffusionModel
from model.sc_ddpm_model import Diffuser as DiffuserSC
from model.sc_ddpm_model import UNet as UnetSC
from train.common import get_ddpm_params, get_sc_ddpm_params
from train.ddpm_trainer import optimize_and_train_ddpm


def main(data_path: str, output_dir: str | Path):
    get_param_func = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on dataset: {data_path}, saving to {output_dir}")
    if Path(data_path).parent.name == "sc":
        train_loader, val_loader, _ = get_sc_dataloader(is_matrix=True)
        get_param_func = get_sc_ddpm_params
        diffuser = DiffuserSC(device=device)
        model_class = UnetSC

    else:
        dataset = GraphDataset(data_path, is_matrix=True)
        train_loader, val_loader, _ = get_dataloaders(dataset)
        get_param_func = get_ddpm_params
        diffuser = Diffuser(device=device)
        model_class = DiffusionModel

    for data in train_loader:
        w, h = data[0].shape[-2], data[0].shape[-1]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_loss, val_loss, best_params, trainer = optimize_and_train_ddpm(
        train_loader,
        val_loader,
        ModelClass=model_class,
        diffuser=diffuser,
        get_params_func=get_param_func,
        output_dir=output_dir,
        n_trials=100,
        num_epochs_optim=500,
        num_epochs_final=1000,
        device=device,
    )

    plot_losses({"Train losses": train_loss, "Val losses": val_loss}, output_dir)

    model = trainer.model
    model.load_state_dict(torch.load(f"{output_dir}/model.pth"))
    generated = diffuser.sample(model, (1000, 1, w, h)).squeeze().cpu().numpy()
    # NOTICE: saved matricies, not vectors
    np.savez_compressed(f"{output_dir}/generated_graphs.npz", generated)
    plot_matrix_grid(generated[:9], output_path_name=f"{output_dir}/generated")

    t = torch.randn(1).to(device)
    input_tensor = torch.randn(1, 1, 64, 64).to(device)

    with open(Path(output_dir) / "architecture.txt", "w") as f:
        f.write(repr(summary(trainer.model, input_data=[input_tensor, t])))


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
