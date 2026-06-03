import argparse
from pathlib import Path

import numpy as np
import torch
from torchinfo import summary

from common.util import plot_matrix_grid
from dataset.graph_dataset import GraphDataset, get_dataloaders, get_sc_dataloader
from gen.generation import generate_graphs_with_gan
from model.gan_model import GanCritic, GanGenerator
from model.sc_gan_model import CriticMLPOpt, GeneratorOpt
from train.common import get_gan_params, get_sc_gan_params
from train.gan_trainer import optimize_and_train_gan


def main(data_path, output_dir):
    get_param_func = None
    print(f"Training on dataset: {data_path}, saving to {output_dir}")
    if Path(data_path).parent.name == "sc":
        train_loader, val_loader, _ = get_sc_dataloader()
        get_param_func = get_sc_gan_params
        generator_class = GeneratorOpt
        critic_class = CriticMLPOpt
    else:
        dataset = GraphDataset(data_path, is_matrix=False)
        train_loader, val_loader, _ = get_dataloaders(dataset)
        get_param_func = get_gan_params
        generator_class = GanGenerator
        critic_class = GanCritic
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for data in train_loader:
        input_dim = data[0].shape[0]
        break

    print("Starting GAN optimization and training...")
    best_params, trainer = optimize_and_train_gan(
        train_loader,
        val_loader,
        GeneratorClass=generator_class,
        CriticClass=critic_class,
        input_dim=input_dim,
        get_params_func=get_param_func,
        output_dir=output_dir,
        n_trials=100,
        num_epochs_optim=500,
        num_epochs_final=1000,
    )

    model = trainer.generator
    model.load_state_dict(
        torch.load(Path(output_dir) / "model.pth", map_location="cpu")
    )
    model.to(trainer.device)
    generated = generate_graphs_with_gan(model, num_samples=1000, device=trainer.device)
    np.savez_compressed(f"{output_dir}/generated_graphs.npz", generated)
    plot_matrix_grid(generated[:9], output_path_name=f"{output_dir}/generated")

    summary(
        trainer.generator,
        input_data=(torch.randn(2, best_params["latent_dim"]).to(trainer.device),),
    )
    summary(trainer.critic, input_data=(torch.randn(2, input_dim).to(trainer.device),))

    with open(Path(output_dir) / "architecture.txt", "w") as f:
        f.write(
            repr(
                summary(
                    trainer.generator,
                    input_data=(
                        torch.randn(2, best_params["latent_dim"]).to(trainer.device),
                    ),
                )
            )
        )
        f.write("\n\n")
        f.write(
            repr(
                summary(
                    trainer.critic,
                    input_data=(torch.randn(2, input_dim).to(trainer.device),),
                )
            )
        )

    # paths_manager = PathManager("thresh_path_list.json")
    # paths = paths_manager.get_all_original_paths()

    # for path in paths:
    #     set_seed(42)
    #     print(f"Training on dataset: {path}")
    #     dataset = GraphDataset(path, is_matrix=False)
    #     train_loader, val_loader, _ = get_dataloaders(dataset)
    #     output_dir = "checkpoints/gan/" + Path(path).stem
    #     Path(output_dir).mkdir(parents=True, exist_ok=True)

    #     print("Starting GAN optimization and training...")
    #     best_params, trainer = optimize_and_train_gan(
    #         train_loader,
    #         val_loader,
    #         input_dim=2016,
    #         TrainerClass=GanTrainer,
    #         get_params_func=get_gan_params,
    #         output_dir = output_dir
    #     )

    #     model = trainer.generator
    #     model.load_state_dict(torch.load(Path(output_dir) / "model.pth", map_location="cpu"))
    #     model.to(trainer.device)
    #     generated = generate_graphs_with_gan(model, num_samples = 1000, device=trainer.device)
    #     np.savez_compressed(f"{output_dir}/generated_graphs.npz", generated)
    #     plot_matrix_grid(generated[:9], output_path_name=f"{output_dir}/generated")


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
