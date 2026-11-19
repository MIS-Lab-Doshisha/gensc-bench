import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib.colors import PowerNorm
from torchinfo import summary

from dataset.graph_dataset import setup_dataloader_from_config

matplotlib.use("Agg")

from common.util import plot_losses, set_seed, setup_logger
from common.visualize import plot_matrix_grid
from gen.generation import generate_graphs_with_gan
from model.gan_model import GanCritic, GanGenerator
from model.sc_gan_model import CriticMLPOpt, GeneratorOpt
from train.common import (
    filter_params,
    get_gan_params,
    get_sc_gan_params,
)

# from train.gan_trainer import GanTrainer, optimize_and_train_gan
from train.gan_trainer_new import GanTrainer, optimize_and_train_gan

GAN_MODEL_CLASSES = {
    "synthetic": (GanGenerator, GanCritic, get_gan_params),
    "sc": (GeneratorOpt, CriticMLPOpt, get_sc_gan_params),
    # "hcp": (HcpGeneratorOpt, HcpCriticMLPOpt, get_hcp_gan_params),
}


def main(config, output_dir):
    print(
        f"Training on dataset: {config['dataset']['type']}, {config['dataset'].get('name', '')}, saving to {output_dir}"
    )

    train_loader, val_loader, *_ = setup_dataloader_from_config(config, is_matrix=False)
    generator_class, critic_class, get_param_func = GAN_MODEL_CLASSES[
        config["dataset"]["type"]
    ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for data in train_loader:
        input_dim = data[0].shape[0]
        break

    print("Starting GAN optimization and training...")
    best_params, trainer = optimize_and_train_gan(
        train_loader,
        val_loader,
        config=config,
        GeneratorClass=generator_class,
        CriticClass=critic_class,
        input_dim=input_dim,
        get_params_func=get_param_func,
        output_dir=output_dir,
    )

    output_dir_run = Path(output_dir) / "run_0"

    model = trainer.generator
    model.load_state_dict(torch.load(f"{output_dir_run}/model.pth", map_location="cpu"))
    model.to(trainer.device)
    generated = generate_graphs_with_gan(model, num_samples=1000, device=trainer.device)
    np.savez_compressed(f"{output_dir_run}/generated_graphs.npz", generated)
    plot_matrix_grid(generated[:9], output_path_name=f"{output_dir_run}/generated")

    summary(
        trainer.generator,
        input_data=(torch.randn(1, best_params["latent_dim"]).to(trainer.device),),
    )
    summary(trainer.critic, input_data=(torch.randn(1, input_dim).to(trainer.device),))

    with open(output_dir_run / "architecture.txt", "w") as f:
        f.write(
            repr(
                summary(
                    trainer.generator,
                    input_data=(
                        torch.randn(1, best_params["latent_dim"]).to(trainer.device),
                    ),
                )
            )
        )
        f.write("\n\n")
        f.write(
            repr(
                summary(
                    trainer.critic,
                    input_data=(torch.randn(1, input_dim).to(trainer.device),),
                )
            )
        )


def main_seed(config, output_dir):
    print(f"Training on dataset: {config['dataset']['type']}, saving to {output_dir}")

    generator_class, critic_class, get_param_func = GAN_MODEL_CLASSES[
        config["dataset"]["type"]
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, run in enumerate(config["training"]["runs"]):
        run_seed = run["seed"]
        fold = run["fold"]
        output_dir_seed = Path(output_dir) / f"run_{i + 1}"

        logger = setup_logger(output_dir_seed / "training.log")
        logger.info(f"Running training with seed: {run_seed}, fold: {fold}")

        train_loader, val_loader, *_ = setup_dataloader_from_config(
            config, is_matrix=False, fold=fold
        )
        for data in train_loader:
            input_dim = data[0].shape[0]
            break

        Path(output_dir_seed).mkdir(parents=True, exist_ok=True)

        fixed_params = config["model"]
        with open(Path(config["output"]["path"]) / "best_params.json", "r") as f:
            best_params = json.load(f)

        params = {**fixed_params, **best_params}

        generator_params = filter_params(params, generator_class)
        critic_params = filter_params(params, critic_class)
        trainer_params = filter_params(params, GanTrainer)

        set_seed(run_seed)

        generator = generator_class(**generator_params)
        critic = critic_class(**critic_params)

        trainer = GanTrainer(
            **trainer_params,
            generator=generator,
            critic=critic,
            num_epochs=config["training"]["num_epoch_final"],
            device=device,
            output_dir=output_dir_seed,
            logger=logger,
        )

        trainer.train(train_loader, val_loader)
        plot_losses(
            {
                "Generator losses": trainer.generator_train_losses,
                "Critic train losses": trainer.critic_train_losses,
                "Critic val losses": trainer.critic_val_losses,
            },
            output_dir_seed,
        )
        plot_losses({"MMD": trainer.mmds}, output_dir_seed, name="mmd_losses")

        generator.load_state_dict(
            torch.load(Path(output_dir_seed) / "model.pth", map_location="cpu")
        )
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        generated = generate_graphs_with_gan(
            generator, num_samples=1000, device=trainer.device
        )

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MiB
            logger.info(f"Peak GPU memory usage: {peak_memory:.4f} MiB")
        elapsed_time = end_time - start_time
        logger.info(f"Sampling time for 1000 samples: {elapsed_time:.4f} seconds")

        np.savez_compressed(f"{output_dir_seed}/generated_graphs.npz", generated)

        norm = (
            PowerNorm(gamma=0.3) if config["dataset"]["type"] != "synthetic" else None
        )
        plot_matrix_grid(
            generated[:9], output_path_name=f"{output_dir_seed}/generated", norm=norm
        )

        with open(Path(output_dir) / "architecture.txt", "w") as f:
            f.write(
                repr(
                    summary(
                        trainer.generator,
                        input_data=(
                            torch.randn(1, best_params["latent_dim"]).to(
                                trainer.device
                            ),
                        ),
                    )
                )
            )
            f.write("\n\n")
            f.write(
                repr(
                    summary(
                        trainer.critic,
                        input_data=(torch.randn(1, input_dim).to(trainer.device),),
                    )
                )
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to save outputs.",
    )
    argparser.add_argument(
        "--optimize",
        action="store_true",
        help="If set, will run hyperparameter optimization before training.",
    )
    args = argparser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.output_dir is None:
        output_dir = config["output"]["path"]
    else:
        output_dir = args.output_dir

    config["dataset"]["name"] = Path(
        args.config_path
    ).stem  # Use the config file name as the dataset name

    if args.optimize:
        main(config, output_dir)
    else:
        main_seed(config, output_dir)
