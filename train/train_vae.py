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
from gen.generation import generate_graphs_with_vae
from model.sc_vae_model import VAE as SCVAE
from model.vae_model import VaeModel
from train.common import (
    filter_params,
    get_sc_vae_params,
    get_vae_params,
)
from train.vae_trainer import VaeTrainer, optimize_and_train_vae

VAE_MODEL_CLASSES = {
    "synthetic": (VaeModel, get_vae_params),
    "sc": (SCVAE, get_sc_vae_params),
    # "hcp": (SCVAE, get_sc_vae_params),
}


def main_optimize(config, output_dir):
    print(
        f"Training on dataset: {config['dataset']['type']}, {config['dataset'].get('name', '')}, saving to {output_dir}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, *_ = setup_dataloader_from_config(config, is_matrix=False)
    model_class, get_param_func = VAE_MODEL_CLASSES[config["dataset"]["type"]]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for data in train_loader:
        input_dim = data[0].shape[0]
        break

    train_loss, val_loss, best_params, trainer = optimize_and_train_vae(
        train_loader,
        val_loader,
        config=config,
        ModelClass=model_class,
        get_params_func=get_param_func,
        output_dir=output_dir,
        input_dim=input_dim,
        device=device,
    )

    plot_losses({"Train losses": train_loss, "Val losses": val_loss}, output_dir)

    model = trainer.model
    run_dir = Path(output_dir) / "run_0"
    model.load_state_dict(torch.load(run_dir / "model.pth"))
    generated = generate_graphs_with_vae(model, num_samples=1000, device=trainer.device)
    np.savez_compressed(run_dir / "generated_graphs.npz", generated)
    plot_matrix_grid(generated[:9], output_path_name=run_dir / "generated")

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


def main_seed(config, output_dir):
    print(
        f"Training on dataset: {config['dataset']['type']}, {config['dataset'].get('name', '')}, saving to {output_dir}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class, get_param_func = VAE_MODEL_CLASSES[config["dataset"]["type"]]

    for i, run in enumerate(config["training"]["runs"]):
        run_seed = run["seed"]
        fold = run["fold"]
        output_dir_seed = Path(output_dir) / f"run_{i + 1}"

        logger = setup_logger(output_dir_seed / "training.log")
        logger.info(f"Running training with seed: {run_seed}, fold: {fold}")

        train_loader, val_loader, *_ = setup_dataloader_from_config(
            config, is_matrix=False, fold=fold
        )

        Path(output_dir_seed).mkdir(parents=True, exist_ok=True)

        fixed_params = config["model"]
        with open(Path(config["output"]["path"]) / "best_params.json", "r") as f:
            best_params = json.load(f)

        params = {**fixed_params, **best_params}

        model_params = filter_params(params, model_class)
        trainer_params = filter_params(params, VaeTrainer)

        set_seed(run_seed)

        model = model_class(**model_params)
        model.to(device)

        trainer = VaeTrainer(
            **trainer_params,
            model=model,
            num_epochs=config["training"]["num_epoch_final"],
            device=device,
            output_dir=output_dir_seed,
            logger=logger,
        )

        trainer.train(train_loader, val_loader)

        plot_losses(
            {"Train losses": trainer.train_losses, "Val losses": trainer.val_losses},
            output_dir_seed,
        )

        model.load_state_dict(torch.load(f"{output_dir_seed}/model.pth"))

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        generated = generate_graphs_with_vae(
            model, num_samples=1000, device=trainer.device
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

        with open(Path(output_dir_seed) / "architecture.txt", "w") as f:
            f.write(
                repr(
                    summary(
                        trainer.model.encoder,
                        input_data=(
                            torch.randn(1, config["model"]["input_dim"]).to(
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
                        trainer.model.decoder,
                        input_data=(
                            torch.randn(1, best_params["latent_dim"]).to(
                                trainer.device
                            ),
                        ),
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
        main_optimize(config, output_dir)
    else:
        main_seed(config, output_dir)
