import argparse
import json
import os
import time

# needed for DDPM to ensure deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib.colors import PowerNorm
from torchinfo import summary

matplotlib.use("Agg")

from common.util import plot_losses, set_seed, setup_logger
from common.visualize import plot_matrix_grid
from dataset.graph_dataset import setup_dataloader_from_config
from model.ddpm_model import DiffusionModel
from model.sc_ddpm_model import Diffuser
from model.sc_ddpm_model import UNet as UnetSC
from train.common import (
    filter_params,
    get_ddpm_params,
    get_sc_ddpm_params,
)
from train.ddpm_trainer import DiffusionTrainer, optimize_and_train_ddpm

DDPM_MODEL_CLASSES = {
    "synthetic": (DiffusionModel, get_ddpm_params),
    "sc": (UnetSC, get_sc_ddpm_params),
}


def main(config: dict, output_dir: str | Path):
    # needed for DDPM to ensure deterministic behavior
    torch.use_deterministic_algorithms(True)
    print(
        f"Training on dataset: {config['dataset']['type']}, {config['dataset'].get('name', '')}, saving to {output_dir}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, *_ = setup_dataloader_from_config(config, is_matrix=True)
    model_class, get_param_func = DDPM_MODEL_CLASSES[config["dataset"]["type"]]

    diffuser = Diffuser(num_timesteps=config["model"]["timesteps"], device=device)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_loss, val_loss, best_params, trainer = optimize_and_train_ddpm(
        train_loader,
        val_loader,
        config=config,
        ModelClass=model_class,
        diffuser=diffuser,
        get_params_func=get_param_func,
        output_dir=output_dir,
        device=device,
    )

    output_dir_run = Path(output_dir) / "run_0"

    plot_losses({"Train losses": train_loss, "Val losses": val_loss}, output_dir_run)

    model = trainer.model
    model.load_state_dict(torch.load(f"{output_dir_run}/model.pth"))
    set_seed(91)
    generated = (
        diffuser.sample(
            model, (1000, 1, config["dataset"]["size"], config["dataset"]["size"])
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    # NOTICE: saved matricies, not vectors
    np.savez_compressed(f"{output_dir_run}/generated_graphs.npz", generated)
    plot_matrix_grid(generated[:9], output_path_name=f"{output_dir_run}/generated")

    t = torch.randn(1).to(device)
    input_tensor = torch.randn(
        1, 1, config["dataset"]["size"], config["dataset"]["size"]
    ).to(device)

    with open(Path(output_dir) / "architecture.txt", "w") as f:
        f.write(repr(summary(trainer.model, input_data=[input_tensor, t])))


def main_seed(config, output_dir):
    torch.use_deterministic_algorithms(True)
    print(
        f"Training on dataset: {config['dataset']['type']}, {config['dataset'].get('name', '')}, saving to {output_dir}"
    )

    model_class, _ = DDPM_MODEL_CLASSES[config["dataset"]["type"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, run in enumerate(config["training"]["runs"]):
        run_seed = run["seed"]
        fold = run["fold"]
        output_dir_seed = Path(output_dir) / f"run_{i + 1}"

        logger = setup_logger(output_dir_seed / "training.log")
        logger.info(f"Running training with seed: {run_seed}, fold: {fold}")

        train_loader, val_loader, *_ = setup_dataloader_from_config(
            config, fold=fold, is_matrix=True
        )
        Path(output_dir_seed).mkdir(parents=True, exist_ok=True)

        fixed_params = config["model"]
        with open(Path(config["output"]["path"]) / "best_params.json", "r") as f:
            best_params = json.load(f)

        params = {**fixed_params, **best_params}

        model_params = filter_params(params, model_class)
        trainer_params = filter_params(params, DiffusionTrainer)

        diffuser = Diffuser(num_timesteps=config["model"]["timesteps"], device=device)

        set_seed(run_seed)

        model = model_class(**model_params)
        model.to(device)

        trainer = DiffusionTrainer(
            diffuser=diffuser,
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

        model = trainer.model
        model.load_state_dict(torch.load(f"{output_dir_seed}/model.pth"))
        set_seed(91)

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        generated = (
            diffuser.sample(
                model,
                (1000, 1, config["dataset"]["size"], config["dataset"]["size"]),
                chunk_size=config["training"].get("sampling_chunk_size", 500),
            )
            .squeeze()
            .cpu()
            .numpy()
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
                        trainer.model,
                        input_data=(
                            torch.randn(
                                1,
                                1,
                                config["dataset"]["size"],
                                config["dataset"]["size"],
                            ).to(trainer.device),
                            torch.randn(1).to(trainer.device),
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

    config["dataset"]["name"] = Path(args.config_path).stem

    if args.optimize:
        main(config, output_dir)
    else:
        main_seed(config, output_dir)
