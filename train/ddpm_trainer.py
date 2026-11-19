import json
import os
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from common.util import set_seed, setup_logger
from train.common import EarlyStopping, filter_params


class DiffusionTrainer:
    def __init__(
        self,
        diffuser,
        model: nn.Module,
        lr: float = 1e-4,
        num_epochs: int = 100,
        seed: int | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str | Path | None = None,
        logger: Any = None,
    ):
        self.diffuser = diffuser
        self.learning_rate = lr
        self.timesteps = diffuser.num_timesteps
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = device
        self.output_dir = output_dir
        self.logger = logger

        if self.seed is not None:
            set_seed(self.seed)

        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.log(
            f"Initialized DiffusionTrainer with model: {self.model.__class__.__name__}, learning rate: {self.learning_rate}, timesteps: {self.timesteps}, num_epochs: {self.num_epochs}, seed: {self.seed}, device: {self.device}"
        )

    def log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_step(self, data_loader: DataLoader):
        self.model.train()
        total_loss = 0.0

        for data in data_loader:
            if type(data) is list:
                data, _ = data
            data = data.to(self.device)
            t = torch.randint(
                1, self.timesteps + 1, (data.size(0),), device=self.device
            )

            x_noisy, noise = self.diffuser.add_noise(data, t)
            noise_pred = self.model(x_noisy, t)
            loss = self.calc_loss(noise_pred, noise, mask_tril=False)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)

        return avg_loss

    def val_step(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data in data_loader:
                if type(data) is list:
                    data, _ = data
                data = data.to(self.device)
                t = torch.randint(
                    1, self.timesteps + 1, (data.size(0),), device=self.device
                )

                x_noisy, noise = self.diffuser.add_noise(data, t)
                noise_pred = self.model(x_noisy, t)
                loss = self.calc_loss(noise_pred, noise, mask_tril=False)

                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)

        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        early_stopping = EarlyStopping(patience=30, window_size=5)
        train_losses = []
        val_losses = []

        start_time = time.perf_counter()

        for epoch in range(self.num_epochs):
            train_loss = self.train_step(train_loader)
            train_losses.append(train_loss)

            val_loss = self.val_step(val_loader)
            val_losses.append(val_loss)

            self.log(
                f"Epoch [{epoch + 1}/{self.num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if early_stopping(val_loss):
                self.log(f"Early stopping at epoch {epoch + 1}")
                break
            if self.output_dir is not None and early_stopping.save_model:
                torch.save(self.model.state_dict(), Path(self.output_dir) / "model.pth")

        if self.device == "cuda":
            torch.cuda.synchronize()

        training_time = time.perf_counter() - start_time

        if self.device == "cuda":
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)

            self.log(f"Training completed in {training_time:.4f} seconds.")
            self.log(f"Peak GPU memory usage: {max_memory_allocated:.4f} MiB")
        else:
            self.log(f"Training completed in {training_time:.4f} seconds.")

        self.train_losses = train_losses
        self.val_losses = val_losses

        return train_losses, val_losses

    def calc_loss(self, pred, target, mask_tril=False):
        if mask_tril:
            dim = pred.size(-1)
            mask = torch.tril(torch.ones(dim, dim), diagonal=-1).to(self.device)
            mask = mask.unsqueeze(0).expand_as(pred)

            diff = (pred - target) * mask

            loss = (diff**2).sum() / mask.sum()
        else:
            loss = F.mse_loss(pred, target)

        return loss


# class DiffusionTrainerSC(DiffusionTrainer):
#     def __init__(
#         self,
#         diffuser,
#         input_dim: int = 1,
#         base_ch: int = 16,
#         lr: float = 1e-4,
#         num_epochs: int = 100,
#         seed: int = 39,
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#         output_dir: str | None = None,
#     ):
#         self.diffuser = diffuser
#         self.input_dim = input_dim
#         self.base_ch = base_ch
#         self.learning_rate = lr
#         self.timesteps = diffuser.num_timesteps
#         self.num_epochs = num_epochs
#         self.seed = seed
#         self.device = device
#         self.output_dir = output_dir
#         set_seed(self.seed)

#         self.model = DiffusionModelSC(in_ch=self.input_dim, base_ch=self.base_ch).to(
#             self.device
#         )

#         self.hparams = {"lr": self.learning_rate, "base_ch": self.base_ch}

#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=self.learning_rate
#         )


def ddpm_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: dict,
    ModelClass: type,
    diffuser,
    get_params_func: Callable,
):
    num_epochs_optim = config["training"]["num_epoch_optim"]
    optimized_params = get_params_func(trial)
    fixed_params = config.get("model", {})
    params = {**optimized_params, **fixed_params}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet_params = filter_params(params, ModelClass)
    trainer_params = filter_params(params, DiffusionTrainer)

    set_seed(config["training"]["seed"])

    model = ModelClass(**unet_params)

    trainer = DiffusionTrainer(
        model=model,
        diffuser=diffuser,
        **trainer_params,
        device=device,
        num_epochs=num_epochs_optim,
        # seed=config["training"]["seed"],
    )

    try:
        train_losses, val_losses = trainer.train(train_dataloader, val_dataloader)
    except optuna.exceptions.TrialPruned:
        print("Trial was pruned.")
        raise

    return np.min(val_losses)


def optimize_ddpm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    ModelClass: type,
    diffuser: type,
    get_params_func: Callable,
    output_dir: str | os.PathLike | None = None,
):
    n_trials = config["training"]["n_trials"]
    # optuna logging setup
    # optuna.logging.disable_default_handler()
    # optuna.logging.set_verbosity(optuna.logging.INFO)
    # logging.getLogger("optuna").setLevel(logging.INFO)
    # if output_dir:
    #     log_file_path = f"{output_dir}/ddpm_optuna_study.log"
    #     setup_logger("optuna", log_file_path, mode="w")

    objective_func = partial(
        ddpm_objective,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        ModelClass=ModelClass,
        diffuser=diffuser,
        get_params_func=get_params_func,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1),
        study_name="ddpm_hyperparameter_optimization",
        storage=f"sqlite:///{output_dir}/ddpm_optuna_study.db",
        load_if_exists=True,
    )

    study.optimize(objective_func, n_trials=n_trials, gc_after_trial=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    if output_dir is not None:
        try:
            ax = optuna.visualization.matplotlib.plot_param_importances(study)
            fig = ax.figure
            fig.savefig(Path(output_dir) / "param_importance.png", bbox_inches="tight")
            plt.close(fig)

            axes = optuna.visualization.matplotlib.plot_slice(study)
            fig = axes.flat[0].figure
            fig.savefig(Path(output_dir) / "slice_plot.png")
            plt.close(fig)
        except Exception as e:
            print(f"Could not create plots: {e}")

    return study.best_params


def optimize_and_train_ddpm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    ModelClass: type,
    diffuser,
    get_params_func: Callable,
    output_dir: str | os.PathLike,
    device: str = "cpu",
):
    num_epochs_final = config["training"]["num_epoch_final"]

    best_params = optimize_ddpm(
        train_loader,
        val_loader,
        config=config,
        ModelClass=ModelClass,
        diffuser=diffuser,
        get_params_func=get_params_func,
        output_dir=output_dir,
    )

    with open(Path(output_dir) / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    output_dir_run = Path(output_dir) / "run_0"

    Path(output_dir_run).mkdir(parents=True, exist_ok=True)

    fixed_params = config.get("model", {})
    params = {**best_params, **fixed_params}

    unet_params = filter_params(params, ModelClass)
    trainer_params = filter_params(params, DiffusionTrainer)

    logger = setup_logger(log_file=f"{output_dir_run}/ddpm_training.log")

    set_seed(config["training"]["seed"])

    model = ModelClass(**unet_params)

    trainer = DiffusionTrainer(
        model=model,
        diffuser=diffuser,
        **trainer_params,
        num_epochs=num_epochs_final,
        output_dir=output_dir_run,
        device=device,
        logger=logger,
    )

    train_losses, val_losses, *_ = trainer.train(train_loader, val_loader)

    return train_losses, val_losses, best_params, trainer
