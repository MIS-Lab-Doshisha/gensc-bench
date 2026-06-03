import json
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.util import set_seed
from model.sc_ddpm_model import UNet as DiffusionModelSC
from train.common import EarlyStopping, filter_params


class DiffusionTrainer:
    def __init__(
        self,
        diffuser,
        model: type[nn.Module],
        lr: float = 1e-4,
        num_epochs: int = 100,
        seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str | None = None,
    ):
        self.diffuser = diffuser
        self.learning_rate = lr
        self.timesteps = diffuser.num_timesteps
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = device
        self.output_dir = output_dir
        set_seed(self.seed)

        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

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
        early_stopping = EarlyStopping(patience=30, window_size=5)
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            train_loss = self.train_step(train_loader)
            train_losses.append(train_loss)

            val_loss = self.val_step(val_loader)
            val_losses.append(val_loss)

            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            if self.output_dir is not None and early_stopping.save_model:
                torch.save(self.model.state_dict(), Path(self.output_dir) / "model.pth")

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


class DiffusionTrainerSC(DiffusionTrainer):
    def __init__(
        self,
        diffuser,
        input_dim: int = 1,
        base_ch: int = 16,
        lr: float = 1e-4,
        num_epochs: int = 100,
        seed: int = 39,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str | None = None,
    ):
        self.diffuser = diffuser
        self.input_dim = input_dim
        self.base_ch = base_ch
        self.learning_rate = lr
        self.timesteps = diffuser.num_timesteps
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = device
        self.output_dir = output_dir
        set_seed(self.seed)

        self.model = DiffusionModelSC(in_ch=self.input_dim, base_ch=self.base_ch).to(
            self.device
        )

        self.hparams = {"lr": self.learning_rate, "base_ch": self.base_ch}

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )


def ddpm_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    ModelClass: type,
    diffuser,
    get_params_func: Callable,
    num_epochs_optim: int = 500,
):
    params = get_params_func(trial)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet_params = filter_params(params, ModelClass)
    trainer_params = filter_params(params, DiffusionTrainer)

    model = ModelClass(**unet_params)

    trainer = DiffusionTrainer(
        model=model,
        diffuser=diffuser,
        **trainer_params,
        device=device,
        num_epochs=num_epochs_optim,
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
    ModelClass: type,
    diffuser,
    get_params_func: Callable,
    output_dir: str | None = None,
    n_trials: int = 100,
    num_epochs_optim: int = 500,
):
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
        ModelClass=ModelClass,
        diffuser=diffuser,
        get_params_func=get_params_func,
        num_epochs_optim=num_epochs_optim,
    )
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=1)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=1)
    )

    study.optimize(objective_func, n_trials=n_trials)  # type: ignore

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
    ModelClass: type,
    diffuser,
    get_params_func: Callable,
    output_dir: str,
    device: str = "cpu",
    n_trials: int = 100,
    num_epochs_optim: int = 500,
    num_epochs_final: int = 1000,
):
    best_params = optimize_ddpm(
        train_loader,
        val_loader,
        ModelClass=ModelClass,
        diffuser=diffuser,
        get_params_func=get_params_func,
        output_dir=output_dir,
        n_trials=n_trials,
        num_epochs_optim=num_epochs_optim,
    )

    with open(Path(output_dir) / "params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    unet_params = filter_params(best_params, ModelClass)
    trainer_params = filter_params(best_params, DiffusionTrainer)

    model = ModelClass(**unet_params)

    trainer = DiffusionTrainer(
        model=model,
        diffuser=diffuser,
        **trainer_params,
        num_epochs=num_epochs_final,
        seed=42,
        output_dir=output_dir,
        device=device,
    )

    train_losses, val_losses, *_ = trainer.train(train_loader, val_loader)

    return train_losses, val_losses, best_params, trainer

