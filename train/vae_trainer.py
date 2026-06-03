import json
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.util import set_seed
from train.common import EarlyStopping, filter_params


class VaeTrainer:
    def __init__(
        self,
        model: type[nn.Module],
        lr: float = 1e-4,
        num_epochs: int = 100,
        seed: int = 42,
        trial: optuna.Trial | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str | None = None,
    ):
        self.learning_rate = lr
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = device
        self.output_dir = output_dir
        self.trial = trial
        set_seed(self.seed)

        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def train_step(self, data_loader: DataLoader, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_div = 0.0

        for data in data_loader:
            # data = transform(data)
            data = data.to(self.device)

            self.optimizer.zero_grad()

            x_hat, mu, std = self.model(data)
            loss, recon, kl = self.calc_loss(data, x_hat, mu, std, epoch)
            total_recon_loss += recon
            total_kl_div += kl

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_recon_loss = total_recon_loss / len(data_loader)
        avg_kl_div = total_kl_div / len(data_loader)

        avg_loss = total_loss / len(data_loader)

        return avg_loss, avg_recon_loss, avg_kl_div

    def val_step(self, dataloader: DataLoader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_div = 0.0
        diffs = []
        mus = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                x_hat, mu, std = self.model(data)
                mus.append(mu.cpu())

                z = mu
                z_shuffled = z[torch.randperm(z.size(0))]
                x_hat_shuffled = self.model.decoder(z_shuffled)
                diffs.append(((x_hat - x_hat_shuffled) ** 2).mean().item())

                loss, recon, kl = self.calc_loss(data, x_hat, mu, std, epoch)
                total_loss += loss.item()
                total_recon_loss += recon
                total_kl_div += kl

            mus = torch.cat(mus, dim=0)
            var = torch.var(mus)
            print("mean: ", mus.mean().item(), "var: ", var.item())

        avg_loss = total_loss / len(dataloader)
        recon = total_recon_loss / len(dataloader)
        kl = total_kl_div / len(dataloader)

        print(sum(diffs) / len(diffs))

        return avg_loss, recon, kl

    def train(self, train_dataloader, val_dataloader):
        train_losses = []
        val_losses = []
        early_stopping = EarlyStopping(patience=30, window_size=5)

        for epoch in range(self.num_epochs):
            train_loss, recon, kl = self.train_step(train_dataloader, epoch)
            val_loss, val_recon, val_kl = self.val_step(val_dataloader, epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f} (recon: {recon:.4f}, kl: {kl:.4f}), Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            # avoid KL vanishing
            if epoch > 30 and self.trial is not None:
                if kl < 0.001:
                    print("Trial pruned due to low KL divergence.")
                    raise optuna.exceptions.TrialPruned()
            if self.output_dir is not None and early_stopping.save_model:
                torch.save(self.model.state_dict(), Path(self.output_dir) / "model.pth")

        print("Training complete.")

        return train_losses, val_losses, val_recon, val_kl

    def calc_loss(self, x, x_hat, mu, std, epoch):
        batch_size = x.size(0)

        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")

        std = torch.clamp(std, min=1e-6)  # Avoid log(0)
        # kl_div = - (1 + 2 * std.log() - mu.pow(2) - std.pow(2))
        # kl_div = torch.clamp(kl_div, min=0.3)
        # kl_div = torch.sum(kl_div)
        kl_div = -torch.sum(1 + 2 * std.log() - mu.pow(2) - std.pow(2))
        beta = 1.0
        loss = (recon_loss + beta * kl_div) / batch_size  # Normalize by batch size

        return loss, recon_loss.item() / batch_size, kl_div.item() / batch_size


def vae_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    ModelClass: type,
    get_params_func: Callable,
    input_dim: int,
    num_epochs_optim: int,
):
    params = get_params_func(trial)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_params = filter_params(params, ModelClass)
    trainer_params = filter_params(params, VaeTrainer)

    model = ModelClass(**model_params, input_dim=input_dim)

    trainer = VaeTrainer(
        model=model,
        **trainer_params,
        device=device,
        trial=trial,
        num_epochs=num_epochs_optim,
    )

    try:
        train_losses, val_losses, recon, kl = trainer.train(
            train_dataloader, val_dataloader
        )
    except optuna.exceptions.TrialPruned:
        print("Trial was pruned.")
        raise

    # return np.mean(val_losses[-3:])
    return np.min(val_losses)


def optimize_vae(
    train_loader,
    val_loader,
    ModelClass: type,
    get_params_func: Callable,
    input_dim: int,
    n_trials: int = 100,
    num_epochs_optim: int = 500,
    output_dir=None,
):
    # optuna.logging.disable_default_handler()
    # optuna.logging.set_verbosity(optuna.logging.INFO)
    # logging.getLogger("optuna").setLevel(logging.INFO)

    # log_file_path = f"{output_dir}/vae_optuna_study.log"
    # setup_logger("optuna", log_file_path, mode="w")

    objective_func = partial(
        vae_objective,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        ModelClass=ModelClass,
        get_params_func=get_params_func,
        input_dim=input_dim,
        num_epochs_optim=num_epochs_optim,
    )
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=1)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=1)
    )
    study.optimize(objective_func, n_trials=n_trials)

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


def optimize_and_train_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    ModelClass: type,
    get_params_func: Callable,
    output_dir: str,
    input_dim: int,
    n_trials: int = 100,
    num_epochs_optim: int = 500,
    num_epochs_final: int = 1000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_params = optimize_vae(
        train_loader,
        val_loader,
        ModelClass=ModelClass,
        get_params_func=get_params_func,
        input_dim=input_dim,
        n_trials=n_trials,
        num_epochs_optim=num_epochs_optim,
        output_dir=output_dir,
    )

    with open(Path(output_dir) / "best_params.json", "w") as f:
        json.dump(best_params, f)

    model_params = filter_params(best_params, ModelClass)
    trainer_params = filter_params(best_params, VaeTrainer)

    model = ModelClass(**model_params, input_dim=input_dim)

    trainer = VaeTrainer(
        model=model,
        **trainer_params,
        num_epochs=num_epochs_final,
        seed=42,
        output_dir=output_dir,
        device=device,
    )

    train_losses, val_losses, *_ = trainer.train(train_loader, val_loader)

    return train_losses, val_losses, best_params, trainer


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )
