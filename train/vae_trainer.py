import json
import os
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from optuna.trial import FixedTrial
from torch import nn
from torch.utils.data import DataLoader

from common.util import (
    plot_losses,
    set_seed,
    setup_logger,
)
from train.common import (
    EarlyStopping,
    filter_params,
)


def grad_norm(loss, paramters, retain_graph=True):
    grads = torch.autograd.grad(
        loss, paramters, retain_graph=retain_graph, allow_unused=True
    )

    total = torch.zeros((), device=loss.device)

    for g in grads:
        if g is not None:
            total += g.detach().square().sum()

    return total.sqrt()


def measure_gradient_balance(model, x, beta=2.0):
    model.eval()

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]

    x_reconstructed, mu, std = model(x)

    recon_loss = (
        torch.nn.functional.mse_loss(x_reconstructed, x, reduction="sum") / x.shape[0]
    )
    kl_loss = -torch.sum(1 + 2 * std.log() - mu.pow(2) - std.pow(2)) / x.shape[0]

    recon_grad_norm = grad_norm(recon_loss, encoder_params, retain_graph=True)
    kl_grad_norm = grad_norm(kl_loss, encoder_params, retain_graph=False)
    weighted_kl_grad = beta * kl_grad_norm

    model.train()
    model.zero_grad()
    return {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "recon_grad_norm": recon_grad_norm.item(),
        "kl_grad_norm": kl_grad_norm.item(),
        "weighted_kl_grad_norm": weighted_kl_grad.item(),
        "grad_ratio": recon_grad_norm.item() / (weighted_kl_grad.item() + 1e-8),
    }


class VaeTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        num_epochs: int = 100,
        seed: int | None = None,
        trial: optuna.Trial | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str | os.PathLike | None = None,
        logger=None,
    ):
        self.learning_rate = lr
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = device
        self.output_dir = output_dir
        self.trial = trial
        self.logger = logger
        self.mmd_threshold_patience = 3
        self.beta = 2.0
        if self.seed is not None:
            set_seed(self.seed)

        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.mmd_components = []

        self.fixed_train = None  # For gradient balance evaluation

        self.log(
            f"Initialized VaeTrainer with learning_rate={lr}, num_epochs={num_epochs}, "
            f"seed={seed}, device={device}, output_dir={output_dir}"
        )

    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

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

        return avg_loss, recon, kl

    def evaluate_gradient_balance(
        self, train_dataloader: DataLoader, val_dataloader: DataLoader
    ):
        if self.fixed_train is None:
            train_iter = next(iter(train_dataloader)).to(self.device)
            self.fixed_train = train_iter
            del train_iter

        print("Evaluating gradient balance on fixed training batch...")
        print(measure_gradient_balance(self.model, self.fixed_train, beta=self.beta))

        print("Evaluating gradient balance on validation batch...")
        val_iter = next(iter(val_dataloader)).to(self.device)
        self.fixed_val = val_iter
        del val_iter
        print(measure_gradient_balance(self.model, self.fixed_val, beta=self.beta))

    def train(self, train_dataloader, val_dataloader):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        train_losses = []
        val_losses = []
        mmds = []

        early_stopping = EarlyStopping(patience=30, window_size=5)

        start_time = time.perf_counter()

        for epoch in range(self.num_epochs):
            train_loss, recon, kl = self.train_step(train_dataloader, epoch)
            val_loss, val_recon, val_kl = self.val_step(val_dataloader, epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.log(
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

        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        if self.device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MiB

            self.log(f"Training completed in {end_time - start_time:.4f} seconds.")
            self.log(f"Peak GPU memory usage: {peak_memory:.4f} MiB")
        else:
            self.log(f"Training completed in {end_time - start_time:.4f} seconds.")

        self.log("Training complete.")

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.val_recon = val_recon
        self.val_kl = val_kl

        return train_losses, val_losses, val_recon, val_kl, mmds

    def calc_loss(self, x, x_hat, mu, std, epoch):
        batch_size = x.size(0)

        recon_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")

        std = torch.clamp(std, min=1e-6)  # Avoid log(0)
        kl_div = -torch.sum(1 + 2 * std.log() - mu.pow(2) - std.pow(2))
        beta = self.beta
        loss = (recon_loss + beta * kl_div) / batch_size  # Normalize by batch size

        return loss, recon_loss.item() / batch_size, kl_div.item() / batch_size


def vae_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    ModelClass: type,
    config: dict,
    get_params_func: Callable,
    input_dim: int,
    num_epochs_optim: int,
):
    fixed_params = config.get("model", {})
    optim_params = get_params_func(trial)
    params = {**fixed_params, **optim_params}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_params = filter_params(params, ModelClass)
    trainer_params = filter_params(params, VaeTrainer)

    set_seed(config["training"]["seed"])
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

    return np.min(val_losses)


def optimize_vae(
    train_loader,
    val_loader,
    config: dict,
    ModelClass: type,
    get_params_func: Callable,
    input_dim: int,
    output_dir=None,
):
    num_epochs_optim = config["training"]["num_epoch_optim"]
    n_trials = config["training"]["n_trials"]

    objective_func = partial(
        vae_objective,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        ModelClass=ModelClass,
        get_params_func=get_params_func,
        input_dim=input_dim,
        num_epochs_optim=num_epochs_optim,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1),
        study_name="hcp_vae_optuna",
        storage=f"sqlite:///{output_dir}/vae_optuna_study.db",
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


def optimize_and_train_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    ModelClass: type,
    get_params_func: Callable,
    output_dir: str,
    input_dim: int,
    device: str = "cpu",
):
    output_dir_path = Path(output_dir)
    num_epochs_final = config["training"]["num_epoch_final"]

    best_params = optimize_vae(
        train_loader,
        val_loader,
        config=config,
        ModelClass=ModelClass,
        get_params_func=get_params_func,
        input_dim=input_dim,
        output_dir=output_dir,
    )

    with open(output_dir_path / "best_params.json", "w") as f:
        json.dump(best_params, f)

    best_trial = FixedTrial(best_params)
    optimized_params = get_params_func(best_trial)
    fixed_params = config.get("model", {})

    final_all_params = {**optimized_params, **fixed_params}

    model_params = filter_params(final_all_params, ModelClass)
    trainer_params = filter_params(final_all_params, VaeTrainer)

    final_training_output_dir = output_dir_path / "run_0"

    logger = setup_logger(
        log_file=f"{final_training_output_dir}/vae_trainer.log",
    )

    set_seed(config["training"]["seed"])
    model = ModelClass(
        **model_params,
        input_dim=input_dim,
    )

    trainer = VaeTrainer(
        model=model,
        **trainer_params,
        num_epochs=num_epochs_final,
        output_dir=final_training_output_dir,
        device=device,
        logger=logger,
    )

    train_losses, val_losses, _, _ = trainer.train(train_loader, val_loader)

    plot_losses(
        {"Train losses": train_losses, "Val losses": val_losses},
        output_dir=final_training_output_dir,
    )

    return train_losses, val_losses, best_params, trainer
