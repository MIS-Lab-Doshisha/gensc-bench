import json
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader

from common.graph_util import get_vecs_from_matrices
from common.util import (
    extract_tensor_from_dataloaders,
    plot_losses,
    plot_matrix_grid,
)
from eval.mmd import compute_mmd
from gen.generation import generate_graphs_with_gan
from train.common import EarlyStopping, filter_params


class GanTrainer:
    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        gp_weight: float = 10.0,
        lr_critic: float = 1e-4,
        lr_generator: float = 1e-4,
        n_critic: int = 3,
        batch_size: int = 64,
        num_epochs: int = 100,
        seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str | None = None,
    ):
        self.gp_weight = gp_weight
        self.lr_critic = lr_critic
        self.lr_generator = lr_generator
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = device
        self.output_dir = output_dir

        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)

        # self.hparams = {
        #     "latent_dim": self.generator.latent_dim,
        #     "lr_critic": self.lr_critic,
        #     "lr_generator": self.lr_generator,
        # }

        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic, betas=(0.0, 0.9)
        )
        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_generator, betas=(0.0, 0.9)
        )

    def critic_train_step(self, data_loader: DataLoader):
        self.generator.eval()
        self.critic.train()
        total_loss = 0.0
        fake_scores = 0.0
        real_scores = 0.0
        gradient_penalty = 0.0

        for data in data_loader:
            real_data = data.to(self.device)
            self.optimizer_critic.zero_grad()

            z = torch.randn(data.shape[0], self.generator.latent_dim).to(self.device)
            fake_data = self.generator(z)

            real_score = self.critic(real_data)
            fake_score = self.critic(fake_data.detach())

            gradient_penalty = self.gradient_penalty(real_data, fake_data)
            critic_loss = (
                fake_score.mean()
                - real_score.mean()
                + self.gp_weight * gradient_penalty
            )
            fake_scores += fake_score.mean().item()
            real_scores += real_score.mean().item()
            gradient_penalty += gradient_penalty.item()

            critic_loss.backward()
            self.optimizer_critic.step()
            total_loss += critic_loss.item()

        avg_loss = total_loss / len(data_loader)
        fake_scores /= len(data_loader)
        real_scores /= len(data_loader)
        gradient_penalty /= len(data_loader)
        print(
            f"TRAIN fake score: {fake_scores:.4f}, real score: {real_scores:.4f}, gradient penalty: {gradient_penalty:.4f}"
        )

        return avg_loss

    def generator_train_step(self, data_loader: DataLoader):
        self.generator.train()
        self.critic.eval()
        total_loss = 0.0

        for data in data_loader:
            self.optimizer_generator.zero_grad()
            data = data.to(self.device)

            z = torch.randn(data.shape[0], self.generator.latent_dim).to(self.device)
            fake_data = self.generator(z)

            fake_score = self.critic(fake_data)
            generator_loss = -fake_score.mean()

            generator_loss.backward()
            self.optimizer_generator.step()

            total_loss += generator_loss.item()

        avg_loss = total_loss / len(data_loader)

        return avg_loss

    def critic_val_step(self, data_loader: DataLoader, epoch):
        self.critic.eval()
        self.generator.eval()
        total_loss = 0.0
        fake_scores = 0.0
        real_scores = 0.0
        total_gradient_penalty = 0.0

        for data in data_loader:
            with torch.no_grad():
                real_data = data.to(self.device)
                self.optimizer_critic.zero_grad()

                z = torch.randn(data.shape[0], self.generator.latent_dim).to(
                    self.device
                )
                fake_data = self.generator(z)

                real_score = self.critic(real_data)
                fake_score = self.critic(fake_data.detach())

            gradient_penalty = self.gradient_penalty(real_data, fake_data)
            critic_loss = (
                fake_score.mean()
                - real_score.mean()
                + self.gp_weight * gradient_penalty
            )
            fake_scores += fake_score.mean().item()
            real_scores += real_score.mean().item()
            total_gradient_penalty += gradient_penalty.item()

            total_loss += critic_loss.item()

        avg_loss = total_loss / len(data_loader)
        fake_scores /= len(data_loader)
        real_scores /= len(data_loader)
        total_gradient_penalty /= len(data_loader)
        print(
            f"VAL fake score: {fake_scores:.4f}, real score: {real_scores:.4f}, gradient penalty: {total_gradient_penalty:.4f}"
        )

        return avg_loss

    def train(self, train_dataloader, val_dataloader):
        critic_train_losses = []
        generator_train_losses = []
        critic_val_losses = []
        mmds = []

        early_stopping = EarlyStopping(patience=30, window_size=5)

        for epoch in range(self.num_epochs):
            # train critic
            critic_loss_n = []
            for _ in range(self.n_critic):
                loss = self.critic_train_step(train_dataloader)
                critic_loss_n.append(loss)

            critic_loss = np.mean(critic_loss_n)
            critic_train_losses.append(critic_loss)

            # train generator
            generator_loss = self.generator_train_step(train_dataloader)
            generator_train_losses.append(generator_loss)

            # validate
            critic_val_loss = self.critic_val_step(val_dataloader, epoch)
            critic_val_losses.append(critic_val_loss)

            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Critic Train Loss: {critic_loss:.4f},  Gen Train Loss: {generator_loss:.4f}, Critic Val Loss: {critic_val_loss:.4f}"
            )

            if epoch % 100 == 0 and self.output_dir is not None:
                generated = generate_graphs_with_gan(
                    self.generator, num_samples=9, device=self.device
                )
                plot_matrix_grid(
                    generated,
                    output_path_name=f"{self.output_dir}/wba_dence_{epoch + 1}",
                )

            mmd = self.eval_objective(val_dataloader)
            mmds.append(mmd)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], MMD on val set: {mmd:.4f}")

            if early_stopping(mmd):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            if self.output_dir is not None and early_stopping.save_model:
                torch.save(
                    self.generator.state_dict(), Path(self.output_dir) / "model.pth"
                )

        return critic_train_losses, generator_train_losses, critic_val_losses, mmds

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data).to(self.device)

        # Interpolate between real and fake data
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).to(self.device)
        interpolated.requires_grad_(True)

        # Get critic output for interpolated data
        critic_output = self.critic(interpolated)
        gradients = grad(
            outputs=critic_output,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_output.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def eval_objective(self, val_dataloader: DataLoader):
        self.generator.eval()
        generated_graphs = generate_graphs_with_gan(
            self.generator, num_samples=500, device=self.device
        )
        generated = get_vecs_from_matrices(generated_graphs)

        val_data = extract_tensor_from_dataloaders([val_dataloader], squeeze=True)
        print(generated.shape)
        print(val_data.shape)
        mmd = compute_mmd(generated, val_data, is_hist=False)

        return mmd


def get_min_rolling_mean(scores: list[float], window_size: int = 3) -> float:
    if len(scores) < window_size:
        return float("inf")

    weight = np.ones(window_size) / window_size

    rolling_means = np.convolve(scores, weight, mode="valid")
    return np.min(rolling_means)


def gan_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    GeneratorClass: type[nn.Module],
    CriticClass: type[nn.Module],
    get_params_func: Callable,
    input_dim: int,
    num_epochs_optim: int,
) -> float:
    params = get_params_func(trial)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator_params = filter_params(params, GeneratorClass)
    critic_params = filter_params(params, CriticClass)
    trainer_params = filter_params(params, GanTrainer)

    generator = GeneratorClass(**generator_params, output_dim=input_dim)
    critic = CriticClass(**critic_params, input_dim=input_dim)

    trainer = GanTrainer(
        generator=generator,
        critic=critic,
        **trainer_params,
        num_epochs=num_epochs_optim,
        seed=42,
        device=device,
    )

    _, _, _, mmds = trainer.train(train_dataloader, val_dataloader)

    objective_value = get_min_rolling_mean(mmds, window_size=3)

    return objective_value


def optimize_gan(
    train_loader,
    val_loader,
    GeneratorClass: type,
    CriticClass: type,
    get_params_func: Callable,
    input_dim: int,
    output_dir: str,
    n_trials: int = 100,
    num_epochs_optim: int = 500,
) -> dict[str, Any]:
    # optuna.logging.disable_default_handler()
    # optuna.logging.set_verbosity(optuna.logging.INFO)
    # logging.getLogger("optuna").setLevel(logging.INFO)

    # log_file_path = f"{output_dir}/gan_optuna_study.log"
    # setup_logger("optuna", log_file_path, mode="w")

    objective_func = partial(
        gan_objective,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        GeneratorClass=GeneratorClass,
        CriticClass=CriticClass,
        get_params_func=get_params_func,
        input_dim=input_dim,
        num_epochs_optim=num_epochs_optim,
    )
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


def optimize_and_train_gan(
    train_loader: DataLoader,
    val_loader: DataLoader,
    GeneratorClass: type,
    CriticClass: type,
    get_params_func: Callable,
    output_dir: str,
    input_dim=2016,
    n_trials: int = 100,
    num_epochs_optim: int = 500,
    num_epochs_final: int = 1000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_params = optimize_gan(
        train_loader,
        val_loader,
        output_dir=output_dir,
        GeneratorClass=GeneratorClass,
        CriticClass=CriticClass,
        get_params_func=get_params_func,
        input_dim=input_dim,
        n_trials=n_trials,
        num_epochs_optim=num_epochs_optim,
    )

    with open(Path(output_dir) / "best_params.json", "w") as f:
        json.dump(best_params, f)

    generator_params = filter_params(best_params, GeneratorClass)
    critic_params = filter_params(best_params, CriticClass)
    trainer_params = filter_params(best_params, GanTrainer)

    generator = GeneratorClass(**generator_params, output_dim=input_dim)
    critic = CriticClass(**critic_params, input_dim=input_dim)

    trainer = GanTrainer(
        generator=generator,
        critic=critic,
        **trainer_params,
        num_epochs=num_epochs_final,
        seed=42,
        device=device,
        output_dir=output_dir,
    )

    losses = trainer.train(train_loader, val_loader)

    plot_losses(
        {
            "Critic train loss": losses[0],
            "Generator train loss": losses[1],
            "Critic validation loss": losses[2],
        },
        output_dir=output_dir,
    )

    return best_params, trainer
