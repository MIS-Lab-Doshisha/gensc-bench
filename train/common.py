import inspect
import json
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
from tqdm import tqdm

from common.graph_util import get_vecs_from_matrices
from common.util import extract_tensor_from_dataloaders
from eval.core.mmd import compute_mmd


def get_common_params(trial: optuna.Trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
    }

    return params


def get_vae_params(trial: optuna.Trial):
    vae_params = {
        "latent_dim": trial.suggest_categorical("latent_dim", [2, 4, 8, 16]),
    }

    common_params = get_common_params(trial)
    vae_params.update(common_params)
    return vae_params


def get_gan_params(trial: optuna.Trial):
    gan_params = {
        "lr_generator": trial.suggest_float("lr_generator", 1e-5, 1e-2, log=True),
        "lr_critic": trial.suggest_float("lr_critic", 1e-5, 1e-2, log=True),
        "latent_dim": trial.suggest_categorical("latent_dim", [4, 8, 16, 32]),
    }

    return gan_params


def get_ddpm_params(trial: optuna.Trial):
    # ddpm_params = {
    #     "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    # }

    # common_params = get_common_params(trial)
    # ddpm_params.update(common_params)

    ddpm_params = get_common_params(trial)

    return ddpm_params


def get_sc_vae_params(trial: optuna.Trial):
    sc_vae_params = {
        "latent_dim": trial.suggest_categorical("latent_dim", [4, 8, 16, 32, 64]),
        "encoder_n_blocks": trial.suggest_int("encoder_n_blocks", 2, 5),
        "decoder_n_blocks": trial.suggest_int("decoder_n_blocks", 2, 5),
    }

    common_params = get_common_params(trial)
    sc_vae_params.update(common_params)
    return sc_vae_params


def get_sc_gan_params(trial: optuna.Trial):
    sc_gan_params = {
        "lr_generator": trial.suggest_float("lr_generator", 1e-5, 1e-3, log=True),
        "lr_critic": trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True),
        "latent_dim": trial.suggest_categorical("latent_dim", [4, 8, 16, 32, 64]),
        "generator_n_blocks": trial.suggest_int("generator_n_blocks", 2, 5),
        "critic_n_blocks": trial.suggest_int("critic_n_blocks", 2, 5),
    }

    return sc_gan_params


def get_sc_ddpm_params(trial: optuna.Trial):
    sc_ddpm_params = {
        "base_ch": trial.suggest_categorical("base_ch", [8, 16, 32]),
    }

    common_params = get_common_params(trial)
    sc_ddpm_params.update(common_params)
    return sc_ddpm_params


class EarlyStopping:
    def __init__(
        self, patience: int = 10, window_size: int = 1, min_delta: float = 0.0
    ):
        self.patience = patience
        self.window_size = window_size
        self.min_delta = min_delta

        self.counter = 0
        self.early_stop = False
        self.ma_loss_min = np.inf
        self.val_loss_history = deque(maxlen=window_size)

        self.save_model = False
        self.best_val_loss = np.inf

    def __call__(self, val_loss: float):
        self.save_model = False
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_model = True
            print("Best model found")

        self.val_loss_history.append(val_loss)
        if len(self.val_loss_history) < self.window_size:
            return self.early_stop

        ma_loss = np.mean(self.val_loss_history)
        if self._is_significant_improvement(ma_loss):
            self.ma_loss_min = ma_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_significant_improvement(self, current_loss):
        if not np.isfinite(self.ma_loss_min):
            return True
        return current_loss < self.ma_loss_min - self.min_delta


def get_constructor_args(cls: type):
    try:
        sig = inspect.signature(cls.__init__)
        return set(list(sig.parameters.keys())[1:])
    except:
        return set()


def filter_params(all_params: dict, cls: type):
    constructor_args = get_constructor_args(cls)
    filtered_params = {k: v for k, v in all_params.items() if k in constructor_args}
    return filtered_params


def get_reference_sigmas(X_ref, base_sigmas=np.array([0.25, 0.5, 1.0, 2.0, 4.0])):
    dists = (
        sklearn.metrics.pairwise_distances(X_ref, X_ref, metric="euclidean", n_jobs=8)
        ** 2
    )
    mask = ~np.eye(len(X_ref), dtype=bool)
    mean_dist = dists[mask].mean()

    if mean_dist <= 0:
        raise ValueError(
            "Mean pairwise distance is non-positive, cannot compute sigmas."
        )

    return base_sigmas * np.sqrt(mean_dist)


@dataclass
class MMDThreshold:
    n_bootstrap: int = 1000
    quantile: float = 0.95
    sigmas_: Sequence[float] | None = None
    random_state: int = 42

    def save(self, filepath):
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Call fit() before save().")

        with open(filepath, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        self.threshold_ = data["threshold"]
        self.bootstrap_values = np.array(data["bootstrap_values"])
        self.sigmas_ = np.array(data["sigmas_"])
        self.feature_dim_ = data["feature_dim"]
        self.n_bootstrap = data["n_bootstrap"]
        self.quantile = data["quantile"]
        self.n_gen = data.get("n_gen", None)

    def _to_numpy_2d(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError(f"Input data must be 2D, but got shape {x.shape}.")

        return x

    def fit(self, x_train, x_val, n_gen=None):
        rng = np.random.default_rng(self.random_state)
        mmd_values = []

        X_ref = np.concatenate([x_train, x_val], axis=0)
        n_val = len(x_val)
        if n_gen is None:
            self.n_gen = n_val
        else:
            self.n_gen = n_gen

        if self.sigmas_ is None:
            self.sigmas_ = get_reference_sigmas(X_ref)
        else:
            self.sigmas_ = np.asarray(self.sigmas)

        for _ in tqdm(range(self.n_bootstrap)):
            idx_a = rng.choice(len(X_ref), size=n_val, replace=True)
            idx_b = rng.choice(len(X_ref), size=n_gen, replace=True)

            A = X_ref[idx_a]
            B = X_ref[idx_b]

            mmd = compute_mmd(
                A, B, is_hist=False, sigmas=self.sigmas_, pad=False, reduction="mean"
            )
            mmd_values.append(mmd)

        self.bootstrap_values = np.array(mmd_values)
        self.threshold_ = float(np.quantile(self.bootstrap_values, self.quantile))

        self.feature_dim_ = X_ref.shape[1]
        self.x_val = x_val.detach().cpu().clone()

        return self

    def score(self, x_gen, x_val=None, return_components=False):
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Call fit() before score().")

        if x_val is None:
            if not hasattr(self, "x_val"):
                raise ValueError("x_val must be provided if not stored during fit.")
            x_val = self.x_val
        else:
            x_val = self._to_numpy_2d(x_val)

        x_gen = self._to_numpy_2d(x_gen)
        x_val = self._to_numpy_2d(x_val)

        if x_gen.shape[1] != self.feature_dim_ or x_val.shape[1] != self.feature_dim_:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.feature_dim_}, "
                f"but got {x_gen.shape[1]} for x_gen and {x_val.shape[1]} for x_val."
            )

        return compute_mmd(
            x_gen,
            x_val,
            is_hist=False,
            sigmas=self.sigmas_,
            pad=False,
            reduction="mean",
            return_components=return_components,
        )

    def reached(self, mmd):
        return mmd <= self.threshold_

    def ratio(self, x_gen, x_val):
        score = self.score(x_gen, x_val)
        return score / self.threshold_

    def summary(self):
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Call fit() before summary().")

        return {
            "threshold": self.threshold_,
            "bootstrap_std": float(np.std(self.bootstrap_values)),
            "bootstrap_max": float(np.max(self.bootstrap_values)),
            "bootstrap_min": float(np.min(self.bootstrap_values)),
            "bootstrap_mean": float(np.mean(self.bootstrap_values)),
            "sigmas_": self.sigmas_.tolist(),
            "feature_dim": self.feature_dim_,
            "n_bootstrap": self.n_bootstrap,
            "quantile": self.quantile,
            "n_gen": self.n_gen,
        }


class MMDEvaluatorForTrainer:
    def __init__(
        self,
        graph_generator: Callable,
        *,
        mmd_threshold: MMDThreshold | None = None,
        n_gen: int = 300,
        logger: Callable | None = None,
    ):
        self.graph_generator = graph_generator
        self.mmd_threshold = mmd_threshold
        self.n_gen = n_gen
        self.logger = logger if logger is not None else print

        self.history = []

    def _get_num_sample(self):
        if self.mmd_threshold is None:
            return self.n_gen

        if not hasattr(self.mmd_threshold, "_threshold"):
            raise RuntimeError(
                "MMDThreshold must be fitted before using MMDEvaluatorForTrainer."
            )

        return self.mmd_threshold.n_gen

    @torch.no_grad()
    def calculate(self, *, model, device, val_dataloader, epoch: int):
        model.eval()
        generated_graphs = self.graph_generator(
            model, num_samples=self._get_num_sample(), device=device
        )

        generated = get_vecs_from_matrices(generated_graphs)
        self.logger(f"Generated shape: {generated.shape}")

        if self.mmd_threshold is not None:
            mmd, components = self.mmd_threshold.score(
                generated, return_components=True
            )
        else:
            val_data = extract_tensor_from_dataloaders([val_dataloader], squeeze=True)
            self.logger(f"Validation data shape: {val_data.shape}")

            mmd, components = compute_mmd(
                generated,
                val_data,
                reduction="mean",
                is_hist=False,
                sigma_dist="RR",
                verbose=True,
                return_components=True,
            )

        if components is not None:
            self.history.append(
                {
                    "epoch": epoch,
                    "components": components,
                }
            )

        return mmd

    def plot_components(self, output_dir):
        if not self.history:
            raise ValueError("No history available. Call calculate() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self._create_components_dataframe()

        for sigma_label, sub_df in df.groupby("sigma_label", sort=False):
            sub_df = sub_df.sort_values("epoch")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sub_df["epoch"], sub_df["u_GG"], label="u_GG")
            ax.plot(sub_df["epoch"], sub_df["u_RR"], label="u_RR")
            ax.plot(sub_df["epoch"], sub_df["K_GR"], label="K_GR")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_title(f"MMD Components for {sigma_label}")
            ax.legend()

            fig.tight_layout()
            fig.savefig(output_dir / f"mmd_components_{sigma_label}.pdf")
            plt.close(fig)

    def _create_components_dataframe(self):
        rows = []

        for record in self.history:
            epoch = record["epoch"]
            components = record["components"]

            if not components:
                continue

            target_indices = self._get_target_indices(len(components))

            for index, sigma_label in target_indices.items():
                item = components[index]

                rows.append(
                    {
                        "epoch": epoch,
                        "sigma_label": sigma_label,
                        "u_GG": item["u_GG"],
                        "u_RR": item["u_RR"],
                        "K_GR": item["K_GR"],
                        "mmd": item["u_GG"] + item["u_RR"] - 2 * item["K_GR"],
                    }
                )

        if not rows:
            raise ValueError("No components data found in history.")

        return pd.DataFrame(rows)

    @staticmethod
    def _get_target_indices(num_components):
        candidates = [
            (0, "min_sigma"),
            (num_components // 2, "median_sigma"),
            (num_components - 1, "max_sigma"),
        ]

        result = {}
        for index, label in candidates:
            result.setdefault(index, label)

        return result
