import inspect
from collections import deque

import numpy as np
import optuna


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
    def __init__(self, patience: int = 10, window_size: int = 1):
        self.patience = patience
        self.window_size = window_size

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
        if ma_loss < self.ma_loss_min:
            self.ma_loss_min = ma_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


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
