import logging
import sys
from pathlib import Path

import optuna
import torch

from common.path_manager import PathManager
from train.train_ddpm import main as train_ddpm
from train.train_gan import main as train_gan
from train.train_vae import main as train_vae

if __name__ == "__main__":
    paths_manager = PathManager("path_list.json")
    path_keys = paths_manager.get_all_keys()
    output_base = Path("checkpoints/artificial_2")
    optuna_logger = logging.getLogger("optuna")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for key in path_keys:
        target = paths_manager.get_original_path(key)
        print(f"Training on dataset: {target}")
        key = "planar"

        # VAE Training
        try:
            vae_log = output_base / key / "vae" / "vae.log"
            vae_log.parent.mkdir(parents=True, exist_ok=True)

            vae_optuna_log_handler = None
            with open(vae_log, "w") as f:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = f, f
                try:
                    optuna.logging.set_verbosity(logging.INFO)
                    optuna_logger.setLevel(logging.INFO)
                    vae_optuna_log_path = output_base / key / "vae" / "vae_optuna_study.log"
                    vae_optuna_log_handler = logging.FileHandler(
                        vae_optuna_log_path, mode="w"
                    )
                    vae_optuna_log_handler.setFormatter(
                        logging.Formatter(
                            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                        )
                    )
                    optuna_logger.addHandler(vae_optuna_log_handler)
                    print(f"Training VAE on dataset: {target}")
                    train_vae(target, output_base / key / "vae")
                finally:
                    if vae_optuna_log_handler:
                        optuna_logger.removeHandler(vae_optuna_log_handler)
                        vae_optuna_log_handler.close()
                    sys.stdout, sys.stderr = old_stdout, old_stderr
            print(f"Finished VAE training on dataset: {target}")
        except Exception as e:
            print(f"Error during VAE training on dataset {target}: {e}")

        # -----------------------
        # GAN Training
        # -----------------------
        try:
            gan_log = output_base / key / "gan" / "gan.log"
            gan_log.parent.mkdir(parents=True, exist_ok=True)
            gan_optuna_log_handler = None
            with open(gan_log, "w") as f:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = f, f
                try:
                    optuna.logging.set_verbosity(logging.INFO)
                    optuna_logger.setLevel(logging.INFO)
                    gan_optuna_log_path = output_base / key / "gan" / "gan_optuna_study.log"
                    gan_optuna_log_handler = logging.FileHandler(
                        gan_optuna_log_path, mode="w"
                    )
                    gan_optuna_log_handler.setFormatter(
                        logging.Formatter(
                            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                        )
                    )
                    optuna_logger.addHandler(gan_optuna_log_handler)
                    print(f"Training GAN on dataset: {target}")
                    train_gan(target, output_base / key / "gan")
                finally:
                    if gan_optuna_log_handler:
                        optuna_logger.removeHandler(gan_optuna_log_handler)
                        gan_optuna_log_handler.close()
                    sys.stdout, sys.stderr = old_stdout, old_stderr
            print(f"Finished GAN training on dataset: {target}")
        except Exception as e:
            print(f"Error during GAN training on dataset {target}: {e}")

        # -----------------------
        # DDPM Training
        # -----------------------
        try:
            ddpm_log = output_base / key / "ddpm" / "ddpm.log"
            ddpm_log.parent.mkdir(parents=True, exist_ok=True)
            ddpm_optuna_log_handler = None
            with open(ddpm_log, "w") as f:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = f, f
                try:
                    optuna.logging.set_verbosity(logging.INFO)
                    optuna_logger.setLevel(logging.INFO)
                    ddpm_optuna_log_path = (
                        output_base / key / "ddpm" / "ddpm_optuna_study.log"
                    )
                    ddpm_optuna_log_handler = logging.FileHandler(
                        ddpm_optuna_log_path, mode="w"
                    )
                    ddpm_optuna_log_handler.setFormatter(
                        logging.Formatter(
                            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                        )
                    )
                    optuna_logger.addHandler(ddpm_optuna_log_handler)
                    print(f"Training DDPM on dataset: {target}")
                    train_ddpm(target, output_base / key / "ddpm")
                finally:
                    if ddpm_optuna_log_handler:
                        optuna_logger.removeHandler(ddpm_optuna_log_handler)
                        ddpm_optuna_log_handler.close()
                    sys.stdout, sys.stderr = old_stdout, old_stderr
            print(f"Finished DDPM training on dataset: {target}")
        except Exception as e:
            print(f"Error during DDPM training on dataset {target}: {e}")
