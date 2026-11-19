import logging
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_reconstruction_from_npz(file_path) -> np.ndarray:
    ref = np.load(file_path, allow_pickle=True)
    ref_list = list(ref.values())[0]
    print(f"loaded {ref_list.shape} from {file_path}")
    return ref_list


def extract_tensor_from_dataloaders(
    dataloaders: list[DataLoader], squeeze: bool = True
) -> torch.Tensor:
    all_data = []

    for loader in dataloaders:
        for batch in loader:
            all_data.append(batch)

    if not all_data:
        warnings.warn("No data found in the provided dataloaders.")
        return torch.Tensor([])

    concatenated_data = torch.cat(all_data, dim=0)

    if squeeze:
        concatenated_data = torch.squeeze(concatenated_data)

    return concatenated_data


def plot_losses(
    losses_dict: dict[str, list[float]], output_dir: str | Path, name: str = "loss_plot"
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Linear plot
    for label, losses in losses_dict.items():
        ax1.plot(losses, label=label)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid()
    ax1.text(
        0.5,
        -0.15,
        "Losses (Linear Scale)",
        ha="center",
        va="center",
        transform=ax1.transAxes,
    )

    # Log plot
    for label, losses in losses_dict.items():
        ax2.plot(losses, label=label)

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid()
    ax2.text(
        0.5,
        -0.15,
        "Losses (Log Scale)",
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{name}.pdf")
    plt.close()


def setup_logger(log_file, level=logging.INFO, mode="w"):
    """
    Configure a logger with the given settings.

    Parameters:
    -----------
    logger_name : str
        Unique name of the logger.
    log_file : str
        Full path to the log file.
    level : int, optional
        Logging level (e.g., logging.INFO or logging.DEBUG).

    Returns:
    --------
    logging.Logger
        Configured logger.
    """
    log_dir = Path(log_file).parent
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    handler = logging.FileHandler(log_file, mode=mode)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)  # Use the same format as the file handler.
    logger.addHandler(stream_handler)

    return logger
