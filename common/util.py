import json
import logging
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

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

def get_reconstruction_from_npz(file_path) -> np.ndarray:
    ref = np.load(file_path, allow_pickle=True)
    ref_list = list(ref.values())[0]
    print(f"loaded {ref_list.shape} from {file_path}")
    return ref_list


def extract_tensor_from_dataloaders(
    dataloaders: List[DataLoader], squeeze: bool = True
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


def plot_losses(losses_dict: Dict[str, List[float]], output_dir: str):
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
    plt.savefig(f"{output_dir}/loss_plot.pdf")
    plt.close()


def setup_logger(logger_name, log_file, level=logging.INFO, mode="w"):
    """
    setup a logger

    Parameters:
    -----------
    logger_name : str
    log_file : str
    level : int, optional (ex: logging.INFO, logging.DEBUG)

    Returns:
    --------
    logging.Logger
    """
    log_dir = Path(log_file).parent
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, mode=mode)
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


@dataclass
class PathPair:
    key: str
    original_path: str
    reconstruction_path: str

    def __repr__(self):
        return (
            f"PathPair(key='{self.key}', "
            f"ref='{self.original_path}', "
            f"recon='{self.reconstruction_path}')"
        )


class PathListIterator:
    def __init__(self, json_path):
        """
        Parameters:
        -----------
        json_path : str

        """
        self.json_path = Path(json_path)

        if not self.json_path.exists():
            raise FileNotFoundError(f"Error: {json_path} not found.")

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

    def __iter__(self) -> Iterator[PathPair]:
        """
        iterate over all path pairs in the JSON file

        Yields:
        -------
        PathPair : (key, original_path, reconstruction_path)
        """
        for key in self.data.keys():
            ref_path = self.data[key].get("original", "")
            reconstructions = self.data[key].get("reconstructions", [])

            for recon_path in reconstructions:
                yield PathPair(
                    key=key, original_path=ref_path, reconstruction_path=recon_path
                )

    def return_original_path(self, key: str) -> str:
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in {self.json_path}")
        return self.data[key].get("original", "")

    def iter_by_key(self, key: str) -> Iterator[PathPair]:
        """
        iterate over path pairs for a specific key

        Parameters:
        -----------
        key : str

        Yields:
        -------
        PathPair : (key, original_path, reconstruction_path)
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in {self.json_path}")

        ref_path = self.data[key].get("original", "")
        reconstructions = self.data[key].get("reconstructions", [])

        for recon_path in reconstructions:
            yield PathPair(
                key=key, original_path=ref_path, reconstruction_path=recon_path
            )

    def iter_grouped_by_key(self) -> Iterator[Tuple[str, List[PathPair]]]:
        """
        iterate over path pairs grouped by key

        Yields:
        -------
        tuple : (key, list of PathPair)
        """
        for key in self.data.keys():
            pairs = list(self.iter_by_key(key))
            yield key, pairs

    def get_all_pairs(self) -> List[PathPair]:
        """
        Get all path pairs as a list

        Returns:
        --------
        list of PathPair : list of all path pairs
        """
        return list(self)

    def get_keys(self) -> List[str]:
        """
        Get a list of available keys

        Returns:
        --------
        list of str : list of keys available in the JSON file
        """
        return list(self.data.keys())
