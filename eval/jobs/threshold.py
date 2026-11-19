import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from common.graph_util import get_thresholded_matrix
from common.path_resolver import get_generated_path, get_thresholded_path
from common.util import get_reconstruction_from_npz
from dataset.graph_dataset import GraphDataset, get_dataloaders
from eval.core.threshold import find_threshold


def get_training_matrices(original_path, fold):
    dataset = GraphDataset(
        original_path,
        is_matrix=True,
    )

    train_loader, *_ = get_dataloaders(dataset, fold=fold, batch_size=1)
    training_matrices = np.concatenate([data.numpy() for data in train_loader], axis=0)

    return training_matrices


def save_thresholded_matrices(
    recon_path: str,
    original_path: str,
    matrix_output_dir: os.PathLike,
    fold: int,
    is_weighted: bool,
):
    target = get_training_matrices(original_path, fold)
    recon = get_reconstruction_from_npz(recon_path)

    # thresh = np.linspace(0.1, 0.49, 40)
    # thresh = np.linspace(0.01, 0.69, 70)
    thresh = np.linspace(0.01, 0.99, 100)
    thresh = np.round(thresh, decimals=2)

    best_thresh_results = find_threshold(target, recon, thresh_list=thresh)
    best_thresh = best_thresh_results["best_threshold"]

    matrices = get_thresholded_matrix(recon, best_thresh, is_weighted=is_weighted)

    with open(Path(matrix_output_dir) / "best_threshold.json", "w") as f:
        json.dump(
            {
                "original_path": str(original_path),
                "recon_path": str(recon_path),
                "fold": fold,
                **best_thresh_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=float,
        )

    np.savez_compressed(Path(matrix_output_dir) / "thresholded_matrices.npz", matrices)


def save_all_thresholded_matrices(path_config, datasets, models=None):
    if models is None:
        models = path_config["models"]
    for dataset in datasets:
        dataset_config = path_config["datasets"][dataset]
        if not dataset_config["thresholding"]:
            continue

        for model in models:
            for run, run_config in path_config["runs"].items():
                recon_path = get_generated_path(path_config, dataset, model, run)
                matrix_output_dir = get_thresholded_path(
                    path_config, dataset, model, run
                ).parent
                matrix_output_dir.mkdir(exist_ok=True, parents=True)
                original_path = dataset_config["original"]

                fold = run_config["fold"]

                save_thresholded_matrices(
                    recon_path,
                    original_path,
                    matrix_output_dir,
                    fold,
                    is_weighted=dataset_config["weighted"],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save thresholded matrices")
    parser.add_argument(
        "--path_config",
        type=str,
        default="config/paths.yaml",
        help="Path to the path_config yaml file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="List of datasets to process",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of models to process",
    )

    args = parser.parse_args()

    with open(args.path_config, "r") as f:
        path_config = yaml.safe_load(f)

    save_all_thresholded_matrices(path_config, args.datasets, args.models)
