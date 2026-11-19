from pathlib import Path

import torch
import yaml

from dataset.graph_dataset import setup_dataloader


def indices_to_list(indices):
    """Convert Subset.indices to a list of integers that can be saved as JSON."""
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()

    return [int(i) for i in indices]


def main():
    with open("config/paths.yaml", "r") as f:
        path_config = yaml.safe_load(f)

    output = {}

    for dataset_name, dataset_config in path_config["datasets"].items():
        dataset_type = dataset_config["type"]
        path = dataset_config["original"]

        print(f"Processing: {dataset_name}")

        # Fold 0 is sufficient because only the test and discovery splits are needed.
        train_loader, val_loader, test_loader = setup_dataloader(
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            path=path,
            is_matrix=True,
            fold=0,
            shuffle=False,
        )

        train_indices = indices_to_list(train_loader.dataset.indices)
        val_indices = indices_to_list(val_loader.dataset.indices)
        test_indices = indices_to_list(test_loader.dataset.indices)

        # Combine the training and validation indices into the discovery split. Their order is irrelevant.
        discovery_indices = train_indices + val_indices

        assert set(test_indices).isdisjoint(discovery_indices), (
            f"{dataset_name}: test and discovery overlap"
        )

        assert len(set(test_indices)) + len(set(discovery_indices)) == len(
            test_loader.dataset.dataset
        ), f"{dataset_name}: indices do not cover the whole dataset"

        output[dataset_name] = {
            "test": test_indices,
            "discovery": discovery_indices,
        }

        print(f"  discovery: {len(discovery_indices)}, test: {len(test_indices)}")

    output_path = Path("split_indices.yaml")

    with open(output_path, "w") as f:
        yaml.safe_dump(output, f, sort_keys=False, default_flow_style=None)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
