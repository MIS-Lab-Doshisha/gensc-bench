import pickle

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from common.graph_util import get_vecs_from_matrices, vec_to_adj_matrix
from common.util import extract_tensor_from_dataloaders, get_reconstruction_from_npz


class GraphDataset(Dataset):
    def __init__(self, npz_file, is_matrix=True):
        self.graphs = get_reconstruction_from_npz(npz_file)
        if not is_matrix:
            self.graphs = get_vecs_from_matrices(self.graphs.squeeze())

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        x = self.graphs[idx]
        data = torch.tensor(x, dtype=torch.float32)
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        return data


def get_dataloaders(
    dataset, batch_size=64, seed=60, val_ratio=0.2, test_ratio=0.2, fold=0, shuffle=True
):
    if not 0 <= fold < 4:
        raise ValueError("fold must be in range [0, 3]")

    generator = torch.Generator().manual_seed(seed)
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    original_train, original_val, original_test = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    if fold == 0:
        train_set = original_train
        val_set = original_val
    else:
        train_parts = torch.tensor_split(torch.tensor(original_train.indices), 3)

        val_indices = train_parts[fold - 1]
        train_indices = torch.cat(
            [
                torch.tensor(original_val.indices),
                *[train_parts[i] for i in range(3) if i != fold - 1],
            ]
        )

        train_set = Subset(dataset, train_indices.tolist())
        val_set = Subset(dataset, val_indices.tolist())

    print(
        f"dataset split for fold {fold}: train={len(train_set)}, val={len(val_set)}, test={len(original_test)}"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(original_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def GetAllSC(
    data_path: str = "data/sc/input_dict.pkl",
    is_matrix: bool = False,
    is_tensor: bool = False,
):
    """
    Returns all SC data.
    is_matrix determines if SC data is matrix or vector.
    is_tensor determines if SC data is tensor or ndarray.
    """

    with open(data_path, "rb") as f:
        data_dict = pickle.load(
            f
        )  # key: participant_id, value: tuple(connectivity vector, fluid intelligence)

    sorted_values = sorted(data_dict.values(), key=lambda value: value[1])

    if is_matrix:
        data_list = np.array(
            [vec_to_adj_matrix(value[0]) for value in sorted_values]
        ).reshape(-1, 1, 116, 116)
    else:
        data_list = np.array([value[0] for value in sorted_values])

    if is_tensor:
        data_list = torch.from_numpy(data_list).to(torch.float32)

    return data_list


def get_sc_dataloader(
    batch_size: int | None = None,
    is_matrix: bool = False,
    shuffle: bool = True,
    fold: int = 0,  # 0, 1, 2, 3
):
    """
    Returns DataLoader.
    """
    if fold not in [0, 1, 2, 3]:
        raise ValueError("fold must be 0, 1, 2, or 3")

    data_list = GetAllSC(is_matrix=is_matrix, is_tensor=True)
    test_indices = np.arange(1, 106, 4) - 1

    # indices for spliting dataset
    discovery_indices = []
    train_indices = []
    val_indices = []
    for i in range(1, 107):
        if (i % 4) != 0:
            discovery_indices.append(i)
    for iterations, i in enumerate(discovery_indices):
        if iterations % 4 == fold:
            val_indices.append(i)
        else:
            train_indices.append(i)

    print("train indices:", train_indices)
    print("val indices:", val_indices)
    print("test indices:", test_indices)

    # split dataset
    train_dataset = Subset(data_list, train_indices)
    val_dataset = Subset(data_list, val_indices)
    test_dataset = Subset(data_list, test_indices)

    # data loader
    train_dataloader = DataLoader(
        train_dataset, batch_size=20 if not batch_size else batch_size, shuffle=shuffle
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset) if not batch_size else batch_size,
        shuffle=shuffle,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset) if not batch_size else batch_size,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_data_as_numpy(
    path_config, dataset_name, split, fold=0, is_matrix=True
) -> np.ndarray:
    dataset_type = path_config["datasets"][dataset_name]["type"]
    path = path_config["datasets"][dataset_name]["original"]

    train, val, test = setup_dataloader(
        dataset_type,
        dataset_name=dataset_name,
        path=path,
        is_matrix=True,
        fold=fold,
    )
    if split == "train":
        data = extract_tensor_from_dataloaders([train]).cpu().numpy()
    elif split == "val":
        data = extract_tensor_from_dataloaders([val]).cpu().numpy()
    elif split == "discovery":
        data = extract_tensor_from_dataloaders([train, val]).cpu().numpy()
    elif split == "test":
        data = extract_tensor_from_dataloaders([test]).cpu().numpy()
    else:
        raise ValueError(
            f"Invalid split: {split}. Must be 'train', 'val', 'discovery', or 'test'."
        )

    if not is_matrix:
        data = get_vecs_from_matrices(data.squeeze())

    return data


def _make_fold_indices(
    total_len: int, fold: int, seed: int, n_folds: int, test_ratio: float = 0.2
):
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in range [0, {n_folds - 1}]")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_len, generator=generator)

    test_len = int(total_len * test_ratio)
    test_indices = indices[:test_len]
    discovery_indices = indices[test_len:]

    fold_indices = torch.tensor_split(discovery_indices, n_folds)
    val_indices = fold_indices[fold]

    train_indices = torch.cat([fold_indices[i] for i in range(n_folds) if i != fold])

    return train_indices, val_indices, test_indices


def get_hcp_dataloader(
    fold=0,
    batch_size=64,
    shuffle=True,
    is_matrix=False,
):
    with open("config/paths.yaml", "r") as f:
        path_config = yaml.safe_load(f)
    orig_path = path_config["datasets"]["hcp"]["original"]
    data = np.load(orig_path)["adj_matrix"]

    if not is_matrix:
        data = get_vecs_from_matrices(data.squeeze())
    else:
        data = np.expand_dims(data, axis=1)

    dataset = torch.from_numpy(data).to(torch.float32)

    total_len = len(dataset)
    train_indices, val_indices, test_indices = _make_fold_indices(
        total_len, fold=fold, seed=42, n_folds=4, test_ratio=0.2
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # scaler_kwargs = scaler_kwargs or {}
    # scaler = Scaler(**scaler_kwargs)

    # train_dataset = scaler.transform(train_dataset)
    # val_dataset = scaler.transform(val_dataset)
    # test_dataset = scaler.transform(test_dataset)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"HCP Dataset split: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test, total: {len(dataset)}"
    )

    # if return_scaler:
    #     return train_loader, val_loader, test_loader, scaler

    return train_loader, val_loader, test_loader


def setup_dataloader(
    dataset_type, dataset_name=None, path=None, is_matrix=False, **dataloader_kwargs
):
    if dataset_type == "sc":
        if dataset_name == "nimh":
            return get_sc_dataloader(is_matrix=is_matrix, **dataloader_kwargs)
        elif dataset_name == "hcp":
            return get_hcp_dataloader(is_matrix=is_matrix, **dataloader_kwargs)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    elif dataset_type == "synthetic":
        if path is None:
            raise ValueError("Path must be provided for synthetic dataset.")
        dataset = GraphDataset(path, is_matrix=is_matrix)
        return get_dataloaders(dataset, **dataloader_kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def setup_dataloader_from_config(config, is_matrix=False, **dataloader_kwargs):
    return setup_dataloader(
        dataset_type=config["dataset"]["type"],
        dataset_name=config["dataset"].get("name"),
        path=config["dataset"].get("path"),
        is_matrix=is_matrix,
        batch_size=config["training"]["batch_size"],
        **dataloader_kwargs,
    )


if __name__ == "__main__":
    GetAllSC(is_matrix=True, is_tensor=False)
