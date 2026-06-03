import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from common.graph_util import get_vecs_from_matrices, matrix_to_vec, vec_to_adj_matrix
from common.path_manager import PathManager
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
    dataset, batch_size=64, seed=60, val_ratio=0.2, test_ratio=0.2, shuffle=True
):
    generator = torch.Generator().manual_seed(seed)
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    print(
        f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_kanji_dataloaders(
    batch_size=64, seed=60, val_ratio=0.2, directory="data/kkanji"
):
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: matrix_to_vec(x.squeeze(0))),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(root=directory, transform=transform)
    length = len(dataset)
    print(f"Dataset size: {length} images")
    print(f"Data size: {dataset[0][0].shape}")

    val_size = int(length * val_ratio)
    train_size = length - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def get_mnist_dataloaders(batch_size=64, zoom=True, vectorize=True):
    if vectorize:
        if zoom:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (64, 64), interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: matrix_to_vec(x.squeeze(0))),
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor(), torch.flatten])
    else:
        if zoom:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (64, 64), interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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

    if is_matrix == True:
        data_list = np.array(
            [vec_to_adj_matrix(value[0]) for value in sorted_values]
        ).reshape(-1, 1, 116, 116)
    else:
        data_list = np.array([value[0] for value in sorted_values])

    if is_tensor == True:
        data_list = torch.from_numpy(data_list).to(torch.float32)

    return data_list


def get_sc_dataloader(
    batch_size: int | None = None, is_matrix: bool = False, shuffle: bool = True
):
    """
    Returns DataLoader.
    """
    data_list = GetAllSC(is_matrix=is_matrix, is_tensor=True)

    # indices for spliting dataset
    discovery_indices = []
    train_indices = []
    val_indices = []
    for i in range(1, 107):
        if (i % 4) != 0:
            discovery_indices.append(i)
    for iterations, i in enumerate(discovery_indices):
        if iterations % 4 == 0:
            val_indices.append(i)
        else:
            train_indices.append(i)
    test_indices = np.arange(1, 106, 4) - 1

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


def create_dataloaders(key, shuffle=False):
    manager = PathManager("processed_path_list.json")
    keys = manager.get_all_keys()

    if key not in keys:
        raise ValueError(f"Key {key} not found in processed_path_list.json")

    if key == "sc":
        return get_sc_dataloader(is_matrix=True, shuffle=shuffle)
    else:
        original = manager.get_original_path(key)
        dataset = GraphDataset(original, is_matrix=True)
        return get_dataloaders(dataset, shuffle=shuffle)


def get_testdata_as_numpy(dataset_name: str) -> np.ndarray:
    _, _, test = create_dataloaders(dataset_name)
    test = extract_tensor_from_dataloaders([test]).cpu().numpy()
    return test
