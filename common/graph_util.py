import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import Normalize


def vec_to_adj_matrix(vec):
    is_tensor = False
    if type(vec) == torch.Tensor:
        is_tensor = True
        vec = vec.detach().cpu().numpy()

    discriminant = 1 + 8 * len(vec)
    sqrt = math.isqrt(discriminant)
    if sqrt**2 != discriminant or (-1 + sqrt) % 2 != 0:
        raise ValueError("length of the input vector is not a triangular number")

    n = int((-1 + np.sqrt(1 + 8 * len(vec))) / 2 + 1)

    matrix = np.zeros((n, n))
    matrix[np.tril_indices_from(matrix, k=-1)] = vec
    result = np.tril(matrix, k=-1) + np.tril(matrix, k=-1).T

    if is_tensor:
        result = torch.from_numpy(result).to(torch.float32)

    return result


def matrix_to_vec(matrix):
    is_tensor = False
    if type(matrix) == torch.Tensor:
        is_tensor = True
        matrix = matrix.detach().cpu().numpy()

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Input is not a square matrix. matrix shape: {matrix.shape}")

    tril_indices = np.tril_indices_from(matrix, k=-1)
    result = matrix[tril_indices]

    if is_tensor:
        result = torch.from_numpy(result).to(torch.float32)

    return result


def get_vecs_from_matrices(matrices):
    is_tensor = False
    if type(matrices) == list:
        matrices = np.array(matrices)
    if type(matrices) == torch.Tensor:
        is_tensor = True
        matrices = matrices.detach().cpu().numpy()
    if len(matrices.shape) != 3:
        raise ValueError(
            f"Input is not a list of matrices. matrices shape: {matrices.shape}"
        )

    vecs = []
    for matrix in matrices:
        vec = matrix_to_vec(matrix)
        vecs.append(vec)
    vecs = np.array(vecs)

    if is_tensor:
        vecs = torch.from_numpy(vecs).to(torch.float32)

    return vecs


def get_matrices_from_vecs(vecs):
    is_tensor = False
    if type(vecs) == list:
        matrices = np.array(vecs)
    if type(vecs) == torch.Tensor:
        is_tensor = True
        vecs = vecs.detach().cpu().numpy()
    if len(vecs.shape) != 2:
        raise ValueError(f"Input is not a list of vectors. vecs shape: {vecs.shape}")

    matrices = []
    for vec in vecs:
        matrix = vec_to_adj_matrix(vec)
        matrices.append(matrix)
    matrices = np.array(matrices)

    if is_tensor:
        matrices = torch.from_numpy(matrices).to(torch.float32)

    return matrices


def min_max_normalization(matrix, to_01=True):
    is_tensor = False
    if type(matrix) == list:
        matrix = np.array(matrix)
    if type(matrix) == torch.Tensor:
        is_tensor = True
        matrix = matrix.detach().cpu().numpy()

    min_val = matrix.min()
    max_val = matrix.max()
    if max_val - min_val < 1e-6:
        normalized = matrix - min_val
    else:
        normalized = (matrix - min_val) / (max_val - min_val)

    if not to_01:
        normalized = normalized * 2 - 1  # Scale to [-1, 1]

    if is_tensor:
        normalized = torch.from_numpy(normalized).to(torch.float32)

    return normalized


def get_thresholded_matrix(matrices, threshold=0.0, is_weighted=False):
    is_tensor = False
    if type(matrices) == list:
        matrices = np.array(matrices)
    if type(matrices) == torch.Tensor:
        is_tensor = True
        matrices = matrices.detach().cpu().numpy()

    if is_weighted:
        threholded = matrices.copy()
        threholded = np.where(threholded > threshold, threholded, 0)
    else:
        threholded = matrices.copy()
        threholded = np.where(threholded > threshold, 1, 0)

    if is_tensor:
        threholded = torch.from_numpy(threholded).to(torch.float32)

    return threholded


def plot_weighted_graph(g: nx.Graph, path=None):
    fig, ax = plt.subplots()
    edges, weights = zip(*nx.get_edge_attributes(g, "weight").items())
    pos = nx.arf_layout(g)  # positions for all nodes
    nx.draw(
        g,
        pos,
        ax=ax,
        with_labels=True,
        edge_color=weights,
        edge_cmap=plt.cm.cool,
        edgelist=edges,
        node_size=200,
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.cool, norm=Normalize(vmin=min(weights), vmax=max(weights))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Edge Weight")
    plt.savefig(path)


def get_top_n_edges(adj, n, weighted=True):
    """
    Args:
        adj (ndarray): Adjacency matrix
        n: ratio

    Returns:
        A vector that has top n edges from input adjacency matrix
    """
    if n is None:
        return adj
    if n <= 0 or n > 1:
        raise ValueError("n should be in (0, 1]")
    vec = matrix_to_vec(adj)

    weights_sorted = np.sort(vec)[::-1]
    num = int(len(vec) * n)
    threshold = weights_sorted[num - 1]

    if weighted:
        masked_vec = np.where(vec < threshold, 0, vec)
    else:
        masked_vec = np.where(vec < threshold, 0, 1)

    adj = vec_to_adj_matrix(masked_vec)

    return adj


def get_topn_adj_list(adj_list, top_n):
    """
    Args:
        adj_list (list of ndarray): List of adjacency matrices
        thresh: ratio

    Returns:
        A list of adjacency matrices that have top n edges from input adjacency matrices
    """
    if top_n is None:
        return adj_list
    else:
        return [get_top_n_edges(adj, top_n) for adj in adj_list]


def get_topns(dataset_name):
    if dataset_name == "sc":
        return [0.05, 0.1, 0.2, 0.3]
    else:
        return [None]
