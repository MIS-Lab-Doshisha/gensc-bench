import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from common.graph_util import get_top_n_edges, get_vecs_from_matrices


def get_hubs_spearman(adj1, adj2):
    """
    Args:
        adj1 (ndarray): Adjacency matrix 1
        adj2 (ndarray): Adjacency matrix 2
        n: ratio

    Returns:
        Spearman correlation between node strengths of two adjacency matrices
    """
    strength1 = np.sum(adj1, axis=0)
    strength2 = np.sum(adj2, axis=0)

    correlation, p_value = spearmanr(strength1, strength2)

    return correlation, p_value


def get_edge_jaccard(adj1, adj2, n=None):
    adj1 = get_top_n_edges(adj1, n, weighted=False)
    adj2 = get_top_n_edges(adj2, n, weighted=False)

    intersection = np.logical_and(adj1 > 0, adj2 > 0).sum()
    union = np.logical_or(adj1 > 0, adj2 > 0).sum()

    jaccard = intersection / union if union != 0 else 0.0

    return jaccard


def calc_pairwise_distances(samples: np.ndarray):
    nsamples = samples.shape[0]

    flattened = get_vecs_from_matrices(samples)
    distances = pdist(flattened, metric="euclidean")

    results = []
    k = 0

    for i in range(nsamples):
        for j in range(i + 1, nsamples):
            results.append((i, j, distances[k]))
            k += 1

    return results
