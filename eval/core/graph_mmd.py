import os
import subprocess as sp

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigvalsh

import eval.core.mmd_gpu as mmd

""" Degree """


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    if is_parallel:
        sample_ref = Parallel(n_jobs=-1)(
            delayed(degree_worker)(G) for G in graph_ref_list
        )
        sample_pred = Parallel(n_jobs=-1)(
            delayed(degree_worker)(G) for G in graph_pred_list_remove_empty
        )
    else:
        sample_ref = [degree_worker(G) for G in graph_ref_list]
        sample_pred = [degree_worker(G) for G in graph_pred_list_remove_empty]

    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred)

    return mmd_dist


"""Clustering Coefficient"""


def clustering_worker(graph, bins):
    if graph.number_of_nodes() == 0:
        return np.zeros(bins)

    clustering_coeffs = list(nx.clustering(graph).values())
    hist, _ = np.histogram(
        clustering_coeffs, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if G.number_of_nodes() > 0
    ]

    def process_graphs(graph_list):
        if is_parallel:
            return Parallel(n_jobs=-1)(
                delayed(clustering_worker)(g, bins) for g in graph_list
            )
        else:
            return [clustering_worker(g, bins) for g in graph_list]

    sample_ref = process_graphs(graph_ref_list)
    sample_pred = process_graphs(graph_pred_list_remove_empty)

    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred)

    return mmd_dist


"""orbit"""
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts: \n"


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_fname = "eval/orca/tmp.txt"
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    f.writelines(str(u) + " " + str(v) + "\n" for u, v in edge_list_reindexed(graph))
    f.close()

    output = sp.check_output(
        ["./eval/orca/orca", "node", "4", "eval/orca/tmp.txt", "std"]
    )
    output = output.decode("utf8").strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list_remove_empty:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, is_hist=False)

    print("-------------------------")
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print("...")
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print("-------------------------")
    return mmd_dist


"""Spectral"""


def spectral_worker(G, n_eigvals=-1):
    G_main = G.copy()
    G_main.remove_nodes_from(list(nx.isolates(G_main)))

    # Skip graphs with fewer than two nodes.
    if G_main.number_of_nodes() < 2:
        return np.zeros(200)

    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        print("eigvalsh failed, returning zero array")
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """
    Compute the MMD between the spectral distributions (eigenvalue histograms) of two sets of graphs.

    Args:
        graph_ref_list (list): Reference graphs as NetworkX objects.
        graph_pred_list (list): Generated graphs as NetworkX objects.
        is_parallel (bool): Whether to enable parallel processing.
        n_eigvals (int): Number of eigenvalues to use. Use -1 to include all eigenvalues.
        compute_emd (bool): Unused here. Kept for interface consistency.
    """
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if G.number_of_nodes() > 0
    ]

    def process_graph_list(graph_list):
        """Apply spectral_worker to each graph and return a list of PMFs."""
        if is_parallel:
            return Parallel(n_jobs=-1)(
                delayed(spectral_worker)(G, n_eigvals) for G in graph_list
            )
        else:
            return [spectral_worker(G, n_eigvals) for G in graph_list]

    sample_ref = process_graph_list(graph_ref_list)
    sample_pred = process_graph_list(graph_pred_list_remove_empty)

    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred)

    return mmd_dist


def strength_worker(matrix):
    """
    Compute the node strengths from a weighted adjacency matrix.
    Each node's strength is the sum of its row in the matrix.
    """
    strengths = np.sum(matrix, axis=1)
    return strengths[strengths > 0]


def strength_stats(matrix_ref_list, matrix_pred_list, is_parallel=True):
    """
    Compute the maximum mean discrepancy (MMD) between the node strength
    distributions of two lists of weighted adjacency matrices.

    Parameters:
        matrix_ref_list (list): Reference adjacency matrices.
        matrix_pred_list (list): Predicted adjacency matrices.
        is_parallel (bool): Enable parallel processing when True. Defaults to True.

    Returns:
        float: MMD between the reference and predicted node strength distributions.
    """
    matrix_pred_list_remove_empty = [m for m in matrix_pred_list if m.sum() > 0]

    def process_matrix_list(matrix_list):
        if is_parallel:
            return Parallel(n_jobs=-1)(delayed(strength_worker)(G) for G in matrix_list)
        else:
            return [strength_worker(G) for G in matrix_list]

    sample_ref = process_matrix_list(matrix_ref_list)
    sample_pred = process_matrix_list(matrix_pred_list_remove_empty)
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred)

    return mmd_dist
