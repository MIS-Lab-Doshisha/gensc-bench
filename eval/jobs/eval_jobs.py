from typing import Any

import networkx as nx
import numpy as np

from common.graph_util import get_thresholded_matrix, get_topn_adj_list
from eval.core.graph_mmd import (
    clustering_stats,
    degree_stats,
    orbit_stats_all,
    spectral_stats,
)
from eval.core.graph_statistics import get_planarity_rate


def evaluate_graphs(
    ref_list: np.ndarray,
    generated_list: np.ndarray,
    logger,
    thresh: float | None = None,
    top_n_edges: float | None = None,
    calculate_planarity: bool = False,
) -> dict[str, Any]:
    """
    Evaluate the graphs.

    Args:
        ref_list (np.ndarray): List of reference (ground-truth) data.
        generated_list (np.ndarray): List of generated data.
        logger: Logger.
        thresh (float, optional): Threshold. If specified, the matrices are binarized.
        top_n_edges (float, optional): Consider the top n edges.
        calculate_planarity (bool, optional): Whether to calculate the planarity rate.
    """

    is_parallel = False

    print(f"Reference data shape: {ref_list.shape}")
    print(f"Generated data shape: {generated_list.shape}")
    logger.info(f"thresh: {thresh}, top_n_edges: {top_n_edges}")

    if thresh is not None:
        ref_list = get_thresholded_matrix(ref_list, thresh)
        generated_list = get_thresholded_matrix(generated_list, thresh)

    if top_n_edges is not None:
        ref_list = get_topn_adj_list(ref_list, top_n_edges)
        generated_list = get_topn_adj_list(generated_list, top_n_edges)

    # Convert adjacency matrices to NetworkX graphs
    ref_graphs = [nx.from_numpy_array(adj) for adj in ref_list]
    pred_graphs = [nx.from_numpy_array(adj) for adj in generated_list]

    degree_mmd = degree_stats(ref_graphs, pred_graphs, is_parallel)
    logger.info(f"Degree MMD: {degree_mmd}")

    clustering_mmd = clustering_stats(
        ref_graphs, pred_graphs, bins=100, is_parallel=is_parallel
    )
    logger.info(f"Clustering Coefficient MMD: {clustering_mmd}")

    orbit_mmd = orbit_stats_all(ref_graphs, pred_graphs)
    logger.info(f"Orbit MMD: {orbit_mmd}")

    spectral_mmd = spectral_stats(ref_graphs, pred_graphs, is_parallel=is_parallel)
    logger.info(f"Spectral MMD: {spectral_mmd}")

    if calculate_planarity:
        planar_validity_rate = get_planarity_rate(generated_list)
        logger.info(f"Planarity rate: {planar_validity_rate}")

    results = {
        "Degree MMD": degree_mmd,
        "Clustering MMD": clustering_mmd,
        "Orbit MMD": orbit_mmd,
        "Spectral MMD": spectral_mmd,
    }
    if calculate_planarity:
        results["planar_validity_rate"] = planar_validity_rate

    return results
