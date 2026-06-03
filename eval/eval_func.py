import logging
import multiprocessing
import os
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

from common.graph_util import get_thresholded_matrix, get_top_n_edges, get_topn_adj_list
from common.util import (
    PathListIterator,
    get_reconstruction_from_npz,
    set_seed,
    setup_logger,
)
from eval.graph_mmd import (
    clustering_stats,
    degree_stats,
    orbit_stats_all,
    spectral_stats,
)


def calc_planarity(matrices):
    n = len(matrices)
    planar_count = 0

    for matrix in matrices:
        G = nx.from_numpy_array(matrix)
        is_planar, _ = nx.check_planarity(G)
        if is_planar:
            planar_count += 1

    return planar_count / n


def calculate_degree_statistics(adjacency_matrices):
    mean_degrees = []

    for adj_matrix in adjacency_matrices:
        # 隣接行列をNumPy配列に変換
        adj = np.array(adj_matrix)

        # 各ノードの次数を計算 (行列の各行の和が次数となる)
        # axis=1で行ごとに合計する
        degrees = np.sum(adj, axis=1)

        # このグラフの平均次数を計算し、リストに追加
        mean_deg = np.mean(degrees)
        mean_degrees.append(mean_deg)

    # 全グラフの平均次数リストから、全体の平均と標準偏差を計算
    overall_mean = np.mean(mean_degrees)

    # グラフが1つの場合は標準偏差が計算できないため0を返す
    if len(mean_degrees) > 1:
        overall_std = np.std(mean_degrees, ddof=1)  # 標本標準偏差
    else:
        overall_std = 0

    return overall_mean, overall_std


def find_threshold(
    target, recon, thresh_list=np.arange(0.0, 1.0, 0.1), output_path=None
):
    """
    Find the optimal threshold for binarizing a reconstruction matrix based on the mean degree of a target graph.
    Parameters:
        target (array-like): The target graph represented as a matrix or similar structure.
        recon (array-like): The reconstructed graph to be evaluated.
        thresh_list (array-like, optional): A list of thresholds to evaluate. Defaults to np.arange(0.0, 1.0, 0.1).
        output_path (str, optional): The path to save logs. If None, logs will be printed to the console.
    Returns:
        float: The best threshold that minimizes the difference in mean degree between the target and the reconstructed graph.
    """
    logger = logging.getLogger("thresh")
    log_func = logger.info

    target_mean_degree, target_std_degree = calculate_degree_statistics(target)
    log_func(f"Target mean degree: {target_mean_degree:.4f} ± {target_std_degree:.4f}")

    best_thresh = None
    best_diff = float("inf")
    best_recon_mean = None

    for thresh in thresh_list:
        binarized_recon = get_thresholded_matrix(recon, thresh)
        recon_mean_degree, recon_std_degree = calculate_degree_statistics(
            binarized_recon
        )

        diff = abs(recon_mean_degree - target_mean_degree)
        # print(f"Threshold: {thresh:.3f}, Recon mean degree: {recon_mean_degree:.4f} ± {recon_std_degree:.4f}, Diff: {diff:.4f}")

        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
            best_recon_mean = recon_mean_degree

    log_func(
        f"Best threshold: {best_thresh:.3f}, Recon mean degree: {best_recon_mean:.4f} ± {recon_std_degree:.4f}, Diff: {best_diff:.4f}"
    )

    return best_thresh


def get_best_thresholded_matrices(target, recon, thresh_list=np.arange(0.0, 1.0, 0.1)):
    """
    Get the best thresholded matrix based on the target and reconstruction matrices.
    Parameters:
        target (np.ndarray): The target matrix used for comparison.
        recon (np.ndarray): The reconstructed matrix to be thresholded.
        thresh_list (np.ndarray, optional): A list of thresholds to evaluate. Defaults to np.arange(0.0, 1.0, 0.1).
    Returns:
        np.ndarray: The best thresholded matrix derived from the reconstruction matrix.
    Logs:
        - Information about whether the target matrices are weighted.
        - The best threshold determined during the evaluation.
    """

    logger = logging.getLogger("thresh")
    log_func = logger.info

    is_weighted = len(np.unique(target)) > 2
    if is_weighted:
        log_func("Target matrices are weighted.")

    best_thresh = find_threshold(target, recon, thresh_list=thresh_list)
    log_func(f"Best threshold determined: {best_thresh}")
    best_thresholded = get_thresholded_matrix(
        recon, best_thresh, is_weighted=is_weighted
    )
    return best_thresholded


def save_all_thresholded_matrices():
    iterator = PathListIterator("path_list.json")
    output_path = "log/thresholds.log"
    setup_logger("thresh", output_path)
    logger = logging.getLogger("thresh")
    for path in iterator:
        target_path = path.reference_path
        recon_path = path.reconstruction_path
        logger.info(f"Reference data: {target_path}, predicted data: {recon_path}")

        target = get_reconstruction_from_npz(target_path)
        recon = get_reconstruction_from_npz(recon_path)
        thresh = np.linspace(0.1, 0.49, 40)
        thresh = np.round(thresh, decimals=2)
        matrices = get_best_thresholded_matrices(target, recon, thresh_list=thresh)

        parts = Path(recon_path).parts
        last_two_dirs = parts[-3:-1]
        output_dir = Path("output/thresholded").joinpath(*last_two_dirs)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(output_dir / "thresholded_matrices.npz", matrices)


def calc_single_sigma(matrix):
    """
    Worker function to calculate small-worldness for a single adjacency matrix.
    """
    pid = os.getpid()
    size_ratio = 1.0
    try:
        G = nx.from_numpy_array(matrix)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

            size_ratio = len(largest_cc) / matrix.shape[0]
            print(
                f"[PID:{pid}] Graph not connected. Using LCC with size ratio: {size_ratio:.4f}"
            )

        # sigma = nx.algorithms.smallworld.sigma(G, seed=42)
        sigma = sigma_np(G, seed=42)
        print(f"[PID:{pid}] Calculated sigma: {sigma:.4f}")
        return sigma, size_ratio

    except Exception as e:
        # エラーが発生した場合はNoneを返し、処理を続行
        pid = os.getpid()
        print(f"[PID:{pid}] Error processing a matrix: {e}")
        return None, None


def calc_smallworldness(matrices, logger=None, output_path=None):
    log_func = print
    if logger:
        log_func = logger.info
    with multiprocessing.Pool(processes=20) as pool:
        results_raw = pool.map(calc_single_sigma, matrices)

    valid_results = [res for res in results_raw if not np.isnan(res[0])]
    sigma_values, size_ratio_values = zip(*valid_results)
    log_func(f"mean sigma: {np.mean(sigma_values)}, std sigma: {np.std(sigma_values)}")
    log_func(f"mean size ratio: {np.mean(size_ratio_values)}")

    return np.mean(sigma_values)


def calculate_density(G, community):
    comm1, comm2 = community
    n1, n2 = len(comm1), len(comm2)
    if n1 == 0 or n2 == 0:
        raise ValueError("One of the communities is empty.")

    subgraph1 = G.subgraph(comm1)
    subgraph2 = G.subgraph(comm2)
    edges_intra = subgraph1.number_of_edges() + subgraph2.number_of_edges()

    max_edges_intra = (n1 * (n1 - 1) / 2) + (n2 * (n2 - 1) / 2)
    p_in = edges_intra / max_edges_intra

    edges_between = G.number_of_edges() - edges_intra
    max_edges_between = n1 * n2
    p_out = edges_between / max_edges_between

    return p_in, p_out


def calc_modularities(matrices: np.ndarray):
    modularities = []
    for matrix in matrices:
        G = nx.from_numpy_array(matrix)
        isolated = list(nx.isolates(G))
        if isolated:
            G.remove_nodes_from(isolated)
            print(f"Removed {len(isolated)} isolated nodes.")
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)

        modularities.append(modularity)

    return modularities


def calc_wasserstein_distance(target, recon):
    return wasserstein_distance(target, recon)


def evaluate_graphs(
    ref_list: np.ndarray,
    pred_list: np.ndarray,
    logger,
    thresh: float | None = None,
    top_n_edges: float | None = None,
    calculate_planarity: bool = False,
):
    """
    グラフ評価を実行する関数。

    Args:
        ref_path (str): 参照（正解）データのファイルパス。
        pred_path (str): 予測データのファイルパス。
        thresh (float, optional): しきい値。指定した場合、行列を二値化する。
        calculate_planarity (bool, optional): Planarity rateを計算するかどうか。
        log_file (str, optional): ログファイルのパス。
    """

    is_parallel = False

    print(f"Reference data shape: {ref_list.shape}")
    print(f"Predicted data shape: {pred_list.shape}")

    logger.info(f"thresh: {thresh}, top_n_edges: {top_n_edges}")

    if thresh is not None:
        ref_list = get_thresholded_matrix(ref_list, thresh)
        pred_list = get_thresholded_matrix(pred_list, thresh)

    if top_n_edges is not None:
        ref_list = get_topn_adj_list(ref_list, top_n_edges)
        pred_list = get_topn_adj_list(pred_list, top_n_edges)

    # Convert adjacency matrices to NetworkX graphs
    ref_graphs = [nx.from_numpy_array(adj) for adj in ref_list]
    pred_graphs = [nx.from_numpy_array(adj) for adj in pred_list]

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
        planar_validity_rate = calc_planarity(pred_list)
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


def random_reference(G, niter=1, connectivity=True, seed=None):
    """Compute a random graph by swapping edges of a given graph.
    NOTICE: rewrited from networkx to fix the connectivity checking algorithm.

    Parameters
    ----------
    G : graph
        An undirected graph with 4 or more nodes.

    niter : integer (optional, default=1)
        An edge is rewired approximately `niter` times.

    connectivity : boolean (optional, default=True)
        When True, ensure connectivity for the randomized graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
        The randomized graph.

    Raises
    ------
    NetworkXError
        If there are fewer than 4 nodes or 2 edges in `G`

    Notes
    -----
    The implementation is adapted from the algorithm by Maslov and Sneppen
    (2002) [1]_.

    References
    ----------
    .. [1] Maslov, Sergei, and Kim Sneppen.
           "Specificity and stability in topology of protein networks."
           Science 296.5569 (2002): 910-913.
    """
    if seed is None:
        seed = np.random.default_rng()
    else:
        seed = np.random.default_rng(seed)

    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
    if len(G.edges) < 2:
        raise nx.NetworkXError("Graph has fewer that 2 edges")

    from networkx.utils import cumulative_distribution, discrete_sequence

    G = G.copy()
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = cumulative_distribution(degrees)  # cdf of degree
    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0

    for i in range(niter):
        n = 0
        while n < ntries:
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ai, ci) = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ai == ci:
                continue  # same source, skip
            a = keys[ai]  # convert index to label
            c = keys[ci]
            # choose target uniformly from neighbors
            b = seed.choice(list(G.neighbors(a)))
            d = seed.choice(list(G.neighbors(c)))
            if b in [a, c, d] or d in [a, b, c]:
                continue  # all vertices should be different

            # don't create parallel edges
            if (d not in G[a]) and (b not in G[c]):
                G.add_edge(a, d)
                G.add_edge(c, b)
                G.remove_edge(a, b)
                G.remove_edge(c, d)

                # Check if the graph is still connected
                if connectivity and not nx.is_connected(G):
                    # Not connected, revert the swap
                    G.remove_edge(a, d)
                    G.remove_edge(c, b)
                    G.add_edge(a, b)
                    G.add_edge(c, d)
                else:
                    swapcount += 1
                    break
            n += 1
    return G


def sigma_np(G, niter=100, nrand=10, seed=42):
    set_seed(seed)

    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    randMetrics = {"C": [], "L": []}
    for i in range(nrand):
        Gr = random_reference(G, niter=niter, seed=seed)
        # print("generated random graph")
        randMetrics["C"].append(nx.transitivity(Gr))
        # print("calculated transitivity")
        randMetrics["L"].append(
            nx.average_shortest_path_length(Gr, method="floyd-warshall-numpy")
        )
        # print("calculated average shortest path length")

    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G, method="floyd-warshall-numpy")
    Cr = np.mean(randMetrics["C"])
    Lr = np.mean(randMetrics["L"])

    sigma = (C / Cr) / (L / Lr)

    return float(sigma)


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


def get_planarity_rate(adj_matrices):
    planar_count = 0
    total_count = len(adj_matrices)

    for adj in adj_matrices:
        G = nx.from_numpy_array(adj)
        is_planar, _ = nx.check_planarity(G)
        if is_planar:
            planar_count += 1

    planarity_rate = planar_count / total_count if total_count > 0 else 0.0
    return planarity_rate
