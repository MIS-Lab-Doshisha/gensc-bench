import numpy as np

from common.graph_util import get_thresholded_matrix


def densities_from_adjs(adjs):
    n_nodes = adjs.shape[1]

    i, j = np.triu_indices(n_nodes, k=1)
    edge_counts = np.count_nonzero(adjs[:, i, j], axis=1)

    return edge_counts / len(i)


def calculate_degree_statistics(adjacency_matrices):
    mean_degrees = []

    for adj_matrix in adjacency_matrices:
        adj = np.array(adj_matrix)

        # Sum each row to obtain the corresponding node's degree.
        degrees = np.sum(adj, axis=1)

        mean_deg = np.mean(degrees)
        mean_degrees.append(mean_deg)

    overall_mean = np.mean(mean_degrees)

    # Use zero when there is only one graph, since the sample standard deviation is undefined.
    if len(mean_degrees) > 1:
        overall_std = np.std(mean_degrees, ddof=1)  # Sample standard deviation
    else:
        overall_std = 0

    return overall_mean, overall_std


def find_threshold(
    target: np.ndarray, recon: np.ndarray, thresh_list=np.arange(0.0, 1.0, 0.1)
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
    binarized_target = get_thresholded_matrix(target, 0.0)
    target_mean_degree, _ = calculate_degree_statistics(binarized_target)

    best_thresh = None
    best_diff = float("inf")
    best_recon_mean = None

    densities = densities_from_adjs(recon)
    mean_density_before_thresholding = densities.mean()

    for thresh in thresh_list:
        binarized_recon = get_thresholded_matrix(recon, thresh)
        recon_mean_degree, recon_std_degree = calculate_degree_statistics(
            binarized_recon
        )

        diff = abs(recon_mean_degree - target_mean_degree)
        print(
            f"Threshold: {thresh:.3f}, Recon mean degree: {recon_mean_degree:.4f} ± {recon_std_degree:.4f}, Diff: {diff:.4f}"
        )

        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
            best_recon_mean = recon_mean_degree

            densities_after_thresholding = densities_from_adjs(binarized_recon)
            mean_density_after_thresholding = densities_after_thresholding.mean()

    results = {
        "best_threshold": best_thresh,
        "target_mean_degree": target_mean_degree,
        "recon_mean_degree": best_recon_mean,
        "mean_degree_diff": best_diff,
        "density_before_thresholding": mean_density_before_thresholding,
        "density_after_thresholding": mean_density_after_thresholding,
    }

    return results
