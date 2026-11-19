import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def calc_modularities(matrices: np.ndarray | list[np.ndarray]):
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


def calc_edge_variability(matrices: np.ndarray) -> np.ndarray:
    if len(matrices) == 0:
        raise ValueError("The input matrices array is empty.")

    edge_variability = np.std(matrices, axis=0, ddof=1)

    return edge_variability


def calc_nearest_neighbor(
    generated: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    generated_flat = generated.reshape(generated.shape[0], -1)
    test_flat = test.reshape(test.shape[0], -1)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(test_flat)

    distances, indices = nn.kneighbors(generated_flat)

    # i = np.argmin(distances[:, 0])
    # j = indices[i, 0]

    # min_distance = distances[i, 0]

    return distances[:, 0], indices[:, 0]
