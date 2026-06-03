import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from common.graph_util import matrix_to_vec
from common.util import set_seed
from dataset.gen_graph import weighted_barabasi_albert_graph

"""
Sort graph nodes and return adjacency matrix.
"""


def get_bfs_ordered_adjacency_matrix(graph, start_node=None):
    """
    Returns the adjacency matrix of a given graph with nodes reordered by BFS traversal.

    Args:
        graph (nx.Graph): A NetworkX graph object.
        start_node: The starting node for BFS. If None, a random node is chosen.

    Returns:
        np.ndarray: The BFS-ordered adjacency matrix.
        list: The new order of nodes determined by BFS.
    """
    if start_node is None:
        start_node = sorted(list(graph.nodes()))[0]
        print(f"BFS starting node: {start_node} (randomly selected)")
    else:
        if start_node not in graph:
            raise ValueError(
                f"Specified start node '{start_node}' does not exist in the graph."
            )
        print(f"BFS starting node: {start_node} (specified node)")

    bfs_order = list(nx.bfs_tree(graph, source=start_node).nodes())

    adj_matrix = nx.to_numpy_array(graph, nodelist=bfs_order)

    return adj_matrix, bfs_order


def get_dfs_ordered_adjacency_matrix(graph, start_node=None):
    """
    Returns the adjacency matrix of a given graph with nodes reordered by DFS traversal.

    Args:
        graph (nx.Graph): A NetworkX graph object.
        start_node: The starting node for DFS. If None, the first node in the sorted node list is used.

    Returns:
        np.ndarray: The DFS-ordered adjacency matrix.
        list: The new order of nodes determined by DFS.
    """
    if not graph.nodes():
        return np.array([]), []

    if start_node is None:
        start_node = sorted(list(graph.nodes()))[0]
    else:
        if start_node not in graph:
            raise ValueError(
                f"Specified start node '{start_node}' does not exist in the graph."
            )

    dfs_order = list(nx.dfs_preorder_nodes(graph, source=start_node))

    adj_matrix = nx.to_numpy_array(graph, nodelist=dfs_order)

    return adj_matrix, dfs_order


def get_degree_ordered_adjacency_matrix(graph):
    """
    Returns the adjacency matrix of a given graph with nodes reordered by degree in descending order.
    Nodes with the same degree are sorted by their original node ID in ascending order (tie-breaking).

    Args:
        graph (nx.Graph): A NetworkX graph object.

    Returns:
        np.ndarray: The degree-ordered adjacency matrix.
        list: The new order of nodes determined by degree.
    """
    if not graph.nodes():
        return np.array([]), []

    degree_dict = dict(graph.degree())

    nodes_with_degrees = [(node, degree_dict[node]) for node in graph.nodes()]

    sorted_nodes_with_degrees = sorted(
        nodes_with_degrees, key=lambda item: (-item[1], item[0])
    )

    nodes_sorted_by_degree = [node for node, degree in sorted_nodes_with_degrees]

    adj_matrix = nx.to_numpy_array(graph, nodelist=nodes_sorted_by_degree)

    return adj_matrix, nodes_sorted_by_degree


def get_kcore_ordered_adjacency_matrix(graph):
    """
    Returns the adjacency matrix of a given graph with nodes reordered based on k-core number and degree.
    Nodes with higher k-core numbers are prioritized. For nodes with the same k-core number, those with higher degrees are prioritized.
    If both k-core number and degree are the same, nodes are sorted by their original node ID in ascending order.

    Args:
        graph (nx.Graph): A NetworkX graph object.

    Returns:
        np.ndarray: The k-core and degree-ordered adjacency matrix.
        list: The new order of nodes determined by k-core and degree.
        dict: A dictionary mapping nodes to their k-core numbers.
    """
    if not graph.nodes():
        return np.array([]), [], {}

    core_numbers = nx.core_number(graph)
    degrees = dict(graph.degree())

    print("k-core numbers for each node:", core_numbers)
    print("Degrees for each node:", degrees)

    nodes = list(graph.nodes())

    kcore_degree_order = sorted(
        nodes, key=lambda node: (-core_numbers[node], -degrees[node], node)
    )

    adj_matrix = nx.to_numpy_array(graph, nodelist=kcore_degree_order)

    return adj_matrix, kcore_degree_order


def get_community_ordered_adjacency_matrix(graph):
    k_core_values = nx.core_number(graph)

    node_data = []
    for node in graph.nodes():
        community_id = graph.nodes[node]["block"]
        core_value = k_core_values[node]
        degree = graph.degree(node)
        node_data.append((node, community_id, core_value, degree))

    sorted_node_data = sorted(node_data, key=lambda x: (x[1], -x[2], -x[3], x[0]))

    sorted_nodes = [data[0] for data in sorted_node_data]
    ordered_matrix = nx.to_numpy_array(graph, nodelist=sorted_nodes)

    return ordered_matrix, sorted_nodes


"""
Save binary Barabasi-Albert, Watts-Strogatz, Stochastic Block Model, Planar graphs' adjacency matrix.
"""


def save_ba_adj_matrix(num=500, n=64, m=4, output_path="data/raw"):
    ba = []
    for i in range(num):
        graph = nx.barabasi_albert_graph(n, m)
        ba.append(graph)
    ba_adj_matrix = [get_dfs_ordered_adjacency_matrix(graph)[0] for graph in ba]
    ba_adj_matrix = np.array(ba_adj_matrix)
    print(f"BA graph generated: {ba_adj_matrix.shape}")

    np.savez(f"{output_path}/ba_dfs_adj_matrix.npz", adj_matrix=ba_adj_matrix)


def save_ws_adj_matrix(num=500, n=64, k=6, p=0.05, output_path="data/raw"):
    ws = []
    for i in range(num):
        graph = nx.watts_strogatz_graph(n, k, p)
        ws.append(graph)
    ws_adj_matrix = [get_kcore_ordered_adjacency_matrix(graph)[0] for graph in ws]
    ws_adj_matrix = np.array(ws_adj_matrix)
    print(f"WS graph generated: {ws_adj_matrix.shape}")

    np.savez(f"{output_path}/ws_kcore_adj_matrix.npz", adj_matrix=ws_adj_matrix)


def save_sbm_adj_matrix(num=500, n=64, p_in=0.3, p_out=0.05, output_path="data/raw"):
    num_communities = 2
    block_matrix = np.full((num_communities, num_communities), p_out)
    np.fill_diagonal(block_matrix, p_in)

    sbm = []
    for i in range(num):
        community_node = random.randint(27, 37)
        sbm.append(
            nx.stochastic_block_model(
                [community_node, 64 - community_node], block_matrix
            )
        )
    sbm_adj_matrix = [get_community_ordered_adjacency_matrix(graph)[0] for graph in sbm]
    sbm_adj_matrix = np.array(sbm_adj_matrix)
    print(f"SBM graph generated: {sbm_adj_matrix.shape}")

    np.savez(f"{output_path}/sbm_adj_matrix.npz", adj_matrix=sbm_adj_matrix)


def save_planar_adj_matrix(output_path="data/raw"):
    planar = []
    data = torch.load("data/raw/planar_64_200.pt")
    dataset = data[0]
    dataset = [t.detach().cpu().numpy() for t in dataset]
    dataset = np.array(dataset)
    print("planar dataset loaded:", dataset.shape)

    planar = [nx.from_numpy_array(matrix) for matrix in dataset]
    planar_adj_matrix = [get_dfs_ordered_adjacency_matrix(graph)[0] for graph in planar]
    planar_adj_matrix = np.array(planar_adj_matrix)

    print(f"Planar graph generated: {planar_adj_matrix.shape}")

    np.savez(f"{output_path}/planar_dfs_adj_matrix.npz", adj_matrix=planar_adj_matrix)


def save_all_adj_matrices(num=500, output_path="data/raw", seed=64):
    set_seed(seed)
    save_ba_adj_matrix(num=num, output_path=output_path)
    save_ws_adj_matrix(num=num, output_path=output_path)
    save_sbm_adj_matrix(num=num, output_path=output_path)
    save_planar_adj_matrix(output_path=output_path)


"""
Save weighted Barabasi-Albert graphs' adjacency matrix.
"""


def save_wba_adj_matrix(num=500, n=64, m=4, output_path="data/continuous/raw"):
    set_seed(66)
    ba = []
    for i in range(num):
        graph = weighted_barabasi_albert_graph(n, m)
        ba.append(graph)
    ba_adj_matrix = [get_dfs_ordered_adjacency_matrix(graph)[0] for graph in ba]
    ba_adj_matrix = np.array(ba_adj_matrix)

    print(f"BA graph generated: {ba_adj_matrix.shape}")

    np.savez(f"{output_path}/wba_dence_dfs_adj_matrix.npz", adj_matrix=ba_adj_matrix)

def save_vectorized_adj_matrix(data, dataname, output_path="data/processed"):
    # data: list of adjacency matrix
    num, node = len(data), len(data[0])

    vec_length = node * (node - 1) // 2
    adj_vec = np.zeros((num, vec_length))
    for i, matrix in enumerate(data):
        adj_vec[i] = matrix_to_vec(matrix)
        assert len(adj_vec[i]) == vec_length, (
            f"Vector length mismatch: {len(adj_vec[i])} != {vec_length}"
        )

    print(f"Vectorized adjacency matrix shape: {adj_vec.shape}")

    # save vectorized adjacency matrix
    np.savez(f"{output_path}/{dataname}_matrix.npz", adj_matrix=adj_vec)


def show_graphs(npz_path, name, num_graphs=9):
    data = np.load(npz_path, allow_pickle=True)
    adj_matrices = data["adj_matrix"]

    if len(adj_matrices) < num_graphs:
        raise ValueError(f"Number of graphs in the file is less than {num_graphs}.")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_graphs):
        ax = axes[i]
        ax.matshow(adj_matrices[i], cmap="viridis")
        ax.set_title(f"Graph {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{name}_adj_visualization.pdf")
    plt.show()


def show_avg(npz_path, name):
    data = np.load(npz_path, allow_pickle=True)
    adj_matrices = data["adj_matrix"]

    avg_matrix = np.mean(adj_matrices, axis=0)

    plt.matshow(avg_matrix, cmap="gist_heat")
    plt.savefig(f"{name}_avg_adj_visualization.pdf")



