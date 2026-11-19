import os

import networkx as nx
import numpy as np

from common.util import set_seed


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
                # if connectivity and not nx.is_connected(G):
                if connectivity and not nx.has_path(G, a, b):
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
        rand_seed = seed + i
        Gr = random_reference(G, niter=niter, seed=rand_seed)
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


def calc_single_sigma(matrix) -> tuple[float, float]:
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

        sigma = sigma_np(G, seed=42)
        print(f"[PID:{pid}] Calculated sigma: {sigma:.4f}")
        return sigma, size_ratio

    except Exception as e:
        # Return None so processing can continue after an error.
        pid = os.getpid()
        print(f"[PID:{pid}] Error processing a matrix: {e}")
        return np.nan, np.nan
