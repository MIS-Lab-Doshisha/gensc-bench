import random
import itertools
import networkx as nx
from networkx.utils import py_random_state
from networkx.generators.classic import complete_graph, empty_graph, path_graph, star_graph, cycle_graph

def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


@py_random_state(2)
#@nx._dispatchable(graphs=None, returns_graph=True)
def weighted_barabasi_albert_graph(n, m, seed=None, initial_graph=None, *, create_using=None):
    """Returns a random graph using Barabási–Albert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for Barabási–Albert algorithm.
        It should be a connected graph for most use cases.
        A copy of `initial_graph` is used.
        If None, starts from a star graph on (m+1) nodes.
    create_using : Graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        Multigraph and directed types are not supported and raise a ``NetworkXError``.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``, or
        the initial graph number of nodes m0 does not satisfy ``m <= m0 <= n``.

    """
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )

    if initial_graph is None:
        # Default initial graph : star graph on (m + 1) nodes
        G = cycle_graph(m+1)
        for u, v in G.edges():
            G[u][v]['weight'] = seed.uniform(0.1, 0.25)
        
    else:
        if len(initial_graph) < m or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"Barabási–Albert initial graph needs between m={m} and n={n} nodes"
            )
        G = initial_graph.copy()


    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        # TODO:calculate edge weights
        degree_view = G.degree(targets)
        sum_degree = sum(degree for _, degree in degree_view)
        weights = [degree / sum_degree for _, degree in degree_view]
        # Add edges to m nodes from the source.
        G.add_weighted_edges_from(zip([source] * m, targets, weights))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)

        source += 1

    max_w = max(nx.get_edge_attributes(G, "weight").values())
    for u, v in G.edges():
        G[u][v]['weight'] /= max_w

    print("generated weighted barabasi graph with", n, "nodes and", m, "edges per node")
    return G

@py_random_state(3)
def weighted_sbm(sizes, p, dist, seed=None):
    if len(sizes) != len(p):
        raise nx.NetworkXException("'sizes' and 'p' do not match.")
    # Check for probability symmetry (undirected) and shape (directed)
    for row in p:
        if len(p) != len(row):
            raise nx.NetworkXException("'p' must be a square matrix.")
        
    p_transpose = [list(i) for i in zip(*p)]
    for i in zip(p, p_transpose):
        for j in zip(i[0], i[1]):
            if abs(j[0] - j[1]) > 1e-08:
                raise nx.NetworkXException("'p' must be symmetric.")
    # Check for probability range
    for row in p:
        for prob in row:
            if prob < 0 or prob > 1:
                raise nx.NetworkXException("Entries of 'p' not in [0,1].")
    # Check for nodelist consistency
    nodelist = range(sum(sizes))

    # Setup the graph conditionally to the directed switch.
    block_range = range(len(sizes))

    g = nx.Graph()
    block_iter = itertools.combinations_with_replacement(block_range, 2)
    # Split nodelist in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(len(sizes) + 1)]
    g.graph["partition"] = [
        set(nodelist[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(len(size_cumsum) - 1)
    ]
    # Setup nodes and graph name
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            g.add_node(node, block=block_id)

    g.name = "stochastic_block_model"

    # Test for edge existence
    parts = g.graph["partition"]
    for i, j in block_iter:
        if i == j:
            edges = itertools.combinations(parts[i], 2)
            for e in edges:
                if seed.random() < p[i][j]:
                    w = dist[i][j].rvs()
                    g.add_edge(*e, weight=w)
        else:
            edges = itertools.product(parts[i], parts[j])
            for e in edges:
                if seed.random() < p[i][j]:
                    w = dist[i][j].rvs()
                    g.add_edge(*e, weight=w)  # __safe
    return g