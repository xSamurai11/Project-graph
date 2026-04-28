import pandas as pd
import collections
import sys

INPUT_CSV = "data/total.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")


# ── helpers ────────────────────────────────────────────────────────────────────

def build_graph(df):
    """Build adjacency list {node: [(neighbour, dist), ...]} from a dataframe."""
    adj = collections.defaultdict(list)
    for _, row in df.iterrows():
        u, v, d = row["station_a"], row["station_b"], float(row["distance_km"])
        adj[u].append((v, d))
        adj[v].append((u, d))
    return adj


# ── task functions ─────────────────────────────────────────────────────────────

def average_degree(df):
    """
    Returns the average degree of the nodes in the graph.
    Each undirected edge contributes 1 to the degree of each endpoint,
    so sum of degrees = 2 * |E|.
    Time complexity: O(E)
    """
    nodes = set(df["station_a"]).union(set(df["station_b"]))
    num_edges = len(df)
    num_nodes = len(nodes)
    # sum of all degrees = 2 * number_of_edges (handshaking lemma)
    return (2 * num_edges) / num_nodes


def number_of_bridges(df):
    """
    Returns the number of bridges in the graph using Tarjan's algorithm.
    A bridge is an edge whose removal increases the number of connected components.
    Time complexity: O(V + E)
    """
    adj = build_graph(df)
    nodes = set(df["station_a"]).union(set(df["station_b"]))

    disc = {}          # discovery time
    low  = {}          # lowest disc reachable via subtree
    visited = set()
    bridges = []
    timer = [0]

    # Iterative Tarjan to avoid recursion limit on large graphs
    # We simulate the call stack explicitly.
    sys.setrecursionlimit(300_000)

    def dfs(start):
        stack = [(start, None, iter(adj[start]))]  # (node, parent, iterator)
        disc[start] = low[start] = timer[0]
        timer[0] += 1
        visited.add(start)

        while stack:
            u, parent, children = stack[-1]
            try:
                v, _ = next(children)
                if v not in visited:
                    visited.add(v)
                    disc[v] = low[v] = timer[0]
                    timer[0] += 1
                    stack.append((v, u, iter(adj[v])))
                elif v != parent:
                    low[u] = min(low[u], disc[v])
            except StopIteration:
                stack.pop()
                if stack:
                    pu, _, _ = stack[-1]
                    low[pu] = min(low[pu], low[u])
                    if low[u] > disc[pu]:
                        bridges.append((pu, u))

    for n in nodes:
        if n not in visited:
            dfs(n)

    return len(bridges)


def number_of_local_bridges(df):
    """
    Returns the number of local bridges in the graph.
    A local bridge is an edge (u, v) where u and v share NO common neighbour
    (i.e. their neighbourhoods — excluding each other — are disjoint).
    Time complexity: O(E * average_degree)
    """
    adj = build_graph(df)
    count = 0
    for _, row in df.iterrows():
        u, v = row["station_a"], row["station_b"]
        nbrs_u = {nb for nb, _ in adj[u]} - {v}
        nbrs_v = {nb for nb, _ in adj[v]} - {u}
        if nbrs_u.isdisjoint(nbrs_v):
            count += 1
    return count
