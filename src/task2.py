import pandas as pd
import collections

INPUT_CSV = "data/total.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")


# ── helpers ────────────────────────────────────────────────────────────────────

def build_graph(df):
    """Build adjacency list and signed adjacency dict from a dataframe.
    sign: edge weight < 1 km  → positive (+1)
          edge weight >= 1 km → negative (-1)
    Returns:
        adj      : {node: [(neighbour, dist), ...]}
        sign_adj : {node: {neighbour: +1/-1}}
    """
    adj = collections.defaultdict(list)
    sign_adj = collections.defaultdict(dict)
    for _, row in df.iterrows():
        u, v, d = row["station_a"], row["station_b"], float(row["distance_km"])
        adj[u].append((v, d))
        adj[v].append((u, d))
        s = 1 if d < 1.0 else -1
        sign_adj[u][v] = s
        sign_adj[v][u] = s
    return adj, sign_adj


# ── task functions ─────────────────────────────────────────────────────────────

def clustering_coefficient(df, station):
    """
    Returns the local clustering coefficient of `station`.
    CC(A) = (# pairs of A's neighbours that are connected)
            / (k*(k-1)/2)   where k = degree(A)
    Time complexity: O(k^2) per node, k = degree(station).
    """
    adj, _ = build_graph(df)
    neighbours = [nb for nb, _ in adj[station]]
    k = len(neighbours)
    if k < 2:
        return 0.0

    nb_set = set(neighbours)
    nb_adj = {nb: {n for n, _ in adj[nb]} for nb in neighbours}

    linked = 0
    for i in range(k):
        for j in range(i + 1, k):
            if neighbours[j] in nb_adj[neighbours[i]]:
                linked += 1

    max_pairs = k * (k - 1) // 2
    return linked / max_pairs


def number_of_triangles(df):
    """
    Returns the total number of triangles in the graph.
    Uses the degree-ordered adjacency intersection method to count each
    triangle exactly once.
    Time complexity: O(E * sqrt(E))  (standard triangle counting bound)
    """
    adj, _ = build_graph(df)
    nodes = list(adj.keys())

    # Assign index for ordering
    idx = {n: i for i, n in enumerate(nodes)}

    # Build forward adjacency (only to higher-indexed neighbours) as sets
    fwd = {u: set() for u in nodes}
    for u in nodes:
        for v, _ in adj[u]:
            if idx[v] > idx[u]:
                fwd[u].add(v)

    triangles = 0
    for u in nodes:
        for v in fwd[u]:
            # count common forward neighbours
            triangles += len(fwd[u] & fwd[v])

    return triangles


def number_of_balanced_triangles(df):
    """
    Returns the number of balanced triangles.
    A triangle is balanced iff it has 0 or 2 negative edges (even number of −).
    Sign rule: distance < 1 km → positive; distance >= 1 km → negative.
    Time complexity: O(E * sqrt(E))
    """
    adj, sign_adj = build_graph(df)
    nodes = list(adj.keys())
    idx = {n: i for i, n in enumerate(nodes)}

    fwd = {u: set() for u in nodes}
    for u in nodes:
        for v, _ in adj[u]:
            if idx[v] > idx[u]:
                fwd[u].add(v)

    balanced = 0
    for u in nodes:
        fwd_u = fwd[u]
        for v in fwd_u:
            common = fwd_u & fwd[v]
            s_uv = sign_adj[u][v]
            for w in common:
                s_uw = sign_adj[u][w]
                s_vw = sign_adj[v][w]
                neg = sum(1 for s in (s_uv, s_uw, s_vw) if s == -1)
                if neg % 2 == 0:      # 0 or 2 negatives → balanced
                    balanced += 1

    return balanced


def number_of_unbalanced_triangles(df):
    """
    Returns the number of unbalanced triangles (1 or 3 negative edges).
    Time complexity: O(E * sqrt(E))
    """
    total = number_of_triangles(df)
    balanced = number_of_balanced_triangles(df)
    return total - balanced


def gcc(df):
    """
    Returns the Global Clustering Coefficient (GCC).
    GCC = (number of closed triplets) / (number of open triplets)
        = 3 * triangles / (all connected triplets of length 2)
    A triplet centred on u with neighbours v, w is:
      - closed if (v, w) is also an edge
      - open   if (v, w) is not an edge
    Time complexity: O(V * d^2) where d = max degree
    """
    adj, _ = build_graph(df)

    closed_triplets = 0
    open_triplets   = 0

    for u in adj:
        nbrs = [v for v, _ in adj[u]]
        k = len(nbrs)
        if k < 2:
            continue
        nb_set = {v: {n for n, _ in adj[v]} for v in nbrs}
        for i in range(k):
            for j in range(i + 1, k):
                open_triplets += 1
                if nbrs[j] in nb_set[nbrs[i]]:
                    closed_triplets += 1

    if open_triplets == 0:
        return 0.0
    return closed_triplets / open_triplets
