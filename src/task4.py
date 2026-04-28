import pandas as pd
import collections
import heapq
import math

INPUT_CSV = "data/belgium.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")


# ── helpers ────────────────────────────────────────────────────────────────────

def build_graph(df):
    """Returns adj {node: [(neighbour, dist), ...]} and edge_dist {(u,v): dist}."""
    adj = collections.defaultdict(list)
    edge_dist = {}
    for _, row in df.iterrows():
        u, v, d = row["station_a"], row["station_b"], float(row["distance_km"])
        adj[u].append((v, d))
        adj[v].append((u, d))
        edge_dist[(u, v)] = d
        edge_dist[(v, u)] = d
    return adj, edge_dist


def _compute_all_scores(df):
    """
    Compute score(n) for every node.

    score(n) = deg(n)
               * sum_{i in Neighbors(n)}  [ deg(i) / d(n,i)
                                            * sum_{j in Neighbors(i)\{n}} deg(j)*d(i,j) ]

    Time complexity: O(V * d_max^2)  where d_max = max degree
    """
    adj, edge_dist = build_graph(df)
    nodes = list(adj.keys())
    deg = {n: len(adj[n]) for n in nodes}

    scores = {}
    for n in nodes:
        deg_n = deg[n]
        outer = 0.0
        for i, d_ni in adj[n]:
            deg_i = deg[i]
            inner = sum(
                deg[j] * d_ij
                for j, d_ij in adj[i]
                if j != n
            )
            outer += (deg_i / d_ni) * inner
        scores[n] = deg_n * outer

    return scores


# Cache so repeated calls don't recompute
_score_cache = None

def _get_scores(df):
    global _score_cache
    if _score_cache is None:
        _score_cache = _compute_all_scores(df)
    return _score_cache


# ── task functions ─────────────────────────────────────────────────────────────

def score(df, station):
    """
    Returns score(station) as defined in the project.
    Time complexity: O(V * d_max^2) on first call, O(1) thereafter.
    """
    return _get_scores(df).get(station, 0.0)


def gain_from_split(df, station_a, station_b):
    """
    Returns the criterion value for splitting edge (station_a, station_b)
    by inserting a new station M exactly in the middle.

    When we split edge (u, v) with distance d:
      - Remove edge (u, v)
      - Add node M with edges (u, M, d/2) and (M, v, d/2)

    The criterion is:
        criterion(u,v) = 0                               if score(u)+score(v) == 0
                       = score_gain / (score(u)+score(v)) * d(u,v) * 100  otherwise

    where score_gain = sum of new scores for ALL affected nodes
                       minus sum of their old scores.

    Affected nodes: u, v, M (new), and all neighbours of u and v (their
    scores change because u and v's degrees don't change but the inner
    sums change; M is new).

    For simplicity (and as stated: "evaluated on the same original graph"),
    we compute the score gain = new_score(u) + new_score(v) + new_score(M)
    minus old_score(u) - old_score(v).

    Time complexity: O(d_max^2) per call after initial score computation.
    """
    adj, edge_dist = build_graph(df)

    key = (station_a, station_b)
    if key not in edge_dist:
        key = (station_b, station_a)
    if key not in edge_dist:
        return 0.0

    u, v = key
    d = edge_dist[(u, v)]

    scores = _get_scores(df)
    old_score_u = scores.get(u, 0.0)
    old_score_v = scores.get(v, 0.0)

    total_old = old_score_u + old_score_v
    if total_old == 0:
        return 0.0

    # Build modified adjacency for the split:
    # Remove (u,v), add M with edges (u,M,d/2) and (M,v,d/2)
    M = f"__MID_{u}_{v}__"
    half = d / 2.0

    mod_adj = collections.defaultdict(list)
    for node in adj:
        for nb, w in adj[node]:
            if (node == u and nb == v) or (node == v and nb == u):
                continue          # remove original edge
            mod_adj[node].append((nb, w))
    # add midpoint
    mod_adj[u].append((M, half))
    mod_adj[v].append((M, half))
    mod_adj[M].append((u, half))
    mod_adj[M].append((v, half))

    mod_deg = {n: len(mod_adj[n]) for n in mod_adj}

    def node_score(node, a, ed):
        dn = ed[node]
        outer = 0.0
        for i, d_ni in a[node]:
            di = ed[i]
            inner = sum(ed[j] * dij for j, dij in a[i] if j != node)
            outer += (di / d_ni) * inner
        return dn * outer

    new_score_u = node_score(u, mod_adj, mod_deg)
    new_score_v = node_score(v, mod_adj, mod_deg)
    new_score_M = node_score(M, mod_adj, mod_deg)

    gain = (new_score_u + new_score_v + new_score_M) - (old_score_u + old_score_v)

    return (gain / total_old) * d * 100
