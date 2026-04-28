import pandas as pd
import collections
import heapq
import math

INPUT_CSV = "data/belgium.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")


# ── helpers ────────────────────────────────────────────────────────────────────

def build_graph(df):
    adj = collections.defaultdict(list)
    for _, row in df.iterrows():
        u, v, d = row["station_a"], row["station_b"], float(row["distance_km"])
        adj[u].append((v, d))
        adj[v].append((u, d))
    return adj


def _all_betweenness(df):
    """
    Compute betweenness centrality for ALL nodes using Brandes' algorithm
    with Dijkstra-based shortest paths (weighted graph).

    For each source s:
      1. Run Dijkstra to find distances and count shortest paths (sigma).
      2. Back-propagate dependency scores (delta).

    The raw score for node v is:
        BC(v) = sum over s != v of (sum over t != v,s of sigma(s,t|v)/sigma(s,t))
    Divided by 2 for undirected graphs (each pair counted twice).

    Time complexity: O(V * (V + E) * log V)
    """
    adj = build_graph(df)
    nodes = list(adj.keys())
    bc = {n: 0.0 for n in nodes}

    for s in nodes:
        # --- Dijkstra with path counting ---
        dist  = {n: math.inf for n in nodes}
        sigma = {n: 0        for n in nodes}   # number of shortest paths from s
        pred  = {n: []       for n in nodes}   # predecessors on shortest paths

        dist[s]  = 0.0
        sigma[s] = 1
        heap     = [(0.0, s)]
        visited  = set()
        order    = []                          # nodes in settlement order

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            order.append(u)

            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v]  = nd
                    sigma[v] = sigma[u]
                    pred[v]  = [u]
                    heapq.heappush(heap, (nd, v))
                elif math.isclose(nd, dist[v], rel_tol=1e-9):
                    sigma[v] += sigma[u]
                    pred[v].append(u)

        # --- Back-propagation (reverse settlement order) ---
        delta = {n: 0.0 for n in nodes}
        while order:
            w = order.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # Each undirected pair counted twice
    bc = {n: v / 2.0 for n, v in bc.items()}
    return bc


# Cache so we don't recompute for every call to betweenness_centrality(df, station)
_bc_cache = None

def _get_bc(df):
    global _bc_cache
    if _bc_cache is None:
        _bc_cache = _all_betweenness(df)
    return _bc_cache


# ── task function ──────────────────────────────────────────────────────────────

def betweenness_centrality(df, station):
    """
    Returns the (normalised) betweenness centrality score of `station`.

    Raw BC = number of shortest paths (between all other pairs) passing through station.
    Normalised BC = raw BC / C   where C = (V-1)(V-2) / 2

    Time complexity: O(V * (V + E) * log V)
    """
    bc = _get_bc(df)
    nodes = set(df["station_a"]).union(set(df["station_b"]))
    V = len(nodes)
    C = (V - 1) * (V - 2) / 2
    raw = bc.get(station, 0.0)
    return raw / C if C > 0 else 0.0
