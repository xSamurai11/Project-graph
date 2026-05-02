import pandas as pd
import collections
import heapq
import math

INPUT_CSV = "data/belgium.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")


def build_graph(df):
    adj = collections.defaultdict(list)
    for _, row in df.iterrows():
        u, v, d = row["station_a"], row["station_b"], float(row["distance_km"])
        adj[u].append((v, d))
        adj[v].append((u, d))
    return adj

def _all_betweenness(df):
    adj = build_graph(df)
    nodes = list(adj.keys())
    bc = {n: 0.0 for n in nodes}

    for s in nodes:
        dist  = {n: math.inf for n in nodes}
        sigma = {n: 0 for n in nodes}
        pred  = {n: [] for n in nodes}
        dist[s] = 0.0
        sigma[s] = 1
        heap = [(0.0, s)]
        visited = set()
        order = []

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            order.append(u)
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    sigma[v] = sigma[u]
                    pred[v] = [u]
                    heapq.heappush(heap, (nd, v))
                elif math.isclose(nd, dist[v], rel_tol=1e-9):
                    sigma[v] += sigma[u]
                    pred[v].append(u)

        delta = {n: 0.0 for n in nodes}
        while order:
            w = order.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    return {n: v / 2.0 for n, v in bc.items()}

_bc_cache = None

def _get_bc(df):
    global _bc_cache
    if _bc_cache is None:
        _bc_cache = _all_betweenness(df)
    return _bc_cache

def betweenness_centrality(df, station):
    bc = _get_bc(df)
    V = len(set(df["station_a"]).union(set(df["station_b"])))
    C = (V - 1) * (V - 2) / 2
    return bc.get(station, 0.0) / C if C > 0 else 0.0


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)

    print("TASK 3")
    V = len(set(df["station_a"]) | set(df["station_b"]))
    C = (V - 1) * (V - 2) / 2
    raw_bc  = _get_bc(df)
    norm_bc = {n: v / C for n, v in raw_bc.items()}
    top     = max(norm_bc, key=norm_bc.get)
    print(f"  Top station             : {top}")
    print(f"  Raw BC score            : {raw_bc[top]:.4f}")
    print(f"  Normalised BC score     : {norm_bc[top]:.6f}")

    plt.figure(figsize=(7, 3.5))
    plt.hist(sorted(norm_bc.values()), bins=60, color="darkorange", edgecolor="black")
    plt.xlabel("Normalised betweenness centrality")
    plt.ylabel("Number of stations")
    plt.title("Betweenness centrality distribution – Belgian railway network")
    plt.tight_layout()
    plt.savefig("outputs/task3.png", dpi=150)
    plt.close()
    print(f"  Diagram generated in outputs/task3.png")
