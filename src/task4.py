import pandas as pd
import collections

INPUT_CSV = "data/belgium.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")


def build_graph(df):
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
    adj, _ = build_graph(df)
    deg = {n: len(adj[n]) for n in adj}
    scores = {}
    for n in adj:
        outer = sum(
            (deg[i] / d_ni) * sum(deg[j] * d_ij for j, d_ij in adj[i] if j != n)
            for i, d_ni in adj[n]
        )
        scores[n] = deg[n] * outer
    return scores

_score_cache = None

def _get_scores(df):
    global _score_cache
    if _score_cache is None:
        _score_cache = _compute_all_scores(df)
    return _score_cache

def score(df, station):
    return _get_scores(df).get(station, 0.0)

def gain_from_split(df, station_a, station_b):
    adj, edge_dist = build_graph(df)
    key = (station_a, station_b) if (station_a, station_b) in edge_dist else (station_b, station_a)
    if key not in edge_dist:
        return 0.0
    u, v = key
    d = edge_dist[(u, v)]
    scores = _get_scores(df)
    old_u, old_v = scores.get(u, 0.0), scores.get(v, 0.0)
    total_old = old_u + old_v
    if total_old == 0:
        return 0.0
    M, half = f"__MID_{u}_{v}__", d / 2.0
    mod_adj = collections.defaultdict(list)
    for node in adj:
        for nb, w in adj[node]:
            if not ((node == u and nb == v) or (node == v and nb == u)):
                mod_adj[node].append((nb, w))
    mod_adj[u].append((M, half))
    mod_adj[v].append((M, half))
    mod_adj[M].append((u, half))
    mod_adj[M].append((v, half))
    mod_deg = {n: len(mod_adj[n]) for n in mod_adj}

    def node_score(node):
        dn = mod_deg[node]
        outer = sum(
            (mod_deg[i] / d_ni) * sum(mod_deg[j] * dij for j, dij in mod_adj[i] if j != node)
            for i, d_ni in mod_adj[node]
        )
        return dn * outer

    gain = node_score(u) + node_score(v) + node_score(M) - old_u - old_v
    return (gain / total_old) * d * 100


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)

    print("TASK 4")
    all_scores    = _get_scores(df)
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    print("  Top-5 stations (highest score):")
    for i, (st, sc) in enumerate(sorted_scores[:5], 1):
        print(f"    {i}. {st} → {sc:.4f}")
    print("  Bottom-5 stations (lowest score):")
    for i, (st, sc) in enumerate(sorted_scores[-5:][::-1], 1):
        print(f"    {i}. {st} → {sc:.4f}")

    plt.figure(figsize=(7, 3.5))
    plt.hist([v for v in all_scores.values() if v > 0], bins=60, color="seagreen", edgecolor="black")
    plt.xlabel("Score")
    plt.ylabel("Number of stations")
    plt.title("Score distribution – Belgian railway network")
    plt.tight_layout()
    plt.savefig("outputs/task4.png", dpi=150)
    plt.close()

    edge_criteria = sorted(
        [(row["station_a"], row["station_b"], gain_from_split(df, row["station_a"], row["station_b"]))
         for _, row in df.iterrows()],
        key=lambda x: x[2], reverse=True
    )
    print("  Top-5 edges for midpoint insertion:")
    for i, (u, v, c) in enumerate(edge_criteria[:5], 1):
        print(f"    {i}. {u} -- {v}  →  criterion = {c:.4f}")
    print(f"  Diagram generated in outputs/task4.png")
