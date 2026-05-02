import pandas as pd
import collections

INPUT_CSV = "data/total.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

def build_graph(df):
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

def clustering_coefficient(df, station):
    adj, _ = build_graph(df)
    neighbours = [nb for nb, _ in adj[station]]
    k = len(neighbours)
    if k < 2:
        return 0.0
    nb_adj = {nb: {n for n, _ in adj[nb]} for nb in neighbours}
    linked = sum(
        1 for i in range(k) for j in range(i + 1, k)
        if neighbours[j] in nb_adj[neighbours[i]]
    )
    return linked / (k * (k - 1) // 2)

def number_of_triangles(df):
    adj, _ = build_graph(df)
    nodes = list(adj.keys())
    idx = {n: i for i, n in enumerate(nodes)}
    fwd = {u: {v for v, _ in adj[u] if idx[v] > idx[u]} for u in nodes}
    return sum(len(fwd[u] & fwd[v]) for u in nodes for v in fwd[u])

def number_of_balanced_triangles(df):
    adj, sign_adj = build_graph(df)
    nodes = list(adj.keys())
    idx = {n: i for i, n in enumerate(nodes)}
    fwd = {u: {v for v, _ in adj[u] if idx[v] > idx[u]} for u in nodes}
    balanced = 0
    for u in nodes:
        for v in fwd[u]:
            for w in fwd[u] & fwd[v]:
                neg = sum(1 for s in (sign_adj[u][v], sign_adj[u][w], sign_adj[v][w]) if s == -1)
                if neg % 2 == 0:
                    balanced += 1
    return balanced

def number_of_unbalanced_triangles(df):
    return number_of_triangles(df) - number_of_balanced_triangles(df)

def gcc(df):
    adj, _ = build_graph(df)
    closed = open_ = 0
    for u in adj:
        nbrs = [v for v, _ in adj[u]]
        k = len(nbrs)
        if k < 2:
            continue
        nb_sets = {v: {n for n, _ in adj[v]} for v in nbrs}
        for i in range(k):
            for j in range(i + 1, k):
                open_ += 1
                if nbrs[j] in nb_sets[nbrs[i]]:
                    closed += 1
    return closed / open_ if open_ else 0.0

if __name__ == "__main__":
    stations = [
        "Berlin_Westhafen",
        "Krakow_Gowny",
        "Amsterdam_Transformatorweg_Aansl.",
        "ROMA_TERMINI",
    ]
    print("TASK 2")
    for s in stations:
        print(f"  CC({s}) = {clustering_coefficient(df, s):.6f}")
    tri  = number_of_triangles(df)
    bal  = number_of_balanced_triangles(df)
    print(f"  Total triangles         : {tri}")
    print(f"  Balanced triangles      : {bal}")
    print(f"  Unbalanced triangles    : {tri - bal}")
    print(f"  GCC                     : {gcc(df):.6f}")
