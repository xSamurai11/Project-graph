import pandas as pd
import collections

CSV = "data/total.csv"
df = pd.read_csv(CSV, dtype=str).fillna("")

def build_graph(df):
    adj = collections.defaultdict(list)
    for _, row in df.iterrows():
        u, v, d = row["station_a"], row["station_b"], float(row["distance_km"])
        adj[u].append((v, d))
        adj[v].append((u, d))
    return adj

def average_degree(df):
    nodes = set(df["station_a"]).union(set(df["station_b"]))
    return (2 * len(df)) / len(nodes)

def number_of_bridges(df):
    adj = build_graph(df)
    nodes = set(df["station_a"]).union(set(df["station_b"]))
    disc, low, visited, bridges, timer = {}, {}, set(), [], [0]

    def dfs(start):
        stack = [(start, None, iter(adj[start]))]
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
    adj = build_graph(df)
    count = 0
    for _, row in df.iterrows():
        u, v = row["station_a"], row["station_b"]
        if ({nb for nb, _ in adj[u]} - {v}).isdisjoint({nb for nb, _ in adj[v]} - {u}):
            count += 1
    return count

if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("outputs", exist_ok=True)
    print("TASK 1")
    print(f"  Average degree          : {average_degree(df):.6f}")
    adj_t = collections.defaultdict(list)
    for _, row in df.iterrows():
        adj_t[row["station_a"]].append(row["station_b"])
        adj_t[row["station_b"]].append(row["station_a"])
    all_nodes = set(df["station_a"]) | set(df["station_b"])
    deg_count = collections.Counter(len(adj_t[n]) for n in all_nodes)
    x = list(range(11))
    y = [deg_count.get(d, 0) for d in x]
    plt.figure(figsize=(7, 3.5))
    plt.bar(x, y, color="steelblue", edgecolor="black")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Degree distribution (degree ≤ 10) – European railway network")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig("outputs/task1.png", dpi=150)
    plt.close()
    print(f"  Degree distribution     : {dict(zip(x, y))}")
    print(f"  Bridges                 : {number_of_bridges(df)}")
    print(f"  Local bridges           : {number_of_local_bridges(df)}")
    print(f"  Diagram generated in outputs/task1.png")