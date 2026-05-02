import pandas as pd
import collections
import heapq
import math

INPUT_CSV = "data/total.csv"
df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")


def build_graph(df):
    adj = collections.defaultdict(list)
    for _, row in df.iterrows():
        u, v, d = row["station_a"], row["station_b"], float(row["distance_km"])
        adj[u].append((v, d))
        adj[v].append((u, d))
    return adj

def bfs_component(adj, start):
    visited = {start}
    queue = collections.deque([start])
    while queue:
        u = queue.popleft()
        for v, _ in adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return visited

def dijkstra(adj, source):
    dist = collections.defaultdict(lambda: math.inf)
    prev = {}
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev

def reconstruct_path(prev, source, target):
    path = []
    cur = target
    while cur != source:
        if cur not in prev:
            return []
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path


def number_of_nodes(df):
    return len(set(df["station_a"]).union(set(df["station_b"])))

def number_of_edges(df):
    return len(df)

def number_of_components(df):
    adj = build_graph(df)
    nodes = set(df["station_a"]).union(set(df["station_b"]))
    visited = set()
    components = 0
    for n in nodes:
        if n not in visited:
            visited.update(bfs_component(adj, n))
            components += 1
    return components

def only_path(df):
    src, dst = "Ahlbeck_Grenze", "Peenemunde"
    adj = build_graph(df)
    _, prev = dijkstra(adj, src)
    return reconstruct_path(prev, src, dst)

def length_of_path(df):
    src, dst = "Ahlbeck_Grenze", "Peenemunde"
    adj = build_graph(df)
    dist, _ = dijkstra(adj, src)
    return dist[dst]

def shortest_path(df):
    src, dst = "Portarlington_Junction", "Foyens_Junction"
    adj = build_graph(df)

    def dfs(node, target, visited, current_dist):
        if node == target:
            return [current_dist]
        results = []
        for nb, w in adj[node]:
            if nb not in visited:
                visited.add(nb)
                results.extend(dfs(nb, target, visited, current_dist + w))
                visited.remove(nb)
        return results

    all_lengths = dfs(src, dst, {src}, 0.0)
    return min(all_lengths) if all_lengths else math.inf


if __name__ == "__main__":
    print("TASK 0")
    print(f"  Distinct nodes          : {number_of_nodes(df)}")
    print(f"  Edges                   : {number_of_edges(df)}")
    comps = number_of_components(df)
    print(f"  Connected               : {'Yes' if comps == 1 else 'No'}")
    print(f"  Components              : {comps}")
    path = only_path(df)
    print(f"  Path Ahlbeck→Peenemunde : {' → '.join(path)}")
    print(f"  Path length             : {length_of_path(df):.4f} km")
    print(f"  Shortest Portarlington→Foyens : {shortest_path(df):.4f} km")
