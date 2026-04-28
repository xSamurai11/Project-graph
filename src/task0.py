import pandas as pd
import collections
import heapq
import math

# Task 0 uses the FULL European dataset
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


def bfs_component(adj, start):
    """Return the set of nodes reachable from start via BFS."""
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
    """
    Single-source shortest paths (Dijkstra, non-negative weights).
    Returns dist dict and prev dict.
    Time complexity: O((V + E) log V)
    """
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
    """Reconstruct ordered path list from prev dict."""
    path = []
    cur = target
    while cur != source:
        if cur not in prev:
            return []          # unreachable
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path


# ── task functions ─────────────────────────────────────────────────────────────

def number_of_nodes(df):
    """
    Returns the number of distinct nodes in the graph.
    Time complexity: O(E)
    """
    nodes = set(df["station_a"]).union(set(df["station_b"]))
    return len(nodes)


def number_of_edges(df):
    """
    Returns the number of edges in the graph.
    Time complexity: O(E)
    """
    return len(df)


def number_of_components(df):
    """
    Returns the number of connected components in the graph.
    Uses BFS from each unvisited node.
    Time complexity: O(V + E)
    """
    adj = build_graph(df)
    nodes = set(df["station_a"]).union(set(df["station_b"]))
    visited = set()
    components = 0
    for n in nodes:
        if n not in visited:
            comp = bfs_component(adj, n)
            visited.update(comp)
            components += 1
    return components


def only_path(df):
    """
    Returns the list of stations on the only path from Ahlbeck_Grenze to Peenemunde.
    Uses Dijkstra (unique path, so shortest = only path).
    Time complexity: O((V + E) log V)
    """
    src = "Ahlbeck_Grenze"
    dst = "Peenemunde"
    adj = build_graph(df)
    _, prev = dijkstra(adj, src)
    return reconstruct_path(prev, src, dst)


def length_of_path(df):
    """
    Returns the rail distance (km) of the only path from Ahlbeck_Grenze to Peenemunde.
    Time complexity: O((V + E) log V)
    """
    src = "Ahlbeck_Grenze"
    dst = "Peenemunde"
    adj = build_graph(df)
    dist, _ = dijkstra(adj, src)
    return dist[dst]


def shortest_path(df):
    """
    Returns the length (km) of the shortest path from Portarlington_Junction to Foyens_Junction.
    Finds all paths (there are 11) via DFS and returns the minimum length.
    Time complexity: O(paths * path_length) — feasible as only 11 paths exist.
    """
    src = "Portarlington_Junction"
    dst = "Foyens_Junction"
    adj = build_graph(df)

    # DFS to enumerate all simple paths
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
