import itertools
import time
from typing import Dict, List, Set, Tuple, Optional
from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc
from hyper_simulation.hypergraph.linguistic import QueryType, Pos, Tag, Dep, Entity
import numpy as np
import logging
from hyper_simulation.component.embedding import get_embedding_batch, get_cosine_similarity_batch, get_similarity_batch, get_similarity, cosine_similarity
from hyper_simulation.component.nli import get_nli_label, get_nli_labels_batch
from hyper_simulation.utils.log import getLogger
from hyper_simulation.component.denial import get_top_k_matched_vertices
from collections import deque
from itertools import product
from hyper_simulation.hypergraph.path import find_shortest_hyperpaths, find_shortest_hyperpaths_local

def abstraction_lca(query: list[str], data: list[str]) -> tuple[str, int]:
    """ Docstring removed due to garbled encoding. """
    if not query or not data:
        return '', -1
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    if query[0] != data[0]:
        return '', -1
        
    lca = query[0]
    depth = 0
    
    min_len = min(len(query), len(data))
    for i in range(min_len):
        if query[i] == data[i]:
            lca = query[i]
            depth = i
        else:
            break
            
    return lca, depth

def _vertex_sort_key(vertex: Vertex) -> tuple[int, str]:
    return (vertex.id, vertex.text())


def _hyperedge_signature(hyperedge: Hyperedge) -> tuple[int, int, int, str]:
    root_id = hyperedge.root.id if hyperedge.root else -1
    return (root_id, hyperedge.start, hyperedge.end, hyperedge.desc)


# def _path_sort_key(path: Path) -> tuple:
#     sig = [_hyperedge_signature(he) for he in path.hyperedges]
#     return (len(path.hyperedges), sig)


def _cluster_sort_key(cluster: 'SemanticCluster') -> tuple:
    return cluster.signature()


class TarjanLCA:
    def __init__(self, edges: list[tuple[Node, Node]], queries: list[tuple[Node, Node]]) -> None:
        # build adjacency list (directed) and node set
        self.adj: dict[Node, list[Node]] = {}
        self.nodes: set[Node] = set()
        
        # Comment removed due to garbled encoding.
        in_degree: dict[Node, int] = {}

        for a, b in edges:
            self.nodes.add(a)
            self.nodes.add(b)
            if a not in self.adj:
                self.adj[a] = []
            self.adj[a].append(b)
            
            # Comment removed due to garbled encoding.
            if a not in in_degree: in_degree[a] = 0
            if b not in in_degree: in_degree[b] = 0
            in_degree[b] += 1

        # store queries and build per-node query map
        self.queries = list(queries)
        self.query_map: dict[Node, list[tuple[Node, int]]] = {}
        
        for i, (u, v) in enumerate(self.queries):
            self.nodes.add(u)
            self.nodes.add(v)
            if u not in in_degree: in_degree[u] = 0
            if v not in in_degree: in_degree[v] = 0

            # Comment removed due to garbled encoding.
            if u not in self.query_map: self.query_map[u] = []
            if v not in self.query_map: self.query_map[v] = []
            
            self.query_map[u].append((v, i))
            if u != v:
                self.query_map[v].append((u, i))

        # union-find parent and ancestor used by Tarjan's algorithm
        self.uf_parent: dict[Node, Node] = {}
        self.ancestor: dict[Node, Node] = {}
        self.visited: set[Node] = set()
        self.res: list[Node | None] = [None] * len(self.queries)

        # Comment removed due to garbled encoding.
        self.node_roots: dict[Node, Node] = {}

        # initialize union-find for all nodes
        for n in list(self.nodes):
            self.uf_parent[n] = n
            self.ancestor[n] = n

        # run Tarjan on each component (forest support)
        # Comment removed due to garbled encoding.
        sorted_nodes = sorted(list(self.nodes), key=lambda n: in_degree.get(n, 0))
        
        for n in sorted_nodes:
            if n not in self.visited:
                # Comment removed due to garbled encoding.
                self.tarjan(n, None, n)
        
    # union-find's find
    def find(self, x):
        if x not in self.uf_parent:
            self.uf_parent[x] = x
            return x
        if self.uf_parent[x] != x:
            self.uf_parent[x] = self.find(self.uf_parent[x])
        return self.uf_parent[x]
    
    # union-find's union
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        self.uf_parent[ry] = rx
    
    # Comment removed due to garbled encoding.
    def tarjan(self, u, p, root_id):
        # Comment removed due to garbled encoding.
        self.node_roots[u] = root_id

        self.ancestor[u] = u 
        
        for v in self.adj.get(u, []):
            if v == p: 
                continue
            if v in self.visited:
                continue
            
            # Comment removed due to garbled encoding.
            self.tarjan(v, u, root_id)
            self.union(u, v)
            self.ancestor[self.find(u)] = u

        self.visited.add(u)

        for other, qi in self.query_map.get(u, []):
            # Comment removed due to garbled encoding.
            # Comment removed due to garbled encoding.
            if other in self.visited:
                if self.node_roots.get(other) == root_id:
                    self.res[qi] = self.ancestor[self.find(other)]

    def lca(self) -> list[Node | None]:
        return self.res

class SemanticCluster:
    def __init__(self, hyperedges: list[Hyperedge], doc: LocalDoc, is_query: bool=False) -> None:
        self.hyperedges = hyperedges
        self.doc = doc
        self.vertices: list[Vertex] = []
        self.contained_hyperedges: dict[Vertex, list[Hyperedge]] = {}
        self.embedding: np.ndarray | None = None
        self.text_cache: str | None = None
        
        self.vertices_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
        self.node_paths_cache: dict[tuple[Node, Node], tuple[str, int]] = {}
        
        self.is_query = is_query
        self._signature: tuple | None = None
        
        self.vertices_paths_within_hyperedges: dict[tuple[Node, Node, Node], tuple[str, int]] = {}
        
        # Comment removed due to garbled encoding.
        self._hyperedge_groups: list[list[Hyperedge]] | None = None
        self._group_intersections: dict[tuple[int, int], set[Node]] | None = None
        self._hyperedge_to_group: dict[Hyperedge, int] | None = None
        self._vertex_to_hyperedges: dict[Vertex, list[Hyperedge]] | None = None
        self._node_pair_nearest_root: dict[tuple[Node, Node], Node] | None = None
        
    # Comment removed due to garbled encoding.
    def _build_hyperedge_groups(self) -> tuple[list[list[Hyperedge]], dict[Hyperedge, int]]:
        """ Docstring removed due to garbled encoding. """
        if self._hyperedge_groups is not None and self._hyperedge_to_group is not None:
            return self._hyperedge_groups, self._hyperedge_to_group

        # Comment removed due to garbled encoding.
        # Comment removed due to garbled encoding.
        ultimate_root_cache: dict[Node, Node] = {}

        def get_ultimate_root(start: Node) -> Node:
            # Comment removed due to garbled encoding.
            if start in ultimate_root_cache:
                return ultimate_root_cache[start]

            current = start
            visited: set[Node] = set()
            trace: list[Node] = []

            while True:
                # Comment removed due to garbled encoding.
                if current in ultimate_root_cache:
                    ultimate = ultimate_root_cache[current]
                    break

                # Comment removed due to garbled encoding.
                if current in visited:
                    ultimate = current
                    break

                visited.add(current)
                trace.append(current)

                head = current.head
                if head is None or head == current:
                    ultimate = current
                    break

                current = head

            # Comment removed due to garbled encoding.
            for node in trace:
                ultimate_root_cache[node] = ultimate
            return ultimate

        groups_dict: dict[Node, list[Hyperedge]] = {}
        for he in self.hyperedges:
            root = he.current_node(he.root)
            if root is None:
                continue
            ultimate_root = get_ultimate_root(root)
            if ultimate_root not in groups_dict:
                groups_dict[ultimate_root] = []
            groups_dict[ultimate_root].append(he)
        
        groups = list(groups_dict.values())
        
        # Comment removed due to garbled encoding.
        he_to_group: dict[Hyperedge, int] = {}
        for group_idx, group in enumerate(groups):
            for he in group:
                he_to_group[he] = group_idx
        
        self._hyperedge_groups = groups
        self._hyperedge_to_group = he_to_group
        
        # Comment removed due to garbled encoding.
        node_pairs_to_roots: dict[tuple[Node, Node], Node] = {}
        for group in groups:
            # Comment removed due to garbled encoding.
            node_to_roots: dict[Node, set[Node]] = {}
            group_nodes: set[Node] = set()
            for he in group:
                root = he.current_node(he.root)
                if root is None:
                    continue
                for vertex in he.vertices:
                    node = he.current_node(vertex)
                    if node is None:
                        continue
                    group_nodes.add(node)
                    if node not in node_to_roots:
                        node_to_roots[node] = set()
                    node_to_roots[node].add(root)

            # Comment removed due to garbled encoding.
            root_chain_cache: dict[Node, list[Node]] = {}
            # Comment removed due to garbled encoding.
            root_depth_cache: dict[Node, dict[Node, int]] = {}
            # Comment removed due to garbled encoding.
            root_pair_nearest_cache: dict[tuple[Node, Node], Node | None] = {}

            def get_root_chain(root: Node) -> list[Node]:
                if root in root_chain_cache:
                    return root_chain_cache[root]
                chain: list[Node] = []
                current = root
                visited: set[Node] = set()
                while current is not None and current not in visited:
                    visited.add(current)
                    chain.append(current)
                    current = current.head
                root_chain_cache[root] = chain
                return chain

            def get_root_depth_map(root: Node) -> dict[Node, int]:
                if root in root_depth_cache:
                    return root_depth_cache[root]
                depth_map: dict[Node, int] = {}
                for depth, ancestor in enumerate(get_root_chain(root)):
                    depth_map[ancestor] = depth
                root_depth_cache[root] = depth_map
                return depth_map

            def nearest_common_root(root1: Node, root2: Node) -> Node | None:
                if root1 == root2:
                    return root1
                key = (root1, root2)
                if key in root_pair_nearest_cache:
                    return root_pair_nearest_cache[key]

                ancestors1 = get_root_depth_map(root1)
                nearest: Node | None = None
                for ancestor in get_root_chain(root2):
                    if ancestor in ancestors1:
                        nearest = ancestor
                        break

                root_pair_nearest_cache[(root1, root2)] = nearest
                root_pair_nearest_cache[(root2, root1)] = nearest
                return nearest

            group_nodes_list = list(group_nodes)
            for i in range(len(group_nodes_list)):
                for j in range(i + 1, len(group_nodes_list)):
                    node1 = group_nodes_list[i]
                    node2 = group_nodes_list[j]

                    roots1 = node_to_roots.get(node1, set())
                    roots2 = node_to_roots.get(node2, set())
                    if not roots1 or not roots2:
                        continue

                    best_root: Node | None = None
                    best_score: int | None = None

                    # Comment removed due to garbled encoding.
                    for root1 in roots1:
                        depth1 = get_root_depth_map(root1)
                        for root2 in roots2:
                            common = nearest_common_root(root1, root2)
                            if common is None:
                                continue
                            depth2 = get_root_depth_map(root2)
                            score = depth1.get(common, 10**9) + depth2.get(common, 10**9)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_root = common

                    if best_root is not None:
                        node_pairs_to_roots[(node1, node2)] = best_root
                        node_pairs_to_roots[(node2, node1)] = best_root
        
        self._node_pair_nearest_root = node_pairs_to_roots
        
        # Comment removed due to garbled encoding.
        vertex_to_groups: dict[Vertex, set[int]] = {}
        for he in self.hyperedges:
            group_idx = he_to_group.get(he)
            if group_idx is None:
                continue
            
            for vertex in he.vertices:
                if vertex not in vertex_to_groups:
                    vertex_to_groups[vertex] = set()
                vertex_to_groups[vertex].add(group_idx)
        
        self._vertex_to_groups_cache = vertex_to_groups
        
        # Comment removed due to garbled encoding.
        # Comment removed due to garbled encoding.
        def get_group_nodes(group: list[Hyperedge]) -> set[Node]:
            """ Docstring removed due to garbled encoding. """
            nodes = set()
            for he in group:
                for vertex in he.vertices:
                    node = he.current_node(vertex)
                    if node is not None:
                        nodes.add(node)
            return nodes
        
        # Comment removed due to garbled encoding.
        group_adjacency: dict[int, set[int]] = {}
        inter_group_bridges: dict[tuple[int, int], set[Node]] = {}
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                nodes_i = get_group_nodes(groups[i])
                nodes_j = get_group_nodes(groups[j])
                bridges = nodes_i & nodes_j
                
                if bridges:
                    group_adjacency.setdefault(i, set()).add(j)
                    group_adjacency.setdefault(j, set()).add(i)
                    inter_group_bridges[(i, j)] = bridges
                    inter_group_bridges[(j, i)] = bridges
        
        self._inter_group_bridge_cache = inter_group_bridges
        
        # Comment removed due to garbled encoding.
        # Comment removed due to garbled encoding.
        inter_group_distances: dict[tuple[int, int], list[int]] = {}
        
        for start_group in range(len(groups)):
            # Comment removed due to garbled encoding.
            dist: dict[int, int] = {start_group: 0}
            parent: dict[int, int | None] = {start_group: None}
            queue = deque([start_group])
            
            while queue:
                current = queue.popleft()
                for neighbor in group_adjacency.get(current, set()):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        parent[neighbor] = current
                        queue.append(neighbor)
            
            # Comment removed due to garbled encoding.
            for end_group, d in dist.items():
                if end_group == start_group:
                    inter_group_distances[(start_group, end_group)] = [start_group]
                else:
                    # Comment removed due to garbled encoding.
                    path = []
                    current = end_group
                    while current is not None:
                        path.append(current)
                        current = parent.get(current)
                    path.reverse()
                    inter_group_distances[(start_group, end_group)] = path
        
        self._inter_group_distances_cache = inter_group_distances
        
        return groups, he_to_group
    
    # Comment removed due to garbled encoding.
    def _find_group_intersections(self) -> dict[tuple[int, int], set[Node]]:
        """ Docstring removed due to garbled encoding. """
        if self._group_intersections is not None:
            return self._group_intersections
        
        groups, _ = self._build_hyperedge_groups()
        
        def get_group_nodes(group: list[Hyperedge]) -> set[Node]:
            """ Docstring removed due to garbled encoding. """
            nodes = set()
            for he in group:
                for vertex in he.vertices:
                    node = he.current_node(vertex)
                    if node is not None:
                        nodes.add(node)
            return nodes
        
        intersections: dict[tuple[int, int], set[Node]] = {}
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                nodes_i = get_group_nodes(groups[i])
                nodes_j = get_group_nodes(groups[j])
                intersection = nodes_i & nodes_j
                if intersection:
                    intersections[(i, j)] = intersection
                    intersections[(j, i)] = intersection  # Comment removed due to garbled encoding.
        
        self._group_intersections = intersections
        return intersections
    
    # Comment removed due to garbled encoding.
    def _get_nearest_root_for_node_pair(self, node1: Node, node2: Node) -> Node | None:
        """ Docstring removed due to garbled encoding. """
        # Comment removed due to garbled encoding.
        self._build_hyperedge_groups()
        
        # Comment removed due to garbled encoding.
        if self._node_pair_nearest_root is not None:
            return self._node_pair_nearest_root.get((node1, node2))
        
        return None
    
    # Comment removed due to garbled encoding.
    def _build_vertex_to_groups(self) -> dict[Vertex, set[int]]:
        """ Docstring removed due to garbled encoding. """
        # Comment removed due to garbled encoding.
        self._build_hyperedge_groups()
        
        if not hasattr(self, '_vertex_to_groups_cache'):
            self._vertex_to_groups_cache = {}
        
        return self._vertex_to_groups_cache
    
    # Comment removed due to garbled encoding.
    def _get_vertex_groups(self, v: Vertex) -> set[int]:
        """ Docstring removed due to garbled encoding. """
        vertex_to_groups = self._build_vertex_to_groups()
        return vertex_to_groups.get(v, set())
    
    # Comment removed due to garbled encoding.
    def _build_inter_group_bridge_map(self) -> dict[tuple[int, int], set[Node]]:
        """ Docstring removed due to garbled encoding. """
        # Comment removed due to garbled encoding.
        self._build_hyperedge_groups()
        
        if not hasattr(self, '_inter_group_bridge_cache'):
            self._inter_group_bridge_cache = {}
        
        return self._inter_group_bridge_cache
    
    # Comment removed due to garbled encoding.
    def _find_shortest_group_path(self, g1: int, g2: int) -> list[int]:
        """ Docstring removed due to garbled encoding. """
        if g1 == g2:
            return [g1]
        
        # Comment removed due to garbled encoding.
        self._build_hyperedge_groups()
        
        # Comment removed due to garbled encoding.
        if hasattr(self, '_inter_group_distances_cache'):
            return self._inter_group_distances_cache.get((g1, g2), [])
        
        return []
    
    # Comment removed due to garbled encoding.
    def _find_closest_group_pair(self, v1_groups: set[int], v2_groups: set[int]) -> tuple[int, int, list[int]] | None:
        """ Docstring removed due to garbled encoding. """
        if not v1_groups or not v2_groups:
            return None
        
        # Comment removed due to garbled encoding.
        common_groups = v1_groups & v2_groups
        if common_groups:
            g = next(iter(common_groups))
            return (g, g, [g])
        
        # Comment removed due to garbled encoding.
        self._build_hyperedge_groups()
        
        # Comment removed due to garbled encoding.
        shortest_path: list[int] | None = None
        best_length = float('inf')
        best_pair: tuple[int, int] | None = None
        
        if hasattr(self, '_inter_group_distances_cache'):
            distances = self._inter_group_distances_cache
            
            for g1 in v1_groups:
                for g2 in v2_groups:
                    path = distances.get((g1, g2), [])
                    if path and len(path) < best_length:
                        best_length = len(path)
                        best_pair = (g1, g2)
                        shortest_path = path
        
        if best_pair is None or shortest_path is None:
            return None
        
        return (best_pair[0], best_pair[1], shortest_path)
    
    # Comment removed due to garbled encoding.
    # def _find_shortest_inter_group_paths(self, g_from: int, g_to: int,
    #                                     groups: list[list[Hyperedge]],
    #                                     intersections: dict[tuple[int, int], set[Node]]
    #                                     ) -> list[tuple[set[Node], set[Node]]]:
    #     """
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #     """
    #     if g_from == g_to:
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    #     queue = deque([(g_from, [])])  # (current_group, path_so_far)
    #     visited = {g_from}
    #     found_paths = []
    #     min_length = float('inf')
        
    #     while queue:
    #         current, path = queue.popleft()
            
    #         if current == g_to:
    #             if len(path) < min_length:
    #                 min_length = len(path)
    #                 found_paths = [path]
    #             elif len(path) == min_length:
    #                 found_paths.append(path)
    #             continue
            
    # Comment removed due to garbled encoding.
    #         if len(path) >= min_length:
    #             continue
            
    # Comment removed due to garbled encoding.
    #         for (g1, g2), intersection in intersections.items():
    #             if g1 == current and g2 not in visited:
    #                 new_path = path + [(current, g2, intersection)]
    #                 queue.append((g2, new_path))
    #                 visited.add(g2)
    #             elif g2 == current and g1 not in visited:
    #                 new_path = path + [(current, g1, intersection)]
    #                 queue.append((g1, new_path))
    #                 visited.add(g1)
        
    # Comment removed due to garbled encoding.
    #     result: list[tuple[set[Node], set[Node]]] = []
    #     for path in found_paths:
    #         if not path:  # g_from == g_to
    #             result.append((set(), set()))
    #         else:
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #             bridges = [inter for _, _, inter in path]
    # Comment removed due to garbled encoding.
    #             result.append((bridges[0] if bridges else set(), bridges[-1] if bridges else set()))
        
    #     return result if result else [(set(), set())]
    
    # Comment removed due to garbled encoding.
    # def _construct_full_sequences(self, v1: Vertex, v2: Vertex,
    #                                 groups: list[list[Hyperedge]]
    #                                 ) -> list[list[Node]]:
    #     """
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #     """
    #     logger = getLogger("semantic_cluster")
        
    # Comment removed due to garbled encoding.
    #     hyperedges_v1 = [he for he in self.hyperedges if v1 in he.vertices]
    #     hyperedges_v2 = [he for he in self.hyperedges if v2 in he.vertices]
        
    #     if not hyperedges_v1 or not hyperedges_v2:
    #         return []
        
    #     _, he_to_group = self._build_hyperedge_groups()
    #     intersections = self._find_group_intersections()
        
    #     sequences = []
        
    # Comment removed due to garbled encoding.
    #     for he_v1 in hyperedges_v1:
    #         node_v1 = he_v1.current_node(v1)
    #         if node_v1 is None:
    #             continue
            
    #         g_v1 = he_to_group.get(he_v1)
    #         if g_v1 is None:
    #             continue
            
    #         for he_v2 in hyperedges_v2:
    #             node_v2 = he_v2.current_node(v2)
    #             if node_v2 is None:
    #                 continue
                
    #             g_v2 = he_to_group.get(he_v2)
    #             if g_v2 is None:
    #                 continue
                
    # Comment removed due to garbled encoding.
    #             if g_v1 == g_v2:
    #                 path = self._construct_path_within_group(node_v1, node_v2, groups[g_v1])
    #                 if path:
    #                     sequences.append(path)
    #             else:
    # Comment removed due to garbled encoding.
    #                 big_step_sequences = self._construct_path_across_groups(
    #                     node_v1, node_v2, g_v1, g_v2, groups, intersections
    #                 )
                    
    # Comment removed due to garbled encoding.
    #                 for big_steps in big_step_sequences:
    #                     full_path = []
                        
    # Comment removed due to garbled encoding.
    #                     for step_idx, (start_node, group_idx, end_node) in enumerate(big_steps):
    # Comment removed due to garbled encoding.
    #                         small_path = self._construct_path_within_group(start_node, end_node, groups[group_idx])
                            
    #                         if small_path is None:
    #                             logger.debug(f"[_construct_full_sequences] Failed to construct path in group {group_idx}")
    # Comment removed due to garbled encoding.
                            
    # Comment removed due to garbled encoding.
    #                         if not full_path:
    #                             full_path.extend(small_path)
    #                         else:
    #                             full_path.extend(small_path[1:])
    #                     else:
    # Comment removed due to garbled encoding.
    #                         if full_path:
    #                             sequences.append(full_path)
        
    #     return sequences
    
    # Comment removed due to garbled encoding.
    # def _construct_path_within_group(self, node_v1: Node, node_v2: Node,
    #                                   group: list[Hyperedge]) -> list[Node] | None:
    #     """
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #     """
    #     logger = getLogger("semantic_cluster")
        
    # Comment removed due to garbled encoding.
    #     nearest_root = self._get_nearest_root_for_node_pair(node_v1, node_v2)
    #     if nearest_root is None:
    #         logger.debug(f"[_construct_path_within_group] No common root for '{node_v1.text}' and '{node_v2.text}'")
    #         return None
        
    # Comment removed due to garbled encoding.
    #     path_v1_to_root = []
    #     current = node_v1
    #     visited = set()
    #     while current is not None and current not in visited:
    #         visited.add(current)
    #         path_v1_to_root.append(current)
    #         if current == nearest_root:
    #             break
    #         current = current.head
        
    #     if not path_v1_to_root or path_v1_to_root[-1] != nearest_root:
    #         logger.debug(f"[_construct_path_within_group] Failed to trace node_v1 to root")
    #         return None
        
    # Comment removed due to garbled encoding.
    #     path_v2_to_root = []
    #     current = node_v2
    #     visited = set()
    #     while current is not None and current not in visited:
    #         visited.add(current)
    #         path_v2_to_root.append(current)
    #         if current == nearest_root:
    #             break
    #         current = current.head
        
    #     if not path_v2_to_root or path_v2_to_root[-1] != nearest_root:
    #         logger.debug(f"[_construct_path_within_group] Failed to trace node_v2 to root")
    #         return None
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #     path_v2_to_root_reversed = path_v2_to_root[:-1]
    #     path_v2_to_root_reversed.reverse()
        
    #     full_path = path_v1_to_root + path_v2_to_root_reversed
    #     return full_path if full_path else None
    
    # Comment removed due to garbled encoding.
    # def _construct_path_across_groups(self, node_v1: Node, node_v2: Node,
    #                                 g_v1: int, g_v2: int,
    #                                 groups: list[list[Hyperedge]],
    #                                 intersections: dict[tuple[int, int], set[Node]]
    #                                 ) -> list[list[tuple[Node, int, Node]]]:
    #     """
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #     """
    #     logger = getLogger("semantic_cluster")

    # Comment removed due to garbled encoding.
    #     group_paths = self._find_all_shortest_inter_group_paths_bfs(g_v1, g_v2, intersections)

    #     if not group_paths:
    #         logger.debug(f"[_construct_path_across_groups] No path between group {g_v1} and {g_v2}")
    #         return []

    #     all_big_step_sequences: list[list[tuple[Node, int, Node]]] = []

    #     for group_path in group_paths:
    #         if len(group_path) < 2:
    #             continue

    # Comment removed due to garbled encoding.
    #         bridge_candidates_per_hop: list[list[Node]] = []
    #         valid_group_path = True
    #         for i in range(len(group_path) - 1):
    #             g_curr = group_path[i]
    #             g_next = group_path[i + 1]
    #             inter_key = (g_curr, g_next) if (g_curr, g_next) in intersections else (g_next, g_curr)
    #             bridge_nodes = intersections.get(inter_key)
    #             if not bridge_nodes:
    #                 valid_group_path = False
    #                 break
    #             bridge_candidates_per_hop.append(list(bridge_nodes))

    #         if not valid_group_path or not bridge_candidates_per_hop:
    #             continue

    # Comment removed due to garbled encoding.
    #         for bridge_combo in product(*bridge_candidates_per_hop):
    #             big_steps: list[tuple[Node, int, Node]] = []

    # Comment removed due to garbled encoding.
    #             big_steps.append((node_v1, group_path[0], bridge_combo[0]))

    # Comment removed due to garbled encoding.
    #             for i in range(1, len(bridge_combo)):
    #                 big_steps.append((bridge_combo[i - 1], group_path[i], bridge_combo[i]))

    # Comment removed due to garbled encoding.
    #             big_steps.append((bridge_combo[-1], group_path[-1], node_v2))

    #             all_big_step_sequences.append(big_steps)

    #     return all_big_step_sequences

    # def _find_all_shortest_inter_group_paths_bfs(self, g_from: int, g_to: int,
    #                                             intersections: dict[tuple[int, int], set[Node]]
    #                                             ) -> list[list[int]]:
    #     """
    # Comment removed due to garbled encoding.
    #     """
    #     if g_from == g_to:
    #         return [[g_from]]

    # Comment removed due to garbled encoding.
    #     adjacency: dict[int, set[int]] = {}
    #     for g1, g2 in intersections.keys():
    #         adjacency.setdefault(g1, set()).add(g2)

    # Comment removed due to garbled encoding.
    #     dist: dict[int, int] = {g_from: 0}
    #     queue = deque([g_from])
    #     while queue:
    #         current = queue.popleft()
    #         for nxt in adjacency.get(current, set()):
    #             if nxt not in dist:
    #                 dist[nxt] = dist[current] + 1
    #                 queue.append(nxt)

    #     if g_to not in dist:
    #         return []

    #     shortest_len = dist[g_to]

    # Comment removed due to garbled encoding.
    #     all_paths: list[list[int]] = []

    #     def dfs(node: int, path: list[int]) -> None:
    #         if node == g_from:
    #             all_paths.append(path[::-1])
    #             return

    #         for prev in adjacency.get(node, set()):
    #             if prev in dist and dist[prev] == dist[node] - 1:
    #                 dfs(prev, path + [prev])

    #     dfs(g_to, [g_to])

    # Comment removed due to garbled encoding.
    #     return [p for p in all_paths if len(p) - 1 == shortest_len]
    
    # Comment removed due to garbled encoding.
    # def _find_shortest_inter_group_path_bfs(self, g_from: int, g_to: int,
    #                                         intersections: dict[tuple[int, int], set[Node]]
    #                                         ) -> list[int]:
    #     """
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #     """
    #     if g_from == g_to:
    #         return [g_from]
        
    #     queue = deque([(g_from, [g_from])])
    #     visited = {g_from}
        
    #     while queue:
    #         current, path = queue.popleft()
            
    #         if current == g_to:
    #             return path
            
    # Comment removed due to garbled encoding.
    #         for (g1, g2) in intersections.keys():
    #             if g1 == current and g2 not in visited:
    #                 queue.append((g2, path + [g2]))
    #                 visited.add(g2)
    #             elif g2 == current and g1 not in visited:
    #                 queue.append((g1, path + [g1]))
    #                 visited.add(g1)
        
    #     return []
    
    # Comment removed due to garbled encoding.
    # def _path_to_string(self, path: list[Node]) -> tuple[str, int]:
    #     """
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    #     - (description: str, length: int)
    #     """
    #     if not path:
    #         return "", 0
        
    # Comment removed due to garbled encoding.
    #     node_texts = []
    #     for node in path:
    #         if hasattr(node, 'text'):
    #             node_texts.append(node.text)
    #         else:
    #             node_texts.append(str(node))
        
    #     description = " ".join(node_texts)
    #     return description, len(path)

    def get_path_node_steps(self, v1: Vertex, v2: Vertex) -> tuple[list[list[Node]], Node | None, Node | None]:
        """ Docstring removed due to garbled encoding. """
        logger = getLogger("semantic_cluster")
        
        if v1 is None or v2 is None:
            return [], None, None
        
        try:
            # Comment removed due to garbled encoding.
            v1_groups = self._get_vertex_groups(v1)
            v2_groups = self._get_vertex_groups(v2)
            
            if not v1_groups or not v2_groups:
                logger.debug(f"[get_path_node_steps] v1 or v2 not in any group")
                return [], None, None
            
            # Comment removed due to garbled encoding.
            closest_result = self._find_closest_group_pair(v1_groups, v2_groups)
            if closest_result is None:
                logger.debug(f"[get_path_node_steps] No path between v1_groups={v1_groups} and v2_groups={v2_groups}")
                return [], None, None
            
            g1, gn, group_path = closest_result
            logger.debug(f"[get_path_node_steps] Found path: {group_path}")
            
            # Comment removed due to garbled encoding.
            groups, he_to_group = self._build_hyperedge_groups()
            
            n1 = None
            for he in self.hyperedges:
                if v1 in he.vertices and he_to_group.get(he) == g1:
                    n1 = he.current_node(v1)
                    if n1 is not None:
                        break
            
            if n1 is None:
                logger.debug(f"[get_path_node_steps] Cannot find node for v1 in group {g1}")
                return [], None, None
            
            # Comment removed due to garbled encoding.
            n2 = None
            for he in self.hyperedges:
                if v2 in he.vertices and he_to_group.get(he) == gn:
                    n2 = he.current_node(v2)
                    if n2 is not None:
                        break
            
            if n2 is None:
                logger.debug(f"[get_path_node_steps] Cannot find node for v2 in group {gn}")
                return [], None, None
            
            # Comment removed due to garbled encoding.
            triple_sequence: list[tuple[Node, int, Node]] = []
            bridge_map = self._build_inter_group_bridge_map()
            
            # Comment removed due to garbled encoding.
            if g1 == gn:
                triple_sequence.append((n1, g1, n2))
            else:
                # Comment removed due to garbled encoding.
                current_node = n1
                for i in range(len(group_path) - 1):
                    gi = group_path[i]
                    next_gi = group_path[i + 1]
                    
                    # Comment removed due to garbled encoding.
                    bridge_key = (gi, next_gi) if (gi, next_gi) in bridge_map else (next_gi, gi)
                    bridges = bridge_map.get(bridge_key, set())
                    
                    if not bridges:
                        logger.debug(f"[get_path_node_steps] No bridges between {gi} and {next_gi}")
                        return [], None, None
                    
                    # Comment removed due to garbled encoding.
                    next_node = next(iter(bridges))
                    triple_sequence.append((current_node, gi, next_node))
                    current_node = next_node
                
                # Comment removed due to garbled encoding.
                triple_sequence.append((current_node, gn, n2))
            
            # Comment removed due to garbled encoding.
            result: list[list[Node]] = []
            
            for node_a, group_idx, node_b in triple_sequence:
                # Comment removed due to garbled encoding.
                # Comment removed due to garbled encoding.
                assert self._node_pair_nearest_root
                
                nearest_root = self._node_pair_nearest_root.get((node_a, node_b))
                
                if nearest_root is None:
                    # Comment removed due to garbled encoding.
                    nearest_root = self._node_pair_nearest_root.get((node_b, node_a))
                
                if nearest_root is None:
                    logger.debug(f"[get_path_node_steps] Cannot find common root for {node_a.text} and {node_b.text} in group {group_idx}")
                    return [], None, None
                
                # Comment removed due to garbled encoding.
                path_a: list[Node] = []
                current = node_a
                visited_a: set[Node] = set()
                while current is not None and current not in visited_a:
                    visited_a.add(current)
                    path_a.append(current)
                    if current == nearest_root:
                        break
                    current = current.head
                
                # Comment removed due to garbled encoding.
                path_b: list[Node] = []
                current = node_b
                visited_b: set[Node] = set()
                while current is not None and current not in visited_b:
                    visited_b.add(current)
                    path_b.append(current)
                    if current == nearest_root:
                        break
                    current = current.head
                
                # Comment removed due to garbled encoding.
                if not path_a or path_a[-1] != nearest_root:
                    logger.debug(f"[get_path_node_steps] Failed to trace {node_a.text} to root {nearest_root.text}")
                    return [], None, None
                
                if not path_b or path_b[-1] != nearest_root:
                    logger.debug(f"[get_path_node_steps] Failed to trace {node_b.text} to root {nearest_root.text}")
                    return [], None, None
                
                # Comment removed due to garbled encoding.
                # path_a: [a, ..., root]
                # path_b: [b, ..., root]
                # Comment removed due to garbled encoding.
                merged_path = path_a + path_b[-2::-1]  # Comment removed due to garbled encoding.
                
                # Comment removed due to garbled encoding.
                sorted_nodes = sorted(merged_path, key=lambda node: node.index if hasattr(node, 'index') else float('inf'))
                
                result.append(sorted_nodes)
            
            return result, n1, n2
            
        except Exception as e:
            logger.exception(f"[get_path_node_steps] Error finding path: {e}")
            return [], None, None
    
    # Comment removed due to garbled encoding.
    # def get_path_description(self, v1: Vertex, v2: Vertex) -> list[tuple[str, int]]:
    #     """
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
        
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    #     """
    #     logger = getLogger("semantic_cluster")
        
    #     if v1 is None or v2 is None:
    #         return []
        
    #     try:
    # Comment removed due to garbled encoding.
    #         groups, _ = self._build_hyperedge_groups()
    #         sequences = self._construct_full_sequences(v1, v2, groups)
            
    #         if not sequences:
    #             logger.debug(f"[get_path_description] No path found between {v1.text()} and {v2.text()}")
    #             return []
            
    # Comment removed due to garbled encoding.
    #         paths = [self._path_to_string(seq) for seq in sequences]
            
    # Comment removed due to garbled encoding.
    #         paths.sort(key=lambda x: x[1])
            
    #         return paths
            
    #     except Exception as e:
    #         logger.exception(f"[get_path_description] Error finding path: {e}")
    #         return []
        
    @staticmethod
    def likely_nodes(nodes1: list[Vertex], nodes2: list[Vertex]) -> dict[Vertex, set[Vertex]]:
        # node is likely if NLI label is entailment or share same pos
        likely_nodes: dict[Vertex, set[Vertex]] = {}
        text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
        for node1 in nodes1:
            for node2 in nodes2:
                text_pair_to_node_pairs[(node1.text(), node2.text())] = (node1, node2)
        text_pairs = list(text_pair_to_node_pairs.keys())
        labels = get_nli_labels_batch(text_pairs)
        for i, text_pair in enumerate(text_pairs):
            node_pair = text_pair_to_node_pairs[text_pair]
            label = labels[i]
            node1, node2 = node_pair
            if label == "entailment" or (label == "neutral" and node1.is_domain(node2)):
                if node1 not in likely_nodes:
                    likely_nodes[node1] = set()
                likely_nodes[node1].add(node2)
        return likely_nodes
    
    
    def is_subset_of(self, other: 'SemanticCluster') -> bool:
        self_edge_set = set(self.hyperedges)
        other_edge_set = set(other.hyperedges)
        return self_edge_set.issubset(other_edge_set)
    
    def get_contained_hyperedges(self, vertex: Vertex) -> list[Hyperedge]:
        if vertex in self.contained_hyperedges:
            return self.contained_hyperedges[vertex]
        contained_edges: list[Hyperedge] = []
        for he in self.hyperedges:
            if vertex in he.vertices:
                contained_edges.append(he)
        self.contained_hyperedges[vertex] = contained_edges
        return contained_edges
    
    def get_vertices(self) -> list[Vertex]:
        if len(self.vertices) > 0:
            return self.vertices
        id_set: set[int] = set()
        ordered_vertices: list[Vertex] = []
        for he in self.hyperedges:
            for v in he.vertices:
                if v.id in id_set:
                    continue
                id_set.add(v.id)
                ordered_vertices.append(v)
        self.vertices = ordered_vertices
        return self.vertices
    
    def get_path_within_hyperedges(self, v1: Node, v2: Node, root: Node) -> tuple[str, int]:
        # Implementation for getting path within hyperedges
        key = (v1, v2, root)
        if key in self.vertices_paths_within_hyperedges:
            return self.vertices_paths_within_hyperedges[key]
        
        # collect the nodes by v1 -> root -> v2
        path_nodes: set[Node] = set()
        path_nodes.add(root)
        path_nodes.add(v1)
        path_nodes.add(v2)
        current = v1.head
        while current and current not in path_nodes:
            path_nodes.add(current)
            current = current.head
        
        current = v2.head
        while current and current not in path_nodes:
            path_nodes.add(current)
            current = current.head
        
        nodes = sorted(list(path_nodes), key=lambda n: n.index)
        
        base_desc = " ".join([f"{n.text}" for n in nodes])
        
        type_v1 = v1.type_str()
        type_v2 = v2.type_str()
        if not type_v1:
            type_v1 = v1.text
        if not type_v2:
            type_v2 = v2.text
        
        if type_v1 == type_v2:
            type_v1 = f"{type_v1}#1"
            type_v2 = f"{type_v2}#2"
        
        desc = base_desc.replace(v1.text, type_v1).replace(v2.text, type_v2)
        return desc, len(nodes)
    
    def get_path_to_root(self, node: Node, root: Node) -> tuple[str, int]:
        # Implementation for getting path to root
        path_nodes: set[Node] = set()
        path_nodes.add(root)
        path_nodes.add(node)
        current = node.head
        while current and current not in path_nodes:
            path_nodes.add(current)
            current = current.head
        nodes = sorted(list(path_nodes), key=lambda n: n.index)
        desc = " ".join([f"{n.text}" for n in nodes])
        
        node_type = node.type_str()
        if not node_type:
            node_type = node.text
        desc = desc.replace(node.text, node_type)
        
        return desc, len(nodes)

    def get_paths_between_vertices(self, v1: Vertex, v2: Vertex) -> tuple[str, int]:
        key = (v1, v2)
        if key in self.vertices_paths:
            return self.vertices_paths[key]
        logger = getLogger("semantic_cluster")
        logger.debug(f"String removed due to garbled encoding.")

        node_vertex: dict[Node, Vertex] = {}
        nodes_in_vertices: set[Node] = set()
        for he in self.hyperedges:
            for v in he.vertices:
                if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                    continue
                nodes_in_vertices.add(he.current_node(v))
                node_vertex[he.current_node(v)] = v
            
        nodes_in_vertices_list = list(nodes_in_vertices)
        queries: list[tuple[Node, Node]] = []
        for i in range(len(nodes_in_vertices_list) - 1):
            for j in range(i + 1, len(nodes_in_vertices_list)):
                u = nodes_in_vertices_list[i]
                v = nodes_in_vertices_list[j]
                queries.append((u, v))
        
        edge_between_nodes: list[tuple[Node, Node]] = []
        saved_nodes: set[Node] = set()
        for he in self.hyperedges:
            root = he.current_node(he.root)
            for i in range(1, len(he.vertices)):
                node = he.current_node(he.vertices[i])
                edge_between_nodes.append((root, node))
                saved_nodes.add(node)
            head = root.head
            current = root
            visited_in_trace = {current}
            while head:
                if head in visited_in_trace:
                    logger.warning(f"Cycle detected in head trace: '{head.text}' is already in trace. Breaking.")
                    break
                visited_in_trace.add(head)
                
                edge_between_nodes.append((head, current))
                if head in saved_nodes:
                    break
                current = head
                # Comment removed due to garbled encoding.
                if head.head == head:
                    logger.warning(f"String removed due to garbled encoding.")
                    break
                head = head.head
            saved_nodes.add(root)
        lca_results = TarjanLCA(edge_between_nodes, queries).lca()
        lca_map: dict[tuple[Node, Node], Node] = {}
        for i, (u, v) in enumerate(queries):
            lca_node = lca_results[i]
            if lca_node:
                lca_map[(u, v)] = lca_node
            
        node_paths: dict[tuple[Vertex, Vertex], list[tuple[str, int]]] = {}
        for (u, v), k in lca_map.items():
            vertex_u = node_vertex[u]
            vertex_v = node_vertex[v]
            
            if u == k:
                text = f"#A -{v.dep.name}-> #B"
                node_paths.setdefault((vertex_u, vertex_v), []).append((text, 1))
                continue
            elif v == k:
                text = f"#A <-{u.dep.name}- #B"
                node_paths.setdefault((vertex_u, vertex_v), []).append((text, 1))
                continue
            
            # Comment removed due to garbled encoding.
            node_cnt = 1
            path_items: list[Node] = []
            current = u
            current_trace: list[str] = [current.text]
            visited_trace: set[Node] = {current}
            while current != k:
                if current in nodes_in_vertices:
                    node_cnt += 1
                    path_items.append(current)
                if current.head is None:
                    logger.warning(f"String removed due to garbled encoding."
                            f"while tracing to LCA '{k.text}' (index={k.index}). "
                            f"String removed due to garbled encoding.")
                    break
                # Comment removed due to garbled encoding.
                if current.head in visited_trace:
                    logger.warning(f"String removed due to garbled encoding.")
                    break
                visited_trace.add(current.head)
                
                if current.head == current:
                    logger.warning(f"String removed due to garbled encoding.")
                    break
                current = current.head
                current_trace.append(current.text)
            else:
                # Comment removed due to garbled encoding.
                path_items.append(k)
                
                # Comment removed due to garbled encoding.
                rev_path_items: list[Node] = []
                current = v
                current_trace = [current.text]
                visited_trace_v: set[Node] = {current}
                while current != k:
                    if current in nodes_in_vertices:
                        node_cnt += 1
                        rev_path_items.append(current)
                    if current.head is None:
                        logger.warning(f"String removed due to garbled encoding."
                                f"while tracing to LCA '{k.text}' (index={k.index}). "
                                f"String removed due to garbled encoding.")
                        break
                    # Comment removed due to garbled encoding.
                    if current.head in visited_trace_v:
                        logger.warning(f"String removed due to garbled encoding.")
                        break
                    visited_trace_v.add(current.head)
                    
                    if current.head == current:
                        logger.warning(f"String removed due to garbled encoding.")
                        break
                    current = current.head
                    current_trace.append(current.text)
                else:
                    # Comment removed due to garbled encoding.
                    rev_path_items = rev_path_items[::-1]
                    path_items.extend(rev_path_items)
                    text = node_sequence_to_text(path_items)
                    text_inv = text.replace("#A", "#TEMP").replace("#B", "#A").replace("#TEMP", "#B")
                    
                    node_paths.setdefault((vertex_u, vertex_v), []).append((text, node_cnt))
                    node_paths.setdefault((vertex_v, vertex_u), []).append((text_inv, node_cnt))
                    continue  # Comment removed due to garbled encoding.
        
        # Comment removed due to garbled encoding.
        for (vertex_u, vertex_v), paths in node_paths.items():
            if paths:
                paths = sorted(paths, key=lambda x: x[1])
                self.vertices_paths[(vertex_u, vertex_v)] = paths[0]
        
        result = self.vertices_paths.get(key, ("", 0))
        logger.debug(f"get_paths_between_vertices result: count={result[1]}, sample='{result[0][:50]}...'")
        return result
    
    def text(self) -> str:
        if self.text_cache is not None:
            return self.text_cache

        if not self.hyperedges:
            return ""
        logger = getLogger("semantic_cluster")
        try:
            # Step 1: Build root_ancestors mapping
            root_ancestors = {}
            for e in self.hyperedges:
                root_node = e.current_node(e.root)
                if root_node is None:
                    logger.error(f"[text] Hyperedge {e} has invalid root node (None). Skipping.")
                    continue
                root_ancestors[root_node] = root_node

            # Resolve ancestor chains (with cycle detection)
            for e in self.hyperedges:
                root = e.current_node(e.root)
                if root is None:
                    continue
                node = root
                visited = set()
                while node.head is not None:
                    if node in visited:
                        logger.warning(f"[text] Detected cycle in ancestor chain starting from {root.text}. Breaking.")
                        break
                    visited.add(node)
                    if node.head in root_ancestors:
                        root_ancestors[root] = root_ancestors[node.head]
                        break
                    node = node.head

            # Step 2: Group nodes by ultimate root
            root_to_nodes: dict[Node, set[Node]] = {}
            for e in self.hyperedges:
                root = e.current_node(e.root)
                if root is None or root not in root_ancestors:
                    continue
                ultimate_root = root_ancestors[root]
                if ultimate_root not in root_to_nodes:
                    root_to_nodes[ultimate_root] = set()
                for vertex in e.vertices:
                    node = e.current_node(vertex)
                    if node is not None:
                        root_to_nodes[ultimate_root].add(node)

            # Step 3: Order sub-clusters by index
            sub_cluster_roots = set(root_ancestors.get(r, r) for r in root_to_nodes.keys())
            sub_clusters = sorted(list(sub_cluster_roots), key=lambda r: getattr(r, 'index', float('inf')))

            # Step 4: Generate text for each sub-cluster
            texts = []
            for root in sub_clusters:
                if root not in root_to_nodes:
                    continue
                nodes = list(root_to_nodes[root])
                if not nodes:
                    continue

                try:
                    start = min(getattr(node, 'index', 0) for node in nodes)
                    end = max(getattr(node, 'index', 0) for node in nodes) + 1
                except Exception as ex:
                    logger.error(f"[text] Failed to compute indices for root {root.text}: {ex}")
                    continue

                sentence_by_range = str(self.doc[start:end]) if self.doc else ""
                sentence_obj = getattr(root, 'sentence', None)
                sentence = str(sentence_obj) if sentence_obj else ""

                # Helper to extract prefix/suffix
                def calc_prefix_suffix(range_text, full_sentence):
                    start_idx = full_sentence.find(range_text)
                    if start_idx != -1:
                        prefix = full_sentence[:start_idx].strip()
                        suffix = full_sentence[start_idx + len(range_text):].strip()
                        return prefix, suffix
                    else:
                        return "", ""

                prefix, suffix = calc_prefix_suffix(sentence_by_range, sentence)

                # Build replacement list
                replacement = []
                for node in nodes:
                    if node == root:
                        continue
                    resolved_text = Vertex.resolved_text(node)
                    original_text = getattr(node, 'text', '')
                    replacement.append((original_text, resolved_text))

                # Add prefix/suffix removal
                if prefix:
                    replacement.append((prefix, ""))
                if suffix:
                    replacement.append((suffix, ""))

                # Apply replacements
                final_sentence = sentence
                for old, new in replacement:
                    if old in final_sentence:
                        final_sentence = final_sentence.replace(old, new)

                cleaned = final_sentence.strip()
                if cleaned:
                    texts.append(cleaned)

            # Step 5: Combine
            text = " ".join(texts).strip()
            self.text_cache = text
            return text

        except Exception as e:
            logger.exception(f"[text] Unexpected error in SemanticCluster.text(): {e}")
            fallback = " ".join(
                str(e.current_node(e.root).text) for e in self.hyperedges
                if e.current_node(e.root) and hasattr(e.current_node(e.root), 'text')
            ).strip()
            self.text_cache = fallback
            return fallback
        
    def virtual_text(self) -> str:
        pass
    
    def _build_signature(self) -> tuple:
        if not self.hyperedges:
            return ()
        items = []
        for he in self.hyperedges:
            root_id = he.root.id if he.root else -1
            items.append((root_id, he.start, he.end, he.desc))
        items.sort()
        return tuple(items)

    def signature(self) -> tuple:
        if self._signature is None:
            self._signature = self._build_signature()
        return self._signature

    def to_triple(self) -> list[tuple[str, list[str]]]:
        """ Docstring removed due to garbled encoding. """
        triples = []
        for he in self.hyperedges:
            root_text = Vertex.resolved_text(he.current_node(he.root))
            
            # Comment removed due to garbled encoding.
            args = []
            for vertex in he.vertices:
                if vertex == he.root:
                    continue
                node = he.current_node(vertex)
                node_text = Vertex.resolved_text(node)
                args.append(node_text)
                
                if node.pos in {Pos.ADJ, Pos.ADV} and node.dep in {Dep.amod, Dep.advmod}:
                    head = node.head
                    if head and head.pos in {Pos.NOUN, Pos.PROPN, Pos.VERB}:
                        head_text = Vertex.resolved_text(head)
                        triples.append(("attr", [head_text, node_text]))
            
            triples.append((root_text, args))
        
        return triples
    
    def to_triple_text(self) -> str:
        """ Docstring removed due to garbled encoding. """
        texts = []
        for root, args in self.to_triple():
            if len(args) == 0:
                texts.append(f"{root}()")
            else:
                texts.append(f"{root}({', '.join(args)})")
        return " & ".join(texts)

    def __hash__(self) -> int:
        return hash((self.is_query, self.signature()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticCluster):
            return False
        if self.is_query != other.is_query:
            return False
        return self.signature() == other.signature()
    
def combine_hyperedges_to_cluster(hypergraph: Hypergraph) -> list[SemanticCluster]:
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    clusters: list[SemanticCluster] = []
    
    # TODO: USE MORE CONVINCED METHOD
    root_to_hyperedge: dict[Node, Hyperedge] = {} # assume one root corresponds to one hyperedge for simplicity
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node is None:
            continue
        root_to_hyperedge[root_node] = he
    
    hyperedge_visited: set[Hyperedge] = set()
    
    
    # Rule 1: child: VERB/AUX + root: non-VERB/AUX -> combine into cluster
    for he in hypergraph.hyperedges:
        if he in hyperedge_visited:
            continue
        
        root_node = he.current_node(he.root)
        if root_node is None:
            continue
        
        if root_node.pos in {Pos.VERB, Pos.AUX}:
            continue
        
        # After that, root must be not a VERB/AUX, we check its children
        
        # children are vertices that not the he.vertices[0]
        children = he.vertices[1:]
        
        descent_hyperedges = []
        
        for child_vertex in children:
            child_node = he.current_node(child_vertex)
            if child_node is None:
                continue
            
            if child_node.pos in {Pos.VERB, Pos.AUX}:
                # This child is a verb, we check if it has a corresponding hyperedge
                child_he = root_to_hyperedge.get(child_node)
                if child_he and child_he not in hyperedge_visited:
                    # We combine he and child_he into a new cluster
                    descent_hyperedges.append(child_he)
                    hyperedge_visited.add(child_he)
                    
        if not descent_hyperedges:
            continue
        
        # Create a cluster for he and its descent hyperedges
        cluster_hyperedges = [he] + descent_hyperedges
        clusters.append(SemanticCluster(cluster_hyperedges, hypergraph.doc, is_query=True))
        hyperedge_visited.add(he)
    
    # Rule 2: Dep relcl: if a hyperedge's root has dep relcl then we combine it with its head hyperedge into a new cluster
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node and root_node.dep == Dep.relcl and root_node.head:
            head_he = root_to_hyperedge.get(root_node.head)
            if head_he and head_he not in hyperedge_visited:
                clusters.append(SemanticCluster([he, head_he], hypergraph.doc, is_query=True))
                hyperedge_visited.add(he)
                hyperedge_visited.add(head_he)
    
    # Rule 3: VERB/AUX-[advcl/ccomp]-VERB/AUX: if a hyperedge's root is VERB/AUX and it's dep is advcl/ccomp and its head is also VERB/AUX, then we combine it with its head hyperedge into a new cluster
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node and root_node.pos in {Pos.VERB, Pos.AUX} and root_node.dep in {Dep.advcl, Dep.ccomp} and root_node.head and root_node.head.pos in {Pos.VERB, Pos.AUX}:
            head_he = root_to_hyperedge.get(root_node.head)
            if head_he and head_he not in hyperedge_visited:
                clusters.append(SemanticCluster([he, head_he], hypergraph.doc, is_query=True))
                hyperedge_visited.add(he)
                hyperedge_visited.add(head_he)
    
    # Rule-last: all single SC
    for he in hypergraph.hyperedges:
        if he not in hyperedge_visited:
            clusters.append(SemanticCluster([he], hypergraph.doc))
    
    return clusters

# NEW IMPLEMENTATION FOR SEMANTIC CLUSTER PAIRS
def calc_semantic_cluster_pairs(
    hypergraph_q: Hypergraph, 
    hypergraph_d: Hypergraph, 
    likely_map: dict[Vertex, set[Tuple[Vertex, float]]], 
    cluster_sim_threshold: float = 0.5,
    branch_threshold: int = 5,
    is_multihop: bool = False, 
    logger: Optional[logging.Logger] = None
) -> list[tuple[SemanticCluster, SemanticCluster, float]]:
    assert logger is not None
    # Comment removed due to garbled encoding.
    logger.info(f"[SemanticClusterPairs] Start: Q_edges={len(hypergraph_q.hyperedges)}, "
                   f"D_edges={len(hypergraph_d.hyperedges)}, likely_map={len(likely_map)}, "
                   f"threshold={cluster_sim_threshold}, multihop={is_multihop}")
    
    pairs: list[tuple[SemanticCluster, SemanticCluster, float]] = []
    
    # Comment removed due to garbled encoding.
    clusters_q = combine_hyperedges_to_cluster(hypergraph_q)
    logger.info(f"[SemanticClusterPairs] Step1-Done: Generated {len(clusters_q)} query clusters from {len(hypergraph_q.hyperedges)} hyperedges")
    
    # Comment removed due to garbled encoding.
    calc_embedding_for_cluster_batch(clusters_q)
    logger.info(f"[SemanticClusterPairs] Step2-Done: Computed embeddings for {len(clusters_q)} query clusters")
    
    # Comment removed due to garbled encoding.
    K_LIKELY = branch_threshold
    # likely_map = get_top_k_matched_vertices(matched_vertices, K_LIKELY)
    matched_count = sum(len(v) for v in likely_map.values())
    logger.info(f"[SemanticClusterPairs] Step3-Done: likely_map built with {len(likely_map)} source vertices, "
                   f"{matched_count} total matches (K={K_LIKELY})")
    
    # Comment removed due to garbled encoding.
    vertices_pairs_need_path: list[tuple[Vertex, Vertex]] = []
    vertices_pairs_to_sc: dict[tuple[Vertex, Vertex], list[SemanticCluster]] = {}
    
    pair_gen_start = time.time() if logger else None  # Comment removed due to garbled encoding.
    
    for sc_q in clusters_q:
        vertices_d_pairs: set[tuple[Vertex, Vertex]] = set()
        for u in sc_q.get_vertices():
            for u_prime in sc_q.get_vertices():
                if u == u_prime:
                    continue
                if Vertex.is_both_verb(u, u_prime):
                    continue
                
                for v, score_v in likely_map.get(u, set()):
                    for v_prime, score_v_prime in likely_map.get(u_prime, set()): 
                        if v == v_prime:
                            continue
                        if Vertex.is_both_verb(v, v_prime):
                            continue
                        
                        vertices_d_pairs.add((v, v_prime))
        
        for pair in vertices_d_pairs:
            vertices_pairs_to_sc.setdefault(pair, []).append(sc_q)
            vertices_pairs_need_path.append(pair)
    
    if logger and pair_gen_start is not None:
        pair_gen_time = time.time() - pair_gen_start
        unique_pairs = len(set(vertices_pairs_need_path))
        logger.info(f"[SemanticClusterPairs] Step4-Done: Generated {len(vertices_pairs_need_path)} raw pairs, "
                   f"{unique_pairs} unique pairs, {len(vertices_pairs_to_sc)} pairs mapped to clusters, "
                   f"time={pair_gen_time:.2f}s")
    
    # Comment removed due to garbled encoding.
    vertices_pairs_need_path = list(set(vertices_pairs_need_path))
    logger.info(f"[SemanticClusterPairs] Step5-Start: Path search for {len(vertices_pairs_need_path)} unique vertex pairs "
                   f"(method={'local' if is_multihop else 'global'})")
    path_search_start = time.time()
    
    if is_multihop:
        path_map = find_shortest_hyperpaths_local(hypergraph_d, vertices_pairs_need_path)
    else:
        path_map = find_shortest_hyperpaths(hypergraph_d, vertices_pairs_need_path)

    path_search_time = time.time() - path_search_start
    reachable = sum(1 for p in path_map.values() if p)
    logger.info(f"[SemanticClusterPairs] Step5-Done: Path search completed in {path_search_time:.2f}s, "
                f"reachable={reachable}/{len(path_map)} pairs")
    
    # Comment removed due to garbled encoding.
    sc_pairs_candidates: list[tuple[SemanticCluster, SemanticCluster]] = []
    sc_d_candidates: list[SemanticCluster] = []
    
    for (v, v_prime), scs in vertices_pairs_to_sc.items():
        path = path_map.get((v, v_prime), [])
        for sc_q in scs:
            sc_d = SemanticCluster(path, hypergraph_d.doc)
            sc_pairs_candidates.append((sc_q, sc_d))
            sc_d_candidates.append(sc_d)
    
    
    logger.info(f"[SemanticClusterPairs] Step6-Done: Built {len(sc_pairs_candidates)} candidate cluster pairs, {len(sc_d_candidates)} document clusters to embed")
    
    # Comment removed due to garbled encoding.
    calc_embedding_for_cluster_batch(sc_d_candidates)
    logger.info(f"[SemanticClusterPairs] Step7-Done: Computed embeddings for {len(sc_d_candidates)} document clusters")
    
    # Comment removed due to garbled encoding.
    filter_start = time.time()
    sim_embedding_pairs = [ (sc_q.embedding, sc_d.embedding) for sc_q, sc_d in sc_pairs_candidates]
    
    if sim_embedding_pairs:
        sim_scores = get_cosine_similarity_batch(sim_embedding_pairs, is_normalized=True)
    else:
        sim_scores = []
    passed_count = 0
    for (sc_q, sc_d), sim_score in zip(sc_pairs_candidates, sim_scores):
        assert sc_q.embedding is not None and sc_d.embedding is not None, "Embedding should have been calculated"    
        # print(f"Q: '{sc_q.text()}'\nD: '{sc_d.text()}'\nSim: {sim_score:.4f} that {'<' if sim_score < cluster_sim_threshold else '>='} {cluster_sim_threshold} \n---")
        if sim_score >= cluster_sim_threshold:
            pairs.append((sc_q, sc_d, sim_score))
            passed_count += 1
    
    filter_time = time.time() - filter_start
    logger.info(f"[SemanticClusterPairs] Step8-Done: Filtered {passed_count}/{len(sc_pairs_candidates)} pairs "
                f"by threshold={cluster_sim_threshold}, time={filter_time:.2f}s")
    
    # Comment removed due to garbled encoding.
    logger.info(f"[SemanticClusterPairs] Return: {len(pairs)} final semantic cluster pairs")
    return pairs

# Comment removed due to garbled encoding.
def build_descendant_cluster(
    vertex: Vertex,
    hg: Hypergraph,
    max_hops: int = 2
) -> 'SemanticCluster':
    """ Docstring removed due to garbled encoding. """
    logger = getLogger("semantic_cluster")
    
    # Comment removed due to garbled encoding.
    node = vertex.nodes[0] if vertex.nodes else None
    if not node or not hasattr(node, 'children'):
        # Comment removed due to garbled encoding.
        direct_edges = [e for e in hg.hyperedges if vertex in e.vertices]
        return SemanticCluster(direct_edges, hg.doc)
    
    # Comment removed due to garbled encoding.
    descendant_nodes = {node}
    queue = [(node, 0)]
    
    while queue:
        curr_node, depth = queue.pop(0)
        if depth >= max_hops:
            continue
        for child in getattr(curr_node, 'children', []):
            if child not in descendant_nodes:
                descendant_nodes.add(child)
                queue.append((child, depth + 1))
    
    # Comment removed due to garbled encoding.
    visited_edges = set()
    node_to_vertex = {}  # Comment removed due to garbled encoding.
    for v in hg.vertices:
        for n in v.nodes:
            node_to_vertex[n] = v
    
    for e in hg.hyperedges:
        for v in e.vertices:
            if any(n in descendant_nodes for n in v.nodes):
                visited_edges.add(e)
                break
    
    logger.debug(
        f"String removed due to garbled encoding."
        f"{len(descendant_nodes)} descendant nodes, {len(visited_edges)} hyperedges"
    )
    return SemanticCluster(list(visited_edges), hg.doc)

def calc_embedding_for_cluster_batch(clusters: list[SemanticCluster]) -> None:
    texts = [sc.text() for sc in clusters]
    embeddings = get_embedding_batch(texts)
    for i, sc in enumerate(clusters):
        sc.embedding = np.array(embeddings[i])

def get_semantic_cluster_pairs(
    query_hg: Hypergraph,
    data_hg: Hypergraph,
    allowed_pairs: Set[Tuple[int, int]],
    vertex_to_sim_id_q: Dict[Vertex, int],
    vertex_to_sim_id_d: Dict[Vertex, int],
    max_hops_query: int = 1,   # Comment removed due to garbled encoding.
    max_hops_data: int = 2,    # Comment removed due to garbled encoding.
    cluster_sim_threshold: float = 0.4,
    logger: Optional[logging.Logger] = None
) -> List[Tuple[SemanticCluster, SemanticCluster, float, Vertex, Vertex]]:
    
    if logger is None:
        logger = getLogger("semantic_cluster")
    
    logger.info(f"String removed due to garbled encoding.")
    start_time = time.time()
    
    # Comment removed due to garbled encoding.
    sim_id_to_vertex_q = {sim_id: v for v, sim_id in vertex_to_sim_id_q.items()}
    sim_id_to_vertex_d = {sim_id: v for v, sim_id in vertex_to_sim_id_d.items()}
    
    cluster_pairs = []
    pair_count = kept_count = 0
    
    for q_sim_id, d_sim_id in allowed_pairs:
        pair_count += 1
        
        q_vertex = sim_id_to_vertex_q.get(q_sim_id)
        d_vertex = sim_id_to_vertex_d.get(d_sim_id)
        if q_vertex is None or d_vertex is None:
            continue
        
        # Comment removed due to garbled encoding.
        sc_q = build_descendant_cluster(q_vertex, query_hg, max_hops=max_hops_query)
        if not sc_q.hyperedges:
            continue
        
        # Comment removed due to garbled encoding.
        sc_d = build_descendant_cluster(d_vertex, data_hg, max_hops=max_hops_data)
        if not sc_d.hyperedges:
            continue
        
        # Comment removed due to garbled encoding.
        calc_embedding_for_cluster_batch([sc_q, sc_d])
        if sc_q.embedding is None or sc_d.embedding is None:
            continue
        
        sim_score = cosine_similarity(sc_q.embedding, sc_d.embedding)
        if sim_score >= cluster_sim_threshold:
            cluster_pairs.append((sc_q, sc_d, sim_score, q_vertex, d_vertex))
            kept_count += 1
            
            q_triples = sc_q.to_triple() or []
            d_triples = sc_d.to_triple() or []
            q_triple_repr = f"({q_triples[0][0]}, {', '.join(q_triples[0][1])})" if q_triples else "(no triple)"
            d_triple_repr = f"({d_triples[0][0]}, {', '.join(d_triples[0][1])})" if d_triples else "(no triple)"
            
            logger.debug(
                f"String removed due to garbled encoding."
                f"  Q: text='{sc_q.text()}' | nodes={len(sc_q.get_vertices())}, edges={len(sc_q.hyperedges)}\n"
                f"     triple={q_triple_repr}\n"
                f"  D: text='{sc_d.text()}' | nodes={len(sc_d.get_vertices())}, edges={len(sc_d.hyperedges)}\n"
                f"     triple={d_triple_repr}"
            )
    
    cost = time.time() - start_time
    logger.info(f"String removed due to garbled encoding.")
    return cluster_pairs

def node_sequence_to_text(nodes: list[Node]) -> str:
    if not nodes:
        return ""
    start, end = nodes[0], nodes[-1]
    nodes = sorted(nodes, key=lambda n: n.index)
    texts = []
    for node in nodes:
        if node == start:
            texts.append("#A")
        elif node == end:
            texts.append("#B")
        elif node.pos in {Pos.ADV, Pos.ADJ, Pos.DET}:
            continue
        elif node.pos in {Pos.NOUN, Pos.PROPN, Pos.PRON}:
            texts.append("some")
        else:
            texts.append(Vertex.resolved_text(node))
    return " ".join(texts)
        

def _formal_text_of(root: Node, node: Node) -> str:
    match (root.pos, node.dep):
        case (Pos.AUX, Dep.nsubj) | (Pos.AUX, Dep.nsubjpass):
            text = "#A is something"
        case (Pos.AUX, Dep.iobj) | (Pos.AUX, Dep.dobj):
            text = "#A is something"
        case (Pos.VERB, Dep.nsubj) | (Pos.VERB, Dep.nsubjpass):
            text = "#A does something"
        case (Pos.VERB, Dep.iobj) | (Pos.VERB, Dep.dobj):
            text = "Someone does #A"
        case _:
            text = f"#A -{node.dep.name}-> something"
    return text

def _better_path(s1: str, s2: str, s2_inv: str) -> bool:
    nli_labels = {"entailment": 3, "neutral": 2, "contradiction": 1}
    label1 = get_nli_label(s1, s2)
    label2 = get_nli_label(s1, s2_inv)
    if nli_labels[label1] > nli_labels[label2]: # s2 is better
        return True
    
    sim1 = get_similarity(s1, s2)
    sim2 = get_similarity(s1, s2_inv)
    return sim1 > sim2 
    

def _legal_vertices(v1: Vertex, v2: Vertex) -> bool:
    # Comment removed due to garbled encoding.
    label = get_nli_label(v1.text(), v2.text())
    if not (label == "entailment" or (label == "neutral" and v1.is_domain(v2))):
        # Comment removed due to garbled encoding.
        return False

    # Comment removed due to garbled encoding.
    dep1 = v1.dep()
    dep2 = v2.dep()

    # Comment removed due to garbled encoding.
    SUBJECT_DEPS = {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.agent}
    OBJECT_DEPS = {Dep.dobj, Dep.iobj, Dep.pobj, Dep.attr}
    MODIFIER_DEPS = {Dep.amod, Dep.nmod, Dep.advmod, Dep.appos}

    # Comment removed due to garbled encoding.
    if (dep1 in SUBJECT_DEPS and dep2 in SUBJECT_DEPS) or (dep1 in OBJECT_DEPS and dep2 in OBJECT_DEPS) or (dep1 in MODIFIER_DEPS and dep2 in MODIFIER_DEPS):
        return True

    # Comment removed due to garbled encoding.
    if {dep1, dep2} <= {Dep.nmod, Dep.dobj}:
        return True

    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    return False

def _path_score(s1: str, cnt1: int, s2: str, cnt2: int, path_score_cache: dict[tuple[str, str], float]) -> float:
    key = (s1, s2)
    if key in path_score_cache:
        return path_score_cache[key]
    sim = get_similarity(s1, s2)
    score = sim / (cnt1 + cnt2)
    path_score_cache[key] = score
    return score

def _get_matched_vertices(vertices1: list[Vertex], vertices2: list[Vertex]) -> dict[Vertex, set[Vertex]]: # Comment removed due to garbled encoding.
    matched_vertices: dict[Vertex, set[Vertex]] = {}
    text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
    for node1 in vertices1:
        for node2 in vertices2:
            text_pair_to_node_pairs[(node1.text(), node2.text())] = (node1, node2)
    text_pairs = list(text_pair_to_node_pairs.keys())
    labels = get_nli_labels_batch(text_pairs)
    for i, text_pair in enumerate(text_pairs):
        node_pair = text_pair_to_node_pairs[text_pair]
        label = labels[i]
        node1, node2 = node_pair
        if label == "entailment" or node1.is_domain(node2):
            if node1 not in matched_vertices:
                matched_vertices[node1] = set()
            matched_vertices[node1].add(node2)
    return matched_vertices

def get_d_match(sc1: SemanticCluster, sc2: SemanticCluster, score_threshold: float = 0.0, force_include: Optional[Tuple[Vertex, Vertex]] = None) -> list[tuple[Vertex, Vertex, float]]:
    dm_logger = getLogger("d_match")

    # Comment removed due to garbled encoding.
    sc1_vertices_all = sc1.get_vertices()
    sc1_vertices_noun = [v for v in sc1_vertices_all if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
    sc1_edges = sc1.hyperedges
    sc1_text = sc1.text()
    sc1_triples = sc1.to_triple() or []
    sc1_triple_repr = str(sc1_triples[0]) if sc1_triples else "(no triple)"

    # Comment removed due to garbled encoding.
    sc2_vertices_all = sc2.get_vertices()
    sc2_vertices_noun = [v for v in sc2_vertices_all if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
    sc2_edges = sc2.hyperedges
    sc2_text = sc2.text()
    sc2_triples = sc2.to_triple() or []
    sc2_triple_repr = str(sc2_triples[0]) if sc2_triples else "(no triple)"

    # Comment removed due to garbled encoding.
    dm_logger.info(
        f"String removed due to garbled encoding."
        f"String removed due to garbled encoding."
        f"   text='{sc1_text}'\n"
        f"   triple={sc1_triple_repr}\n"
        f"   nodes={len(sc1_vertices_all)} (noun={len(sc1_vertices_noun)}), edges={len(sc1_edges)}\n"
        f"String removed due to garbled encoding."
        f"   text='{sc2_text}'\n"
        f"   triple={sc2_triple_repr}\n"
        f"   nodes={len(sc2_vertices_all)} (noun={len(sc2_vertices_noun)}), edges={len(sc2_edges)}"
    )
    matches: list[tuple[Vertex, Vertex]] = []
    # Comment removed due to garbled encoding.
    sc1_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc1.get_vertices()))
    sc2_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc2.get_vertices()))
    
    # logger.info(f"SC1 non-verb vertices: {[v.text() for v in sc1_vertices]}")
    # logger.info(f"SC2 non-verb vertices: {[v.text() for v in sc2_vertices]}")

    index_map: dict[Vertex, int] = {}
    for e in sc1.hyperedges:
        for v in e.vertices:
            if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                continue
            if v not in index_map:
                index_map[v] = e.current_node(v).index
                
    for e in sc2.hyperedges:
        for v in e.vertices:
            if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                continue
            if v not in index_map:
                index_map[v] = e.current_node(v).index
    
    sc1_edges: list[tuple[Vertex, Vertex]] = []
    for he in sc1.hyperedges:
        for i in range(len(he.vertices) - 1):
            for j in range(i + 1, len(he.vertices)):
                if he.have_no_link(he.vertices[i], he.vertices[j]):
                    continue
                if he.is_sub_vertex(he.vertices[i], he.vertices[j]):
                    sc1_edges.append((he.vertices[i], he.vertices[j]))
                else:
                    sc1_edges.append((he.vertices[j], he.vertices[i]))
    
    # logger.info(f"SC1 direct edges: {[(u.text(), v.text()) for u, v in sc1_edges]}")

    # sc1_pairs : list[tuple[Vertex, Vertex]] = []
    # all (u, v) in sc1_edges are in sc1_pairs, and if (u, k), (k, v) in sc1_edges, then (u, v) is also in sc1_pairs
    # calculate then recursively
    
    # Comment removed due to garbled encoding.
    sc1_pairs_set = set(sc1_edges)
    added = True
    tc_loop_count = 0
    while added:
        tc_loop_count += 1
        # Comment removed due to garbled encoding.
        if tc_loop_count == 1 or tc_loop_count % 10 == 0:
            dm_logger.info(f"Transitive Closure Iteyration {tc_loop_count}: current pairs count = {len(sc1_pairs_set)}")
            
        added = False
        # Comment removed due to garbled encoding.
        adj = {}
        for u, v in sc1_pairs_set:
            if u not in adj: adj[u] = []
            adj[u].append(v)
            
        new_edges = set()
        for u in adj:
            for v in adj[u]:
                if v in adj:
                    for w in adj[v]:
                        if u == w: continue # Comment removed due to garbled encoding.
                        if (u, w) not in sc1_pairs_set and (u, w) not in new_edges:
                            new_edges.add((u, w))
                            added = True
        
        if new_edges:
            sc1_pairs_set.update(new_edges)
            
    sc1_pairs = list(sc1_pairs_set)
    
    def _is_pair_in_vertices(u: Vertex, v: Vertex) -> bool:
        if u.pos_equal(Pos.VERB) or u.pos_equal(Pos.AUX):
            return False
        if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
            return False
        return True
    
    sc1_pairs = list(filter(lambda pairs: _is_pair_in_vertices(pairs[0], pairs[1]), sc1_pairs))
    
    # logger.info(f"SC1 path pairs after closure and filter: {[(u.text(), v.text()) for u, v in sc1_pairs]}")

    sc1_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    for u, v in sc1_pairs:
        s, cnt = sc1.get_paths_between_vertices(u, v)
        if cnt == 0:
            continue
        sc1_paths[(u, v)] = (s, cnt)
    
    # logger.info(f"SC1 valid paths count: {len(sc1_paths)}")

    likely_nodes = _get_matched_vertices(sc1_vertices, sc2_vertices)
    
    # Comment removed due to garbled encoding.

    sc2_pairs: list[tuple[Vertex, Vertex]] = []
    sc2_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    
    # Comment removed due to garbled encoding.
    dm_logger.info(f"Start Core Matching Logic: {len(sc1_pairs)} sc1 pairs")
    processed_count = 0
    for u, u_prime in sc1_pairs:
        processed_count += 1
        if processed_count % 10 == 0:
             dm_logger.debug(f"Processing sc1 pair {processed_count}/{len(sc1_pairs)}")
        for v, v_prime in itertools.product(likely_nodes.get(u, set()), likely_nodes.get(u_prime, set())):
            if v == v_prime:
                continue
            s1, cnt1 = sc1_paths[(u, u_prime)]
            dm_logger.debug(f"    Calling sc2.get_paths_between_vertices('{v.text()}', '{v_prime.text()}')")
            s2, cnt2 = sc2.get_paths_between_vertices(v, v_prime)
            dm_logger.debug(f"    Forward path: count={cnt2}, sample='{s2[:50]}...'")

            dm_logger.debug(f"    Calling sc2.get_paths_between_vertices('{v_prime.text()}', '{v.text()}')")
            s2_inv, cnt2_prime = sc2.get_paths_between_vertices(v_prime, v)
            dm_logger.debug(f"    Backward path: count={cnt2_prime}, sample='{s2_inv[:50]}...'")
            
            # Comment removed due to garbled encoding.
            if cnt2 == 0 or s2 == "":
                if cnt2_prime > 0 and s2_inv:
                    sc2_pairs.append((v_prime, v))
                    sc2_paths[(v_prime, v)] = (s2_inv, cnt2_prime)
                continue
            elif cnt2_prime == 0 or s2_inv == "":
                sc2_pairs.append((v, v_prime))
                sc2_paths[(v, v_prime)] = (s2, cnt2)
                continue
            
            # Comment removed due to garbled encoding.
            if not s2 or not s2_inv:
                dm_logger.debug(f"String removed due to garbled encoding.")
                continue
            
            if _better_path(s1, s2, s2_inv):
                sc2_pairs.append((v, v_prime))
                sc2_paths[(v, v_prime)] = (s2, cnt2)
            else:
                sc2_pairs.append((v_prime, v))
                sc2_paths[(v_prime, v)] = (s2_inv, cnt2)

    dm_logger.debug(f"SC2 inferred path pairs: {[(u.text(), v.text()) for u, v in sc2_pairs]}")
    dm_logger.debug(f"SC2 paths count: {len(sc2_paths)}")

    # Comment removed due to garbled encoding.
    match_scores: dict[tuple[Vertex, Vertex], float] = {}
    
    for u, v in itertools.product(sc1_vertices, sc2_vertices):
        if _legal_vertices(u, v):
            matches.append((u, v))
    
    dm_logger.debug(f"Initial legal matches count: {len(matches)}")

    in_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc1_pairs:
        if v not in in_paths_of_sc1:
            in_paths_of_sc1[v] = []
        in_paths_of_sc1[v].append(sc1_paths[(u, v)])
        if u not in out_paths_of_sc1:
            out_paths_of_sc1[u] = []
        out_paths_of_sc1[u].append(sc1_paths[(u, v)])
    
    for vertex in sc1_vertices:
        if vertex in in_paths_of_sc1:
            dm_logger.debug(f"SC1 Vertex '{vertex.text()}' In Paths: {[s for s, _ in in_paths_of_sc1[vertex]]}")
        if vertex in out_paths_of_sc1:
            dm_logger.debug(f"SC1 Vertex '{vertex.text()}' Out Paths: {[s for s, _ in out_paths_of_sc1[vertex]]}")
    
    
    in_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc2_pairs:
        if v not in in_paths_of_sc2:
            in_paths_of_sc2[v] = []
        in_paths_of_sc2[v].append(sc2_paths[(u, v)])
        if u not in out_paths_of_sc2:
            out_paths_of_sc2[u] = []
        out_paths_of_sc2[u].append(sc2_paths[(u, v)])
    
    for vertex in sc2_vertices:
        if vertex in in_paths_of_sc2:
            dm_logger.debug(f"SC2 Vertex '{vertex.text()}' In Paths: {[s for s, _ in in_paths_of_sc2[vertex]]}")
        if vertex in out_paths_of_sc2:
            dm_logger.debug(f"SC2 Vertex '{vertex.text()}' Out Paths: {[s for s, _ in out_paths_of_sc2[vertex]]}")
    
    root_path_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    for e in sc1.hyperedges:
        root = e.root
        root_node = e.current_node(root)
        if not (root_node.pos == Pos.VERB or root_node.pos == Pos.AUX):
            continue
        for v in e.vertices[1:]:
            v_node = e.current_node(v)
            if v_node.pos == Pos.VERB or v_node.pos == Pos.AUX:
                continue
            text = _formal_text_of(root_node, v_node)
            if v not in root_path_of_sc1:
                root_path_of_sc1[v] = []
            root_path_of_sc1[v].append((text, 2))
    root_path_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    for e in sc2.hyperedges:
        root = e.root
        root_node = e.current_node(root)
        if not (root_node.pos == Pos.VERB or root_node.pos == Pos.AUX):
            continue
        for v in e.vertices[1:]:
            v_node = e.current_node(v)
            if v_node.pos == Pos.VERB or v_node.pos == Pos.AUX:
                continue
            text = _formal_text_of(root_node, v_node)
            if v not in root_path_of_sc2:
                root_path_of_sc2[v] = []
            root_path_of_sc2[v].append((text, 2))

    # logger.info(f"SC1 root paths count: {sum(len(ps) for ps in root_path_of_sc1.values())}")
    # logger.info(f"SC2 root paths count: {sum(len(ps) for ps in root_path_of_sc2.values())}")

    
    path_score_cache: dict[tuple[str, str], float] = {}
    path_pair_need_to_calc: set[tuple[str, str]] = set()
    for u, v in matches:
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))
        for s1, cnt1 in root_path_of_sc1.get(u, []):
            for s2, cnt2 in root_path_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))

    # logger.info(f"Path similarity pairs to compute: {len(path_pair_need_to_calc)}")
    if path_pair_need_to_calc:
        dm_logger.info(f"Computing path similarities for {len(path_pair_need_to_calc)} pairs...")
    
    path_list_1: list[str] = []
    path_list_2: list[str] = []
    path_pair_need_to_calc_list = list(path_pair_need_to_calc)
    for s1, s2 in path_pair_need_to_calc_list:
        path_list_1.append(s1)
        path_list_2.append(s2)
    similarities = get_similarity_batch(path_list_1, path_list_2)
    for i, (s1, s2) in enumerate(path_pair_need_to_calc_list):
        path_score_cache[(s1, s2)] = similarities[i]
    
    # logger.info("Path similarity cache populated.")

    for u, v in matches:
        in_score = 0.0
        in_cnt = 0
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                in_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                in_cnt += 1
        if in_cnt > 0:
            in_score /= in_cnt

        out_score = 0.0
        out_cnt = 0
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                out_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                out_cnt += 1
        if out_cnt > 0:
            out_score /= out_cnt
            
        root_score = 0.0
        root_cnt = 0
        for s1, cnt1 in root_path_of_sc1.get(u, []):
            for s2, cnt2 in root_path_of_sc2.get(v, []):
                root_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                root_cnt += 1
        if root_cnt > 0:
            root_score /= root_cnt
        
        match_scores[(u, v)] = in_score + out_score + root_score
        
        # Comment removed due to garbled encoding.
        
    # filter by score_threshold
    matches = list(filter(lambda pair: match_scores.get(pair, 0.0) >= score_threshold, matches))
    # Comment removed due to garbled encoding.
    
    # delete the matches that if (u, v1) and (u, v2) in matches and v1 != v2, keep only the one with highest score
    final_matches: list[tuple[Vertex, Vertex, float]] = []
    matches_by_u: dict[Vertex, list[tuple[Vertex, float]]]  = {}
    for u, v in matches:
        score = match_scores.get((u, v), 0.0)
        if u not in matches_by_u:
            matches_by_u[u] = []
        matches_by_u[u].append((v, score))
    for u, v_scores in matches_by_u.items():
        v_scores = sorted(v_scores, key=lambda x: x[1], reverse=True)
        best_v, best_score = v_scores[0]
        final_matches.append((u, best_v, best_score))
        # if len(v_scores) > 1:
            # logger.info(f"Disambiguation for '{u.text()}': kept '{best_v.text()}' (score={best_score:.3f}), others: {[v.text() for v, s in v_scores[1:]]}")

    # Comment removed due to garbled encoding.
    if final_matches:
        dm_logger.info("String removed due to garbled encoding.")
        for i, (u, v, score) in enumerate(final_matches, 1):
            dm_logger.info(
                f"  [{i}] Q{u.id}: '{u.text()}' "
                f"String removed due to garbled encoding."
                f"(score={score:.4f})"
            )
    else:
        dm_logger.info("String removed due to garbled encoding.")

    if force_include:
        u, v = force_include
        if (u, v, 1.0) not in final_matches:
            final_matches.insert(0, (u, v, 1.0))  # Comment removed due to garbled encoding.

    return final_matches


# Comment removed due to garbled encoding.
# Comment removed due to garbled encoding.
# Comment removed due to garbled encoding.
# Comment removed due to garbled encoding.

# Comment removed due to garbled encoding.

