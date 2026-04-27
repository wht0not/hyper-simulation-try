

import itertools
from platform import node
import time
from typing import Dict, List, Set, Tuple, Optional
from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc
from hyper_simulation.hypergraph.linguistic import QueryType, Pos, Tag, Dep, Entity
import numpy as np
import logging
from hyper_simulation.component.embedding import get_embedding_batch, get_cosine_similarity_batch, get_similarity_batch, get_similarity
from hyper_simulation.component.nli import get_nli_label, get_nli_labels_batch, get_nli_remix_score_batch
from hyper_simulation.utils.log import getLogger
from hyper_simulation.component.denial import get_top_k_matched_vertices

from hyper_simulation.hypergraph.path import find_shortest_hyperpaths, find_shortest_hyperpaths_local

from hyper_simulation.component.semantic_cluster import SemanticCluster
from hyper_simulation.hypergraph.entity import ENT

# QueryType.PERSON -> ENT.PERSON
# QueryType.TIME -> ENT.TEMPORAL
# QueryType.LOCATION -> ENT.GPE/LOC/FAC/ORG
# QueryType.NUMBER -> ENT.NUMBER
# QueryType.BELONGS, QueryType.WHAT, QueryType.WHICH, QueryType.REASON: False
# QueryType.ATTRIBUTE -> POS.ADJ/ADV
def query_same_type(v1: Vertex, v2: Vertex) -> bool:
    if v1.query_type():
        return False
    
    
    qt = v1.query_type()
    v2_type = v2.type()
    if qt == QueryType.PERSON and v2_type:
        return v2_type == ENT.PERSON
    elif qt == QueryType.TIME and v2_type:
        return v2_type == ENT.TEMPORAL
    elif qt == QueryType.LOCATION and v2_type:
        return v2_type in {ENT.GPE, ENT.LOC, ENT.FAC, ENT.ORG}
    elif qt == QueryType.NUMBER and v2_type:
        return v2_type == ENT.NUMBER
    elif qt == QueryType.ATTRIBUTE:
        return v2.pos_equal(Pos.ADJ) or v2.pos_equal(Pos.ADV)
    
    return False

def _construct_description_from_path(path: list[list[Node]], start_node: Node, end_node: Node) -> str:
    """ Docstring removed due to garbled encoding. """
    
    type_nodes_map: dict[str, set[Node]] = {
        "LOCATION": set(),
        "TEMPORAL": set(),
        "ATTRIBUTE": set(),
        "PERSON": set(),
        "COMPONENTS": set(),
        "REASON": set(),
        "CONCEPT": set(),
        "NUMBER": set(),
        "ORGANISM": set(),
        "FOOD": set(),
        "MEDICAL": set(),
        "ANATOMY": set(),
        "SUBSTANCE": set(),
        "ASTRO": set(),
        "AWARD": set(),
        "VEHICLE": set(),
        "COUNTRY": set(),
        "ORGANIZATION": set(),
        "FACILITY": set(),
        "Geopolitical": set(),
        "NORP": set(),
        "PRODUCT": set(),
        "WORK_OF_ART": set(),
        "LAW": set(),
        "LANGUAGE": set(),
        "OCCUPATION": set(),
        "EVENT": set(),
        "THEORY": set(),
        "GROUP": set(),
        "FEATURE": set(),
        "ECONOMIC": set(),
        "SOCIOLOGY": set(),
        "PHENOMENON": set(),
        # "ADJECTIVE": set(),
        # "ADVERB": set(),
    }
    
    node_type_map: dict[Node, str] = {}
    
    start_node_type = start_node.type_str()
    if start_node_type and start_node_type in type_nodes_map:
        type_nodes_map[start_node_type].add(start_node)
        index = len(type_nodes_map[start_node_type])
        node_type_map[start_node] = f"{start_node_type}#{index}"
        
    end_node_type = end_node.type_str()
    if end_node_type and end_node_type in type_nodes_map:
        type_nodes_map[end_node_type].add(end_node)
        index = len(type_nodes_map[end_node_type])
        node_type_map[end_node] = f"{end_node_type}#{index}"
    
    description_parts = []
    for nodes in path:
        if not nodes:
            continue
        
        node_by_index = sorted(nodes, key=lambda n: n.index)
        def node_text(n: Node) -> str:
            if n in node_type_map:
                return node_type_map[n]
            node_type = n.type_str()
            if node_type and node_type in type_nodes_map:
                type_nodes_map[node_type].add(n)
                index = len(type_nodes_map[node_type])
                node_type_map[n] = f"{node_type}#{index}"
                return node_type_map[n]
            return n.text
        node_texts = [node_text(node) for node in node_by_index]
        description_parts.append(" ".join(node_texts))
        
    return ". ".join(description_parts)

def calc_d_match(sc1: SemanticCluster, sc2: SemanticCluster, threshold: float = 0.5) -> list[tuple[Vertex, Vertex, float]]:
    """ Docstring removed due to garbled encoding. """
    # Firstly, set a relation R = { (v1, v2) | v1.type() == v2.type() }
    R: list[tuple[Vertex, Vertex]] = []
    for v1 in sc1.vertices:
        for v2 in sc2.vertices:
            if v1.is_query():
                if query_same_type(v1, v2):
                    R.append((v1, v2))
                continue
            if v1.is_verb() or v2.is_verb():
                continue
            if v1.type() == v2.type():
                R.append((v1, v2))

    # R_map = Dict[Vertex, List[Vertex]]
    R_map: Dict[Vertex, List[Vertex]] = {}
    for v1, v2 in R:
        if v1 not in R_map:
            R_map[v1] = []
        R_map[v1].append(v2)

    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    score_items: list[tuple[str, str, Vertex, Vertex, int]] = []

    for v1, v2 in R:
        root_tuples: list[tuple[Vertex, Vertex]] = []
        other_tuples: list[tuple[Vertex, Vertex]] = []

        # find the hyperedges that v1 belongs to in sc1
        for hyperedge in sc1.hyperedges:
            if v1 not in hyperedge.vertices:
                continue

            if v1 == hyperedge.root:
                for v1_prime in hyperedge.vertices[1:]:
                    root_tuples.append((v1, v1_prime))
                continue

            for v1_prime in hyperedge.vertices[1:]:
                if v1_prime == v1:
                    continue
                other_tuples.append((v1, v1_prime))

        candidate_pairs: list[tuple[Vertex, Vertex]] = []
        for _, v1_prime in root_tuples:
            for v2_prime in R_map.get(v1_prime, []):
                candidate_pairs.append((v1_prime, v2_prime))
        for _, v1_prime in other_tuples:
            for v2_prime in R_map.get(v1_prime, []):
                candidate_pairs.append((v1_prime, v2_prime))

        # Comment removed due to garbled encoding.
        for index, (v1_prime, v2_prime) in enumerate(candidate_pairs):
            path1, v1_node, v1_prime_node = sc1.get_path_node_steps(v1, v1_prime)
            path2, v2_node, v2_prime_node = sc2.get_path_node_steps(v2, v2_prime)

            if not path1 or not path2 or v1_node is None or v2_node is None or v1_prime_node is None or v2_prime_node is None:
                continue

            desc1 = _construct_description_from_path(path1, start_node=v1_node, end_node=v1_prime_node)
            desc2 = _construct_description_from_path(path2, start_node=v2_node, end_node=v2_prime_node)
            score_items.append((desc1, desc2, v1, v2, index))

    if not score_items:
        return []

    # Comment removed due to garbled encoding.
    score_pairs = [(desc1, desc2) for desc1, desc2, _, _, _ in score_items]
    scores = get_nli_remix_score_batch(score_pairs)

    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    # Comment removed due to garbled encoding.
    pair_index_max: dict[tuple[Vertex, Vertex], dict[int, float]] = {}
    for (_, _, v1, v2, index), score in zip(score_items, scores):
        key = (v1, v2)
        if key not in pair_index_max:
            pair_index_max[key] = {}
        prev = pair_index_max[key].get(index)
        if prev is None or score > prev:
            pair_index_max[key][index] = score

    raw_results: list[tuple[Vertex, Vertex, float]] = []
    for (v1, v2), index_max_map in pair_index_max.items():
        if not index_max_map:
            continue
        avg_score = sum(index_max_map.values()) / len(index_max_map)
        if avg_score > threshold:
            raw_results.append((v1, v2, avg_score))

    # Comment removed due to garbled encoding.
    raw_results.sort(key=lambda x: x[2], reverse=True)
    used_v1: set[Vertex] = set()
    used_v2: set[Vertex] = set()
    results: list[tuple[Vertex, Vertex, float]] = []
    for v1, v2, score in raw_results:
        if v1 in used_v1 or v2 in used_v2:
            continue
        used_v1.add(v1)
        used_v2.add(v2)
        results.append((v1, v2, score))

    return results

def calc_d_match_batch(sc_pairs: list[tuple[SemanticCluster, SemanticCluster]], threshold: float = 0.5) -> list[list[tuple[Vertex, Vertex, float]]]:
    
    start_time = time.time()
    if not sc_pairs:
        return []

    # Comment removed due to garbled encoding.
    score_pairs: list[tuple[str, str]] = []
    score_meta: list[tuple[int, Vertex, Vertex, int]] = []

    # Comment removed due to garbled encoding.
    pair_index_max_by_pair: list[dict[tuple[Vertex, Vertex], dict[int, float]]] = [
        {} for _ in range(len(sc_pairs))
    ]

    for pair_idx, (sc1, sc2) in enumerate(sc_pairs):
        # Comment removed due to garbled encoding.
        R: list[tuple[Vertex, Vertex]] = []
        for v1 in sc1.vertices:
            for v2 in sc2.vertices:
                if v1.is_query():
                    if query_same_type(v1, v2):
                        R.append((v1, v2))
                    continue
                if v1.is_verb() or v2.is_verb():
                    continue
                if v1.type() == v2.type():
                    R.append((v1, v2))

        R_map: Dict[Vertex, List[Vertex]] = {}
        for v1, v2 in R:
            if v1 not in R_map:
                R_map[v1] = []
            R_map[v1].append(v2)

        for v1, v2 in R:
            root_tuples: list[tuple[Vertex, Vertex]] = []
            other_tuples: list[tuple[Vertex, Vertex]] = []

            for hyperedge in sc1.hyperedges:
                if v1 not in hyperedge.vertices:
                    continue

                if v1 == hyperedge.root:
                    for v1_prime in hyperedge.vertices[1:]:
                        root_tuples.append((v1, v1_prime))
                    continue

                for v1_prime in hyperedge.vertices[1:]:
                    if v1_prime == v1:
                        continue
                    other_tuples.append((v1, v1_prime))

            candidate_pairs: list[tuple[Vertex, Vertex]] = []
            for _, v1_prime in root_tuples:
                for v2_prime in R_map.get(v1_prime, []):
                    candidate_pairs.append((v1_prime, v2_prime))
            for _, v1_prime in other_tuples:
                for v2_prime in R_map.get(v1_prime, []):
                    candidate_pairs.append((v1_prime, v2_prime))

            for index, (v1_prime, v2_prime) in enumerate(candidate_pairs):
                path1, v1_node, v1_prime_node = sc1.get_path_node_steps(v1, v1_prime)
                path2, v2_node, v2_prime_node = sc2.get_path_node_steps(v2, v2_prime)

                if not path1 or not path2 or v1_node is None or v2_node is None or v1_prime_node is None or v2_prime_node is None:
                    continue

                desc1 = _construct_description_from_path(path1, v1_node, v1_prime_node)
                desc2 = _construct_description_from_path(path2, v2_node, v2_prime_node)
                # print(f"D-match sentence: '{desc1}' <-> '{desc2}' for vertices '{v1.text()}' and '{v2.text()}' with index {index}")
                score_pairs.append((desc1, desc2))
                score_meta.append((pair_idx, v1, v2, index))

    if not score_pairs:
        return [[] for _ in sc_pairs]

    # Comment removed due to garbled encoding.
    
    # print(f"Calculating D-Match for {len(sc_pairs)} pairs with {len(score_pairs)} scoring items...")
    time1 = time.time()
    # print(f"Scoring batch prepared in {time1 - start_time:.2f} seconds, now calling NLI batch API...")
    scores = get_nli_remix_score_batch(score_pairs)
    time2 = time.time()
    # print(f"NLI scoring completed in {time2 - time1:.2f} seconds")
    # Comment removed due to garbled encoding.
    for (pair_idx, v1, v2, index), score in zip(score_meta, scores):
        pair_map = pair_index_max_by_pair[pair_idx]
        key = (v1, v2)
        if key not in pair_map:
            pair_map[key] = {}
        prev = pair_map[key].get(index)
        if prev is None or score > prev:
            pair_map[key][index] = score

    # Comment removed due to garbled encoding.
    all_results: list[list[tuple[Vertex, Vertex, float]]] = []
    for pair_map in pair_index_max_by_pair:
        raw_results: list[tuple[Vertex, Vertex, float]] = []
        for (v1, v2), index_max_map in pair_map.items():
            if not index_max_map:
                continue
            avg_score = sum(index_max_map.values()) / len(index_max_map)
            if avg_score > threshold:
                raw_results.append((v1, v2, avg_score))

        raw_results.sort(key=lambda x: x[2], reverse=True)
        used_v1: set[Vertex] = set()
        used_v2: set[Vertex] = set()
        results: list[tuple[Vertex, Vertex, float]] = []
        for v1, v2, score in raw_results:
            if v1 in used_v1 or v2 in used_v2:
                continue
            used_v1.add(v1)
            used_v2.add(v2)
            results.append((v1, v2, score))

        all_results.append(results)
    return all_results
