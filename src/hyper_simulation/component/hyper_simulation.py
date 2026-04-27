import time
from typing import Dict, List, Set, Tuple
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node, Delta, DMatch
from hyper_simulation.component.semantic_cluster import calc_semantic_cluster_pairs, get_d_match
from hyper_simulation.component.d_match import calc_d_match, calc_d_match_batch
from hyper_simulation.hypergraph.linguistic import Pos
from hyper_simulation.component.denial import get_matched_vertices, compute_allowed_pairs, compute_allowed_pairs_batch, compute_allowed_pairs_batch_with_score, get_top_k_matched_vertices_by_scores
import warnings
from tqdm import tqdm
from hyper_simulation.utils.log import getLogger
import logging
def convert_local_to_sim(
    local_hg: LocalHypergraph,
) -> Tuple[SimHypergraph, Dict[int, str], Dict[int, Vertex], Dict[int, List[SimHyperedge]], Dict[Vertex, int]]:
    """Convert LocalHypergraph to SimHypergraph and return Sim ID mapping."""
    sim_hg = SimHypergraph()
    vertex_id_map: Dict[int, int] = {}
    node_text: Dict[int, str] = {}
    sim_id_to_vertex: Dict[int, Vertex] = {}
    node_to_edges: Dict[int, List[SimHyperedge]] = {}
    vertex_to_sim_id: Dict[Vertex, int] = {}
    
    for idx, vertex in enumerate(sorted(local_hg.vertices, key=lambda v: v.id)):
        sim_hg.add_node(vertex.text())
        vertex_id_map[vertex.id] = idx
        node_text[idx] = vertex.text()
        sim_id_to_vertex[idx] = vertex
        vertex_to_sim_id[vertex] = idx
    
    edge_id = 0
    for local_edge in local_hg.hyperedges:
        node_ids = {vertex_id_map[v.id] for v in local_edge.vertices if v.id in vertex_id_map}
        if not node_ids:
            continue
        sim_edge = SimHyperedge(node_ids, local_edge.desc, edge_id)
        sim_hg.add_hyperedge(sim_edge)
        for nid in node_ids:
            node_to_edges.setdefault(nid, []).append(sim_edge)
        edge_id += 1
    
    return sim_hg, node_text, sim_id_to_vertex, node_to_edges, vertex_to_sim_id

def build_delta_and_dmatch(
    query: SimHypergraph,
    data: SimHypergraph,
    query_texts: Dict[int, str],
    data_texts: Dict[int, str],
    query_node_edges: Dict[int, List[SimHyperedge]],
    data_node_edges: Dict[int, List[SimHyperedge]],
    allowed_pairs: Set[Tuple[int, int]],
    query_local_hg: LocalHypergraph,
    data_local_hg: LocalHypergraph,
    vertex_to_sim_id_q: Dict[Vertex, int],
    vertex_to_sim_id_d: Dict[Vertex, int],
    matched_vertices: dict[Vertex, set[Tuple[Vertex, float]]],
    cluster_sim_threshold: float = 0.75,
    dmatch_threshold: float = 0.3,
    branch_threshold: int = 5,
    is_multihop: bool = False,
) -> Tuple[Delta, DMatch]:
    """
    Construct Delta and D-Match, ensuring 100% coverage of allowed_pairs.
    Key design:
      1. Multi-node clusters: Structural matching (downgrade to empty match on exception).
      2. Single-node clusters: Unconditional fallback (bypass LocalVertex mapping, use Sim ID directly).
      3. D-Match completeness: Every Delta item must have a D-Match item (empty set is also valid).
    """
    
    delta_start = time.time()
    sc_logger = getLogger("semantic_cluster") 
    sc_logger.debug(f"\t\tcalc the delta")
    
    delta = Delta()
    d_delta_matches: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    
    # Step 1: Multi-node semantic clusters (batch structural matching)
    cluster_count = 0
    # === Phase 1: Record original results (from get_semantic_cluster_pairs) ===
    raw_pairs = calc_semantic_cluster_pairs(
        query_local_hg, data_local_hg, matched_vertices, 
        cluster_sim_threshold, branch_threshold, is_multihop, logger=sc_logger
        )
    
    # for i, (sc_q, sc_d, sim_score) in enumerate(raw_pairs):
    #     print(f"SC PAIR [{i}]:\n- Q: {sc_q.text()}\n- D: {sc_d.text()}\n- Score: {sim_score:.3f}\n")
    
    time1 = time.time()
    # print(f"Semantic cluster pair calculation time: {time1 - delta_start:.2f} seconds")
    sc_logger.info(f"Semantic cluster generation complete. Total {len(raw_pairs)} original cluster pairs.")
    # === Phase 2: First iteration - Filter and collect candidate cluster pairs ===
    candidate_cluster_pairs = []  # list of (sc_q, sc_d, sim_score, metadata_dict)
    for sc_q, sc_d, sim_score in raw_pairs:
        # --- Extract structural information ---
        q_vertices = sc_q.get_vertices()
        d_vertices = sc_d.get_vertices()
        q_edges = sc_q.hyperedges
        d_edges = sc_d.hyperedges

        q_triples = sc_q.to_triple() or []
        d_triples = sc_d.to_triple() or []

        # Take the first triple as representative (if exists)
        q_triple_repr = str(q_triples[0]) if q_triples else "(no triple)"
        d_triple_repr = str(d_triples[0]) if d_triples else "(no triple)"

        q_text = sc_q.text()
        d_text = sc_d.text()

        # --- Log: Original cluster details (whether adopted or not) ---
        sc_logger.info(
            f"-> Original cluster pair | score={sim_score:.3f}\n"
            f"  Q: text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)}, edges={len(q_edges)}\n"
            f"  D: text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)}, edges={len(d_edges)}"
        )

        # --- Filtering logic ---
        if sim_score < 0.5:
            sc_logger.info(f"  -> Skip: Low similarity ({sim_score:.3f})")
            continue

        q_vs = [v for v in q_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        d_vs = [v for v in d_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        if not q_vs or not d_vs:
            sc_logger.info(f"  -> Skip: No noun nodes (Q:{len(q_vs)}/{len(q_vertices)}, D:{len(d_vs)}/{len(d_vertices)})")
            continue

        q_rep = min(q_vs, key=lambda v: v.id)
        d_rep = min(d_vs, key=lambda v: v.id)
        q_nid = vertex_to_sim_id_q.get(q_rep)
        d_nid = vertex_to_sim_id_d.get(d_rep)
        if q_nid is None or d_nid is None:
            sc_logger.info(f"  -> Skip: Missing mapping (Q{q_rep.id}->{q_nid}, D{d_rep.id}->{d_nid})")
            continue

        q_es = list({e for v in q_vs if v in vertex_to_sim_id_q for e in query_node_edges.get(vertex_to_sim_id_q[v], []) if e})
        d_es = list({e for v in d_vs if v in vertex_to_sim_id_d for e in data_node_edges.get(vertex_to_sim_id_d[v], []) if e})

        sc_id = delta.add_sematic_cluster_pair(
            Node(q_nid, q_text),
            Node(d_nid, d_text),
            q_es,
            d_es
        )

        # Store metadata for batch processing
        candidate_cluster_pairs.append({
            'sc_q': sc_q,
            'sc_d': sc_d,
            'sc_id': sc_id,
            'q_rep': q_rep,
            'd_rep': d_rep,
            'q_text': q_text,
            'd_text': d_text,
            'q_triple_repr': q_triple_repr,
            'd_triple_repr': d_triple_repr,
            'q_vertices': q_vertices,
            'd_vertices': d_vertices,
            'q_vs': q_vs,
            'd_vs': d_vs,
            'q_edges': q_edges,
            'd_edges': d_edges,
            'sim_score': sim_score,
        })

    # === Phase 3: Batch calculate D-Match ===
    if candidate_cluster_pairs:
        sc_pairs = [(md['sc_q'], md['sc_d']) for md in candidate_cluster_pairs]
        try:
            batch_results = calc_d_match_batch(sc_pairs, dmatch_threshold)
        except (AssertionError, AttributeError, IndexError) as e:
            sc_logger.warning(f"  -> Batch match exception: {type(e).__name__}, downgrading to empty match")
            batch_results = [[] for _ in sc_pairs]
    else:
        batch_results = []

    # === Phase 4: Process batch results and log ===
    for batch_idx, meta in enumerate(candidate_cluster_pairs):
        cluster_count += 1
        sc_id = meta['sc_id']
        q_rep = meta['q_rep']
        d_rep = meta['d_rep']
        q_text = meta['q_text']
        d_text = meta['d_text']
        q_triple_repr = meta['q_triple_repr']
        d_triple_repr = meta['d_triple_repr']
        q_vertices = meta['q_vertices']
        d_vertices = meta['d_vertices']
        q_vs = meta['q_vs']
        d_vs = meta['d_vs']
        q_edges = meta['q_edges']
        d_edges = meta['d_edges']
        sim_score = meta['sim_score']

        # Extract matches for the current cluster pair from batch results
        if batch_idx < len(batch_results):
            matches = {
                (vertex_to_sim_id_q[vq], vertex_to_sim_id_d[vd])
                for vq, vd, _ in batch_results[batch_idx]
                if vq in vertex_to_sim_id_q and vd in vertex_to_sim_id_d
            }
        else:
            matches = set()

        d_delta_matches[(sc_id, sc_id)] = matches

        # --- Log: Adopted cluster (with complete structure) ---
        sc_logger.info(
            f"-> Adopted #{cluster_count} | score={sim_score:.3f}\n"
            f"  Q_rep=Q{q_rep.id}('{q_rep.text()}')\n"
            f"     full_text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)} (noun={len(q_vs)}), edges={len(q_edges)}\n"
            f"  D_rep=D{d_rep.id}('{d_rep.text()}')\n"
            f"     full_text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)} (noun={len(d_vs)}), edges={len(d_edges)}\n"
            f"  D-Match count: {len(matches)}"
        )

    sc_logger.info(f"Semantic cluster construction complete. Original {len(raw_pairs)} -> Valid {cluster_count} cluster pairs.")   
    
    # Step 2: Create single-node cluster for each node pair in allowed_pairs
    # for q_id, d_id in allowed_pairs:
    #     sc_id = delta.add_sematic_cluster_pair(
    #         Node(q_id, query_texts.get(q_id, "")),
    #         Node(d_id, data_texts.get(d_id, "")),
    #         query_node_edges.get(q_id, []),
    #         data_node_edges.get(d_id, [])
    #     )
    #     d_delta_matches[(sc_id, sc_id)] = {(q_id, d_id)}  # Single-node cluster must have self-match
    
    time2 = time.time()
    # print(f"D-Match compute time: {time2 - time1:.2f} seconds")
    
    return delta, DMatch.from_dict(d_delta_matches)


def compute_hyper_simulation(
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph,
    sigma_threshold: float = 0.75,
    b_threshold: int = 5,
    delta_threshold: float = 0.7,
) -> Tuple[Dict[int, Set[int]], Dict[int, Vertex], Dict[int, Vertex]]:
    """
    Execute hypergraph simulation.
    Theoretical guarantee: type_same(u,v)=True -> semantic cluster covers (u,v) (via unconditional fallback).
    """
    sim_logger = getLogger("hyper_simulation")
    sim_logger.debug(f"\tStart Hyper Simulation")
    
    # Convert to SimHypergraph space (get continuous Node ID)
    q_sim, q_texts, q_vertices, q_edges, q_vid_map = convert_local_to_sim(query_hg)
    d_sim, d_texts, d_vertices, d_edges, d_vid_map = convert_local_to_sim(data_hg)
    
    denial_start = time.time()
    sim_logger.debug(f"\tstart denial comment calc")
    # Calculate loose semantic allowability
    dc_logger = getLogger("denial_comment")
    # allowed = compute_allowed_pairs(q_vertices, d_vertices)
    time1 = time.time()
    allowed, confidence_scores = compute_allowed_pairs_batch_with_score(q_vertices, d_vertices)
    time2 = time.time()
    # print(f"DC computation time: {time2 - time1:.2f} seconds")
    q_vertices_list = list(q_vertices.values())
    d_vertices_list = list(d_vertices.values())
    time3 = time.time()
    # calc the match_vertices based on the confidence_scores and q_vertices and d_vertices
    match_vertices = get_top_k_matched_vertices_by_scores(q_vertices, d_vertices, confidence_scores, k=b_threshold)
    # match_vertices = get_matched_vertices(q_vertices_list, d_vertices_list)
    # Define type_same_fn (based on Sim ID space)
    def type_same_fn(x_id: int, y_id: int) -> bool:
        return (x_id, y_id) in allowed
    
    q_sim.set_type_same_fn(type_same_fn)
    d_sim.set_type_same_fn(type_same_fn)
    
    denial_end = time.time()
    dc_logger.info(f"\tdenial comment cost {denial_end - denial_start}s")
    
    sim_logger.debug(f"\tdenial comment cost {denial_end - denial_start}s")
    sim_logger.debug(f"\tstart build delta and d-match")
    
    # Construct Delta/D-Match: 100% coverage guarantee + exception isolation
    delta, d_match = build_delta_and_dmatch(
        q_sim, d_sim, q_texts, d_texts, q_edges, d_edges, allowed,
        query_local_hg=query_hg,
        data_local_hg=data_hg,
        vertex_to_sim_id_q=q_vid_map,
        vertex_to_sim_id_d=d_vid_map,
        matched_vertices=match_vertices,
        dmatch_threshold=delta_threshold,
        cluster_sim_threshold=sigma_threshold,
        branch_threshold=b_threshold
    )
    # time4 = time.time()
    # print(f"Delta and D-Match time: {time4 - time3:.2f} seconds")

    
    # Execute hypergraph simulation
    start_time = time.time()
    sim_logger.info("\tExecuting hypergraph simulation...")
    simulation = SimHypergraph.get_hyper_simulation(q_sim, d_sim, delta, d_match)
    # simulation = SimHypergraph.get_hyper_simulation_strict(q_sim, d_sim, delta, d_match)
    # === New: Structured output of simulation results (INFO level) ===
    sim_logger.info("\t=== Hyper Simulation Mapping ===")
    for q_id, d_ids in sorted(simulation.items()):
        # Query side text
        q_text = q_vertices[q_id].text() if q_id in q_vertices else f"[Q{q_id}]"
        
        # Data side: ID + Text
        if d_ids:
            d_items = []
            for d_id in sorted(d_ids):
                if d_id in d_vertices:
                    d_text = d_vertices[d_id].text()
                    d_items.append(f"D{d_id}: '{d_text}'")
                else:
                    d_items.append(f"D{d_id}")
            targets = ", ".join(d_items)
        else:
            targets = "-"

        sim_logger.info(f"\t  Q{q_id}: '{q_text}' -> {targets}")
    sim_logger.info("\t================================")
    end_time = time.time()
    sim_logger.info(f"\tSimulation complete: {len(simulation)} mappings.")
    sim_logger.info(f"\thyper simulation main cost {end_time - start_time}s")

    return simulation, q_vertices, d_vertices

# Apple / Banana
# Special characters may exist in the middle.

# 1. Cannot handle non-standard characters.
# 2. May mistakenly merge a token with the punctuation following it, impact: program crash or severely affect dependency parsing results.


