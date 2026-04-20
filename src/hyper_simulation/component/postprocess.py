import time
import os
from typing import Dict, List, Set, Tuple
from itertools import product
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex, Hyperedge
from hyper_simulation.hypergraph.dependency import Node
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node as SimNode, Delta, DMatch
from hyper_simulation.component.semantic_cluster import SemanticCluster
from hyper_simulation.hypergraph.linguistic import Pos
from hyper_simulation.component.denial import get_matched_vertices, compute_allowed_pairs, compute_allowed_pairs_batch
from hyper_simulation.hypergraph.path import find_shortest_hyperpaths_bounded, find_shortest_hyperpaths_local_bounded
from hyper_simulation.component.nli import get_nli_labels_batch
import warnings
from tqdm import tqdm
from hyper_simulation.utils.log import getLogger
import logging

def _get_path_description_batch(
    hypergraph: LocalHypergraph,
    pairs: list[tuple[Vertex, Vertex]],
    hops: int,
    max_paths: int = 1000000,
) -> dict[tuple[Vertex, Vertex], str | None]:
    """
    鍦ㄦ暣涓?hypergraph 涓婃壒閲忚幏鍙?(v1, v2) 鐨勮矾寰勬弿杩般€?

    閫昏緫涓?SemanticCluster 鐨?group/intersection/within/across 鎬濊矾涓€鑷达細
    1) 鎸?hyperedge root 鐨?head 閾炬瀯寤?groups
    2) 璁＄畻 groups 闂翠氦闆嗚妭鐐?
    3) 缁勫唴璺緞锛歯ode -> 鏈€杩戝叕鍏辨牴 -> node
    4) 缁勯棿璺緞锛氬厛璧?group 鏈€鐭矾寰勶紝鍐嶅湪姣忚烦浜ら泦涓婂仛缁勫悎
    5) 寰楀埌 list[list[Node]] 鍚庤浆鎴愭枃鏈?

    鎬ц兘锛?
    - 浣跨敤鍑芥暟鍐呯粨鏋勫寲缂撳瓨锛坓roup/浜ら泦/LCA/group shortest path锛?
    - 鍏堢敤 bounded 鏈€鐭?hyperpath 鍋?hops 杩囨护
    """
    if not pairs:
        return {}
    if hops < 0:
        return {pair: None for pair in pairs}

    # 鍑芥暟鍐呯粨鏋勫寲缂撳瓨
    desc_cache: dict[tuple[int, int, int], str | None] = {}
    pair_lca_cache: dict[tuple[int, int], Node | None] = {}

    # 鍘婚噸 pairs锛岄伩鍏嶉噸澶嶈绠?
    unique_pairs = list(dict.fromkeys(pairs))

    # 鍏堝懡涓紦瀛橈紙褰撳墠璋冪敤鍐咃級
    result: dict[tuple[Vertex, Vertex], str | None] = {}
    uncached_pairs: list[tuple[Vertex, Vertex]] = []
    for v1, v2 in unique_pairs:
        if v1 is None or v2 is None:
            result[(v1, v2)] = None
            continue
        key = (v1.id, v2.id, hops)
        if key in desc_cache:
            result[(v1, v2)] = desc_cache[key]
        else:
            uncached_pairs.append((v1, v2))

    # ---------- 0) 鎵归噺 hops 杩囨护锛堟寜缁忚繃 hyperedge 鏁帮級 ----------
    shortest_map = find_shortest_hyperpaths_local_bounded(hypergraph, uncached_pairs, hops) if uncached_pairs else {}

    # ---------- 1) 鏋勫缓 hyperedge groups锛堟寜 founder 鍒嗙粍锛?----------
    # founder 瀹氫箟锛氫粠 root.current_node 涓€鐩存部 head 鍚戜笂锛岀洿鍒?None 鎴?self-loop 鍋滄銆?
    founder_cache: dict[Node, Node] = {}

    def find_founder(node: Node | None) -> Node | None:
        if node is None:
            return None
        if node in founder_cache:
            return founder_cache[node]

        path: list[Node] = []
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            if cur in founder_cache:
                founder = founder_cache[cur]
                for p in path:
                    founder_cache[p] = founder
                return founder

            visited.add(cur)
            path.append(cur)
            nxt = cur.head
            if nxt is None or nxt == cur:
                founder = cur
                for p in path:
                    founder_cache[p] = founder
                return founder
            cur = nxt

        # 閬囧埌鐜椂锛屽洖閫€涓哄綋鍓嶈妭鐐逛綔涓?founder锛岄伩鍏嶆棤闄愬惊鐜€?
        founder = cur if cur is not None else node
        for p in path:
            founder_cache[p] = founder
        return founder

    groups_dict: dict[Node, list[Hyperedge]] = {}
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node is None:
            continue
        founder = find_founder(root_node)
        if founder is None:
            continue
        groups_dict.setdefault(founder, []).append(he)

    groups = list(groups_dict.values())
    he_to_group = {}
    for gi, group in enumerate(groups):
        for he in group:
            he_to_group[he] = gi

    # 姣忎釜 group 鐨勮妭鐐归泦鍚堢紦瀛?
    group_nodes: list[set] = []
    for group in groups:
        nodes = set()
        for he in group:
            for vv in he.vertices:
                nn = he.current_node(vv)
                if nn is not None:
                    nodes.add(nn)
        group_nodes.append(nodes)

    # ---------- 杈呭姪锛氱粍鍐呰矾寰?----------
    ancestor_chain_cache: dict[Node, tuple[list[Node], set[Node]]] = {}

    def get_ancestor_chain_and_set(node: Node) -> tuple[list[Node], set[Node]]:
        cached = ancestor_chain_cache.get(node)
        if cached is not None:
            return cached

        chain: list[Node] = []
        chain_set: set[Node] = set()
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            chain_set.add(cur)
            if cur.head == cur:
                break
            cur = cur.head

        result = (chain, chain_set)
        ancestor_chain_cache[node] = result
        return result

    def nearest_common(node_a: Node, node_b: Node) -> Node | None:
        pair_key: tuple[int, int] = (id(node_a), id(node_b))
        if pair_key in pair_lca_cache:
            return pair_lca_cache[pair_key]

        _, ancestors_a = get_ancestor_chain_and_set(node_a)
        chain_b, _ = get_ancestor_chain_and_set(node_b)

        found = None
        for cur in chain_b:
            if cur in ancestors_a:
                found = cur
                break

        pair_lca_cache[pair_key] = found
        pair_lca_cache[(id(node_b), id(node_a))] = found
        return found

    def path_to_ancestor(node: Node, ancestor: Node) -> list[Node] | None:
        path: list[Node] = []
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path.append(cur)
            if cur == ancestor:
                return path
            if cur.head == cur:
                break
            cur = cur.head
        return None

    def within_group_path(node_a: Node, node_b: Node, group_idx: int) -> list[Node] | None:
        if node_a not in group_nodes[group_idx] or node_b not in group_nodes[group_idx]:
            return None

        root = nearest_common(node_a, node_b)
        if root is None:
            return [node_a, node_b]

        if root == node_a:
            path = path_to_ancestor(node_b, node_a)
            if path:
                return list(reversed(path))
            return [node_a, node_b]

        if root == node_b:
            path = path_to_ancestor(node_a, node_b)
            if path:
                return path
            return [node_a, node_b]

        path_a: list[Node] = []
        cur = node_a
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path_a.append(cur)
            if cur == root:
                break
            cur = cur.head
        if not path_a or path_a[-1] != root:
            return None

        path_b: list[Node] = []
        cur = node_b
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path_b.append(cur)
            if cur == root:
                break
            cur = cur.head
        if not path_b or path_b[-1] != root:
            return None

        tail = list(reversed(path_b[:-1]))
        return path_a + tail

    # ---------- 3) 鎵归噺鐢熸垚 list[list[Node]] 骞惰浆鏂囨湰 ----------
    for v1, v2 in uncached_pairs:
        key = (v1.id, v2.id, hops)

        if v1 == v2:
            desc_cache[key] = v1.text()
            result[(v1, v2)] = desc_cache[key]
            continue

        shortest_hyperedges = shortest_map.get((v1, v2), [])
        # for he in shortest_hyperedges:
        #     he.assert_nodes_reach_root()
        if not shortest_hyperedges or len(shortest_hyperedges) > hops:
            desc_cache[key] = None
            result[(v1, v2)] = None
            continue

        all_segment_paths: list[list[list[Node]]] = []

        # 浠呭熀浜?shortest_hyperedges 鐨勬湁搴忓簭鍒楁瀯閫犺矾寰勩€?
        # 灏嗗叾鍒囨垚杩炵画鐨?group 娈碉細[(gid, [he...]), ...]
        segments: list[tuple[int, list[Hyperedge]]] = []
        valid_shortest = True
        for he in shortest_hyperedges:
            gid = he_to_group.get(he)
            if gid is None:
                valid_shortest = False
                break
            if not segments or segments[-1][0] != gid:
                segments.append((gid, [he]))
            else:
                segments[-1][1].append(he)

        if not valid_shortest or not segments:
            desc_cache[key] = None
            result[(v1, v2)] = None
            continue
        
        def unique_nodes_from_segment(seg_edges: list[Hyperedge]) -> set[Node]:
            nodes: set[Node] = set()
            for he in seg_edges:
                for vv in he.vertices:
                    nn = he.current_node(vv)
                    if nn is not None:
                        nodes.add(nn)
            return nodes

        def vertex_nodes_in_segment(vertex: Vertex, seg_edges: list[Hyperedge]) -> list[Node]:
            nodes: list[Node] = []
            seen = set()
            for he in seg_edges:
                if vertex not in he.vertices:
                    continue
                nn = he.current_node(vertex)
                if nn is None or nn in seen:
                    continue
                seen.add(nn)
                nodes.append(nn)
            return nodes

        # 鐗规畩鎯呭喌锛氬彧鏈変竴涓?segment 鏃讹紙鍗?hops 鎯呭喌锛?
        # 鐩存帴鍦ㄨ group 鍐呭鎵捐矾寰勶紝鏃犻渶娈甸棿杩炴帴
        
        # print segments for debugging
        # print(f"Segments for pair ({v1.text()}, {v2.text()}):")
        # for gid, seg_edges in segments:
        #     print(f"  Group {gid}: Hyperedges {[he.text() for he in seg_edges]}")
        
        if len(segments) == 1:
            gid, seg_edges = segments[0]
            start_candidates = vertex_nodes_in_segment(v1, seg_edges)
            end_candidates = vertex_nodes_in_segment(v2, seg_edges)
            
            if not start_candidates or not end_candidates:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue
            
            # 灏濊瘯鎵€鏈夌粍鍚?
            for start_node in start_candidates:
                for end_node in end_candidates:
                    seg_path = within_group_path(start_node, end_node, gid)
                    if seg_path:
                        all_segment_paths.append([seg_path])
            
            # 鍚屾牱澶勭悊鐩磋繛鍊欓€?
            shared_hes = [he for he in shortest_hyperedges if (v1 in he.vertices and v2 in he.vertices)]
            for he in shared_hes:
                n1_direct = he.current_node(v1)
                n2_direct = he.current_node(v2)
                if n1_direct is not None and n2_direct is not None:
                    gid_direct = he_to_group.get(he)
                    if gid_direct is not None:
                        direct_seg_path = within_group_path(n1_direct, n2_direct, gid_direct)
                        if direct_seg_path:
                            all_segment_paths.append([direct_seg_path])
                            continue
                    all_segment_paths.append([[n1_direct, n2_direct]])
            
            if not all_segment_paths:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue
        else:
            # 澶?segment 鎯呭喌锛氳绠楃浉閭绘浜ら泦
            connector_lists: list[list[Node]] = []
            segment_node_sets = [unique_nodes_from_segment(seg_edges) for _, seg_edges in segments]
            
            connectors_valid = True
            for i in range(len(segments) - 1):
                inter_nodes = list(segment_node_sets[i] & segment_node_sets[i + 1])
                if not inter_nodes:
                    connectors_valid = False
                    break
                connector_lists.append(inter_nodes)


            if not connectors_valid:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue

            # 澶歴egment鎯呭喌鐨剆tart/end鍊欓€?
            start_candidates = vertex_nodes_in_segment(v1, segments[0][1])
            end_candidates = vertex_nodes_in_segment(v2, segments[-1][1])
            if not start_candidates or not end_candidates:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue

            connector_combos = list(product(*connector_lists)) if connector_lists else [()]

            # 涓€璺崇洿杩炲€欓€変紭鍏堜繚鐣欙紙浠嶅彧鏉ユ簮浜?shortest_hyperedges锛?
            shared_hes = [he for he in shortest_hyperedges if (v1 in he.vertices and v2 in he.vertices)]
            for he in shared_hes:
                n1_direct = he.current_node(v1)
                n2_direct = he.current_node(v2)
                if n1_direct is not None and n2_direct is not None:
                    gid_direct = he_to_group.get(he)
                    if gid_direct is not None:
                        direct_seg_path = within_group_path(n1_direct, n2_direct, gid_direct)
                        if direct_seg_path:
                            all_segment_paths.append([direct_seg_path])
                            continue
                    all_segment_paths.append([[n1_direct, n2_direct]])

            for start_node in start_candidates:
                for end_node in end_candidates:
                    for combo in connector_combos:
                        chain_nodes = [start_node] + list(combo) + [end_node]
                        if len(chain_nodes) != len(segments) + 1:
                            continue

                        seg_paths_for_candidate: list[list[Node]] = []
                        ok = True
                        for si, (gid, _) in enumerate(segments):
                            n_from = chain_nodes[si]
                            n_to = chain_nodes[si + 1]
                            seg_path = within_group_path(chain_nodes[si], chain_nodes[si + 1], gid)
                            if not seg_path:
                                ok = False
                                break
                            seg_paths_for_candidate.append(seg_path)

                        if not ok:
                            continue
                        all_segment_paths.append(seg_paths_for_candidate)

        if not all_segment_paths:
            desc_cache[key] = None
            result[(v1, v2)] = None
            continue

        def candidate_node_cost(seg_paths: list[list[Node]]) -> int:
            cost = 0
            for i, seg in enumerate(seg_paths):
                if i == 0:
                    cost += len(seg)
                else:
                    cost += max(0, len(seg) - 1)
            return cost

        min_cost = min(candidate_node_cost(p) for p in all_segment_paths)
        shortest_paths = [p for p in all_segment_paths if candidate_node_cost(p) == min_cost]
        best_segments = shortest_paths[0]

        # 缁熶竴鎸?node.index 鎺掑簭锛屼繚璇?(v,v') 涓?(v',v) 鐨勬弿杩版瀯閫犱竴鑷?
        best_nodes = [n for seg in best_segments for n in seg if getattr(n, "text", None)]
        best_nodes = sorted(best_nodes, key=lambda n: n.index)
        desc = " ".join(n.text for n in best_nodes)
        # print(f"({v1.text()}, {v2.text()}): {desc}")

        desc_cache[key] = desc if desc else None
        result[(v1, v2)] = desc_cache[key]

    # 琛ラ綈閲嶅杈撳叆 pair 鐨勮繑鍥?
    return {pair: result.get(pair) for pair in pairs}

def post_detection(
    query: LocalHypergraph,
    data: LocalHypergraph,
    simulation: list[tuple[Vertex, Vertex]],
    hops: int = 10,
    require_all_neighbors: bool = False,
) -> list[tuple[Vertex, Vertex]]:
    """
    瀵?simulation 杩涜鍚庡鐞嗘鏌ュ拰鏋勯€犱竴鑷存€т笅鐨勬槧灏勩€?
    
     鍙傛暟锛?
     - require_all_neighbors: bool
        * False (榛樿): 瀵逛簬(u, v) in match锛岄拡瀵逛换鎰忓悓杈归偦鎺ョ偣 u'锛?
                   鑻ュ瓨鍦?(u', _) in match锛屽垯蹇呴』瀛樺湪 (u', v') in match
                   涓?(u, u') 涓?(v, v') 鍖归厤锛涘惁鍒欏垹闄?u, v)
        * True: 瀵逛簬(u, v) in match锛寀 鍦ㄨ竟涓殑鎵€鏈夐偦鎺ヨ妭鐐?u' 閮藉繀椤绘湁 (u', _) in match锛?
                  鍚﹀垯鍒犻櫎(u, v)
    
     姝ラ锛?
     1) 鏋氫妇 query 瓒呰竟鍐呯殑鑺傜偣瀵?(u, u')
         - 濡傛灉 (u, v), (u', v') 閮藉湪 simulation锛岃幏鍙栧畠浠殑鎻忚堪
         - 鐢?NLI 妫€鏌?(u, u') 鎻忚堪 鍜?(v, v') 鎻忚堪鏄惁涓嶇煕鐩?
         - 璁板綍鍖归厤鍏崇郴
     2) 鍒濆鍖?match = simulation
     3) 杩涜涓嶅姩鐐硅绠楋紙鏍规嵁 require_all_neighbors 閫夋嫨绛栫暐锛?
    """
    if not simulation:
        return []

    debug_postprocess = os.environ.get("POSTPROCESS_DEBUG", "0") not in {"0", "", "false", "False"}
    debug_focus = {
        term.strip().lower()
        for term in os.environ.get("POSTPROCESS_DEBUG_FOCUS", "").split(",")
        if term.strip()
    }
    trace_causal = os.environ.get("POSTPROCESS_TRACE_CAUSAL", "0") not in {"0", "", "false", "False"}
    
    # 鍒濆鍖?
    filtered_simulation = [(u, v) for u, v in simulation if u is not None and v is not None]
    match: set[tuple[Vertex, Vertex]] = set(filtered_simulation)

    # 鍒濆 simulation 鐨勫鍊肩储寮曪細u -> {v1, v2, ...}
    simulation_by_u: dict[Vertex, set[Vertex]] = {}
    for u, v in filtered_simulation:
        simulation_by_u.setdefault(u, set()).add(v)

    def _should_debug_pair(u: Vertex, v: Vertex) -> bool:
        if not debug_postprocess:
            return False
        if not debug_focus:
            return True
        u_text = u.text().strip().lower()
        v_text = v.text().strip().lower()
        return u_text in debug_focus or v_text in debug_focus
    
    # -------- 绗竴闃舵锛氭灇涓?query 瓒呰竟锛岄獙璇佷竴鑷存€?--------
    # 鏀堕泦鎵€鏈夊湪瓒呰竟鍐呴厤瀵逛笖閮藉湪 simulation 涓殑鍥涘厓缁?(u, u', v, v')
    # 鍚屾椂鏋勫缓閭绘帴鍏崇郴锛氭瘡涓妭鐐瑰湪 query 瓒呰竟涓殑閭绘帴鑺傜偣
    edge_neighbors: dict[Vertex, set[Vertex]] = {}
    uu_vv_quads: list[tuple[Vertex, Vertex, Vertex, Vertex]] = []
    uu_desc_cache: dict[tuple[int, int], str] = {}
    quad_evidence: dict[tuple[int, int, int, int], dict[str, str]] = {}

    def _quad_key(u: Vertex, u_prime: Vertex, v: Vertex, v_prime: Vertex) -> tuple[int, int, int, int]:
        return (u.id, u_prime.id, v.id, v_prime.id)

    def _get_quad_evidence(u: Vertex, u_prime: Vertex, v: Vertex, v_prime: Vertex) -> dict[str, str]:
        direct = quad_evidence.get(_quad_key(u, u_prime, v, v_prime))
        if direct is not None:
            return direct
        reverse = quad_evidence.get(_quad_key(u_prime, u, v_prime, v))
        if reverse is not None:
            return reverse
        return {"reason": "unknown"}

    def _uu_cache_key(u: Vertex, u_prime: Vertex) -> tuple[int, int]:
        return (u.id, u_prime.id) if u.id <= u_prime.id else (u_prime.id, u.id)

    def _path_to_ancestor(node: Node, ancestor: Node) -> list[Node] | None:
        path: list[Node] = []
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path.append(cur)
            if cur == ancestor:
                return path
            if cur.head == cur:
                break
            cur = cur.head
        return None

    def _nearest_common_query_node(node_a: Node, node_b: Node) -> Node | None:
        ancestors_a: set[Node] = set()
        cur = node_a
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            ancestors_a.add(cur)
            if cur.head == cur:
                break
            cur = cur.head

        cur = node_b
        visited = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            if cur in ancestors_a:
                return cur
            if cur.head == cur:
                break
            cur = cur.head
        return None

    def _render_query_node_text(node: Node) -> str | None:
        text = getattr(node, "text", None)
        if not text:
            return None
        if not node.is_query:
            return text
        type_str = node.type_str()
        if type_str:
            return f"The {type_str}"
        return "The ATTRIBUTE"

    def _build_uu_desc_from_hyperedge(he: Hyperedge, u: Vertex, u_prime: Vertex) -> str | None:
        n1 = he.current_node(u)
        n2 = he.current_node(u_prime)
        if n1 is None or n2 is None:
            return None

        lca = _nearest_common_query_node(n1, n2)
        if lca is None:
            return None

        p1 = _path_to_ancestor(n1, lca)
        p2 = _path_to_ancestor(n2, lca)
        if not p1 or not p2:
            return None

        seq = p1 + list(reversed(p2[:-1]))
        # 缁熶竴鎸?node.index 鎺掑簭锛屼繚璇?(u,u') 涓?(u',u) 鐢熸垚鐩稿悓鎻忚堪
        seq = sorted(seq, key=lambda n: n.index)
        tokens = [_render_query_node_text(n) for n in seq]
        desc = " ".join(t for t in tokens if t)
        return desc if desc else None
    
    for he in query.hyperedges:
        vertices = list(he.vertices)
        # 鏋勫缓閭绘帴鍏崇郴
        for v in vertices:
            if v not in edge_neighbors:
                edge_neighbors[v] = set()
            for v_other in vertices:
                if v_other != v:
                    edge_neighbors[v].add(v_other)
        # 鏀堕泦鍥涘厓缁勶紙鏀寔鍚屼竴 u 鐨勫鐩爣鏄犲皠锛?
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, u_prime = vertices[i], vertices[j]

                cache_key = _uu_cache_key(u, u_prime)
                if cache_key not in uu_desc_cache:
                    desc = _build_uu_desc_from_hyperedge(he, u, u_prime)
                    if desc is not None:
                        uu_desc_cache[cache_key] = desc

                v_set = simulation_by_u.get(u)
                v_prime_set = simulation_by_u.get(u_prime)
                if not v_set or not v_prime_set:
                    continue
                for v in v_set:
                    for v_prime in v_prime_set:
                        uu_vv_quads.append((u, u_prime, v, v_prime))
    
    # 瀵逛簬姣忎釜鍥涘厓缁勶紝鑾峰彇 (v, v') 鐨勮矾寰勬弿杩帮紝杩涜 NLI 妫€鏌?
    # 璁板綍 (u, u') 鍙帴鍙楃殑 (v, v')锛歭abel != contradiction
    uu_to_vv_match: dict[tuple[Vertex, Vertex], set[tuple[Vertex, Vertex]]] = {}
    
    def _truncate_desc_between_vertices(desc: str, v: Vertex, v_prime: Vertex) -> str | None:
        """
        鍦ㄦ弿杩版枃鏈腑鏌ユ壘 v 鍜?v_prime锛岃嫢涓よ€呴兘瀛樺湪锛屽垯鎴柇锛?
        - 鍒犲幓 v 绗竴娆″嚭鐜扮殑宸︿晶閮ㄥ垎
        - 鍒犲幓 v_prime 鏈€鍚庝竴娆″嚭鐜扮殑鍙充晶閮ㄥ垎
        淇濈暀涓棿閮ㄥ垎浠ュ噺灏戝櫔澹板 NLI 鐨勫奖鍝嶃€?
        """
        if not desc:
            return None
        
        v_text = v.text()
        v_prime_text = v_prime.text()
        
        if not v_text or not v_prime_text or v_text == v_prime_text:
            return desc
        
        # 鎵?v_text 绗竴娆″嚭鐜扮殑浣嶇疆
        first_v_pos = desc.find(v_text)
        if first_v_pos == -1:
            return desc
        
        # 鎵?v_prime_text 鏈€鍚庝竴娆″嚭鐜扮殑浣嶇疆
        last_v_prime_pos = desc.rfind(v_prime_text)
        if last_v_prime_pos == -1:
            return desc
        
        # 濡傛灉 v 鍦?v_prime 涔嬪悗锛屾棤娉曟埅鏂紝杩斿洖鍘熷€?
        if first_v_pos >= last_v_prime_pos:
            return desc
        
        # 鎴柇锛氫粠 v_text 寮€濮嬪埌 v_prime_text 缁撴潫
        truncated = desc[first_v_pos:last_v_prime_pos + len(v_prime_text)]
        return truncated if truncated else desc
    
    if uu_vv_quads:
        # 鑾峰彇鎵€鏈?(v, v') 鐨勮矾寰勬弿杩?
        v_v_pairs = [(quad[2], quad[3]) for quad in uu_vv_quads]
        path_descs = _get_path_description_batch(data, v_v_pairs, hops)
        # 鏋勯€?NLI 妫€鏌ョ殑鏂囨湰瀵癸紙4 绉嶇粍鍚堬細鍘熸瀯閫犆?鏂瑰悜 + 鎴柇鏋勯€犆?鏂瑰悜锛?
        nli_pairs_list: list[tuple[str, str, str, str, int, int]] = []  # 4 涓?desc + 涓や釜 quad indices
        valid_quads: list[tuple[Vertex, Vertex, Vertex, Vertex]] = []
        valid_quad_keys: list[tuple[int, int, int, int]] = []
        
        for u, u_prime, v, v_prime in uu_vv_quads:
            # (u, u') 浠呬娇鐢?query 瓒呰竟璺緞鎻忚堪锛涙棤娉曟瀯閫犳椂璺宠繃璇ュ洓鍏冪粍
            uu_desc = uu_desc_cache.get(_uu_cache_key(u, u_prime))
            # (v, v') 鐨勬弿杩颁粠璺緞鑾峰彇
            vv_desc_original = path_descs.get((v, v_prime))

            qkey = _quad_key(u, u_prime, v, v_prime)
            if uu_desc is None:
                quad_evidence[qkey] = {
                    "reason": "missing_query_path_desc",
                    "detail": f"(u,u')=({u.text()},{u_prime.text()})",
                }
                continue
            if vv_desc_original is None:
                quad_evidence[qkey] = {
                    "reason": "missing_data_path_desc",
                    "detail": f"(v,v')=({v.text()},{v_prime.text()})",
                }
                continue

            if uu_desc is not None and vv_desc_original is not None:
                # 鐢熸垚鎴柇鐗堟湰
                vv_desc_truncated = _truncate_desc_between_vertices(vv_desc_original, v, v_prime)
                if vv_desc_truncated is None:
                    vv_desc_truncated = vv_desc_original
                
                quad_idx = len(valid_quads)
                nli_pairs_list.append((vv_desc_original, uu_desc, vv_desc_truncated, uu_desc, quad_idx, quad_idx))
                valid_quads.append((u, u_prime, v, v_prime))
                valid_quad_keys.append(qkey)
        
        # 鑾峰彇 NLI 鏍囩锛? 绉嶇粍鍚?(鍘熸瀯閫犆?鏂瑰悜 + 鎴柇鏋勯€犆?鏂瑰悜)
        if nli_pairs_list:
            nli_pairs: list[tuple[str, str]] = []
            nli_pair_to_quad_idx: list[tuple[int, str]] = []  # (quad_idx, desc_type)
            
            for vv_orig, uu_desc, vv_trunc, _, quad_idx, _ in nli_pairs_list:
                # 鍘熸瀯閫狅細A->B 鍜?B->A
                nli_pairs.append((vv_orig, uu_desc))
                nli_pair_to_quad_idx.append((quad_idx, "original_ab"))
                
                nli_pairs.append((uu_desc, vv_orig))
                nli_pair_to_quad_idx.append((quad_idx, "original_ba"))
                
                # 鎴柇鏋勯€狅細A->B 鍜?B->A
                nli_pairs.append((vv_trunc, uu_desc))
                nli_pair_to_quad_idx.append((quad_idx, "truncated_ab"))
                
                nli_pairs.append((uu_desc, vv_trunc))
                nli_pair_to_quad_idx.append((quad_idx, "truncated_ba"))

            labels = get_nli_labels_batch(nli_pairs)
            
            # 鎸?quad 鑱氬悎 4 涓爣绛?
            quad_idx_to_labels: dict[int, dict[str, str]] = {}
            for pair_idx, (quad_idx, desc_type) in enumerate(nli_pair_to_quad_idx):
                if quad_idx not in quad_idx_to_labels:
                    quad_idx_to_labels[quad_idx] = {}
                quad_idx_to_labels[quad_idx][desc_type] = labels[pair_idx]
            
            # 鍒ゆ柇鐭涚浘锛? 绉嶉兘鏄?contradiction 鎵嶈涓虹煕鐩?
            for idx, (u, u_prime, v, v_prime) in enumerate(valid_quads):
                qkey = valid_quad_keys[idx]
                label_dict = quad_idx_to_labels.get(idx, {})
                
                original_ab = label_dict.get("original_ab", "unknown")
                original_ba = label_dict.get("original_ba", "unknown")
                truncated_ab = label_dict.get("truncated_ab", "unknown")
                truncated_ba = label_dict.get("truncated_ba", "unknown")
                
                # 鍙湁 4 涓兘鏄?contradiction锛屾墠璁や负鏄煕鐩?
                is_all_contradiction = (
                    original_ab == 'contradiction' and 
                    original_ba == 'contradiction' and 
                    truncated_ab == 'contradiction' and 
                    truncated_ba == 'contradiction'
                )
                
                if is_all_contradiction:
                    quad_evidence[qkey] = {
                        "reason": "nli_contradiction",
                        "detail": f"original_ab={original_ab}|original_ba={original_ba}|truncated_ab={truncated_ab}|truncated_ba={truncated_ba}",
                    }
                    # print(f"[NLI 4-way contradiction] ({u.text()}, {u_prime.text()}) <-> ({v.text()}, {v_prime.text()})")
                else:
                    quad_evidence[qkey] = {
                        "reason": "nli_non_contradiction",
                        "detail": f"original_ab={original_ab}|original_ba={original_ba}|truncated_ab={truncated_ab}|truncated_ba={truncated_ba}",
                    }

            # 鎸夋柊瑙勫垯鏇存柊 uu_to_vv_match
            for idx, (u, u_prime, v, v_prime) in enumerate(valid_quads):
                qkey = valid_quad_keys[idx]
                label_dict = quad_idx_to_labels.get(idx, {})
                
                # 鍙涓嶆槸鍏ㄩ兘鐭涚浘锛屽氨璁や负闈炵煕鐩?
                original_ab = label_dict.get("original_ab", "unknown")
                original_ba = label_dict.get("original_ba", "unknown")
                truncated_ab = label_dict.get("truncated_ab", "unknown")
                truncated_ba = label_dict.get("truncated_ba", "unknown")
                
                is_non_contradict = not (
                    original_ab == 'contradiction' and 
                    original_ba == 'contradiction' and 
                    truncated_ab == 'contradiction' and 
                    truncated_ba == 'contradiction'
                )
                
                # 璁板綍闈炵煕鐩剧殑鍖归厤
                if is_non_contradict:
                    uu_to_vv_match.setdefault((u, u_prime), set()).add((v, v_prime))
                    uu_to_vv_match.setdefault((u_prime, u), set()).add((v_prime, v))
                    # if debug_postprocess and (_should_debug_pair(u, v) or _should_debug_pair(u_prime, v_prime)):
                    #     print(
                    #         f"[POSTPROCESS DEBUG] NLI ok (4-way): ({u.text()}, {u_prime.text()}) <-> ({v.text()}, {v_prime.text()})"
                    #     )
                # elif debug_postprocess and (_should_debug_pair(u, v) or _should_debug_pair(u_prime, v_prime)):
                #     print(
                #         f"[POSTPROCESS DEBUG] NLI 4-way contradiction: ({u.text()}, {u_prime.text()}) <-> ({v.text()}, {v_prime.text()})"
                #     )
    
    # -------- 绗簩闃舵锛歸orklist 涓嶅姩鐐硅绠?--------
    # 浠呴噸妫€鍙楀奖鍝嶇殑 u锛岄伩鍏嶆瘡杞叏閲忔壂鎻?match銆?

    # u -> 褰撳墠浠嶅湪 match 涓殑 (u, v) 闆嗗悎
    match_by_u: dict[Vertex, set[tuple[Vertex, Vertex]]] = {}
    for u, v in match:
        match_by_u.setdefault(u, set()).add((u, v))

    # 鍙嶅悜閭绘帴锛氬綋鏌愪釜 u 琚垹鏃讹紝鍝簺涓績鑺傜偣浼氬彈褰卞搷
    reverse_neighbors: dict[Vertex, set[Vertex]] = {}
    for center_u, neighbors in edge_neighbors.items():
        for neighbor_u in neighbors:
            reverse_neighbors.setdefault(neighbor_u, set()).add(center_u)

    deleted_pair_causes: dict[tuple[int, int], dict[str, object]] = {}
    deleted_pairs_by_u: dict[Vertex, list[tuple[str, str]]] = {}

    def _print_pair_debug_header(u: Vertex, v: Vertex) -> None:
        print(f"\n[POSTPROCESS DEBUG] pair=({u.text()}, {v.text()})")
        print(f"  query neighbors: {[n.text() for n in sorted(edge_neighbors.get(u, set()), key=lambda x: x.text())]}")
        print("  current neighbors in match:")
        for neighbor in sorted(edge_neighbors.get(u, set()), key=lambda x: x.text()):
            neighbor_pairs = match_by_u.get(neighbor, set())
            formatted_pairs = [
                (u2.text(), v2.text())
                for u2, v2 in sorted(neighbor_pairs, key=lambda p: (p[0].text(), p[1].text()))
            ]
            print(f"    - {neighbor.text()}: {formatted_pairs}")

    def should_remove_pair(u: Vertex, v: Vertex) -> tuple[bool, dict[str, object] | None]:
        debug_this_pair = _should_debug_pair(u, v)
        # if debug_this_pair:
            # _print_pair_debug_header(u, v)

        if require_all_neighbors:
            # 妯″紡2锛歶 鐨勬墍鏈夊悓杈归偦鎺ヨ妭鐐归兘蹇呴』瀛樺湪 (u', _) in match
            for u_neighbor in edge_neighbors.get(u, set()):
                if not match_by_u.get(u_neighbor):
                    reason: dict[str, object] = {
                        "reason": "missing_neighbor_pairs",
                        "neighbor": u_neighbor.text(),
                    }
                    if trace_causal:
                        recent_deleted = deleted_pairs_by_u.get(u_neighbor, [])[-5:]
                        if recent_deleted:
                            reason["upstream_deleted"] = [
                                f"({u_neighbor.text()}, {v_text}): {r_text}"
                                for v_text, r_text in recent_deleted
                            ]
                    # if debug_this_pair:
                    #     print(f"  [REMOVE] missing neighbor support: {u_neighbor.text()} has no surviving match")
                    #     if trace_causal:
                    #         print(f"  [CAUSE] {reason}")
                    return True, reason
            return False, None

        # 妯″紡1锛氬浠绘剰鍚岃竟閭绘帴鐐?u'锛岃嫢瀛樺湪 (u', _) in match锛?
        # 蹇呴』瀛樺湪 v' 浣垮緱 (v, v') 涓?(u, u') 鐨勫叧绯婚潪鐭涚浘銆?
        for u_prime in edge_neighbors.get(u, set()):
            u_prime_pairs = match_by_u.get(u_prime)
            if not u_prime_pairs:
                # if debug_this_pair:
                #     print(f"  [SKIP] neighbor {u_prime.text()} has no surviving pairs")
                continue

            allowed_vv = uu_to_vv_match.get((u, u_prime), set())
            has_support = False
            candidate_failures: list[str] = []
            for _, v_prime in u_prime_pairs:
                if (v, v_prime) in allowed_vv:
                    has_support = True
                    # if debug_this_pair:
                    #     print(f"  [SUPPORT] neighbor={u_prime.text()} supports via ({v.text()}, {v_prime.text()})")
                    break
                evidence = _get_quad_evidence(u, u_prime, v, v_prime)
                candidate_failures.append(
                    f"({v.text()}, {v_prime.text()}): {evidence.get('reason', 'unknown')}"
                )
            if not has_support:
                reason: dict[str, object] = {
                    "reason": "no_support_from_neighbor",
                    "neighbor": u_prime.text(),
                    "candidate_failures": candidate_failures[:12],
                }
                # if debug_this_pair:
                #     print(f"  [REMOVE] no support from neighbor {u_prime.text()}")
                #     print(
                #         f"  allowed pairs for ({u.text()}, {u_prime.text()}): {[(v1.text(), v2.text()) for v1, v2 in sorted(allowed_vv, key=lambda p: (p[0].text(), p[1].text()))]}"
                #     )
                #     print(
                #         f"  surviving pairs for {u_prime.text()}: {[(u2.text(), v2.text()) for u2, v2 in sorted(u_prime_pairs, key=lambda p: (p[0].text(), p[1].text()))]}"
                #     )
                #     if trace_causal:
                #         print(f"  [CAUSE] {reason}")
                return True, reason
        return False, None

    # 鍒濆灏嗘墍鏈夊彲鑳藉彈绾︽潫鐨?u 鍏ラ槦
    dirty_u: set[Vertex] = set(match_by_u.keys())
    
    while dirty_u:
        to_remove: dict[tuple[Vertex, Vertex], dict[str, object]] = {}

        # 鍙壂鎻忚剰鑺傜偣瀵瑰簲鐨勬槧灏勫
        for u in dirty_u:
            for pair in list(match_by_u.get(u, set())):
                pu, pv = pair
                should_remove, cause = should_remove_pair(pu, pv)
                if should_remove:
                    to_remove[pair] = cause or {"reason": "unknown"}

        if not to_remove:
            # if debug_postprocess:
            #     print(f"[POSTPROCESS DEBUG] fixed point reached with {len(match)} pairs")
            break

        next_dirty_u: set[Vertex] = set()
        for (u, v), cause in to_remove.items():
            if (u, v) not in match:
                continue
            match.remove((u, v))
            cause_payload = dict(cause)
            deleted_pair_causes[(u.id, v.id)] = cause_payload
            deleted_pairs_by_u.setdefault(u, []).append((v.text(), str(cause.get("reason", "unknown"))))
            # if trace_causal and (debug_postprocess and _should_debug_pair(u, v)):
            #     print(f"[POSTPROCESS CAUSAL] removed ({u.text()}, {v.text()}) -> {cause}")

            if u in match_by_u:
                match_by_u[u].discard((u, v))
                if not match_by_u[u]:
                    del match_by_u[u]

            # 鍒犻櫎 (u, v) 浼氬奖鍝嶄緷璧?u 鐨勪腑蹇冭妭鐐癸紱u 鑷韩涔熷彲鑳戒粛鏈夊叾瀹冨€欓€夐渶瑕侀噸妫€
            next_dirty_u.update(reverse_neighbors.get(u, set()))
            if u in match_by_u:
                next_dirty_u.add(u)

        dirty_u = next_dirty_u
    
    # if debug_postprocess:
    #     print(f"[POSTPROCESS DEBUG] final match size: {len(match)}")

    return list(match)

def get_simulation_slice(query: LocalHypergraph, data: LocalHypergraph, simulation: list[tuple[Vertex, Vertex]], num: int) -> list[list[tuple[Vertex, Vertex]]]:
    """
    鍩轰簬 Vertex 鐨?provenance 淇℃伅锛屽皢 simulation 鍒囧壊涓哄悇涓師濮?hypergraph 涓嬬殑鍒囩墖銆?
    
    u 鏉ヨ嚜 query锛堝崟涓€鏉ユ簮锛夛紝v 鏉ヨ嚜 data锛堝彲鑳藉睘浜庡涓師濮?hypergraph锛夈€?
    瀵逛簬 simulation 涓殑姣忎釜 (u, v) 瀵癸紝鏍规嵁 v 鐨?provenance 纭畾璇ュ灞炰簬鍝簺鍘熷 hypergraph銆?
    
    鍙傛暟锛?
    - query: 鏌ヨ鐨?LocalHypergraph
    - data: 鏁版嵁鐨?LocalHypergraph
    - simulation: Vertex 瀵圭殑鍖归厤鍒楄〃
    - num: 鍘熷 hypergraph 鐨勬€绘暟閲忥紙浠?鍒皀um锛?
    
    杩斿洖锛?
    - list[list[tuple[Vertex, Vertex]]]: 闀垮害涓?num 鐨勫垪琛紝
      鍏朵腑绱㈠紩 i 瀵瑰簲绗?(i+1) 涓師濮?hypergraph 鐨?simulation 鍒囩墖
    """
    # 鍒濆鍖栫粨鏋滐細姣忎釜鍘熷 hypergraph 瀵瑰簲涓€涓┖鍒楄〃
    slices = [[] for _ in range(num)]
    
    # 閬嶅巻 simulation 涓殑姣忎釜 (u, v) 瀵?
    for u, v in simulation:
        if u is None or v is None:
            continue
        
        # 鑾峰彇 v 鐨?provenance锛堟墍灞炵殑鍘熷 hypergraph id 闆嗗悎锛?
        v_provenance = v.get_provenance()
        
        # 灏嗚瀵规坊鍔犲埌 v 鎵€灞炵殑鎵€鏈夊師濮?hypergraph 瀵瑰簲鐨勫垏鐗囦腑
        for hg_id in v_provenance:
            # hg_id 浠?寮€濮嬶紝鏁扮粍绱㈠紩浠?寮€濮嬶紝鎵€浠ラ渶瑕佸噺1
            slices[hg_id - 1].append((u, v))
    
    return slices

def check_slice_consistency(query: LocalHypergraph, simulation_slice: list[tuple[Vertex, Vertex]], vertex_ids: set[int]) -> bool:
    """
    妫€鏌?simulation_slice 涓殑 (u, v) 瀵规槸鍚︽弧瓒充竴鑷存€ц姹傦細
瀵逛簬 query 涓?id 鍦?vertex_ids 鍐呯殑姣忎釜 vertex u锛宻imulation_slice 涓嚦灏戝瓨鍦ㄤ竴涓?v 浣垮緱 (u, v) 鍦ㄥ叾涓€?
    
    杩斿洖 True 濡傛灉婊¤冻涓€鑷存€?
    """
    
    vertex_map: dict[Vertex, set[Vertex]] = {}
    for u, v in simulation_slice:
        if u not in vertex_map:
            vertex_map[u] = set()
        vertex_map[u].add(v)
    
    vertex_needs: set[Vertex] = set()
    
    for u in query.vertices:
        if u.id in vertex_ids:
            # print(f"- [{u.id}] {u.text()}")
            vertex_needs.add(u)
    # returns True if for all u in vertex_needs, there exists a v such that (u, v) in simulation_slice
    hit_cnt = 0
    for u in vertex_needs:
        if u in vertex_map and len(vertex_map[u]) > 0:
            hit_cnt += 1
    return hit_cnt == len(vertex_needs)

def refine_simulation_slices(query: LocalHypergraph, simulation_slices: list[list[tuple[Vertex, Vertex]]], answer: str) -> list[list[tuple[Vertex, Vertex]]]:
    # 鍩轰簬 answer 瀵?simulation_slices 杩涜杩涗竴姝ヨ繃婊?
    # 鐙珛鎿嶄綔姣忎釜 slice銆?
    
    def _match(v_text: str, answer: str) -> bool:
        # 绠€鍗曠殑鏂囨湰鍖归厤鍑芥暟锛屽垽鏂?v_text 鏄惁涓?answer 鍖归厤
        # 杩欓噷鍙互浣跨敤鏇村鏉傜殑鍖归厤閫昏緫锛屼緥濡傚寘鍚叧绯汇€佸悓涔夎瘝绛?
        return v_text.strip().lower() == answer.strip().lower()
    
    refined_slices: list[list[tuple[Vertex, Vertex]]] = []
    for slice in simulation_slices:
        new_slice: list[tuple[Vertex, Vertex]] = []
        # 棣栧厛妫€鏌?slice 鍐呮槸鍚︽湁 (u, v) 婊¤冻 v.text() 鍜?answer 鍖归厤
        # 鑻ュ瓨鍦ㄥ尮閰嶏紝鍒?淇濈暀 (u, v), 鑰屽垹闄ゅ叾浠栫殑 (u, _)
        # 鍚﹀垯涓嶈繘琛屼慨鏀?
        matched_map: dict[Vertex, Vertex] = {}
        for u, v in slice:
            if v is not None and _match(v.text(), answer):
                matched_map[u] = v
                
        for u, v in matched_map.items():
            new_slice.append((u, v))

        for u, v in slice:
            if u in matched_map:
                continue
            new_slice.append((u, v))
                
        refined_slices.append(new_slice)
    return refined_slices

def ranking_slices(query: LocalHypergraph, simulation_slices: list[list[tuple[Vertex, Vertex]]], vertex_ids: set[int], k: int) -> list[int]:
    """
    瀵?simulation_slices 鍋?soft ranking銆?

    璇勫垎瀹氫箟锛堢簿纭暣鏁版瘮杈冿紝閬垮厤娴偣璇樊锛夛細
    - score = hit_cnt / len(vertex_needs)
    - 鍥犱负 len(vertex_needs) 瀵规墍鏈?slice 鎭掑畾锛屾帓搴忓彲绛変环涓烘寜 hit_cnt 鎺掑簭
    - hit_cnt 涓鸿 slice 涓懡涓殑 query 鐩爣椤剁偣鏁伴噺
    - vertex_needs 涓?query 涓?id 鍦?vertex_ids 鍐呯殑椤剁偣闆嗗悎

    杩斿洖锛?
    - 鎸夊緱鍒嗕粠楂樺埌浣庢帓搴忓悗鐨?slice 绱㈠紩鍒楄〃
    - 鍙栧墠 k 鏃惰嫢鎴柇鍚屽垎椤癸紝鍒欎繚鐣欐墍鏈変笌绗?k 鍚嶅悓鍒嗙殑 slice锛堝彲鑳借秴杩?k锛?
    """
    if k <= 0 or not simulation_slices:
        return []

    vertex_needs: set[Vertex] = {u for u in query.vertices if u.id in vertex_ids}
    need_cnt = len(vertex_needs)

    scored_indices: list[tuple[int, int]] = []
    for idx, simulation_slice in enumerate(simulation_slices):
        present_u: set[Vertex] = {u for u, _ in simulation_slice if u is not None}
        if need_cnt == 0:
            hit_cnt = 1
        else:
            hit_cnt = sum(1 for u in vertex_needs if u in present_u)
        scored_indices.append((idx, hit_cnt))

    # 鍏堟寜鍛戒腑鏁伴檷搴忥紝鍐嶆寜 index 鍗囧簭锛屼繚璇佸悓鍒嗘椂缁撴灉绋冲畾銆?
    scored_indices.sort(key=lambda x: (-x[1], x[0]))

    if len(scored_indices) <= k:
        return [idx for idx, _ in scored_indices]

    kth_hit_cnt = scored_indices[k - 1][1]
    return [idx for idx, hit_cnt in scored_indices if hit_cnt >= kth_hit_cnt]
    

