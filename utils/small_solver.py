"""Small-object constraint solver (dict assets), ported from ablation utils_areas/optimization.py."""
import ast
import re
import torch
import networkx as nx
import numpy as np
from collections import defaultdict
from utils.loss import soft_near_loss, diou_loss_2d

_CONFLICT_ABLATION_MODE = "baseline"
_CONFLICT_ABLATION_CHOICES = {
    "baseline",
    "wo_logic_error",
    "wo_semantic_group",
    "wo_outgoing_completeness",
    "wo_conflict_module",
}
CROSS_EDGE_OCCUPANCY_RATIO = 0.9
CROSS_EDGE_MIN_COUNT_PER_AXIS = 2


def _parse_region_bounds_node(node):
    if isinstance(node, (list, tuple)) and len(node) >= 4:
        try:
            return [float(node[0]), float(node[1]), float(node[2]), float(node[3])]
        except Exception:
            return None
    if isinstance(node, str):
        try:
            val = ast.literal_eval(node)
            if isinstance(val, (list, tuple)) and len(val) >= 4:
                return [float(val[0]), float(val[1]), float(val[2]), float(val[3])]
        except Exception:
            return None
    return None


def _region_width_height(bounds):
    if bounds is None or len(bounds) < 4:
        return None, None
    x_min, x_max, y_min, y_max = bounds[:4]
    return abs(float(x_max) - float(x_min)), abs(float(y_max) - float(y_min))


def _requires_against_edge_for_tableware(name: str) -> bool:
    key = str(name).lower()
    if "dining" in key and "set" in key:
        return True
    tableware_markers = (
        "plate",
        "place_setting",
        "place_setting_set",
        "dining_place_setting",
        "dining_setting",
        "table_setting",
        "cutlery_set",
        "dish_set",
        "serving_dish",
    )
    return any(marker in key for marker in tableware_markers)


def set_conflict_ablation_mode(mode: str):
    global _CONFLICT_ABLATION_MODE
    if mode not in _CONFLICT_ABLATION_CHOICES:
        raise ValueError(f"Unknown conflict ablation mode: {mode}")
    _CONFLICT_ABLATION_MODE = mode


class SmallConstraintSolver:
    def __init__(self):
        self.constraints = []
        self.wall = None
        # 这里不算near
        self.position_constraints = [
            self.place_align.__name__,
            self.against_edge.__name__,
            self.place_align_small.__name__,
            self.center.__name__,
        ]
        self.orientation_constraints = [
            self.align_wall.__name__,
            self.point_towards.__name__,
            self.against_wall.__name__,
            self.against_edge.__name__,
            self.point_to_edge.__name__,
        ]

    def build_wall_assignments(self) -> dict:
        """Small objects are optimized on local support regions, not room walls."""
        return {}

    def add_constraint(self, constraint, *args, **kwargs):
        self.constraints.append((constraint, args, kwargs))
    
    def remove_constraint(self, u, v, edge_type, edge_extra=None):
        """
        删除 self.constraints 中与 u、v 对应且类型为 edge_type 的约束
        """
        def get_id(obj):
            if isinstance(obj, dict):
                return obj.get('id', obj.get('description'))
            return str(obj)

        new_constraints = []
        for c, args, kwargs in self.constraints:
            c_name = c.__name__ if hasattr(c, "__name__") else str(c)
            if len(args) < 2:
                new_constraints.append((c, args, kwargs))
                continue
            if c_name in {"surround", "near_center"} and isinstance(args[1], list):
                new_constraints.append((c, args, kwargs))
                continue
            c_u, c_v = get_id(args[0]), get_id(args[1])

            if (c_u == u and c_v == v) and c_name == edge_type:
                if edge_extra is not None:
                    c_extra = args[2] if len(args) > 2 else None
                    if c_extra != edge_extra:
                        new_constraints.append((c, args, kwargs))
                        continue
                continue
            new_constraints.append((c, args, kwargs))
        self.constraints = new_constraints

    def build_graph(self):
        """
        构建约束图，节点是asset（dict），边是约束
        """
        G = nx.MultiDiGraph()

        def get_node_id(obj):
            if isinstance(obj, dict) and "id" in obj:
                return obj["id"]
            elif isinstance(obj, dict) and "description" in obj:
                return obj['description']
            else:
                return str(obj)

        for constraint, args, kwargs in self.constraints:
            obj_id = get_node_id(args[0])

            # surround 多对象约束
            if (constraint.__name__ == "surround" or constraint.__name__ == "near_center") and isinstance(args[1], list):
                # 兼容历史 near_center 输入：统一按 surround 图边处理
                max_distance = args[2] if len(args) > 2 else kwargs.get("max_distance", 1.0)
                look_mode = args[3] if len(args) > 3 else kwargs.get("look_mode", "axis")
                edge_kwargs = dict(kwargs)
                edge_update = edge_kwargs.get("update", True)

                G.add_node(obj_id)
                for other in args[1]:
                    other_id = get_node_id(other)
                    G.add_node(other_id)
                    G.add_edge(
                        other_id, obj_id,
                        type="surround",
                        fn=self.surround.__name__,
                        arg_ids=[other_id, obj_id],
                        extra_args=[max_distance, look_mode],
                        kwargs=edge_kwargs,
                        update=edge_update
                    )
                continue

            # 普通二元约束
            if len(args) > 1:
                other_id = get_node_id(args[1])
                extra_args = args[2:]
                G.add_node(obj_id)
                G.add_node(other_id)
                G.add_edge(
                    obj_id, other_id,
                    type=constraint.__name__,
                    fn=constraint.__name__,
                    arg_ids=[obj_id, other_id],
                    extra_args=extra_args,
                    kwargs=kwargs,
                    update=True
                )
            else:
                # 单对象约束
                extra_args = args[1:]  # 可能为空
                G.add_node(obj_id)
                G.add_edge(
                    obj_id, obj_id,
                    type=constraint.__name__,
                    fn=constraint.__name__,
                    arg_ids=[obj_id],
                    extra_args=extra_args,
                    kwargs=kwargs,
                    update=True
                )
        return G

    def postprocess_graph(self, G, existing_assets):
        """
        1. 确保每个物体只有一条 position 和 orientation 出边
        2. 根据中心率调整边的权重（组内归一化）
        """
        # --- Step 1: 约束唯一性 ---
        # for node in G.nodes:
        #     out_edges = list(G.out_edges(node, data=True, keys=True))
        #     pos_edges = [e for e in out_edges if e[3]["fn"] in self.position_constraints]
        #     ori_edges = [e for e in out_edges if e[3]["fn"] in self.orientation_constraints]
        #     for e in pos_edges[1:]:
        #         if G.has_edge(e[0], e[1], key=e[2]):
        #             G.remove_edge(e[0], e[1], key=e[2])
        #     for e in ori_edges[1:]:
        #         if G.has_edge(e[0], e[1], key=e[2]):
        #             G.remove_edge(e[0], e[1], key=e[2])

        # --- Step 2: 中心率计算 ---
        centrality = nx.degree_centrality(G)

        # --- Step 3: 按组归一化 ---
        groups = self.group_assets(G, existing_assets)
        node_to_norm_centrality = {}
        for group in groups:
            csum = sum(centrality.get(n, 0.0) for n in group)
            size = len(group)
            if csum == 0:
                # 全部均分
                for n in group:
                    node_to_norm_centrality[n] = 1.0
            else:
                for n in group:
                    node_to_norm_centrality[n] = (centrality.get(n, 0.0) / csum) * size

        return G, groups, node_to_norm_centrality

    def cal_weight(self, G, groups, centrality):

        # --- Step 4: 更新权重 ---
        for u, v, k, data in G.edges(keys=True, data=True):
            base_weight = 2.0 if data["type"] in self.orientation_constraints else 1.0
            data["weight"] = centrality.get(u, 0.0) * base_weight

        return G

    
    def detect_graph_conflicts(self, G, groups, init_assets, id_to_key, bound, verbose=True, first_pass=True):
        """
        检测约束图中的结构性冲突。
        包括：
            1. 有向循环 (位置/朝向冲突)
            2. 距离矛盾
            3. 角度冲突
            4. 重复或互斥约束
        并新增：基于图约束的语义一致性检测 (不依赖 pos/phy)：
            - 语义相关物体必须存在特定类型的边（如 near）
            - 同组物体应 against 同一个 wall（如果规则要求）
            - 每个节点必须至少有一个 position_constraints 类型的出边和一个 orientation_constraints 类型的出边
            - 不使用 init_assets 中的 pos/phy（因为这是优化前的检测）
        """
        removed_edges = []
        log_lines = []
        ablation_mode = _CONFLICT_ABLATION_MODE
        if ablation_mode == "wo_conflict_module":
            return G, removed_edges, [], log_lines
        disable_logic_error = ablation_mode == "wo_logic_error"
        disable_semantic_group = ablation_mode == "wo_semantic_group"
        disable_outgoing_completeness = ablation_mode == "wo_outgoing_completeness"
        logic_conflict_types = {
            "direction_cycle",
            "distance_inconsistent",
            "exclusive_rot_conflict",
            "multi_wall_conflict",
            "multi_edge_conflict",
            "aggregate_cross_edge_conflict",
            "aggregate_wall_conflict",
            "direction_conflict",
        }
        pos_constraint_types = ["near", "against_wall", "place_align", "against_edge", "place_align_small", "surround", "center"]
        orient_constraint_types = ["align_wall", "align_with", "point_towards", "against_wall", "against_edge", "point_to_edge", "surround"]

        def detect_conflicts_in_subgraphs(G, verbose=False):
            """
            仅在每个连通子图内部检测语义冲突。
            """
            all_conflicts = []

            for i, group_nodes in enumerate(groups):
                logged_conflict_keys = set()
                # 深拷贝整个图（保留所有全局节点，如 wall、door、floor 等）
                subG = G.copy()

                # 删除所有「是数字」且不在当前组的节点
                for node in list(subG.nodes):
                    if isinstance(node, (int, np.integer)) and node not in group_nodes:
                        subG.remove_node(node)
                log_start = len(log_lines)
                conflicts = _detect_conflicts(subG, logged_conflict_keys)
                if disable_logic_error or disable_semantic_group:
                    filtered_conflicts = []
                    for c in conflicts:
                        c_type = c.get("type")
                        if disable_logic_error and c_type in logic_conflict_types:
                            continue
                        if disable_semantic_group and c_type == "semantic_conflict":
                            continue
                        filtered_conflicts.append(c)
                    conflicts = filtered_conflicts
                    new_logs = log_lines[log_start:]
                    kept_logs = []
                    for line in new_logs:
                        if disable_logic_error and "[LogicError]" in line:
                            continue
                        if disable_semantic_group and ("[SemanticCommon]" in line or "[SemanticCheckError]" in line):
                            continue
                        kept_logs.append(line)
                    log_lines[log_start:] = kept_logs
                if conflicts:
                    if verbose:
                        print(f"[Group {i}] Detected {len(conflicts)} conflicts.")
                    all_conflicts.extend(conflicts)
                else:
                    if verbose:
                        print(f"[Group {i}] OK — no conflicts.")
            
            return all_conflicts
            
        def _edge_items_between(a, b):
            """返回 a->b 和 b->a 所有边的数据列表 (u,v,data)"""
            items = []
            if G.has_edge(a, b):
                for key, data in G.get_edge_data(a, b).items():
                    items.append((a, b, key, data))
            if G.has_edge(b, a):
                for key, data in G.get_edge_data(b, a).items():
                    items.append((b, a, key, data))
            return items

        def _graph_edge_first_extra_arg(extra_args):
            if not extra_args:
                return None
            try:
                return extra_args[0]
            except (TypeError, IndexError, KeyError):
                return None

        def _has_edge_type_between(a, b, etype):
            for _, _, _, data in _edge_items_between(a, b):
                if data.get("type") == etype:
                    return True
            return False

        def _has_any_edge_type_between(a, b, types):
            for _, _, _, data in _edge_items_between(a, b):
                if data.get("type") in types:
                    return True
            return False

        def _collect_out_edge_types(node):
            return [d.get("type") for _, _, d in G.out_edges(node, data=True)]

        def _get_against_walls(node):
            """
            返回 node 对应的 against_wall 目标名集合（可能是墙节点名或边的 target id）
            这里假设 against_wall 的边目标 v 是墙的 node id / name
            """
            walls = []
            for _, v, d in G.out_edges(node, data=True):
                if d.get("type") == "against_wall":
                    walls.append(v)
            return set(walls)

        def _get_edge_extra_arg(a, b, etype, default=None):
            """若存在 a->b 或 b->a 的 etype 边并且包含 extra_args[0]，返回第一个找到的值"""
            for _, _, _, data in _edge_items_between(a, b):
                if data.get("type") == etype:
                    args = data.get("extra_args", None)
                    if args and len(args) > 0:
                        return args[0]
                    return default
            return default

        def _detect_conflicts(G, logged_conflict_keys, node_order=None):
            conflicts = []

            # --- 1️⃣ 方向循环检测 ---
            direction_edges = [
                (u, v, d["type"]) for u, v, d in G.edges(data=True)
                if d.get("type") in ["place_align", "place_align_small"]
            ]
            # if small_ids:
            #     direction_edges = [
            #         (u, v, t) for (u, v, t) in direction_edges
            #         if (u not in small_ids and v not in small_ids)
            #     ]
            G_dir = nx.DiGraph()
            for u, v, etype in direction_edges:
                G_dir.add_edge(u, v, type=etype)
            try:
                cycles = list(nx.simple_cycles(G_dir))
                for cyc in cycles:
                    if len(cyc) > 1:
                        u, v = cyc[-1], cyc[0]
                        etype = G_dir.edges[u, v]["type"]
                        conflicts.append({
                            "type": "direction_cycle",
                            "edge": (u, v, etype)
                        })
                        if verbose:
                            u_name = id_to_key.get(u, u)
                            v_name = id_to_key.get(v, v)
                            log_lines.append(f"[LogicError] Direction cycle detected: {' → '.join(map(str,cyc))}, removing edge {u_name}->{v_name}")
            except Exception:
                pass

            # --- 2️⃣ 距离冲突检测 (near edges) ---
            # ``extra_args`` may be present but empty (); ``d.get("extra_args", [None])[0]`` then raises.
            dist_edges = []
            for u, v, d in G.edges(data=True):
                if d.get("type") != "near":
                    continue
                ea = d.get("extra_args")
                if not ea:
                    dval = None
                else:
                    dval = ea[0]
                dist_edges.append((u, v, dval))
            # if small_ids:
            #     dist_edges = [
            #         (u, v, dval) for (u, v, dval) in dist_edges
            #         if (u not in small_ids and v not in small_ids)
            #     ]
            if dist_edges:
                Gd = nx.Graph()
                for (a, b, dval) in dist_edges:
                    if dval is not None:
                        Gd.add_edge(a, b, weight=dval)
                for (a, b, dval) in dist_edges:
                    if a in Gd.nodes and b in Gd.nodes:
                        try:
                            sp = nx.shortest_path_length(Gd, a, b, weight='weight')
                            if abs(sp - (dval if dval is not None else sp)) > 1e-3:
                                conflicts.append({
                                    "type": "distance_inconsistent",
                                    "edge": (a, b, "near")
                                })
                                if verbose:
                                    a_name = id_to_key.get(a, a)
                                    b_name = id_to_key.get(b, b)
                                    log_lines.append(f"[LogicError] distance({a_name},{b_name})={dval}, but path={sp}")
                        except nx.NetworkXNoPath:
                            continue

            # --- 3️⃣ 互斥约束检测（墙/对齐） ---
            direction_to_wall = {
                "right": "right_wall(facing -x)",
                "left": "left_wall(facing +x)",
                "up": "back_wall(facing -y)",
                "down": "front_wall(facing +y)"
            }
            wall_to_direction = {v: k for k, v in direction_to_wall.items()}
            dir_alias = {"front": "down", "back": "up"}
            wall_bound = None
            if bound is not None and len(bound) >= 4:
                wall_bound = [bound[1]-bound[0], bound[3]-bound[2]]
            wall_to_axis = {
                "right_wall(facing -x)": 1,
                "left_wall(facing +x)": 1,
                "back_wall(facing -y)": 0,
                "front_wall(facing +y)": 0
            }
            node_iter = node_order if node_order else list(G.nodes)
            out_edges_map = {}
            in_edges_map = {}
            wall_edges_map = {}
            types_present_map = {}
            against_dirs_map = {}
            place_dirs_map = {}
            place_in_edges_map = {}
            for u in node_iter:
                out_edges = [(u, v, k, d) for u, v, k, d in G.edges(u, keys=True, data=True)]
                out_edges_map[u] = out_edges
                in_edges = [(u2, v2, k2, d2) for u2, v2, k2, d2 in G.in_edges(u, keys=True, data=True)]
                in_edges_map[u] = in_edges
                if not out_edges:
                    wall_edges_map[u] = []
                    types_present_map[u] = []
                    against_dirs_map[u] = []
                    place_dirs_map[u] = []
                    place_in_edges_map[u] = []
                    continue

                wall_edges = []
                for _, v, _, d in out_edges:
                    t = d.get("type")
                    if t in {"align_wall", "against_wall", "place_align", "against_edge", "place_align_small"}:
                        if t in {"place_align", "place_align_small"}:
                            direction = _graph_edge_first_extra_arg(d.get("extra_args"))
                        elif t == "against_wall":
                            direction = wall_to_direction.get(v, None)
                        elif t == "against_edge":
                            direction = _graph_edge_first_extra_arg(d.get("extra_args"))
                        else:
                            direction = None
                        if direction in dir_alias:
                            direction = dir_alias[direction]
                        wall_edges.append((v, t, direction))

                wall_edges_map[u] = wall_edges
                types_present_map[u] = [t for _, t, _ in wall_edges]
                against_dirs_map[u] = [dir for _, t, dir in wall_edges if t in {"against_wall", "against_edge"} and dir is not None]
                # collect place_align directions where THIS node is the second object (dst)
                place_in_edges = []
                for src, dst, k, d in in_edges:
                    t = d.get("type")
                    if t in {"place_align", "place_align_small"}:
                        # if src in small_ids:
                        #     continue
                        direction = _graph_edge_first_extra_arg(d.get("extra_args"))
                        if direction in dir_alias:
                            direction = dir_alias[direction]
                        if direction is not None:
                            place_in_edges.append((src, dst, k, d, direction))
                place_in_edges_map[u] = place_in_edges
                place_dirs_map[u] = [direction for _, _, _, _, direction in place_in_edges]

            for u in node_iter:
                # if u in small_ids:
                #     continue
                out_edges = out_edges_map.get(u, [])
                if not out_edges:
                    continue

                wall_edges = wall_edges_map.get(u, [])
                types_present = types_present_map.get(u, [])

                # rotation constraints de-duplication for outgoing edges of the same asset:
                # against_wall > align_wall > point_towards
                rot_priority = {"against_wall": 0, "align_wall": 1, "point_towards": 2}
                rot_types = set(rot_priority.keys())
                rot_type_set = {d.get("type") for _, _, _, d in out_edges if d.get("type") in rot_types}
                if len(rot_type_set) > 1:
                    keep_type = min(rot_type_set, key=lambda t: rot_priority[t])
                    edge_iter = reversed(out_edges) if first_pass else out_edges
                    for _, v, k, d in edge_iter:
                        edge_type = d.get("type")
                        if edge_type in rot_types and edge_type != keep_type:
                            conflicts.append({
                                "type": "exclusive_rot_conflict",
                                "edge": (u, v, edge_type),
                                "reason": f"duplicate rot constraints, keep {keep_type}"
                            })
                            if verbose:
                                u_name = id_to_key.get(u, u)
                                v_name = id_to_key.get(v, v)
                                print(f"[LogicError] Deplicate_rot_constraints: {u_name}-{v_name}, remove {edge_type}, keep {keep_type}")
                                log_lines.append(f"[LogicError] Deplicate_rot_constraints: {u_name}-{v_name}, remove {edge_type}, keep {keep_type}")
                            break

                against_targets = [v for v, t, _ in wall_edges if t == "against_wall"]
                if len(set(against_targets)) > 1:
                    if first_pass:
                        for _, v, k, d in reversed(out_edges):
                            if d.get("type") == "against_wall":
                                conflicts.append({
                                    "type": "multi_wall_conflict",
                                    "edge": (u, v, d["type"]),
                                    "reason": "multiple against_wall to different walls"
                                })
                                if verbose:
                                    u_name = id_to_key.get(u, u)
                                    v_name = id_to_key.get(v, v)
                                    print(f"[LogicError] multi_wall_conflict: {u_name} -> {v_name}, remove against_wall")
                                    log_lines.append(f"[LogicError] multi_wall_conflict: {u_name} -> {v_name}, remove against_wall")
                                break
                    else:
                        for _, v, k, d in out_edges:
                            if d.get("type") == "against_wall":
                                conflicts.append({
                                    "type": "multi_wall_conflict",
                                    "edge": (u, v, d["type"]),
                                    "reason": "multiple against_wall to different walls"
                                })
                                if verbose:
                                    u_name = id_to_key.get(u, u)
                                    v_name = id_to_key.get(v, v)
                                    print(f"[LogicError] multi_wall_conflict: {u_name} -> {v_name}, remove against_wall")
                                    log_lines.append(f"[LogicError] multi_wall_conflict: {u_name} -> {v_name}, remove against_wall")
                                break

                # --- against_edge multi-edge conflict ---
                against_edge_dirs = [dir for _, t, dir in wall_edges if t == "against_edge" and dir is not None]
                if len(set(against_edge_dirs)) > 1:
                    if first_pass:
                        for _, v, k, d in reversed(out_edges):
                            if d.get("type") == "against_edge":
                                conflicts.append({
                                    "type": "multi_edge_conflict",
                                    "edge": (u, v, d["type"]),
                                    "reason": "multiple against_edge to different edges"
                                })
                                if verbose:
                                    u_name = id_to_key.get(u, u)
                                    v_name = id_to_key.get(v, v)
                                    print(f"[LogicError] multi_edge_conflict: {u_name} -> {v_name}, remove against_edge")
                                    log_lines.append(f"[LogicError] multi_edge_conflict: {u_name} -> {v_name}, remove against_edge")
                                break
                    else:
                        for _, v, k, d in out_edges:
                            if d.get("type") == "against_edge":
                                conflicts.append({
                                    "type": "multi_edge_conflict",
                                    "edge": (u, v, d["type"]),
                                    "reason": "multiple against_edge to different edges"
                                })
                                if verbose:
                                    u_name = id_to_key.get(u, u)
                                    v_name = id_to_key.get(v, v)
                                    print(f"[LogicError] multi_edge_conflict: {u_name} -> {v_name}, remove against_edge")
                                    log_lines.append(f"[LogicError] multi_edge_conflict: {u_name} -> {v_name}, remove against_edge")
                                break

                # aggregate wall occupy check (uses init_assets keyed by name)
                wall_occupy = defaultdict(float)
                for u_, v_, d_ in G.edges(data=True):
                    if d_.get("type") == "against_wall" and v_ in wall_to_axis:
                        a_ = wall_to_axis[v_]
                        asset_key = id_to_key.get(u_)
                        if asset_key in init_assets:
                            wall_occupy[v_] += init_assets[asset_key].get('bbox', [0, 0, 0])[a_]

                if wall_bound is not None:
                    for wall_name, total_len in wall_occupy.items():
                        axis = wall_to_axis[wall_name]
                        if total_len > wall_bound[axis]:
                            conflicts.append({
                                "type": "aggregate_wall_conflict",
                                "wall": wall_name,
                                "reason": f"Total occupied length {total_len:.3f} exceeds bound {wall_bound[axis]:.3f}"
                            })
                            if verbose:
                                log_lines.append(f"[LogicError] aggregate_wall_conflict: {wall_name} exceeds {wall_bound[axis]:.3f}")

                place_dirs = place_dirs_map.get(u, [])
                against_dirs = against_dirs_map.get(u, [])
                if place_dirs and against_dirs:
                    overlap = set(place_dirs) & set(against_dirs)
                    if overlap:
                        # remove the incoming place_align edge (other -> u) that conflicts with u's against_wall
                        for src, dst, k, d, direction in reversed(place_in_edges_map.get(u, [])):
                            if d.get("type") in {"place_align", "place_align_small"} and direction in overlap:
                                conflicts.append({
                                    "type": "direction_conflict",
                                    "edge": (src, dst, d.get("type", "place_align")),
                                    "reason": f"direction mismatch: place_align {place_dirs} vs against_wall/edge {against_dirs}"
                                })
                                if verbose:
                                    u_name = id_to_key.get(src, src)
                                    v_name = id_to_key.get(dst, dst)
                                    print(f"[LogicError] direction_conflict: remove place_align: {u_name}-{v_name}:{place_dirs}")
                                    log_lines.append(f"[LogicError] direction_conflict: remove place_align: {u_name}-{v_name}:{place_dirs}")
                                break
                            
            # --- Cross-edge aggregate conflict for open support regions ---
            edge_to_axis = {"front": 0, "back": 0, "left": 1, "right": 1}
            region_entries = defaultdict(list)
            for u_, v_, k_, d_ in G.edges(keys=True, data=True):
                if d_.get("type") != "against_edge":
                    continue
                edge_name = _graph_edge_first_extra_arg(d_.get("extra_args"))
                if edge_name not in edge_to_axis:
                    continue
                bounds = _parse_region_bounds_node(v_)
                if bounds is None:
                    continue
                asset_key = id_to_key.get(u_)
                if asset_key not in init_assets:
                    continue
                bbox = init_assets[asset_key].get("bbox", [0.0, 0.0, 0.0])
                axis = edge_to_axis[edge_name]
                try:
                    occupied_len = float(bbox[axis])
                except Exception:
                    occupied_len = 0.0
                region_entries[str(v_)].append({
                    "node": u_,
                    "region_node": v_,
                    "edge": edge_name,
                    "occupied_len": occupied_len,
                    "bounds": bounds,
                })

            for region_key, entries in region_entries.items():
                bounds = entries[0]["bounds"] if entries else None
                width, height = _region_width_height(bounds)
                if width is None or height is None or width <= 0 or height <= 0:
                    continue
                fb_entries = [e for e in entries if e["edge"] in ("front", "back")]
                lr_entries = [e for e in entries if e["edge"] in ("left", "right")]
                if (
                    len(fb_entries) < CROSS_EDGE_MIN_COUNT_PER_AXIS
                    or len(lr_entries) < CROSS_EDGE_MIN_COUNT_PER_AXIS
                ):
                    continue
                fb_occupy = sum(e["occupied_len"] for e in fb_entries)
                lr_occupy = sum(e["occupied_len"] for e in lr_entries)
                fb_ratio = fb_occupy / width
                lr_ratio = lr_occupy / height
                if (
                    fb_ratio <= CROSS_EDGE_OCCUPANCY_RATIO
                    or lr_ratio <= CROSS_EDGE_OCCUPANCY_RATIO
                ):
                    continue

                remove_axis = "fb" if width <= height else "lr"
                remove_entries = fb_entries if remove_axis == "fb" else lr_entries
                remove_edges = "front/back" if remove_axis == "fb" else "left/right"
                for entry in remove_entries:
                    conflicts.append({
                        "type": "aggregate_cross_edge_conflict",
                        "edge": (entry["node"], entry["region_node"], "against_edge"),
                        "region": region_key,
                        "reason": (
                            f"front/back occupancy {fb_occupy:.3f}/{width:.3f} "
                            f"({fb_ratio:.2f}) and left/right occupancy "
                            f"{lr_occupy:.3f}/{height:.3f} ({lr_ratio:.2f}) are both high"
                        ),
                    })
                    if verbose:
                        u_name = id_to_key.get(entry["node"], entry["node"])
                        print(
                            f"[LogicError] aggregate_cross_edge_conflict: region {region_key} "
                            f"front/back={fb_ratio:.2f}, left/right={lr_ratio:.2f}; "
                            f"remove against_edge on shorter edge pair ({remove_edges}): {u_name}"
                        )
                        log_lines.append(
                            f"[LogicError] aggregate_cross_edge_conflict: region {region_key} "
                            f"front/back={fb_ratio:.2f}, left/right={lr_ratio:.2f}; "
                            f"remove against_edge on shorter edge pair ({remove_edges}): {u_name}"
                        )

            # --- 4️⃣ 语义组内图约束检测（基于边，不依赖 pos/phy） ---
            # 定义语义组及组内应有的边类型要求（仅示例，可扩展）
            groups = {
                "bedroom": {
                    "keywords": ["bed", "nightstand", "night_stand", "bedside_table", "lamp"],
                    # list of required pair rules: (predicate_on_keys, required_edge_types, optional_range)
                    "pair_rules": [
                        # nightstand 与 bed: 必须有 against_wall 且同一面墙
                        (lambda ka, kb: ("nightstand" in ka or "bedside_table" in ka or "night_stand" in ka) and "bed" in kb,
                        ["against_wall"], None, "same_against_wall"),
                        # (lambda ka, kb: ("nightstand" in ka or "night_stand" in ka) and "bed" in kb,
                        # ["against_wall"], None, "same_against_wall"),
                    ]
                },
                "desk_area": {
                    "keywords": ["desk", "chair", "lamp", "bookshelf"],
                    "pair_rules": [
                        # chair - desk: 必须有 near 或 place_align
                        (lambda ka, kb:"chair" in ka and "desk" in kb,
                        ["near"], (0.0, 1.0), None),
                        # bookshelf - desk: 若有 near 则距离不应超过 1.5（若有 extra_args）
                        # (lambda ka, kb: "bookshelf" in ka and "desk" in kb,
                        # ["near"], (0.0, 1.5), None),
                        # (lambda ka, kb: ka.startswith("bookshelf") and kb.startswith("bookshelf"),
                        # ["near"], 1.0, 5.0),
                        # (lambda ka, kb: ka.startswith("bookshelf"),
                        # ["against_wall"], None, None),
                    ]
                },
                "entertainment": {
                    "keywords": ["sofa", "tv", "coffee_table", "armchair", "tv_stand"],
                    "pair_rules": [
                        # sofa - tv: 必须有 near，且如果有 extra_args 则在 [1.5, 3.5]
                        # (lambda ka, kb: "sofa" in ka and "tv" in kb,
                        # ["near"], (1.5, 3.5), None),
                        # sofa - tv_stand: 必须有 place_align
                        (lambda ka, kb: "sofa" in ka and "tv" in kb,
                        ["place_align"], None, None),
                        # (lambda ka, kb: "sofa" in ka and "tv" in kb,
                        # ["point_towards"], None, None),
                        # coffee_table - sofa: 必须有 near
                        # (lambda ka, kb: "coffee_table" in ka and "sofa" in kb,
                        # ["near"], (0.0, 1.2), None),
                        # (lambda ka, kb: "coffee_table" in ka and "sofa" in kb,
                        # ["place_align"], (0.0, 1.2), None),
                    ]
                },
                "dining_area": {
                    "keywords": ["dining_table", "dining_chair", "sideboard"],
                    "pair_rules": [
                        # (lambda ka, kb: ka.startswith("dining_chair") and kb.startswith("dining_table"),
                        # ["near"], (0.0, 0.8), None),
                        # (lambda ka, kb: ka.startswith("sideboard"),
                        # ["against_wall"], None, None),
                        (lambda ka, kb: ka.startswith("dining_table"),
                        [], None, "not_against_wall"),
                    ]
                }
            }
            
            # 遍历每个组中的节点对，仅基于图边
            satisfied_nodes = set()
            nodes = list(G.nodes)
            # print(nodes)
            for i in range(len(nodes)):
                a = nodes[i]
                if a not in id_to_key:
                    continue
                key_a = id_to_key[a]
                if a in satisfied_nodes:
                    continue

                for j in range(i + 1, len(nodes)):
                    b = nodes[j]
                    key_b = id_to_key.get(b, "")

                    for group_name, group in groups.items():
                        keywords = group["keywords"]
                        # 修改关键字匹配为 in
                        if not (any(k in key_a for k in keywords) and any(k in key_b for k in keywords)):
                            continue

                        for predicate, required_types, allowed_range, special in group["pair_rules"]:
                            try:
                                matched = False
                                if predicate(key_a, key_b):
                                    a2, b2 = a, b
                                    key_a2, key_b2 = key_a, key_b
                                    matched = True
                                elif predicate(key_b, key_a):
                                    a2, b2 = b, a
                                    key_a2, key_b2 = key_b, key_a
                                    matched = True

                                if not matched:
                                    continue

                                # --- 检查关系是否满足 ---
                                relation_ok = False
                                for rtype in required_types:
                                    for _, _, _, data in _edge_items_between(a2, b2):
                                        if data.get("type") == rtype:
                                            if allowed_range is not None:
                                                val = _graph_edge_first_extra_arg(data.get("extra_args"))
                                                if val is None:
                                                    continue
                                                min_r, max_r = allowed_range
                                                if min_r <= val <= max_r:
                                                    relation_ok = True
                                                    break
                                            else:
                                                relation_ok = True
                                                break
                                    if relation_ok:
                                        break

                                if relation_ok:
                                    satisfied_nodes.add(a2)
                                    satisfied_nodes.add(b2)
                                    break

                                # --- 处理 special 规则 ---
                                if special == "same_against_wall":
                                    walls_a = _get_against_walls(a2)
                                    walls_b = _get_against_walls(b2)
                                    # print(walls_a, walls_b)
                                    if not walls_a or not walls_b or not (walls_a & walls_b):
                                        reason = f"{key_a2} and {key_b2} are expected to be against the same wall"
                                        conflict = {
                                            "type": "semantic_conflict",
                                            "nodes": (a2, b2),
                                            "reason": reason
                                        }
                                        ckey = ("semantic_conflict", (a2, b2), reason)
                                        if ckey not in logged_conflict_keys:
                                            logged_conflict_keys.add(ckey)
                                            conflicts.append(conflict)
                                            if verbose:
                                                log_lines.append(
                                                    f"[SemanticCommon] {key_a2} vs {key_b2} not against same wall"
                                                )
                                    continue

                                if special == "not_against_wall":
                                    # dining_table 不应贴墙
                                    walls = _get_against_walls(a2)
                                    if walls:
                                        reason = f"{key_a2} should not be against wall"
                                        conflict = {
                                            "type": "semantic_conflict",
                                            "nodes": (a2, walls),
                                            "reason": reason
                                        }
                                        ckey = ("semantic_conflict", (a2, walls), reason)
                                        if ckey not in logged_conflict_keys:
                                            logged_conflict_keys.add(ckey)
                                            conflicts.append(conflict)
                                            if verbose:
                                                log_lines.append(
                                                    f"[SemanticCommon] {key_a2} should not be against wall"
                                                )
                                    continue

                                # 普通的 required_types 检查：至少有一种 required_types 的边存在，并且若指定了 allowed_range 则 extra_args 要满足范围
                                if required_types:
                                    valid = False
                                    for rtype in required_types:
                                        # 检查 a2 <-> b2 是否存在 rtype 边
                                        for _, _, _, data in _edge_items_between(a2, b2):
                                            if data.get("type") == rtype:
                                                # 如果有范围限制并且 extra_args 可用，检查范围
                                                if allowed_range is not None:
                                                    val = _graph_edge_first_extra_arg(data.get("extra_args"))
                                                    if val is None:
                                                        # 没有距离信息，视为不满足范围（选择上可以将其视为满足或不满足，当前视为不满足）
                                                        continue
                                                    min_r, max_r = allowed_range
                                                    if min_r <= val <= max_r:
                                                        valid = True
                                                        break
                                                    else:
                                                        # 存在该类型但值超出
                                                        continue
                                                else:
                                                    valid = True
                                                    break
                                        if valid:
                                            break
                                    if not valid:
                                        reason = f"{key_a2} vs {key_b2} missing/invalid relation {required_types}"
                                        conflict = {
                                            "type": "semantic_conflict",
                                            "nodes": (a2, b2),
                                            "reason": reason
                                        }
                                        ckey = ("semantic_conflict", (a2, b2), reason)
                                        if ckey not in logged_conflict_keys:
                                            logged_conflict_keys.add(ckey)
                                            conflicts.append(conflict)
                                            if verbose:
                                                log_lines.append(
                                                    f"[SemanticCommon] {key_a2} vs {key_b2} missing/invalid relation {required_types}"
                                                )
                            except Exception as e:
                                if verbose:
                                    log_lines.append(f"[SemanticCheckError] group={group_name}, pair=({key_a},{key_b}), err={e}")
                                continue

            return conflicts

        print("start conflict detection...")

        # --- 主循环：冲突检测 + 删除最后边 ---
        while True:
            conflicts = detect_conflicts_in_subgraphs(G, verbose=verbose)
            if not conflicts:
                if verbose:
                    print("[OK] No structural conflicts detected.")
                break

            # 标记是否有 edge 冲突处理过
            edge_removed = False

            for c in conflicts:
                if "edge" in c:
                    u, v, edge_type = c["edge"]
                    edges_to_remove = [
                        (key, data) for key, data in G.get_edge_data(u, v, default={}).items()
                        if data.get("type") == edge_type
                    ]
                    if edges_to_remove:
                        k, data = edges_to_remove[0]
                        G.remove_edge(u, v, key=k)
                        removed_edges.append((u, v, edge_type))
                        edge_removed = True
                        edge_extra = None
                        extra_args = data.get("extra_args")
                        if extra_args:
                            edge_extra = extra_args[0]
                        self.remove_constraint(u, v, edge_type, edge_extra=edge_extra)
                        if verbose:
                            print(f"[Auto-Fix] Removed one conflicting edge: {u} → {v} ({edge_type})")
                else:
                    if verbose:
                        # log_msg = f"[Auto-Fix] Conflict requires manual fix, logged: {c}"
                        print(f"[Auto-Fix] Conflict requires manual fix, logged: {c}")
                        # log_lines.append(log_msg)
            # 如果这次循环没有删除任何 edge，就跳出循环（避免一直卡住）
            if not edge_removed:
                break

        low_outdegree_nodes = []
        if not disable_outgoing_completeness:
            removed_against_edge_nodes = {
                u for u, _v, edge_type in removed_edges if edge_type == "against_edge"
            }
            # --- 5️⃣ 出边类别完整性检测（每个节点必须有 position + orientation 类型的出边） ---
            for node in G.nodes():
                if node in removed_against_edge_nodes:
                    continue
                out_types = _collect_out_edge_types(node)
                if "against_wall" in out_types:
                    continue
                has_pos = any(t in pos_constraint_types for t in out_types)
                has_orient = any(t in orient_constraint_types for t in out_types)
                
                if node not in id_to_key:
                    continue
                node_name = id_to_key[node]

                needs_against_edge = (
                    _requires_against_edge_for_tableware(node_name)
                    and "against_edge" not in out_types
                )

                if not has_pos or not has_orient or needs_against_edge:
                    reason_parts = []
                    if not has_pos:
                        reason_parts.append("missing position-type outgoing constraint")
                    if not has_orient:
                        reason_parts.append("missing orientation-type outgoing constraint")
                    if needs_against_edge:
                        reason_parts.append("missing against_edge constraint for plate/tableware")
                    reason = "; ".join(reason_parts)
                    conflicts.append({
                        "type": "constraint_completeness",
                        "node": node,
                        "reason": f"{node_name}: {reason}"
                    })
                    low_outdegree_nodes.append(node)
                    if verbose:
                        log_lines.append(f"[ConstraintCompleteness] {node_name}: {reason}")
       
        print(log_lines)
        return G, removed_edges, low_outdegree_nodes, log_lines

        
    def update_constraints(self, G, assets, init_assets, id_to_key):
        edge_weights = {}
        edge_update_flags = {}

        def _edge_key(u, v, fn_name, extra_args=None):
            extra_args = extra_args or []
            if fn_name in {"against_edge", "point_to_edge"} and extra_args:
                return (u, v, fn_name, str(extra_args[0]))
            return (u, v, fn_name)

        for u, v, k, data in G.edges(keys=True, data=True):
            update = data['update']
            fn_name = data['type']
            weight = data.get("weight", data.get("kwargs", {}).get("weight", 1.0))
            key = _edge_key(u, v, fn_name, data.get("extra_args"))
            edge_weights[key] = weight
            edge_update_flags[key] = update

        def _resolve_asset(arg):
            if isinstance(arg, dict) and "id" in arg:
                arg_id = arg["id"]
                if arg_id not in id_to_key:
                    return None
                return assets[id_to_key[arg_id]]
            return arg

        updated_constraints = []
        key_in = {}
        init_asset_ids = {asset["id"] for asset in init_assets.values()}

        # --- Step 2: 遍历 constraints 更新 args 和 weight ---
        for fn, args, kwargs in self.constraints:
            fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
            kwargs_new = dict(kwargs)
            updated = False

            if fn_name in {"surround", "near_center"} and len(args) >= 2 and isinstance(args[1], list):
                center_asset = _resolve_asset(args[0])
                if center_asset is None:
                    continue
                center_id = center_asset.get("id", center_asset.get("description"))
                if center_id is None:
                    continue

                surrounding_assets = []
                surrounding_ids = []
                for obj in args[1]:
                    asset = _resolve_asset(obj)
                    if asset is None:
                        continue
                    asset_id = asset.get("id", asset.get("description"))
                    if asset_id is None:
                        continue
                    surrounding_assets.append(asset)
                    surrounding_ids.append(asset_id)
                if not surrounding_assets:
                    continue

                new_args = [center_asset, surrounding_assets]
                new_args.extend(args[2:])

                # surround 使用每个成员到中心的边权重均值回写
                surround_edge_keys = [(sid, center_id, "surround") for sid in surrounding_ids if (sid, center_id, "surround") in edge_weights]
                if not surround_edge_keys:
                    continue

                dedup_key = ("surround", center_id, tuple(sorted(surrounding_ids, key=str)))
                if dedup_key in key_in:
                    continue

                mean_weight = sum(edge_weights[k] for k in surround_edge_keys) / len(surround_edge_keys)
                if "weight" not in kwargs_new:
                    kwargs_new["weight"] = 1.0

                if any(edge_update_flags[k] is False for k in surround_edge_keys):
                    kwargs_new["weight"] = max(mean_weight, 1.0)
                elif center_id in init_asset_ids:
                    kwargs_new["weight"] = max(min(mean_weight, 1.5), 1.0)
                else:
                    kwargs_new["weight"] = max(kwargs_new["weight"] * 0.95, 1.0)

                updated = True
                key_in[dedup_key] = True
                fn_used = self.surround if fn_name == "surround" else fn
            else:
                flag = False
                new_args = []
                for arg in args:
                    asset0 = _resolve_asset(arg)
                    if asset0 is None and isinstance(arg, dict) and "id" in arg:
                        flag = True
                        break
                    new_args.append(asset0)
                if flag or len(new_args) < 2:
                    continue

                u_obj = new_args[0]
                v_obj = new_args[1]
                u_id = u_obj["id"] if isinstance(u_obj, dict) and "id" in u_obj else u_obj.get("description") if isinstance(u_obj, dict) else str(u_obj)
                v_id = v_obj["id"] if isinstance(v_obj, dict) and "id" in v_obj else v_obj.get("description") if isinstance(v_obj, dict) else str(v_obj)
                key_tuple = _edge_key(u_id, v_id, fn_name, args[2:])
                if key_tuple not in edge_weights or key_tuple in key_in:
                    continue

                if "weight" not in kwargs_new:
                    kwargs_new["weight"] = 1.0
                if edge_update_flags[key_tuple] is False:
                    kwargs_new["weight"] = max(edge_weights[key_tuple], 1.0)
                elif u_id in init_asset_ids:
                    kwargs_new["weight"] = max(min(edge_weights[key_tuple], 1.5), 1.0)
                else:
                    kwargs_new["weight"] = max(kwargs_new["weight"] * 0.95, 1.0)
                updated = True
                key_in[key_tuple] = True
                fn_used = fn

            if updated:
                updated_constraints.append((fn_used, tuple(new_args), kwargs_new))

        # --- Step 5: 更新 self.constraints ---
        self.constraints = updated_constraints

    def update_constraints_woweight(self, G, assets, id_to_key):
        edge_weights = {}
        edge_update_flags = {}

        def _edge_key(u, v, fn_name, extra_args=None):
            extra_args = extra_args or []
            if fn_name in {"against_edge", "point_to_edge"} and extra_args:
                return (u, v, fn_name, str(extra_args[0]))
            return (u, v, fn_name)

        for u, v, k, data in G.edges(keys=True, data=True):
            update = data['update']
            fn_name = data['type']
            weight = 1.0
            key = _edge_key(u, v, fn_name, data.get("extra_args"))
            edge_weights[key] = weight
            edge_update_flags[key] = update
        
        def _resolve_asset(arg):
            if isinstance(arg, dict) and "id" in arg:
                arg_id = arg["id"]
                if arg_id not in id_to_key:
                    return None
                return assets[id_to_key[arg_id]]
            return arg

        updated_constraints = []
        key_in = {}
        # --- Step 2: 遍历 constraints 更新 args 和 weight ---
        for fn, args, kwargs in self.constraints:
            fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
            kwargs_new = dict(kwargs)
            updated = False

            if fn_name in {"surround", "near_center"} and len(args) >= 2 and isinstance(args[1], list):
                center_asset = _resolve_asset(args[0])
                if center_asset is None:
                    continue
                center_id = center_asset.get("id", center_asset.get("description"))
                if center_id is None:
                    continue

                surrounding_assets = []
                surrounding_ids = []
                for obj in args[1]:
                    asset = _resolve_asset(obj)
                    if asset is None:
                        continue
                    asset_id = asset.get("id", asset.get("description"))
                    if asset_id is None:
                        continue
                    surrounding_assets.append(asset)
                    surrounding_ids.append(asset_id)
                if not surrounding_assets:
                    continue

                new_args = [center_asset, surrounding_assets]
                new_args.extend(args[2:])

                surround_edge_keys = [(sid, center_id, "surround") for sid in surrounding_ids if (sid, center_id, "surround") in edge_weights]
                if not surround_edge_keys:
                    continue

                dedup_key = ("surround", center_id, tuple(sorted(surrounding_ids, key=str)))
                if dedup_key in key_in:
                    continue

                kwargs_new["weight"] = 1.0
                updated = True
                key_in[dedup_key] = True
                fn_used = self.surround if fn_name == "surround" else fn
            else:
                flag = False
                new_args = []
                for arg in args:
                    asset0 = _resolve_asset(arg)
                    if asset0 is None and isinstance(arg, dict) and "id" in arg:
                        flag = True
                        break
                    new_args.append(asset0)
                if flag or len(new_args) < 2:
                    continue

                u_obj = new_args[0]
                v_obj = new_args[1]
                u_id = u_obj["id"] if isinstance(u_obj, dict) and "id" in u_obj else u_obj.get("description") if isinstance(u_obj, dict) else str(u_obj)
                v_id = v_obj["id"] if isinstance(v_obj, dict) and "id" in v_obj else v_obj.get("description") if isinstance(v_obj, dict) else str(v_obj)
                key_tuple = _edge_key(u_id, v_id, fn_name, args[2:])
                if key_tuple not in edge_weights or key_tuple in key_in:
                    continue

                kwargs_new["weight"] = 1.0
                updated = True
                key_in[key_tuple] = True
                fn_used = fn

            # --- Step 4: 如果约束被更新，加入 updated_constraints ---
            if updated:
                updated_constraints.append((fn_used, tuple(new_args), kwargs_new))
        # --- Step 5: 更新 self.constraints ---
        self.constraints = updated_constraints

    def update_constraints_simple(self, assets, id_to_key):
        """
        直接更新 self.constraints 中所有 asset 的引用，并更新 weight。
        对于 wall / fixed_point 保持原对象不变。
        使用 (u_id, v_id, fn) 唯一标识约束，不依赖 edge_key。
        如果约束没有被更新，则从 self.constraints 中移除。
        """
        updated_constraints = []

        for fn, args, kwargs in self.constraints:
            fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)

            if fn_name == "surround":
                # 处理 surround 类约束
                new_args = []
                # 中心物体
                if isinstance(args[0], dict) and "id" in args[0]:
                    center_asset = assets[id_to_key[args[0]['id']]]
                else:
                    center_asset = args[0]

                if center_asset is None:
                    continue

                # 周边物体
                surrounding_assets = []
                for obj in args[1]:
                    if isinstance(obj, dict) and "id" in obj:
                        asset = assets[id_to_key[obj['id']]]
                        if asset:
                            surrounding_assets.append(asset)

                new_args.append(center_asset)
                new_args.append(surrounding_assets)
                for arg in args[2:]:
                    new_args.append(arg)

                updated_constraints.append((fn, tuple(new_args), kwargs))

            else:
                new_args = []
                for arg in args:
                    if isinstance(arg, dict) and "id" in arg:
                        asset0 = assets[id_to_key[arg['id']]]
                        new_args.append(asset0)
                    else:
                        new_args.append(arg)
                updated_constraints.append((fn, tuple(new_args), kwargs))

        self.constraints = updated_constraints

    def sanitize_graph(self, G):
        for u, v, k, data in G.edges(keys=True, data=True):
            # 转换 torch.Tensor -> float
            if "extra_args" in data:
                clean_args = []
                for a in data["extra_args"]:
                    if isinstance(a, torch.Tensor):
                        a = a.detach().item() if a.numel() == 1 else a.detach().cpu().numpy().tolist()
                    clean_args.append(a)
                data["extra_args"] = tuple(clean_args)
        return G

    def group_assets(self, G, existing_assets):
        """
        忽略 wall / fixed_point，返回分组
        仅保留 group 中第一个 id 在 init_assets.values() 中的非空 group
        """
        # print(G.edges(data=True))
        G = self.sanitize_graph(G)
        G_undirected = G.to_undirected().copy()

        groups = []

        for comp in nx.connected_components(G_undirected):

            # 筛选出 existing_assets 中的 asset
            comp_assets = [n for n in comp if isinstance(n, int)]
            print("existing_assets:", existing_assets)
            existing_in_comp = [n for n in comp_assets if n in existing_assets]
            print("existing_in_comp:", existing_in_comp)
            non_existing_in_comp = [n for n in comp_assets if n not in existing_assets]
            print("non_existing_in_comp:", non_existing_in_comp)
            # 如果 comp_assets 有 existing_assets 的成员，则分组
            if existing_in_comp:
                group_existing = existing_in_comp
                if len(existing_in_comp) > 0:
                    groups.append(group_existing)
                group_non_existing = non_existing_in_comp
                if len(non_existing_in_comp) > 0:
                    groups.append(group_non_existing)
            else:
                asset_group = comp_assets
                if len(asset_group) > 0:
                    groups.append(asset_group)


        print(groups)
        return groups
    
    def add_group_centrality_constraint(self, groups, G, centrality, assets):
        """
        groups: list of sets of node ids (assets)
        G: 可选图，用于获取权重比例和中心率
        """
        for group in groups:
            if len(group) <= 1:
                continue
            center_node = max(group, key=lambda n: centrality.get(n, 0))
            centrality_sum = sum(centrality.get(n, 0.0) for n in group)
            center_centrality = centrality.get(center_node, 0.1)
            normalized_centrality = center_centrality / centrality_sum

            # --- Step 2: 对组内其他物体增加靠近中心约束 ---
            for node in group:
                if node == center_node:
                    continue
                asset = next((a for a in assets.values() if a.get('id') == node), None)
                center_asset = next((a for a in assets.values() if a.get('id') == center_node), None)
                if asset is None or center_asset is None:
                    continue
                kwargs = {"weight": 0.1 * normalized_centrality, "update": True}
                max_dist = torch.norm(torch.tensor(asset["pos"][:2]) - torch.tensor(center_asset["pos"][:2])) * 1.2

                # --- Add to graph ---
                obj_id = node
                center_id = center_node
                G.add_node(obj_id)
                G.add_node(center_id)

                # 检查是否已存在完全相同的边
                duplicate = False
                for _, _, data in G.edges(obj_id, data=True):
                    if (data.get("type") == self.near.__name__ and
                        data.get("arg_ids") == [obj_id, center_id]):
                        duplicate = True
                        break
                if duplicate:
                    continue
                
                self.add_constraint(self.near, asset, center_asset, 0.0, max_dist.item(), weight=kwargs["weight"], update=True)
                G.add_edge(
                    obj_id, center_id,
                    type=self.near.__name__,
                    fn=self.near.__name__,
                    arg_ids=[obj_id, center_id],
                    extra_args=[0.0, max_dist.item()],
                    kwargs=kwargs,
                    update=False
                )
        return G
        
    def get_constraints_repr(self, assets: dict, walls: dict):
        lines = ["solver = SmallConstraintSolver()"]

        def find_asset_by_description(obj):
            if isinstance(obj, dict) and "id" in obj:
                for key, val in assets.items():
                    if val.get("id") == obj["id"]:
                        return f"obj['{key}']"
            elif isinstance(obj, dict):
                for key, val in walls.items():
                    if val.get("description") == obj.get("description"):
                        return f"wall['{key}']"
            return None

        def format_value(val):
            if isinstance(val, torch.Tensor):
                return f"tensor({val.tolist()})"
            if isinstance(val, dict):
                items = []
                for k, v in val.items():
                    items.append(f"{repr(k)}: {format_value(v)}")
                return "{" + ", ".join(items) + "}"
            if isinstance(val, (list, tuple)):
                inner = ", ".join(format_value(v) for v in val)
                return "[" + inner + "]"
            if isinstance(val, str):
                return repr(val)
            return str(val)

        for func, args, kwargs in self.constraints:
            func_name = func.__name__
            arg_strs = []
            for arg in args:
                if isinstance(arg, dict):
                    mapped = find_asset_by_description(arg)
                    if mapped is not None:
                        arg_strs.append(mapped)
                    else:
                        arg_strs.append(format_value(arg))
                elif isinstance(arg, list):
                    a_list = []
                    for a in arg:
                        if isinstance(a, dict):
                            mapped = find_asset_by_description(a)
                            if mapped is not None:
                                a_list.append(mapped)
                            else:
                                a_list.append(format_value(a))
                        else:
                            a_list.append(format_value(a))
                    arg_strs.append(f"[{', '.join(a_list)}]")
                else:
                    arg_strs.append(format_value(arg))
            for k, v in kwargs.items():
                arg_strs.append(f"{k}={format_value(v)}")
            line = f"solver.add_constraint(solver.{func_name}, {', '.join(arg_strs)})"
            lines.append(line)

        return "\n".join(lines)
    
    def on_top_of(self, asset1, asset2, weight=1.0):
        # Ensure asset1 is on top of asset2
        loss_xy = diou_loss_2d(asset1["pos"][:2], asset2["pos"][:2], asset1["corners"], asset2["corners"])
        top_z = (asset2["pos"][2] + asset2["bbox"][2]).detach()
        loss_z = torch.abs(asset1["pos"][2] - top_z)
        return (loss_xy + loss_z) * weight
    
    def place_align(self, asset1, asset2, direction, weight=1.0):
        """
        Place asset1 in a specified direction relative to asset2. The direction here is the direction in the top view, which can be inferred from ImageA.

        direction: one of ["dwon", "up", "left", "right"]
        """
        if 'phy' not in asset2.keys():
            asset2['phy'] = torch.tensor([-90.0 * torch.pi / 180.])
        if isinstance(asset2['pos'], list):
            asset2['pos'] = torch.tensor(asset2['pos'], dtype=torch.float32, requires_grad=False)
        center_1 = asset1["pos"][:2]
        with torch.no_grad():
            center_2 = asset2["pos"][:2]
        rel_vec = center_1 - center_2  # shape: (2,)

        # normalize
        norm = torch.sqrt((rel_vec ** 2).sum() + 1e-8)
        unit_vec = rel_vec / norm

        # semantic direction angles (relative to area-facing)
        semantic_to_angle = {
            "down": -90.0,
            "right": 0.0,
            "up": 90.0,
            "left": 180.0,
        }

        semantic_to_perp = {
            "up": torch.tensor([1., 0.]), 
            "down": torch.tensor([1., 0.]),
            "right": torch.tensor([0., 1.]),
            "left": torch.tensor([0., 1.]),
        }
        if direction not in semantic_to_angle:
            raise ValueError(
                f"place_align: direction must be one of {sorted(semantic_to_angle.keys())!r}, "
                f"got {direction!r}. For wall mounts centered above a floor object, use a fixed_point "
                f"and solver.distance(src, fixed_point, min_d, max_d) instead of place_align(..., 'center')."
            )
        world_angle = semantic_to_angle[direction] % 360.0
        rad = torch.deg2rad(torch.tensor(world_angle, device=rel_vec.device))

        # convert to unit vector in world coords
        target = torch.tensor([torch.cos(rad), torch.sin(rad)], device=rel_vec.device)
        # similarity loss
        angle_loss = 1 - torch.dot(unit_vec, target)

        axis_perp = semantic_to_perp[direction].to(rel_vec.device)
        proj_perp = abs(torch.dot(rel_vec, axis_perp))
        loss = angle_loss * 0.7 + proj_perp * 0.3
        # loss += self.align_wall(asset1, asset2)
        # print("place_align:", loss)
        return loss * weight
    
    def against_wall(self, asset1, wall, weight=1.0):
        '''
        Place an asset again wall w_j, a space defined by four walls oriented along the cardinal directions {w_1, . . . , w_4} ,
        w[0] = {
            'pos': torch.tensor([0, 0, 0, 0]),
            'phy': torch.tensor([1, 0]),}
        
        distance should be 0, if the object is against the wall.
        notice that the object has volume, but the wall is a plane. so the distance should be calculated from the corners of the object to the wall.
        angle should be 0, the object should have the same orientation as the wall.


        TODO: if the object stating from outside the wall.
        '''
        # if isinstance(wall['pos'], torch.Tensor) and wall['pos'].requires_grad:
        #     wall['pos'] = wall['pos'].detach() 
        # if isinstance(wall['phy'], torch.Tensor) and wall['phy'].requires_grad:
        #     wall['phy'] = wall['phy'].detach() 
        pos_i = asset1['pos']
        a_i = asset1['phy']
        v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
        pos_j = wall['pos']
        a_j = wall['phy']
        v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)
        distance_center = torch.abs(torch.dot(pos_i[:2] - pos_j[:2], v_j)) # shape: (4,1)
        gap = asset1['bbox'][0]/2
        dist_loss = abs(distance_center - gap)
        # angle_loss = 1 - torch.dot(v_i, v_j)

        # # TODO: Check: when the angle between two vectors is 180, grad will be error
        if torch.dot(v_i, v_j) < -0.99:
            eps = 8e-2
            v_i = v_i + torch.tensor([eps, -eps])
        angle_loss = 1 - torch.dot(v_i, v_j)
        loss = angle_loss * 1.5 + dist_loss
        # print("aganist_Wall:", loss)
        return loss * weight
    
    def against(self, asset1, wall, height, weight=1.0):
        """
        Place an asset against a wall and set its bottom Z to height.
        """
        pos_i = asset1['pos']
        a_i = asset1['phy']
        v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
        if isinstance(wall['pos'], list):
            wall['pos'] = torch.tensor(wall['pos']).float()
        if isinstance(wall['phy'], list):
            wall['phy'] = torch.tensor(wall['phy']).float()
        pos_j = wall['pos']
        a_j = wall['phy']
        v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)
        distance_center = torch.abs(torch.dot(pos_i[:2] - pos_j[:2], v_j))
        gap = asset1['bbox'][0] / 2
        dist_loss = abs(distance_center - gap)
        angle_loss = 1 - torch.dot(v_i, v_j)
        h_t = height if torch.is_tensor(height) else torch.tensor(height, device=pos_i.device, dtype=pos_i.dtype)
        z_loss = torch.abs(pos_i[2] - h_t)
        loss = dist_loss + angle_loss + z_loss
        return loss * weight
        
    def above(self, asset1, point, height=None, weight=1.0):
        """
        Place an asset above a floor point. If height is None, only apply xy loss.

        ``point`` may be: a dict with ``pos`` (e.g. ``Point(...)`` from exec returns this),
        a length-2+ sequence ``[x, y, ...]``, or any object with numeric ``.x`` / ``.y``.
        """
        if isinstance(point, dict) and "pos" in point:
            p = point["pos"]
            px, py = float(p[0]), float(p[1])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            px, py = float(point[0]), float(point[1])
        elif hasattr(point, "x") and hasattr(point, "y"):
            px, py = float(point.x), float(point.y)
        else:
            raise ValueError(f"Unsupported point type: {type(point)!r}")
        pos_i = asset1['pos']
        target_xy = torch.tensor([px, py], device=pos_i.device, dtype=pos_i.dtype)
        xy_loss = torch.norm(pos_i[:2] - target_xy)
        if height is None:
            return xy_loss * weight
        h_t = height if torch.is_tensor(height) else torch.tensor(height, device=pos_i.device, dtype=pos_i.dtype)
        z_loss = torch.abs(pos_i[2] - h_t)
        return (xy_loss + z_loss) * weight

    def distance(self, asset1, point, min_d, max_d, weight=1.0):
        """
        Radial XY distance from ``asset1`` center to a fixed ``point`` should lie in ``[min_d, max_d]`` (meters).
        ``point`` is a dict with ``pos`` [x,y,z] or a length-≥2 list/tuple (uses x,y only).
        Typical rug pin: ``min_d=0.0``, ``max_d=0.01`` with ``point`` under a bed/sofa center.
        """
        if isinstance(point, dict) and "pos" in point:
            p = point["pos"]
            px, py = float(p[0]), float(p[1])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            px, py = float(point[0]), float(point[1])
        else:
            raise ValueError(f"distance: unsupported point type {type(point)!r}")
        pos_i = asset1["pos"][:2]
        p_xy = torch.tensor([px, py], device=pos_i.device, dtype=pos_i.dtype)
        d = torch.linalg.norm(pos_i - p_xy)
        min_t = torch.as_tensor(float(min_d), device=d.device, dtype=d.dtype)
        max_t = torch.as_tensor(float(max_d), device=d.device, dtype=d.dtype)
        loss_below = torch.relu(min_t - d)
        loss_above = torch.relu(d - max_t)
        return (loss_below + loss_above) * weight

    def near(self, asset1, asset2, min_distance, max_distance, weight=0.9, margin=0.05, update=True):
        '''
        The distance between the edges of the two assets should fall within the range [d min, d max].
        If set the min_distance to 0, it means the edges of the objects can touch but not overlap.
        This calculation considers the bounding box sizes of the assets in the x-y plane.
        '''
        # Ensure positions are torch tensors and require grad if needed
        if isinstance(asset2['pos'], list):
            asset2['pos'] = torch.tensor(asset2['pos'], dtype=torch.float32, requires_grad=False)
            margin = 0.02
        if "bbox" in asset2:
            if update:
                bb1 = asset1['bbox'][:2]
                bb2 = asset2['bbox'][:2]
                min_dist_1 = torch.min(bb1) / 2
                min_dist_2 = torch.min(bb2) / 2
                max_dist_1 = torch.max(bb1) / 2
                max_dist_2 = torch.max(bb2) / 2
                min_distance = torch.max(torch.tensor(min_distance, device=bb1.device), min_dist_1 + min_dist_2)
                max_distance = torch.max(torch.tensor(max_distance, device=bb1.device), max_dist_1 + max_dist_2)
        pos_i = asset1['pos'][:2]
        pos_j = asset2['pos'][:2].detach()

        wall_name = ["front_wall(facing +y)", "right_wall(facing -x)", "back_wall(facing -y)", "left_wall(facing +x)"]
        if 'description' in asset2 and asset2['description'] in wall_name:
            a_j = asset2['phy']
            v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)
            distance = torch.abs(torch.dot(pos_i - pos_j, v_j))
        else:
            distance = torch.norm(pos_i - pos_j)
        term = soft_near_loss(distance, min_distance, max_distance, margin)
        loss = torch.clamp(term, max=10.0) # 防梯度爆炸
        # print("near:", loss)
        return loss * weight

    def align_wall(self, asset1, asset2, angle=0.0, weight=1.0):
        '''
        angle should be in radian.

        the discription in the paper is confusing. Align two assets at a specified angle ϕ . does it means the object should be rotated to the angle ϕ? If then, why don't just set the angle to the object's orientation?

        I think it should be the angle between the two objects. Here I only rotate the asset_j to algin with asset_i with an offset angle ϕ'.
        '''
        if isinstance(asset2['phy'], list):
            asset2['phy'] = torch.tensor(asset2['phy'], dtype=torch.float32, requires_grad=False)
        a_i = asset1['phy'] 
        with torch.no_grad():
            a_j = asset2['phy'] + angle*torch.pi/180.0
        v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
        v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)

        # Compute cosine similarity between the two vectors
        cosine_similarity = torch.dot(v_i, v_j)
        loss = 1 - cosine_similarity
        # print("align_wall:", loss)
        return loss * weight

    def point_towards(self, asset1, asset2, weight=1.0):
        '''
        angle should be in radians.

        rotate asset i to point j with an offset angle ϕ'
        '''
        # Ensure positions are torch tensors and require grad if needed
        if isinstance(asset2['pos'], list):
            asset2['pos'] = torch.tensor(asset2['pos'], dtype=torch.float32, requires_grad=False)

        a_i = asset1['phy']
        # register_hook to check gradient.
        v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
        
        pos_i = asset1['pos']
        pos_j = asset2['pos']

        desired_direction = pos_j[:2].detach() - pos_i[:2]
        align_angle = torch.dot(desired_direction, v_i) / (torch.norm(desired_direction) + 1e-6)
        loss = 1 - align_angle
        # print("point_toward:", loss)
        return loss * weight
    
    def surround_fixed(self, center_asset, surrounding_assets, gap=0.25, dist_tol=0.05, weight=1.0):
        if isinstance(center_asset['pos'], list):
            center_asset['pos'] = torch.tensor(center_asset['pos'], dtype=torch.float32, requires_grad=False)
        center_pos = center_asset['pos'][:2]
        bbox = center_asset['bbox'][:2]
        if torch.is_tensor(bbox):
            bbox_val = bbox.detach().cpu().numpy()
        else:
            bbox_val = np.array(bbox, dtype=float)
        hx = float(bbox_val[0]) / 2.0
        hy = float(bbox_val[1]) / 2.0

        if float(bbox_val[0]) >= float(bbox_val[1]):
            start_angle = 0.0
        else:
            start_angle = 0.5 * np.pi

        n = len(surrounding_assets)
        if n == 0:
            return torch.tensor(0.0, device=center_pos.device, dtype=center_pos.dtype)

        total_loss = torch.tensor(0.0, device=center_pos.device, dtype=center_pos.dtype)
        eps = 1e-6
        gap_t = torch.tensor(gap, device=center_pos.device, dtype=center_pos.dtype)
        for i, asset in enumerate(surrounding_assets):
            angle = start_angle + i * (2.0 * np.pi / n)
            angle_t = torch.tensor(angle, device=center_pos.device, dtype=center_pos.dtype)
            ca = torch.cos(angle_t)
            sa = torch.sin(angle_t)
            bbox_s = asset.get("bbox", [0.0, 0.0])
            if torch.is_tensor(bbox_s):
                bbox_s_val = bbox_s.detach().cpu().numpy()
            else:
                bbox_s_val = np.array(bbox_s, dtype=float)
            hx_s = float(bbox_s_val[0]) / 2.0
            hy_s = float(bbox_s_val[1]) / 2.0
            abs_c = torch.clamp(torch.abs(ca), min=eps)
            abs_s = torch.clamp(torch.abs(sa), min=eps)
            tx = torch.tensor(hx, device=center_pos.device, dtype=center_pos.dtype) / abs_c
            ty = torch.tensor(hy, device=center_pos.device, dtype=center_pos.dtype) / abs_s
            use_x = tx <= ty
            t = torch.where(use_x, tx, ty)
            gap_x = torch.tensor(gap + hx_s, device=center_pos.device, dtype=center_pos.dtype)
            gap_y = torch.tensor(gap + hy_s, device=center_pos.device, dtype=center_pos.dtype)
            offset = torch.where(use_x, gap_x / abs_c, gap_y / abs_s)
            direction = torch.stack([ca, sa])
            edge_point = center_pos + t * direction
            target_point = center_pos + (t + offset) * direction

            dist = torch.norm(asset['pos'][:2] - target_point)
            total_loss += soft_near_loss(dist, 0.0, dist_tol, margin=0.0, alpha=0.005)
            # print(soft_near_loss(dist, 0.0, dist_tol, margin=0.0, alpha=0.005))

            edge_asset = {"pos": edge_point, "description": "surround_edge"}
            total_loss += self.point_towards(asset, edge_asset)

        return total_loss * weight

    def surround(self, center_asset, surrounding_assets, max_distance, look_mode="axis", weight=1.0):
        """
        Make surrounding_assets be near the center_asset from different directions.

        Loss = sum of distances + angular diversity penalty.
        """
        # If center has no bbox (e.g., fixed_point), use max_distance to anchor distance.
        if "bbox" not in center_asset:
            if isinstance(center_asset.get("pos"), list):
                center_asset["pos"] = torch.tensor(center_asset["pos"], dtype=torch.float32, requires_grad=False)
            # --- Step 1: average position constraint ---
            N = len(surrounding_assets)
            pos_j = center_asset["pos"][:2].detach()
            pos_a = surrounding_assets[0]["pos"][:2]
            for asset in surrounding_assets[1:]:
                pos_a = pos_a + asset["pos"][:2]
            distance = torch.norm(pos_a / N - pos_j)
            loss = soft_near_loss(distance, 0.0, 0.01) * 1.5
            if look_mode == "axis":
                for asset in surrounding_assets:
                    loss += self.near(asset, center_asset, 0.0, max_distance, update=True)
                    loss += self.point_towards(asset, center_asset)
            else:
                for asset in surrounding_assets:
                    loss += self.near(asset, center_asset, max_distance, max_distance, update=False)
                    loss += self.point_towards(asset, center_asset)
            return loss * weight

        return self.surround_fixed(center_asset, surrounding_assets, gap=0.25, dist_tol=0.05, weight=weight)
    
    def near_center(self, center_asset, surrounding_assets, weight=1.0):
        """
        Make surrounding_assets be near the center_asset from different directions.

        Loss = sum of distances + angular diversity penalty.
        """
        if isinstance(center_asset['pos'], list):
            center_asset['pos'] = torch.tensor(center_asset['pos'], dtype=torch.float32, requires_grad=False)
        # --- Step 1: 平均位置约束 ---
        N = len(surrounding_assets)
        pos_j = center_asset['pos'][:2].detach()
        pos_a = surrounding_assets[0]["pos"][:2]
        for asset in surrounding_assets[1:]:
            pos_a = pos_a + asset["pos"][:2]
        distance = torch.norm(pos_a / N - pos_j)
        loss = soft_near_loss(distance, 0.0, 0.05)
        return loss * weight

    def align_with(self, asset1, asset2, angle=0.0, weight=1.0):
        '''
        angle should be in radian.

        the discription in the paper is confusing. Align two assets at a specified angle ϕ . does it means the object should be rotated to the angle ϕ? If then, why don't just set the angle to the object's orientation?

        I think it should be the angle between the two objects. Here I only rotate the asset_j to algin with asset_i with an offset angle ϕ'.
        '''
        if isinstance(asset2['phy'], list):
            asset2['phy'] = torch.tensor(asset2['phy'], dtype=torch.float32, requires_grad=False)
        a_i = asset1['phy'] 
        with torch.no_grad():
            a_j = asset2['phy'] + angle*torch.pi/180.0
        v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
        v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)
        cosine_similarity = torch.dot(v_i, v_j)
        # TODO: Check: when the angle between two vectors is 180, grad will be error
        loss = 1 - cosine_similarity
        return loss * weight
    
    def center(self, asset, region, weight=1.0):

        pos_i = asset["pos"][:2]  # (x, y)
        x_min, x_max, y_min, y_max = region
        center = torch.tensor(
            [(x_min + x_max) / 2, (y_min + y_max) / 2],
            device=pos_i.device,
            dtype=pos_i.dtype,
        )
        dist_loss = torch.norm(pos_i - center)
        dist_loss = soft_near_loss(dist_loss, 0.0, 0.01, margin=0.0, alpha=0.005)  * 5.0
        return dist_loss * weight

    
    def place_align_small(self, asset1, asset2, direction, weight=1.0):
        """
        Place asset1 in a specified direction relative to asset2. The direction here is the direction in the top view, which can be inferred from ImageA.

        direction: one of ["front", "back", "left", "right"]
        """
        if isinstance(asset2['pos'], list):
            asset2['pos'] = torch.tensor(asset2['pos'], dtype=torch.float32, requires_grad=False)
        center_1 = asset1["pos"][:2]
        with torch.no_grad():
            center_2 = asset2["pos"][:2]
        rel_vec = center_2 - center_1  # shape: (2,)

        # normalize
        norm = torch.sqrt((rel_vec ** 2).sum() + 1e-6)
        unit_vec = rel_vec / norm

        # semantic direction angles (relative to area-facing)
        semantic_to_angle = {
            "front": 90.0,
            "right": 180.0,
            "back": -90.0,
            "left": 0.0,
        }

        if direction not in semantic_to_angle:
            raise ValueError(
                f"place_align_small: direction must be one of {sorted(semantic_to_angle.keys())!r}, "
                f"got {direction!r}."
            )
        # compute world angle
        local_angle = semantic_to_angle[direction]
        world_angle = local_angle % 360.0
        rad = torch.deg2rad(torch.tensor(world_angle, device=rel_vec.device))

        # convert to unit vector in world coords
        target = torch.tensor([torch.cos(rad), torch.sin(rad)], device=rel_vec.device)

        if torch.dot(unit_vec, target) < -0.999:
            eps = 5e-2
            unit_vec = unit_vec + torch.tensor([eps, -eps])
        # similarity loss
        loss = 1 - torch.dot(unit_vec, target)
        loss = soft_near_loss(loss, 0.0, 0.02, margin=0.0)
        return loss * weight
    
    def against_edge(self, asset, region, edge, weight=1.0):
        pos_i = asset["pos"][:2]  # (x, y)
        angle = asset["phy"]
        v_i = torch.cat([torch.cos(angle), torch.sin(angle)], dim=0)

        if isinstance(asset["bbox"], torch.Tensor):
            bbox_xy = asset["bbox"][:2].to(device=pos_i.device, dtype=pos_i.dtype)
        else:
            bbox_xy = torch.tensor(asset["bbox"][:2], device=pos_i.device, dtype=pos_i.dtype)

        x_min, x_max, y_min, y_max = region

        # 模仿 against_wall：使用中心到边界的距离与半尺寸的差值，避免 min(corners) 带来的非平滑优化
        if edge == "back":      # 物体靠近 y_max
            distance_center = y_max - pos_i[1]
            gap = bbox_xy[0] / 2
        elif edge == "front":   # 靠近 y_min
            distance_center = pos_i[1] - y_min
            gap = bbox_xy[0] / 2
        elif edge == "left":    # 靠近 x_min
            distance_center = pos_i[0] - x_min
            gap = bbox_xy[0] / 2
        elif edge == "right":   # 靠近 x_max
            distance_center = x_max - pos_i[0]
            gap = bbox_xy[0] / 2
        else:
            raise ValueError(f"Unknown edge: {edge}")

        dist_loss = torch.abs(distance_center - gap)

        # semantic direction angles (relative to area-facing)
        semantic_to_angle = {
            "front": -180.0,
            "right": -90.0,
            "back": 0.0,
            "left": 90.0,
        }

        local_angle = semantic_to_angle[edge]
        world_angle = (local_angle - 90.0) % 360.0
        rad = torch.deg2rad(torch.tensor(world_angle, device=pos_i.device, dtype=pos_i.dtype))

        # convert to unit vector in world coords
        target = torch.stack([torch.cos(rad), torch.sin(rad)])

        # similarity loss
        angle_loss = 1 - torch.dot(v_i, target)
        # print(dist_loss + angle_loss)
        return (dist_loss + angle_loss * 1.5) * 2.0 * weight
    
    def point_to_edge(self, asset, region, edge, weight=1.0):
        angle = asset["phy"]
        v_i = torch.cat([torch.cos(angle), torch.sin(angle)], dim=0)

        # semantic direction angles (relative to area-facing)
        semantic_to_angle = {
            "front": -180.0,
            "right": -90.0,
            "back": 0.0,
            "left": 90.0,
        }

        local_angle = semantic_to_angle[edge]
        world_angle = (local_angle + 90.0) % 360.0
        rad = torch.deg2rad(torch.tensor(world_angle))

        # convert to unit vector in world coords
        target = torch.tensor([torch.cos(rad), torch.sin(rad)])

        # similarity loss
        angle_loss = 1 - torch.dot(v_i, target)

        return angle_loss * weight
    
    def place_align_group(self, asset1, asset2, direction, weight=1.0):
        """
        Place asset1 in a specified direction relative to asset2. The direction here is the direction in the top view, which can be inferred from ImageA.

        direction: one of ["dwon", "up", "left", "right"]
        """
        if 'phy' not in asset2.keys():
            asset2['phy'] = torch.tensor([-90.0 * torch.pi / 180.])
        if isinstance(asset2['pos'], list):
            asset2['pos'] = torch.tensor(asset2['pos'], dtype=torch.float32, requires_grad=False)
        center_1 = asset1["pos"][:2]
        with torch.no_grad():
            center_2 = asset2["pos"][:2]
        rel_vec = center_1 - center_2  # shape: (2,)

        # normalize
        norm = torch.sqrt((rel_vec ** 2).sum() + 1e-6)
        unit_vec = rel_vec / norm

        # semantic direction angles (relative to area-facing)
        semantic_to_angle = {
            "down": -90.0,
            "right": 0.0,
            "up": 90.0,
            "left": 180.0,
        }

        if direction not in semantic_to_angle:
            raise ValueError(
                f"place_align_group: direction must be one of {sorted(semantic_to_angle.keys())!r}, "
                f"got {direction!r}."
            )
        world_angle = semantic_to_angle[direction] % 360.0
        rad = torch.deg2rad(torch.tensor(world_angle, device=rel_vec.device))

        # convert to unit vector in world coords
        target = torch.tensor([torch.cos(rad), torch.sin(rad)], device=rel_vec.device)

        if torch.dot(unit_vec, target) < -0.999:
            eps = 5e-2
            unit_vec = -unit_vec + torch.tensor([eps, -eps])
        # similarity loss
        loss = 1 - torch.dot(unit_vec, target)
        # loss += self.align_wall(asset1, asset2)
        return loss * weight * 0.5
    
    def point_towards_group(self, asset1, asset2, weight=1.0):
        '''
        angle should be in radians.

        rotate asset i to point j with an offset angle ϕ'
        '''
        # Ensure positions are torch tensors and require grad if needed
        if isinstance(asset2['pos'], list):
            asset2['pos'] = torch.tensor(asset2['pos'], dtype=torch.float32, requires_grad=False)

        a_i = asset1['phy']
        # register_hook to check gradient.
        v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
        
        pos_i = asset1['pos']
        pos_j = asset2['pos']

        desired_direction = pos_j[:2].detach()  - pos_i[:2]
        align_angle = torch.dot(desired_direction, v_i) / ((torch.norm(desired_direction) * torch.norm(v_i)) + 1e-6)
        loss = 1 - align_angle
        loss = soft_near_loss(loss, 0.0, 0.02, margin=0.0)
        return loss * weight * 0.5

    def align_with_group(self, asset1, asset2, angle=0.0, weight=1.0):
        '''
        angle should be in radian.

        the discription in the paper is confusing. Align two assets at a specified angle ϕ . does it means the object should be rotated to the angle ϕ? If then, why don't just set the angle to the object's orientation?

        I think it should be the angle between the two objects. Here I only rotate the asset_j to algin with asset_i with an offset angle ϕ'.
        '''
        if isinstance(asset2['phy'], list):
            asset2['phy'] = torch.tensor(asset2['phy'], dtype=torch.float32, requires_grad=False)
        a_i = asset1['phy'] 
        with torch.no_grad():
            a_j = asset2['phy'] + angle*torch.pi/180.0
        v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
        v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)

        if torch.dot(v_i, v_j) < -0.999:
            eps = 5e-2
            v_i = v_i + torch.tensor([eps, -eps])
        # Compute cosine similarity between the two vectors
        cosine_similarity = torch.dot(v_i, v_j)
        # TODO: Check: when the angle between two vectors is 180, grad will be error
        loss = 1 - cosine_similarity
        return loss * weight * 0.5

def plot(step, assets):
    print(f"Step {step} Result:")
    for asset in assets.values():
        pos_out = asset['pos']
        rot_out = asset['phy']
        print(f" {asset['description']} position grad:", pos_out.grad)
        print(f" {asset['description']} rotation grad:", rot_out.grad)
        print(f"{asset['description']} final position: {pos_out}, rotation: {rot_out}")


_HAS_SOLVER_ASSIGNMENT = re.compile(
    r"(?m)^\s*solver\s*=\s*ConstraintSolver\s*\(",
)


def _ensure_solver_init_line(code_str, preserve_passed_solver):
    """
    GPT/clean_pattern often drops ``solver = SmallConstraintSolver()`` while the body still
    calls ``solver.add_constraint(...)``. Prepend a fresh line when missing.

    When ``preserve_passed_solver`` is True, the caller bound ``namespace['solver']`` to an
    existing solver (e.g. multi-region CCEA); do not inject, or exec would replace it.
    """
    if preserve_passed_solver:
        return code_str
    if _HAS_SOLVER_ASSIGNMENT.search(code_str or ""):
        return code_str
    return "solver = SmallConstraintSolver()\n" + (code_str or "").lstrip("\n")


def exec_safe(code_str, namespace, preserve_passed_solver=False, inject_point_compat=False):
    """
    Run GPT-produced constraint code as a single Python block (one namespace, top-to-bottom).
    globals and locals are the same dict so plain assignments (e.g. fixed_point_*) stay visible below.

    ``inject_point_compat``: when True (extra assets: ceiling / rug / wall_mount constraint exec),
    define ``Point(x, y, z=0.0)`` as a small helper returning an anchor dict compatible with
    ``above`` / ``distance``. Floor-area constraints should keep this False and use dict/list anchors.
    """
    namespace.update({"exec": lambda *args, **kwargs: None, "eval": lambda *args, **kwargs: None})
    if inject_point_compat and "Point" not in namespace:
        namespace["Point"] = lambda x, y, z=0.0: {
            "pos": [float(x), float(y), float(z)],
            "description": "point",
        }
    code_exec = _ensure_solver_init_line(code_str, preserve_passed_solver)
    try:
        exec(code_exec, namespace, namespace)
    except Exception as e:
        print(f"Error executing code:\n{code_exec}")
        raise e
    return namespace


def ensure_grad(assets, keys=('pos', 'phy'), only_keys=None):
    """If ``only_keys`` is set, only those asset ids get ``requires_grad`` enabled."""
    items = assets.items()
    if only_keys is not None:
        items = ((k, v) for k, v in assets.items() if k in only_keys)
    for _, obj in items:
        for key in keys:
            if key in obj.keys():
                if isinstance(obj[key], torch.Tensor):
                    if not obj[key].requires_grad:
                        obj[key].requires_grad_(True)
                else:
                    obj[key] = torch.tensor(obj[key]).float()
                    obj[key].requires_grad_(True)
    return assets


def freeze_asset_tensors(assets, tensor_keys=('pos', 'phy'), skip_keys=None):
    """
    Detach and disable grad on ``tensor_keys`` for every asset whose key is **not** in ``skip_keys``.
    Used when floor assets are merged into ``obj`` for wall-mount constraints but must not move during GD.
    """
    skip = skip_keys if skip_keys is not None else frozenset()
    for k, obj in assets.items():
        if k in skip:
            continue
        for key in tensor_keys:
            if key not in obj:
                continue
            t = obj[key]
            if isinstance(t, torch.Tensor):
                obj[key] = t.detach().requires_grad_(False)
    return assets
    
def recaculate_bbox(assets):
    '''
    Recalculate the bounding box of the asset based on its position and rotation.
    '''
    for asset in assets.values():
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(asset['phy'][0]), -torch.sin(asset['phy'][0])]),
            torch.stack([torch.sin(asset['phy'][0]),  torch.cos(asset['phy'][0])])
        ])
        device = asset['phy'].device
        local_corners = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], device=device, dtype=asset['phy'].dtype) * asset['bbox'][:2] / 2
        corners = torch.matmul(local_corners, rotation_matrix.T)
        asset['corners'] = corners

def recaculate_bbox_w_remove(assets, region_bound):
    '''
    Recalculate the bounding box of the asset based on its position and rotation.
    '''
    xmin, xmax, ymin, ymax = region_bound
    tol = 0.01
    new_assets = {}
    for key, asset in assets.items():
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(asset['phy'][0]), -torch.sin(asset['phy'][0])]),
            torch.stack([torch.sin(asset['phy'][0]),  torch.cos(asset['phy'][0])])
        ])
        device = asset['phy'].device
        local_corners = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], device=device, dtype=asset['phy'].dtype) * asset['bbox'][:2] / 2
        corners = torch.matmul(local_corners, rotation_matrix.T)
        asset['corners'] = corners
        corners_pos = corners + asset['pos'][:2]
        corner_min = corners_pos.min(dim=1)[0]
        corner_max = corners_pos.max(dim=1)[0]
        if (
            corner_min[0] < (ymin - tol)
            or corner_max[0] > (ymax + tol)
            or corner_min[1] < (xmin - tol)
            or corner_max[1] > (xmax + tol)
        ):
            continue
        new_assets[key] = asset
    return new_assets

def _obj_list_region_namespace_keys(region_idx, assets, region):
    """Expose ``obj_list`` / ``region`` under both str and int keys (GPT code varies)."""

    rs = str(region_idx)
    d_obj = {rs: assets}
    d_reg = {rs: region} if region is not None else {}
    try:
        ri = int(rs)
        d_obj[ri] = assets
        if region is not None:
            d_reg[ri] = region
    except (TypeError, ValueError):
        pass
    return d_obj, d_reg


def load_assets_and_constraints_s(
    code_str,
    wall,
    assets,
    asset_add=None,
    region_idx=None,
    region=None,
    solver=None,
    inject_point_compat=False,
):
    """
    Loads assets and get_constraints from the given code string.
    Parameters:
        code_str: Code string containing the definitions of assets and get_constraints.
        inject_point_compat: if True, exec namespace defines ``Point`` for extra-asset DSL snippets.
    Return:
        (assets, get_constraints)
    """
    namespace  = {
        "torch": torch,
        "wall": wall,
        "tensor": torch.tensor,
        "ConstraintSolver": SmallConstraintSolver
    }
    if region_idx is not None:
        d_obj, d_reg = _obj_list_region_namespace_keys(region_idx, assets, region)
        namespace["obj_list"] = d_obj
        namespace["region"] = d_reg
    else:
        namespace["obj"] = assets

    if solver:
        namespace["solver"] = solver
    else:
        namespace["solver"] = SmallConstraintSolver()
    exec_safe(
        code_str,
        namespace,
        preserve_passed_solver=solver is not None,
        inject_point_compat=inject_point_compat,
    )
    solver = namespace.get("solver")
    assets_all = assets.copy()
    if asset_add:
        asset_add = {key + len(assets): value for key, value in asset_add.items()}
        assets_all.update(asset_add)
    if assets_all is None or solver is None:
        raise ValueError("must define 'assets' and 'get_constraints' in text...")
    
    return assets_all, solver, wall
