"""Small-object graph conflict detection and solver update (ablation-aligned)."""
import ast
import os
import random
from collections import deque

from utils.small_solver import SmallConstraintSolver, load_assets_and_constraints_s
from utils.tool import Point, clean_pattern, parse_ast_value

_SMALL_CONFLICT_ABLATION_MODE = "baseline"
OA_AREA_RATIO = 0.75  # OA: sum of asset bbox areas must fit within this fraction of region

def set_small_conflict_ablation_mode(mode: str):
    global _SMALL_CONFLICT_ABLATION_MODE
    _SMALL_CONFLICT_ABLATION_MODE = mode

def build_group_center_graph(G, groups, centrality, obj_new):
    """
    构建新的中心图 G_centered，只保留每个 group 的中心节点。
    若 group 内有多个高中心率节点，则自动分成多个子组。
    每个节点只属于一个组（归入出边最多的中心节点）。
    """
    center_nodes = {}
    new_groups = []
    obj_new_ids = [a['id'] for a in obj_new.values()]
    i = 0

    for group_keys in groups:
        group_keys = list(group_keys)
        # 情况 1: 单节点组
        if len(group_keys) == 1:
            node = group_keys[0]
            center_nodes[i] = node
            new_groups.append({node})
            i += 1
            continue

        # --- 计算中心率 ---
        scores = {k: centrality.get(k, 0) for k in group_keys}
        high_nodes = [k for k, v in scores.items() if v >= 1.5]

        # --- 情况 2: group 内多个高中心节点 ---
        if len(group_keys) >= 2 and len(high_nodes) >= 2:
            # 初始化子组
            sub_groups = {c: set([c]) for c in high_nodes}

            # 遍历所有节点，为它选最合适的中心节点
            for node in group_keys:
                if node in high_nodes:
                    continue  # 中心节点自己跳过

                # 统计与每个中心节点的出入边数
                link_counts = {}
                for c in high_nodes:
                    out_w = G.out_degree(node, weight=None) if G.has_edge(node, c) else 0
                    in_w = G.in_degree(node, weight=None) if G.has_edge(c, node) else 0
                    link_counts[c] = out_w + in_w

                # 选择与之连接最多的中心节点
                best_center = max(link_counts, key=link_counts.get)
                if link_counts[best_center] > 0:
                    sub_groups[best_center].add(node)

            # 保存子组
            for c_node, members in sub_groups.items():
                center_nodes[i] = c_node
                new_groups.append(members)
                i += 1
            continue

        # --- 情况 3: 其他情况，取中心率最高节点 ---
        node = max(scores, key=scores.get)
        center_nodes[i] = node
        # keep group type stable: downstream code expects set operations
        new_groups.append(set(group_keys))
        i += 1

    # --- 2. 保留中心节点的有向子图 ---
    center_node_set = set(center_nodes.values())
    print(center_node_set)
    G_centered = G.copy()
    nodes_to_remove = [n for n in obj_new_ids if n not in center_node_set]
    G_centered.remove_nodes_from(nodes_to_remove)

    return G_centered, center_nodes, new_groups, nodes_to_remove

    return G_centered, center_nodes, new_groups, nodes_to_remove


def _format_conflict_log_gpt(log_lines):
    semantic_lines = [l for l in log_lines if "[LogicError]" in l]
    semantic_common = [l for l in log_lines if "[SemanticCommon]" in l]
    low_outdegree_lines = [l for l in log_lines if "[ConstraintCompleteness]" in l]
    return "\n".join(
        [
            "Edges conflicts due to semantics logic:",
            *(semantic_lines if semantic_lines else ["(none)"]),
            "Low outdegree nodes:",
            *(low_outdegree_lines if low_outdegree_lines else ["(none)"]),
            "semantic common error:",
            *(semantic_common if semantic_common else ["(none)"]),
        ]
    )


def _arg_key(arg):
    if isinstance(arg, dict):
        if "id" in arg:
            return ("id", arg["id"])
        if "description" in arg:
            return ("desc", arg["description"])
        if "pos" in arg:
            return ("pos", tuple(arg["pos"]))
    if isinstance(arg, Point):
        return ("point", (arg.x, arg.y, arg.z))
    if isinstance(arg, list):
        return ("list", tuple(_arg_key(a) for a in arg))
    return ("raw", str(arg))


def _constraint_key(fn_name, args):
    return (fn_name, tuple(_arg_key(a) for a in args))


def _parse_small_dsl_constraints(code, obj_list, region_bounds, region_idx):
    rels = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return rels
    env = {"obj_list": obj_list or {}}
    region_map = {}
    if region_bounds is not None:
        if isinstance(obj_list, dict) and obj_list:
            for k in obj_list.keys():
                region_map[str(k)] = region_bounds
        elif region_idx is not None:
            region_map[str(region_idx)] = region_bounds
    env["region"] = region_map
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add_constraint":
            if not call.args:
                continue
            fn_node = call.args[0]
            if isinstance(fn_node, ast.Attribute):
                fn_name = fn_node.attr
            elif isinstance(fn_node, ast.Name):
                fn_name = fn_node.id
            else:
                continue
            args = [parse_ast_value(a, env) for a in call.args[1:]]
            rels.append((fn_name, args))
        elif isinstance(call.func, ast.Name):
            fn_name = call.func.id
            args = [parse_ast_value(a, env) for a in call.args]
            rels.append((fn_name, args))
    return rels


def _apply_small_dsl_constraints(dsl_code, solver, assets, obj_list, region_bounds, region_idx):
    existing_keys = set()
    for fn, args, kwargs in solver.constraints:
        fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
        existing_keys.add(_constraint_key(fn_name, args))

    def _resolve_asset(arg):
        if isinstance(arg, dict):
            if "id" in arg:
                return arg
            if "pos" in arg:
                return arg
        if isinstance(arg, str):
            return assets.get(arg)
        return None

    def _resolve_region(arg):
        if isinstance(arg, (list, tuple)) and len(arg) >= 4:
            return list(arg)
        return None

    relations = _parse_small_dsl_constraints(dsl_code, obj_list, region_bounds, region_idx)
    for fn_name, rel_args in relations:
        if fn_name == "center":
            if len(rel_args) < 2:
                continue
            src = _resolve_asset(rel_args[0])
            region = _resolve_region(rel_args[1])
            if src is None or region is None:
                continue
            args = (src, region)
            key = _constraint_key("center", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.center, *args)
            existing_keys.add(key)
            continue

        if fn_name == "against_edge":
            if len(rel_args) < 3:
                continue
            src = _resolve_asset(rel_args[0])
            region = _resolve_region(rel_args[1])
            edge = rel_args[2]
            if src is None or region is None:
                continue
            args = (src, region, edge)
            key = _constraint_key("against_edge", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.against_edge, *args)
            existing_keys.add(key)
            continue

        if fn_name == "point_to_edge":
            if len(rel_args) < 3:
                continue
            src = _resolve_asset(rel_args[0])
            region = _resolve_region(rel_args[1])
            edge = rel_args[2]
            if src is None or region is None:
                continue
            args = (src, region, edge)
            key = _constraint_key("point_to_edge", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.point_to_edge, *args)
            existing_keys.add(key)
            continue

        if fn_name == "place_align_small":
            if len(rel_args) < 3:
                continue
            src = _resolve_asset(rel_args[0])
            dst = _resolve_asset(rel_args[1])
            if src is None or dst is None:
                continue
            direction = rel_args[2]
            args = (src, dst, direction)
            key = _constraint_key("place_align_small", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.place_align_small, *args)
            existing_keys.add(key)
            continue

        if fn_name == "align_with":
            if len(rel_args) < 2:
                continue
            src = _resolve_asset(rel_args[0])
            dst = _resolve_asset(rel_args[1])
            if src is None or dst is None:
                continue
            args = (src, dst)
            key = _constraint_key("align_with", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.align_with, *args)
            existing_keys.add(key)
            continue

        if fn_name == "point_towards":
            if len(rel_args) < 2:
                continue
            src = _resolve_asset(rel_args[0])
            dst = _resolve_asset(rel_args[1])
            if src is None or dst is None:
                continue
            args = (src, dst)
            key = _constraint_key("point_towards", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.point_towards, *args)
            existing_keys.add(key)
            continue

        if fn_name == "distance":
            if len(rel_args) < 4:
                continue
            src = _resolve_asset(rel_args[0])
            dst_val = rel_args[1]
            if isinstance(dst_val, Point):
                dst = {"pos": [dst_val.x, dst_val.y, dst_val.z], "description": "fixed point"}
            elif isinstance(dst_val, dict) and "pos" in dst_val:
                dst = dst_val
            else:
                dst = _resolve_asset(dst_val)
            if src is None or dst is None:
                continue
            args = (src, dst, float(rel_args[2]), float(rel_args[3]))
            key = _constraint_key("near", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.near, *args)
            existing_keys.add(key)
            continue

        if fn_name == "near":
            if len(rel_args) < 4:
                continue
            src = _resolve_asset(rel_args[0])
            dst_val = rel_args[1]
            if isinstance(dst_val, Point):
                dst = {"pos": [dst_val.x, dst_val.y, dst_val.z], "description": "fixed point"}
            elif isinstance(dst_val, dict) and "pos" in dst_val:
                dst = dst_val
            else:
                dst = _resolve_asset(dst_val)
            if src is None or dst is None:
                continue
            args = (src, dst, float(rel_args[2]), float(rel_args[3]))
            key = _constraint_key("near", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.near, *args)
            existing_keys.add(key)
            continue

        if fn_name == "place_align":
            if len(rel_args) < 3:
                continue
            src = _resolve_asset(rel_args[0])
            dst = _resolve_asset(rel_args[1])
            if src is None or dst is None:
                continue
            args = (src, dst, rel_args[2])
            key = _constraint_key("place_align", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.place_align, *args)
            existing_keys.add(key)
            continue

        if fn_name == "surround":
            if len(rel_args) < 3:
                continue
            center = _resolve_asset(rel_args[0])
            assets_list = rel_args[1] if isinstance(rel_args[1], list) else []
            surrounding = [_resolve_asset(s) for s in assets_list]
            surrounding = [s for s in surrounding if s is not None]
            if center is None or not surrounding:
                continue
            max_dist = float(rel_args[2])
            args = (center, surrounding, max_dist)
            key = _constraint_key("surround", args)
            if key in existing_keys:
                continue
            solver.add_constraint(solver.surround, *args)
            existing_keys.add(key)


def _finalize_solver_graph(
    solver,
    assets,
    room_bound,
    if_weight,
    existing_assets=None,
    *,
    open_surface_area_check=True,
):
    G = solver.build_graph()
    G, _, _ = detect_graph_conflicts_bfs_center_oa(
        G,
        assets,
        room_bound,
        solver=solver,
        seed=0,
        verbose=True,
        open_surface_area_check=open_surface_area_check,
    )
    try:
        G, groups, centrality = solver.postprocess_graph(G, existing_assets or {})
    except TypeError:
        G, groups, centrality = solver.postprocess_graph(G)
    if if_weight:
        G = solver.cal_weight(G, groups, centrality)
    id_to_key = {v["id"]: k for k, v in assets.items()}
    solver.update_constraints(G, assets, assets, id_to_key)
    return G, groups, centrality


def update_solver_s(
    assets,
    solver,
    room_bound,
    if_weight,
    gpt_api=None,
    current_code=None,
    areas=None,
    output_dir=None,
    label=None,
    wall=None,
    task=None,
    max_iters=3,
    existing_assets=None,
    region_idx=None,
    use_small_reflection=False,
    region_bounds=None,
    obj_list=None,
    open_surface_area_check=True,
):
    del areas, wall  # kept for Genesis-compatible call sites
    if solver is not None:
        solver.enable_open_surface_area_check = bool(open_surface_area_check)
    id_to_key = {v["id"]: k for k, v in assets.items()}
    existing_assets = existing_assets or {}
    if region_idx is not None:
        active_srcs = {k for k, v in assets.items() if v.get("region_idx") == region_idx}
    else:
        existing_ids = set()
        for ea in existing_assets:
            if isinstance(ea, dict):
                if "id" in ea:
                    existing_ids.add(ea["id"])
                continue
            try:
                existing_ids.add(int(ea))
            except Exception:
                continue
        active_srcs = {k for k, v in assets.items() if v.get("id") not in existing_ids}

    G_init = solver.build_graph()
    try:
        _, groups_fixed, centrality_fixed = solver.postprocess_graph(G_init, existing_assets)
    except TypeError:
        _, groups_fixed, centrality_fixed = solver.postprocess_graph(G_init)

    refine_enabled = gpt_api is not None and output_dir is not None and label is not None
    if not refine_enabled:
        G = solver.build_graph()
        G, _, _, log_lines = solver.detect_graph_conflicts(
            G,
            groups_fixed,
            assets,
            id_to_key,
            room_bound,
            verbose=True,
            first_pass=True,
        )
        if log_lines:
            print(_format_conflict_log_gpt(log_lines))
        return _finalize_solver_graph(
            solver, assets, room_bound, if_weight, existing_assets,
            open_surface_area_check=open_surface_area_check,
        )

    first_pass = True
    max_iters = 8
    for i in range(max_iters):
        G = solver.build_graph()
        groups = groups_fixed
        centrality = centrality_fixed
        G, removed_edges, low_outdegree_nodes, log_lines = solver.detect_graph_conflicts(
            G,
            groups,
            assets,
            id_to_key,
            room_bound,
            verbose=True,
            first_pass=first_pass,
        )
        id_to_key = {v["id"]: k for k, v in assets.items()}
        if not log_lines:
            break

        log_text = _format_conflict_log_gpt(log_lines)
        log_root = os.path.join(output_dir, label, "refine_logs")
        log_dir = (
            os.path.join(log_root, f"region_{region_idx}")
            if region_idx is not None
            else log_root
        )
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"log_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(log_text)

        if use_small_reflection:
            print(log_text)
            small_output_dir = os.path.join(output_dir, label) if label else output_dir
            content_system, content_user = gpt_api.get_reflection_small(
                region_bounds,
                obj_list or {},
                current_code,
                small_output_dir,
                label,
                low_outdegree_nodes,
                log_text,
                task=task,
            )
        else:
            content_system, content_user = gpt_api.get_reflection(
                task,
                {},
                assets,
                current_code,
                log_text,
                output_dir=output_dir,
                low_outdegree_nodes=low_outdegree_nodes,
            )
        refine_constraints = gpt_api(content_system, content_user)
        print(refine_constraints)
        dsl_code = clean_pattern(refine_constraints)
        if not dsl_code:
            break
        with open(os.path.join(log_dir, f"refine_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(dsl_code)
        if use_small_reflection:
            _apply_small_dsl_constraints(
                dsl_code,
                solver,
                assets,
                obj_list,
                region_bounds,
                region_idx,
            )
        first_pass = False

    return _finalize_solver_graph(
        solver, assets, room_bound, if_weight, existing_assets,
        open_surface_area_check=open_surface_area_check,
    )


def detect_graph_conflicts_bfs_center_oa(
    G, assets, room_bound, solver=None, seed=0, verbose=True, open_surface_area_check=True,
):
    if _SMALL_CONFLICT_ABLATION_MODE in {"wo_logic_error", "wo_conflict_module"}:
        return G, [], set()

    removed_edges = []
    removed_asset_ids = set()

    asset_ids = {a.get("id") for a in assets.values() if "id" in a}

    def _node_sort_key(node):
        if node in asset_ids:
            return (0, int(node))
        if isinstance(node, int):
            return (0, node)
        return (1, str(node))

    def _bfs_all_nodes(G_undirected):
        order = []
        visited = set()
        for start in sorted(G_undirected.nodes, key=_node_sort_key):
            if start in visited:
                continue
            q = deque([start])
            visited.add(start)
            while q:
                n = q.popleft()
                order.append(n)
                neighbors = list(G_undirected.neighbors(n))
                neighbors.sort(key=_node_sort_key)
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        return order

    def _parse_region_bounds(node):
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

    def _asset_area(asset):
        bbox = asset.get("bbox", [0.0, 0.0])
        try:
            return float(bbox[0]) * float(bbox[1])
        except Exception:
            return 0.0

    def _region_area(bounds):
        x_min, x_max, y_min, y_max = bounds
        return abs((x_max - x_min) * (y_max - y_min))

    def _oa_area_cap(bounds):
        return _region_area(bounds) * OA_AREA_RATIO

    def _pop_min_area_candidate(candidates, asset_id_to_key):
        if not candidates:
            return None
        best_idx = min(
            range(len(candidates)),
            key=lambda i: _asset_area(assets.get(asset_id_to_key.get(candidates[i]), {})),
        )
        return candidates.pop(best_idx)

    def _collect_region_nodes():
        region_nodes = set()
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("type") in {"center", "against_edge", "point_to_edge"}:
                region_nodes.add(v)
        return region_nodes

    def _cleanup_constraints():
        if solver is None:
            return

        def _get_obj_id(obj):
            if isinstance(obj, dict):
                return obj.get("id", obj.get("description"))
            return None

        new_constraints = []
        for fn, args, kwargs in solver.constraints:
            fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
            if fn_name in {"surround", "near_center"} and len(args) >= 2 and isinstance(args[1], list):
                center_id = _get_obj_id(args[0])
                if center_id in removed_asset_ids:
                    continue
                new_list = []
                for obj in args[1]:
                    obj_id = _get_obj_id(obj)
                    if obj_id in removed_asset_ids:
                        continue
                    new_list.append(obj)
                if not new_list:
                    continue
                new_args = (args[0], new_list, *args[2:])
                new_constraints.append((fn, new_args, kwargs))
                continue

            drop = False
            for arg in args:
                obj_id = _get_obj_id(arg)
                if obj_id in removed_asset_ids:
                    drop = True
                    break
            if drop:
                continue
            new_constraints.append((fn, args, kwargs))
        solver.constraints = new_constraints

    print("start BFS center/OA conflict detection...")

    G_undirected = G.to_undirected()
    bfs_order = _bfs_all_nodes(G_undirected)

    # --- Prefer against_edge over center for the same (u, v) ---
    # If an asset has both against_edge and center to the same region node,
    # remove center edges to avoid contradictory anchoring.
    for node in bfs_order:
        if node not in asset_ids:
            continue
        out_edges = list(G.out_edges(node, keys=True, data=True))
        against_targets = {
            v for _, v, _, d in out_edges if d.get("type") == "against_edge"
        }
        if not against_targets:
            continue
        center_edges = [
            (u, v, k)
            for u, v, k, d in out_edges
            if d.get("type") == "center" and v in against_targets
        ]
        for u, v, k in center_edges:
            if G.has_edge(u, v, key=k):
                G.remove_edge(u, v, key=k)
                removed_edges.append((u, v, "center"))
                if verbose:
                    print(
                        f"[Auto-Fix] Removed center due to against_edge overlap: {u} -> {v}"
                    )

    # --- Center uniqueness per region (BFS-first) ---
    region_to_center = {}
    for node in bfs_order:
        if node not in asset_ids:
            continue
        out_edges = [
            (u, v, k, d)
            for u, v, k, d in G.out_edges(node, keys=True, data=True)
            if d.get("type") == "center"
        ]
        out_edges.sort(key=lambda e: str(e[1]))
        for u, v, k, d in out_edges:
            if v in region_to_center:
                if G.has_edge(u, v, key=k):
                    G.remove_edge(u, v, key=k)
                    removed_edges.append((u, v, "center"))
                    if verbose:
                        print(f"[Auto-Fix] Removed duplicate center: {u} -> {v}")
            else:
                region_to_center[v] = u

    # --- OA detection per region (detect-only; never written to refine log) ---
    if not open_surface_area_check:
        if removed_asset_ids:
            _cleanup_constraints()
        return G, removed_edges, removed_asset_ids

    region_nodes = _collect_region_nodes()

    if not region_nodes:
        # no explicit region nodes; use all assets with room_bound
        bounds = room_bound
        if bounds is None or len(bounds) < 4:
            return G, removed_edges, removed_asset_ids
        asset_id_to_key = {v.get("id"): k for k, v in assets.items()}
        assets_in_region = [aid for aid in asset_ids if aid in asset_id_to_key]
        area_cap = _oa_area_cap(bounds)
        sum_area = sum(_asset_area(assets[asset_id_to_key[aid]]) for aid in assets_in_region)
        candidates = [aid for aid in assets_in_region]
        while sum_area > area_cap and candidates:
            aid = _pop_min_area_candidate(candidates, asset_id_to_key)
            akey = asset_id_to_key.get(aid)
            if akey is None:
                continue
            sum_area -= _asset_area(assets[akey])
            removed_asset_ids.add(aid)
            del assets[akey]
            asset_ids.discard(aid)
            if G.has_node(aid):
                G.remove_node(aid)
                G_undirected.remove_node(aid)
        _cleanup_constraints()
        return G, removed_edges, removed_asset_ids

    # order regions by BFS where possible
    bfs_set = set(bfs_order)
    region_order = [n for n in bfs_order if n in region_nodes]
    for n in sorted(region_nodes, key=_node_sort_key):
        if n not in bfs_set:
            region_order.append(n)

    for region_node in region_order:
        if region_node not in G_undirected:
            continue
        bounds = _parse_region_bounds(region_node)
        if bounds is None:
            bounds = room_bound
        if bounds is None or len(bounds) < 4:
            continue

        # BFS from region node
        asset_id_to_key = {v.get("id"): k for k, v in assets.items()}
        q = deque([region_node])
        visited = {region_node}
        assets_in_region = set()
        while q:
            n = q.popleft()
            if n in asset_ids and n in asset_id_to_key:
                assets_in_region.add(n)
            for nb in G_undirected.neighbors(n):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        if not assets_in_region:
            continue

        area_cap = _oa_area_cap(bounds)
        sum_area = 0.0
        for aid in list(assets_in_region):
            akey = asset_id_to_key.get(aid)
            if akey is None:
                assets_in_region.discard(aid)
                continue
            sum_area += _asset_area(assets[akey])

        if sum_area <= area_cap:
            continue

        center_id = region_to_center.get(region_node)
        non_center = [aid for aid in assets_in_region if aid != center_id]
        while sum_area > area_cap and non_center:
            aid = _pop_min_area_candidate(non_center, asset_id_to_key)
            akey = asset_id_to_key.get(aid)
            if akey is None:
                continue
            sum_area -= _asset_area(assets[akey])
            removed_asset_ids.add(aid)
            del assets[akey]
            asset_ids.discard(aid)
            if G.has_node(aid):
                G.remove_node(aid)
                if G_undirected.has_node(aid):
                    G_undirected.remove_node(aid)
            assets_in_region.discard(aid)

        if sum_area > area_cap:
            remaining = [aid for aid in list(assets_in_region) if aid in assets]
            while sum_area > area_cap and remaining:
                aid = _pop_min_area_candidate(remaining, asset_id_to_key)
                akey = asset_id_to_key.get(aid)
                if akey is None:
                    continue
                sum_area -= _asset_area(assets[akey])
                removed_asset_ids.add(aid)
                del assets[akey]
                asset_ids.discard(aid)
                if G.has_node(aid):
                    G.remove_node(aid)
                    if G_undirected.has_node(aid):
                        G_undirected.remove_node(aid)
                assets_in_region.discard(aid)
                if aid == center_id and region_node in region_to_center:
                    del region_to_center[region_node]

    if removed_asset_ids:
        _cleanup_constraints()

    return G, removed_edges, removed_asset_ids
