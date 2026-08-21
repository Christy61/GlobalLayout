
from typing import Any, List, Union, Optional, Set
import copy
import torch
import re
import inspect
from dataclasses import dataclass
import ast
import networkx as nx
from utils.tool import parse_ast_value, Point
from utils.loss import project_loss, semantic_loss, physics_loss, semantic_loss_group
from tqdm import tqdm
import sys
from utils.constraints import CONSTRAINT_REGISTRY
from utils.node import Object, ObjectSet
from utils.graph_conflict import build_graph_from_edge_center
import numpy as np


from utils.geometry import compute_rotated_corners, compute_rotated_corners_np


def get_device_with_index(index=0):
    """
    Get the appropriate device with index (CUDA if available, otherwise CPU).
    
    Args:
        index (int): CUDA device index (ignored if CUDA not available)
        
    Returns:
        str: Device string ('cuda:0' or 'cpu')
    """
    if torch.cuda.is_available():
        return f"cuda:{index}"
    else:
        return "cpu"


_GRAPH_ORIENTATION_TYPES = frozenset({"align_with", "point_towards", "against", "surround"})


def build_ea_constraint_graph(
    scene_graph,
    active_srcs: Set[str],
    *,
    context_srcs: Optional[Set[str]] = None,
    use_weight: bool = True,
):
    """
    Build weighted constraint graph for EA from ``SceneGraph.edge_center`` (ablation ``G_ea``).

    Include neighbors (walls, existing assets, etc.) so edge weights match full DSL constraints.
    """
    active_srcs = set(active_srcs or [])
    context_srcs = set(context_srcs or [])
    if not active_srcs:
        return nx.MultiDiGraph(), {}

    available_asset_keys = set(active_srcs) | context_srcs
    node_set = set(available_asset_keys)
    for u, v in scene_graph.edge_center.edges():
        u_is_asset = u in getattr(scene_graph, "nodes", {})
        v_is_asset = v in getattr(scene_graph, "nodes", {})
        if u_is_asset and u not in available_asset_keys:
            continue
        if v_is_asset and v not in available_asset_keys:
            continue
        if u in available_asset_keys or v in available_asset_keys:
            node_set.add(u)
            node_set.add(v)

    G_sub = scene_graph.edge_center.subgraph(node_set)
    G = build_graph_from_edge_center(G_sub)
    centrality = scene_graph.compute_centrality(active_srcs=available_asset_keys)

    if use_weight:
        for _u, _v, _k, data in G.edges(keys=True, data=True):
            base_weight = 2.0 if data.get("type") in _GRAPH_ORIENTATION_TYPES else 1.0
            data["weight"] = centrality.get(_u, 0.0) * base_weight
    return G, centrality


def prepare_solver_for_eval(
    init_solver: "ConstraintSolver",
    G_ea: Optional[nx.MultiDiGraph],
    assets: dict[str, Union[Object, ObjectSet]],
    ea_asset_keys: Set[str],
    *,
    use_weight: bool = True,
) -> "ConstraintSolver":
    """Deep-copy solver and refresh constraint weights from graph (ablation EA fitness prep)."""
    eval_solver = copy.deepcopy(init_solver)
    if G_ea is None:
        return eval_solver
    if use_weight:
        eval_solver.update_constraints(G_ea, assets, ea_asset_keys)
    else:
        eval_solver.update_constraints_woweight(G_ea, assets, ea_asset_keys)
    return eval_solver

class Constraint:
    def __init__(self, constraint_name, constraint_func, *args, **kargs):
        self.constraint_name = constraint_name
        self.constraint_func = constraint_func
        self.args = args
        self.kwargs = kargs

    def get_wall(self, wall=None, edge=None):
        self.wall = wall
        self.edge = edge

    def evaluate(self, assets: dict[str, Union[Object, ObjectSet]], device=None, fallback_assets=None):
        if device is None:
            device = get_device_with_index()
        # # Ensure device is available
        # if device.startswith('cuda') and not torch.cuda.is_available():
        #     device = 'cpu'
        if self.constraint_name == "surround":
            src_list = []
            for name in self.args[0]:
                if name in assets:
                    src_list.append(assets[name])
                else:
                    print(f"Warning: src {name} not in assets for surround, skipping constraint.")
                    return torch.tensor(0.0)
            dst_key = self.args[1]
            if isinstance(dst_key, Point):
                dst = dst_key
            elif dst_key in assets:
                dst = assets[dst_key]
            elif fallback_assets is not None and dst_key in fallback_assets:
                dst = fallback_assets[dst_key]
            else:
                print(f"Warning: dst {dst_key} not in assets, using zero tensor instead.")
                return torch.tensor(0.0)
            return self.constraint_func(src_list, dst, *self.args[2:], **self.kwargs)
        else:
            src_key = self.args[0]
            if src_key not in assets:
                print(f"Warning: src {src_key} not in assets, skipping constraint.")
                return torch.tensor(0.0)
            src = assets[src_key]
            if isinstance(self.args[1], Point):
                dst = self.args[1]
            elif self.args[1] in assets:
                dst = assets[self.args[1]]
            elif fallback_assets is not None and self.args[1] in fallback_assets:
                dst = fallback_assets[self.args[1]]
            elif self.args[1].endswith('_wall') and self.args[1] in self.wall:
                dst = self.wall[self.args[1]]
            elif self.args[1].endswith('_edge') and self.args[1] in self.edge:
                dst = self.edge[self.args[1]]
            else:
                print(f"Warning: dst {self.args[1]} not in assets, using zero tensor instead.")
                return torch.tensor(0.0)
            return self.constraint_func(src, dst, *self.args[2:], **self.kwargs)

    def __repr__(self):
        def _fmt(v):
            if isinstance(v, str):
                return f"'{v}'"
            return repr(v)

        args_str = ", ".join(_fmt(a) for a in self.args)
        kwargs_str = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(self.kwargs.items()))
        if args_str and kwargs_str:
            params = f"{args_str}, {kwargs_str}"
        else:
            params = args_str or kwargs_str
        return f"Constraint(name={self.constraint_name}, params={params})"


class ConstraintSolver:
    def __init__(self, dsl_code: str, wall: dict):
        """
        Initialize from LLM-generated DSL.
        Each line must be a single constraint call.
        """
        self.constraints: List[Constraint] = []
        self.orientation_constraints = {"align_with", "point_towards", "against", "surround"}
        self.asset_fallback: Optional[dict[str, Union[Object, ObjectSet]]] = None
        self.wall = wall

        tree = ast.parse(dsl_code)
        env = {}

        for node in tree.body:
            if isinstance(node, ast.Assign):
                value = parse_ast_value(node.value, env)
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        raise ValueError("Only simple assignments are allowed (e.g. name = Point(...))")
                    env[target.id] = value
                continue

            if isinstance(node, ast.AnnAssign):
                if node.value is None or not isinstance(node.target, ast.Name):
                    raise ValueError("Only simple annotated assignments are allowed (e.g. name: Point = Point(...))")
                env[node.target.id] = parse_ast_value(node.value, env)
                continue

            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue

            call = node.value

            # ---- function name ----
            if not isinstance(call.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")

            name = call.func.id
            # print(CONSTRAINT_REGISTRY)
            if name in CONSTRAINT_REGISTRY:
                func = CONSTRAINT_REGISTRY[name]
            else:
                raise ValueError(f"Unknown constraint: {name}")

            # ---- positional args ----
            args = [parse_ast_value(arg, env) for arg in call.args]

            # ---- keyword args ----
            kwargs = {
                kw.arg: parse_ast_value(kw.value, env)
                for kw in call.keywords
            }
            constraint = Constraint(name, func, *args, **kwargs)
            constraint.get_wall(wall)
            self.constraints.append(constraint)

    def set_asset_fallback(self, assets: Optional[dict[str, Union[Object, ObjectSet]]]):
        self.asset_fallback = assets

    def build_wall_assignments(self) -> dict[str, str]:
        """
        Return mapping: asset_key -> wall_key for wall-mounted assets.

        Only `against(src, <wall>, ...)` is treated as a wall assignment.
        """
        assignments: dict[str, str] = {}
        for constraint in self.constraints:
            if constraint.constraint_name != "against":
                continue
            if len(constraint.args) < 2:
                continue
            src_key = constraint.args[0]
            dst_key = constraint.args[1]
            if not isinstance(src_key, str) or not isinstance(dst_key, str):
                continue
            if dst_key.endswith("_wall") and isinstance(self.wall, dict) and dst_key in self.wall:
                assignments[src_key] = dst_key
        return assignments
    
    def update_constraint_weights(
        self,
        centrality: dict[str, float],
        *,
        use_weight: bool = True,
        ea_asset_keys: Optional[set[str]] = None,
    ):
        """
        Assign constraint weights from group-normalized degree centrality (ablation ``cal_weight``).

        weight = centrality[src] * base_weight, base_weight=2 for orientation constraints else 1.
        ``surround`` uses the mean centrality of all surrounding src nodes; when the center (dst)
        is in ``ea_asset_keys``, clamp weight to [0.75, 2.0] (ablation ``update_constraints``).
        """
        ea_asset_keys = ea_asset_keys or set()

        def _existing_src_factor(src_keys):
            if not ea_asset_keys or not src_keys:
                return 1.0
            factors = [1.0 if k in ea_asset_keys else 0.9 for k in src_keys]
            return sum(factors) / len(factors)

        for constraint in self.constraints:
            src = constraint.args[0] if constraint.args else None
            dst = constraint.args[1] if len(constraint.args) > 1 else None
            base_weight = (
                2.0 if constraint.constraint_name in self.orientation_constraints else 1.0
            )

            if not use_weight:
                cent_factor = 1.0
                src_factor_keys = []
            elif constraint.constraint_name == "surround" and isinstance(src, (list, tuple)):
                src_keys = [s for s in src if isinstance(s, str)]
                src_factor_keys = src_keys
                if src_keys:
                    cent_factor = sum(
                        float(centrality.get(k, 0.0)) for k in src_keys
                    ) / len(src_keys)
                else:
                    cent_factor = 0.0
            else:
                src_key = src if isinstance(src, str) else None
                src_factor_keys = [src_key] if src_key is not None else []
                cent_factor = (
                    float(centrality.get(src_key, 0.0)) if src_key is not None else 0.0
                )

            try:
                sig = inspect.signature(constraint.constraint_func)
                params = list(sig.parameters.keys())
                if "weight" in params:
                    weight_idx = params.index("weight")
                    if len(constraint.args) > weight_idx:
                        continue
            except (ValueError, TypeError):
                pass

            if use_weight:
                weight_val = float(cent_factor * base_weight)
                if (
                    constraint.constraint_name == "surround"
                    and isinstance(dst, str)
                    and dst in ea_asset_keys
                ):
                    weight_val = max(min(weight_val, 2.0), 0.75)
                weight_val *= _existing_src_factor(src_factor_keys)
                constraint.kwargs["weight"] = weight_val
            else:
                constraint.kwargs["weight"] = 1.0

    def _collect_graph_edge_meta(self, G: nx.MultiDiGraph):
        edge_weights = {}
        edge_update_flags = {}
        for u, v, _k, data in G.edges(keys=True, data=True):
            fn_name = data.get("type", data.get("fn", ""))
            weight = data.get("weight", data.get("kwargs", {}).get("weight", 1.0))
            edge_weights[(u, v, fn_name)] = weight
            edge_update_flags[(u, v, fn_name)] = data.get("update", True)
        return edge_weights, edge_update_flags

    def _append_updated_constraint(
        self,
        updated_constraints: List["Constraint"],
        constraint: "Constraint",
        fn_name: str,
        new_args: tuple,
        kwargs_new: dict,
    ):
        new_c = Constraint(fn_name, constraint.constraint_func, *new_args, **kwargs_new)
        new_c.get_wall(self.wall)
        updated_constraints.append(new_c)

    def update_constraints(
        self,
        G: nx.MultiDiGraph,
        assets: dict[str, Union[Object, ObjectSet]],
        ea_asset_keys: Optional[Set[str]] = None,
    ):
        """
        Refresh constraint weights from graph edges (ablation ``update_constraints``).

        Updates weights in-place; constraints without a matching graph edge are kept unchanged
        so DSL constraints (walls, points, etc.) are never dropped.
        """
        ea_asset_keys = set(ea_asset_keys or [])

        def _existing_src_factor(src_keys):
            if not ea_asset_keys or not src_keys:
                return 1.0
            factors = [1.0 if k in ea_asset_keys else 0.9 for k in src_keys]
            return sum(factors) / len(factors)

        edge_weights, edge_update_flags = self._collect_graph_edge_meta(G)
        key_in: dict = {}

        for constraint in self.constraints:
            fn_name = constraint.constraint_name
            args = constraint.args
            kwargs_new = dict(constraint.kwargs)

            if fn_name == "surround" and len(args) >= 2 and isinstance(args[0], (list, tuple)):
                src_list = [s for s in args[0] if isinstance(s, str)]
                center_key = args[1] if isinstance(args[1], str) else None
                if center_key is None or center_key not in assets:
                    continue
                surrounding_keys = [s for s in src_list if s in assets]
                if not surrounding_keys:
                    continue

                surround_edge_keys = [
                    (sid, center_key, "surround")
                    for sid in surrounding_keys
                    if (sid, center_key, "surround") in edge_weights
                ]
                if not surround_edge_keys:
                    continue

                dedup_key = ("surround", center_key, tuple(sorted(surrounding_keys, key=str)))
                if dedup_key in key_in:
                    continue

                mean_weight = sum(edge_weights[k] for k in surround_edge_keys) / len(surround_edge_keys)
                if "weight" not in kwargs_new:
                    kwargs_new["weight"] = 0.5
                if any(edge_update_flags.get(k, True) is False for k in surround_edge_keys):
                    kwargs_new["weight"] = mean_weight
                elif center_key in ea_asset_keys or any(k in ea_asset_keys for k in surrounding_keys):
                    kwargs_new["weight"] = max(min(mean_weight, 2.0), 0.75)
                else:
                    kwargs_new["weight"] = mean_weight
                kwargs_new["weight"] *= _existing_src_factor(surrounding_keys)
                constraint.kwargs.update(kwargs_new)
                key_in[dedup_key] = True
                continue

            if len(args) < 2:
                continue
            src_key = args[0] if isinstance(args[0], str) else None
            if src_key is None or src_key not in assets:
                continue

            dst_arg = args[1]
            if isinstance(dst_arg, Point):
                continue
            if isinstance(dst_arg, str):
                if dst_arg in assets:
                    dst_key = dst_arg
                elif dst_arg.endswith("_wall") and isinstance(self.wall, dict) and dst_arg in self.wall:
                    dst_key = dst_arg
                else:
                    continue
            else:
                continue

            key_tuple = (src_key, dst_key, fn_name)
            if key_tuple not in edge_weights or key_tuple in key_in:
                continue

            if "weight" not in kwargs_new:
                kwargs_new["weight"] = 0.5
            if edge_update_flags.get(key_tuple, True) is False:
                kwargs_new["weight"] = edge_weights[key_tuple]
            elif src_key in ea_asset_keys or dst_key in ea_asset_keys:
                kwargs_new["weight"] = max(min(edge_weights[key_tuple], 2.0), 0.75)
            else:
                kwargs_new["weight"] = edge_weights[key_tuple]
            kwargs_new["weight"] *= _existing_src_factor([src_key])
            constraint.kwargs.update(kwargs_new)
            key_in[key_tuple] = True

    def update_constraints_woweight(
        self,
        G: nx.MultiDiGraph,
        assets: dict[str, Union[Object, ObjectSet]],
        ea_asset_keys: Optional[Set[str]] = None,
    ):
        """Set matched constraint weights to 1.0 in-place; unmatched constraints are kept."""
        _ = ea_asset_keys
        edge_weights, _edge_update_flags = self._collect_graph_edge_meta(G)
        key_in: dict = {}

        for constraint in self.constraints:
            fn_name = constraint.constraint_name
            args = constraint.args

            if fn_name == "surround" and len(args) >= 2 and isinstance(args[0], (list, tuple)):
                center_key = args[1] if isinstance(args[1], str) else None
                if center_key is None or center_key not in assets:
                    continue
                surrounding_keys = [s for s in args[0] if isinstance(s, str) and s in assets]
                if not surrounding_keys:
                    continue
                surround_edge_keys = [
                    (sid, center_key, "surround")
                    for sid in surrounding_keys
                    if (sid, center_key, "surround") in edge_weights
                ]
                if not surround_edge_keys:
                    continue
                dedup_key = ("surround", center_key, tuple(sorted(surrounding_keys, key=str)))
                if dedup_key in key_in:
                    continue
                constraint.kwargs["weight"] = 1.0
                key_in[dedup_key] = True
                continue

            if len(args) < 2:
                continue
            src_key = args[0] if isinstance(args[0], str) else None
            if src_key is None or src_key not in assets:
                continue
            dst_arg = args[1]
            if isinstance(dst_arg, Point):
                continue
            if isinstance(dst_arg, str):
                if dst_arg in assets:
                    dst_key = dst_arg
                elif dst_arg.endswith("_wall") and isinstance(self.wall, dict) and dst_arg in self.wall:
                    dst_key = dst_arg
                else:
                    continue
            else:
                continue
            key_tuple = (src_key, dst_key, fn_name)
            if key_tuple not in edge_weights or key_tuple in key_in:
                continue
            constraint.kwargs["weight"] = 1.0
            key_in[key_tuple] = True

    # ------------------ Solver ------------------
    def solve(self, assets: dict[str, Union[Object, ObjectSet]]):
        """
        Iterate through all constraints and apply the methods.
        """
        loss = 0.0
        for i, constraint in enumerate(self.constraints):
            loss += constraint.evaluate(assets, fallback_assets=self.asset_fallback)
        return loss

    def solve_group(self, groups: List[set], assets: dict[str, Union[Object, ObjectSet]]):
        """
        Iterate through all constraints and apply the methods.
        """
        loss_g = torch.zeros(len(groups))
        for constraint in self.constraints:
            args = constraint.args
            if not args:
                continue
            src = args[0]
            dst = args[1] if len(args) > 1 else None
            src_group = None
            dst_group = None
            for idx, group in enumerate(groups):
                if isinstance(src, (list, tuple, set)):
                    src_in_group = any(s in group for s in src)
                else:
                    src_in_group = src in group
                if src_in_group and src_group is None:
                    src_group = idx
                if isinstance(dst, str) and dst in group and dst_group is None:
                    dst_group = idx
            # EA mutates only new-asset groups. Attribute to src's group first;
            # use dst's group only for existing->new constraints.
            matched_group = src_group if src_group is not None else dst_group
            if matched_group is not None:
                loss_g[matched_group] += constraint.evaluate(assets, fallback_assets=self.asset_fallback)
        return loss_g


def build_constraint_z_locks(
    assets: dict[str, Union[Object, ObjectSet]],
    solver: Optional[ConstraintSolver],
    fixed_assets: Optional[dict[str, Union[Object, ObjectSet]]] = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Build z locks from explicit constraint heights (no parent required).

    Rules:
    - against(src, wall, height!=0): lock src.z = height
    - above(src, dst, height!=0): lock src.z = (top_z(dst) + height)
      * if dst is Point: top_z = dst.z
      * if dst is asset: top_z = dst.pos[2] + dst.bbox[2]
    """
    if solver is None:
        return {}
    fixed_assets = fixed_assets or {}

    def _get_height(c: Constraint) -> Optional[float]:
        if "height" in c.kwargs:
            h = c.kwargs.get("height")
        elif len(c.args) >= 3:
            h = c.args[2]
        else:
            h = None
        if h is None:
            return None
        try:
            return float(h)
        except (TypeError, ValueError):
            return None

    def _asset_by_key(key: str):
        return assets.get(key) or fixed_assets.get(key)

    z_locks: dict[str, float] = {}
    for c in solver.constraints:
        if len(c.args) < 2:
            continue
        src_key = c.args[0]
        dst_ref = c.args[1]
        if not isinstance(src_key, str):
            continue
        if src_key not in assets:
            continue
        height = _get_height(c)
        if height is None or abs(height) <= eps:
            continue

        if c.constraint_name == "against":
            z_locks[src_key] = float(height)
            continue

        if c.constraint_name == "above":
            # Point anchor
            if isinstance(dst_ref, Point):
                z_locks[src_key] = float(dst_ref.z + height)
                continue
            if not isinstance(dst_ref, str):
                continue
            dst_asset = _asset_by_key(dst_ref)
            if dst_asset is None:
                continue
            dst_pos = _asset_pos_value(dst_asset)
            dst_bbox = _asset_bbox_value(dst_asset)
            if dst_pos is None:
                continue
            dst_pos = _ensure_tensor(dst_pos)
            if dst_bbox is not None:
                dst_bbox = _ensure_tensor(dst_bbox, device=dst_pos.device, dtype=dst_pos.dtype)
                top_z = dst_pos[2] + dst_bbox[2]
            else:
                top_z = dst_pos[2]
            z_locks[src_key] = float(top_z.item() + height)
            continue

    return z_locks

class Scene:
    def __init__(self, name: str, regions: dict[str, Any], room_bound: List[float]):
        self.name = name
        self.regions = regions
        self.bound = room_bound
        # Hierarchical storage:
        # per-region: region_idx -> depth -> {"solver": ConstraintSolver, "assets": {name: Object}}
        self.layers: dict[str, dict[int, dict[str, Any]]] = {}
        # global: depth -> {"solver": ConstraintSolver, "assets": {name: Object}}
        self.layers_by_depth: dict[int, dict[str, Any]] = {}
    
    # per region init
    def _init_assets(self, region_idx: str, assets: dict[str, Union[Object, ObjectSet]]):
        self.regions[region_idx]['assets'] = assets

    # per region per depth init
    def _init_layer(self, region_idx: str, depth: int, solver: ConstraintSolver, assets: dict[str, Union[Object, ObjectSet]]):
        self.layers.setdefault(region_idx, {})[int(depth)] = {"solver": solver, "assets": assets}

    # per depth global init
    def _init_global_layer(self, depth: int, solver: ConstraintSolver, assets: dict[str, Union[Object, ObjectSet]]):
        self.layers_by_depth[int(depth)] = {"solver": solver, "assets": assets}


# For debug
def plot(step, assets):
    print(f"Step {step} Result:")
    for asset in assets.values():
        if isinstance(asset, dict):
            pos_out = asset['pos']
            rot_out = asset['phy']
            desc = asset.get('description', '')
        else:
            pos_out = asset.pos
            rot_out = asset.rot
            desc = getattr(asset, 'key', '')
        print(f" {desc} position grad:", pos_out.grad)
        print(f" {desc} rotation grad:", rot_out.grad)
        print(f"{desc} final position: {pos_out}, rotation: {rot_out}")


def _asset_pos_ref(asset):
    if isinstance(asset, dict):
        return asset['pos']
    return asset.pos


def _asset_rot_ref(asset):
    if isinstance(asset, dict):
        if 'phy' in asset:
            return asset['phy']
        return asset.get('rot')
    return asset.rot


def _clone_tensor_or_none(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _snapshot_asset(asset):
    snap = copy.copy(asset)
    if hasattr(asset, 'pos'):
        snap.pos = _clone_tensor_or_none(asset.pos)
    if hasattr(asset, 'rot'):
        snap.rot = _clone_tensor_or_none(asset.rot)
    if hasattr(asset, 'bbox'):
        snap.bbox = _clone_tensor_or_none(asset.bbox)
    if hasattr(asset, 'corners'):
        snap.corners = _clone_tensor_or_none(asset.corners)
    return snap


def exec_safe(code_str, namespace):
    namespace.update({'exec': lambda *args, **kwargs: None, 'eval': lambda *args, **kwargs: None})
    try:
        exec(code_str, namespace)
    except Exception as e:
        print(f"Error executing code:\n{code_str}")
        raise e
    return namespace


def ensure_grad(assets, keys=('pos', 'phy', 'rot')):
    for obj in assets.values():
        if isinstance(obj, dict):
            for key in keys:
                if key not in obj:
                    continue
                if isinstance(obj[key], torch.Tensor):
                    if not obj[key].requires_grad:
                        obj[key].requires_grad_(True)
                else:
                    obj[key] = torch.tensor(obj[key]).float()
                    obj[key].requires_grad_(True)
        else:
            for attr in ('pos', 'rot'):
                val = getattr(obj, attr, None)
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    if not val.requires_grad:
                        val.requires_grad_(True)
                else:
                    setattr(obj, attr, torch.tensor(val, dtype=torch.float32, requires_grad=True))
    return assets
    
def recaculate_bbox(assets):
    '''
    Recalculate the bounding box of the asset based on its position and rotation.
    '''
    for asset in assets.values():
        if isinstance(asset, dict):
            rot = asset.get('phy', asset.get('rot'))
            bbox = asset.get('bbox')
            pos = asset.get('pos')
            # ensure tensors and matching device/dtype
            if isinstance(pos, torch.Tensor):
                device = pos.device
                dtype = pos.dtype
            else:
                device = None
                dtype = None
            if rot is None or bbox is None:
                continue
            if not isinstance(rot, torch.Tensor):
                rot = torch.as_tensor(rot, dtype=dtype or torch.float32, device=device)
                asset['phy'] = rot
            if not isinstance(bbox, torch.Tensor):
                bbox = torch.as_tensor(bbox, dtype=rot.dtype, device=rot.device)
                asset['bbox'] = bbox
            # compute local corners (relative offsets); do NOT add position here
            asset['corners'] = compute_rotated_corners(rot, bbox)
        else:
            # ensure tensors for node objects
            if getattr(asset, 'rot', None) is None or getattr(asset, 'bbox', None) is None:
                continue
            if not torch.is_tensor(asset.rot):
                asset.rot = torch.as_tensor(asset.rot, dtype=torch.float32)
            if not torch.is_tensor(asset.bbox):
                asset.bbox = torch.as_tensor(asset.bbox, dtype=asset.rot.dtype, device=asset.rot.device)
            asset.corners = compute_rotated_corners(asset.rot, asset.bbox)

def recaculate_bbox_np(assets):
    """
    Recalculate the bounding box of the asset based on its position and rotation.
    Supports both torch.Tensor and numpy.ndarray.
    """
    for asset in assets.values():
        rot = asset.rot
        bbox = asset.bbox
        asset.corners = compute_rotated_corners_np(
            np.asarray(rot, dtype=float).reshape(-1),
            np.asarray(bbox, dtype=float).reshape(-1),
        )

def recaculate_bbox_w_remove(assets, region_bound):
    '''
    Recalculate the bounding box of the asset based on its position and rotation.
    '''
    xmin, xmax, ymin, ymax = region_bound
    tol = 0.03
    new_assets = {}
    for key, asset in assets.items():
        if isinstance(asset, dict):
            rot = asset.get('phy', asset.get('rot'))
            pos = asset['pos']
            bbox = asset['bbox']
        else:
            rot = asset.rot
            pos = asset.pos
            bbox = asset.bbox
        corners = compute_rotated_corners(rot, bbox)
        if isinstance(asset, dict):
            asset['corners'] = corners
        else:
            asset.corners = corners
        corners_pos = corners + pos[:2]
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


def _asset_pos_value(asset):
    if isinstance(asset, dict):
        return asset.get("pos")
    return getattr(asset, "pos", None)


def _asset_rot_value(asset):
    if isinstance(asset, dict):
        if "phy" in asset:
            return asset.get("phy")
        return asset.get("rot")
    return getattr(asset, "rot", None)


def _asset_bbox_value(asset):
    if isinstance(asset, dict):
        return asset.get("bbox")
    return getattr(asset, "bbox", None)


def _ensure_tensor(value, device=None, dtype=None):
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, device=device, dtype=dtype)


def _ensure_corners(asset):
    if isinstance(asset, dict):
        if asset.get("corners") is not None:
            return asset["corners"]
        rot = _asset_rot_value(asset)
        bbox = _asset_bbox_value(asset)
        if rot is None or bbox is None:
            return None
        rot = _ensure_tensor(rot)
        bbox = _ensure_tensor(bbox, device=rot.device, dtype=rot.dtype)
        corners = compute_rotated_corners(rot, bbox)
        asset["corners"] = corners
        return corners
    if getattr(asset, "corners", None) is not None:
        return asset.corners
    rot = _asset_rot_value(asset)
    bbox = _asset_bbox_value(asset)
    if rot is None or bbox is None:
        return None
    rot = _ensure_tensor(rot)
    bbox = _ensure_tensor(bbox, device=rot.device, dtype=rot.dtype)
    corners = compute_rotated_corners(rot, bbox)
    asset.corners = corners
    return corners


def build_parent_bounds(assets: dict, fixed_assets: Optional[dict] = None):
    bounds_by_key = {}
    fixed_assets = fixed_assets or {}
    for key, asset in assets.items():
        parent_key = getattr(asset, "parent", None)
        if not parent_key:
            continue
        # print("fixed_assets:", fixed_assets.keys())
        parent = assets.get(parent_key) or fixed_assets.get(parent_key)
        if parent is None:
            # print(f"[Warning] parent '{parent_key}' not found for '{key}'.")
            continue
        corners = _ensure_corners(parent)
        pos = _asset_pos_value(parent)
        if corners is None or pos is None:
            continue
        pos = _ensure_tensor(pos, device=corners.device, dtype=corners.dtype)
        corners_world = corners + pos[:2]
        min_xy = corners_world.min(dim=0).values
        max_xy = corners_world.max(dim=0).values
        bounds_by_key[key] = (float(min_xy[0]), float(max_xy[0]), float(min_xy[1]), float(max_xy[1]))
    return bounds_by_key


def build_parent_z_locks(assets: dict, fixed_assets: Optional[dict] = None):
    z_locks = {}
    fixed_assets = fixed_assets or {}
    for key, asset in assets.items():
        parent_key = getattr(asset, "parent", None)
        if not parent_key:
            continue
        print("fixed_assets:", fixed_assets.keys())
        parent = assets.get(parent_key) or fixed_assets.get(parent_key)
        if parent is None:
            print(f"[Warning] parent '{parent_key}' not found for '{key}'.")
            continue
        pos = _asset_pos_value(parent)
        bbox = _asset_bbox_value(parent)
        if pos is None or bbox is None:
            continue
        pos = _ensure_tensor(pos)
        bbox = _ensure_tensor(bbox, device=pos.device, dtype=pos.dtype)
        z_top = pos[2] + bbox[2]
        z_locks[key] = float(z_top)
    return z_locks


def init_assets(assets, code_str):
    """
    Loads assets and get_constraints from the given code string.
    Parameters:
        code_str: Code string containing the definitions of assets and get_constraints.
    Return:
        (assets, get_constraints)
    """
    code_str = re.search(r"```python\n(.+?)```", code_str, re.DOTALL).group(1)

    namespace  = {
        "torch": torch,
        "tensor": torch.tensor,
        "obj": assets
    }

    exec_safe(code_str, namespace)
    assets = namespace.get("obj")    
    return assets


def cal_single_loss(assets, region_bound):
    recaculate_bbox(assets)
    with torch.no_grad():
        loss_physics = physics_loss(assets, region_bound)
        loss_project = project_loss(assets, None, x_min=region_bound[0], x_max=region_bound[1], y_min=region_bound[2], y_max=region_bound[3], z_min=0.0)
    return loss_project + loss_physics

def cal_initial_loss(assets, solver, room_bound, other_assets=None, region_center=None, z_min=0.0, weight_semantic=1.0, weight_physics=1.0, bounds_by_key=None):
    # if other_assets:
    #     recaculate_bbox(other_assets)
    recaculate_bbox(assets)
    if solver:
        loss_semantic = semantic_loss(solver, assets) * weight_semantic
    else:
        loss_semantic = 0.0
    wall_assignments = solver.build_wall_assignments() if solver is not None else None
    walls = getattr(solver, "wall", None) if solver is not None else None
    loss_physics = physics_loss(assets, room_bound, other_assets, wall_assignments=wall_assignments, walls=walls) * weight_physics
    loss_project = project_loss(assets, region_center, x_min=room_bound[0], x_max=room_bound[1], y_min=room_bound[2], y_max=room_bound[3], z_min=z_min, bounds_by_key=bounds_by_key, weight=weight_physics)
    return loss_semantic + loss_project + loss_physics

def cal_initial_loss_group(assets, solver, groups, room_bound, other_assets=None, region_center=None, z_min=0.0, weight_semantic=1.0, weight_physics=1.0, weight_project=1.0, bounds_by_key=None):
    # if other_assets:
    #     recaculate_bbox(other_assets)
    recaculate_bbox(assets)
    if solver:
        loss_g, total_loss_s = semantic_loss_group(solver, groups, assets)
        loss_semantic = total_loss_s * weight_semantic
    else:
        loss_g = torch.zeros(len(groups))
        loss_semantic = 0.0
    wall_assignments = solver.build_wall_assignments() if solver is not None else None
    walls = getattr(solver, "wall", None) if solver is not None else None
    loss_physics = physics_loss(assets, room_bound, other_assets, wall_assignments=wall_assignments, walls=walls) * weight_physics
    loss_project = project_loss(assets, region_center, x_min=room_bound[0], x_max=room_bound[1], y_min=room_bound[2], y_max=room_bound[3], z_min=z_min, bounds_by_key=bounds_by_key) * weight_project
    total_loss = loss_semantic + loss_project + loss_physics
    return loss_g, total_loss

def optimize_pose(
    iterations,
    name,
    assets,
    solver,
    room_bound,
    region_center,
    weight,
    z_min=0.0,
    verbose=False,
    glb_paths=[],
    vis=False,
    door=None,
    bounds_by_key=None,
    vis_cb=None,
    vis_every=None,
    z_locks=None,
):
    first_iter = 1
    params = []
    for asset in assets.values():
        pos_ref = _asset_pos_ref(asset)
        rot_ref = _asset_rot_ref(asset)
        if pos_ref is not None and getattr(pos_ref, "requires_grad", False):
            params.append({'params': pos_ref, 'lr': 5e-2 * weight})
        if rot_ref is not None and getattr(rot_ref, "requires_grad", False):
            params.append({'params': rot_ref, 'lr': 1e-1 * weight})
    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.92)
    
    if vis:
        from utils.visualization import render_scene_sync
        objs, scene, cam_top, cam_front, render_proc, render_q = render_scene_sync(
            assets, name, scene=None, objs=[], cam_top=None, cam_front=None, init=True, data_path=glb_paths
        )
    # if other_assets:
    #     recaculate_bbox(other_assets)

    best_loss = float('inf')
    best_assets = None
    best_grad_dict = None
    recaculate_bbox(assets)
    wall_assignments = solver.build_wall_assignments() if solver is not None else None
    walls = getattr(solver, "wall", None) if solver is not None else None
    if z_locks:
        with torch.no_grad():
            for key, z_val in z_locks.items():
                if key in assets:
                    pos_ref = _asset_pos_ref(assets[key])
                    if pos_ref is not None:
                        pos_ref[2] = z_val
    # door = None
    # print(list(assets.keys()))
    if door:
        recaculate_bbox(door)
    for step in tqdm(range(first_iter, iterations + 1), desc="optimization progress"):
        optimizer.zero_grad()
        loss_semantic = semantic_loss(solver, assets)
        loss_physics = physics_loss(assets, room_bound, door, wall_assignments=wall_assignments, walls=walls)
        loss_project = project_loss(assets, region_center, x_min=room_bound[0], x_max=room_bound[1], y_min=room_bound[2], y_max=room_bound[3], z_min=z_min, bounds_by_key=bounds_by_key)
        loss = loss_semantic + loss_project + loss_physics
        loss.backward()

        curr_loss_val = loss.item()
        if curr_loss_val < best_loss:
            if verbose:
                print(f"[Best Update] step {step}: loss improved {best_loss:.6f} -> {curr_loss_val:.6f}")
            # print(f"Step {step}, Loss Item: {loss.item()}, loss_semantic: {loss_semantic.item()}, loss_physics: {loss_physics.item()}")
            best_loss = curr_loss_val
            # best_loss_list = [loss_semantic.detach().clone(), loss_project.detach().clone(), loss_physics.detach().clone()]

            # save parameters snapshot (detach + clone)
            best_assets = {k: _snapshot_asset(v) for k, v in assets.items()}

            # save gradients snapshot
            grad_dict = {}
            for obj_idx, asset in assets.items():
                pos_ref = _asset_pos_ref(asset)
                rot_ref = _asset_rot_ref(asset)
                if pos_ref is None:
                    pos_grad = None
                else:
                    pos_grad = pos_ref.grad.clone().detach() if pos_ref.grad is not None else torch.zeros_like(pos_ref)
                if rot_ref is None:
                    rot_grad = None
                else:
                    rot_grad = rot_ref.grad.clone().detach() if rot_ref.grad is not None else torch.zeros_like(rot_ref)
                grad_dict[obj_idx] = {'pos': pos_grad, 'phy': rot_grad}
            best_grad_dict = grad_dict
            # print(grad_dict)
        # update parameters
        optimizer.step()
        if z_locks:
            with torch.no_grad():
                for key, z_val in z_locks.items():
                    if key in assets:
                        pos_ref = _asset_pos_ref(assets[key])
                        if pos_ref is not None:
                            pos_ref[2] = z_val
        scheduler.step()
        recaculate_bbox(assets)
        if vis_cb is not None and vis_every is not None:
            if step % vis_every == 0 or step == iterations:
                vis_cb(step, assets)
        if step % 50 == 0 or step == iterations:
            tqdm.write(f"Step {step} - Current Loss: {loss.item():.7f}")
            if verbose:
                print(f"Step {step}, Loss Item: {loss.item()}, loss_semantic: {loss_semantic.item()}, loss_physics: {loss_physics.item()}")
                plot(step, assets)

        if step % 50 == 0 and step != iterations and vis:
            _, scene, cam_top, cam_front, _, _ = render_scene_sync(
                assets, name, scene, objs, cam_top, cam_front, init=False, render_queue=render_q, data_path=glb_paths
            )
    if vis:
        render_scene_sync(assets, name, scene, objs, cam_top, cam_front, stop=True, init=False, render_queue=render_q, data_path=glb_paths)
        if render_proc is not None:
            render_proc.join()
        del sys.modules["utils.visualization"]
    print("SGD_loss:", best_loss)
    return best_assets, best_grad_dict, best_loss, solver

def optimize_pose_region(
    iterations,
    name,
    assets,
    solver,
    room_bound,
    region_center,
    weight,
    z_min=0.0,
    verbose=False,
    glb_paths=[],
    vis=False,
    door=None,
    pure_gd=False,
    trainable_keys=None,
    bounds_by_key=None,
    z_locks=None,
    vis_cb=None,
    vis_every=None,
):
    first_iter = 1
    params = []
    for asset in assets.values():
        pos_ref = _asset_pos_ref(asset)
        rot_ref = _asset_rot_ref(asset)
        if pos_ref is not None and getattr(pos_ref, "requires_grad", False):
            params.append({'params': pos_ref, 'lr': 5e-3 * weight})
        if rot_ref is not None and getattr(rot_ref, "requires_grad", False):
            params.append({'params': rot_ref, 'lr': 1e-1 * weight})
    if pure_gd:
        # optimizer = torch.optim.AdamW(params, betas=(0.85, 0.999), eps=1e-8, weight_decay=3e-4)
        # optimizer = torch.optim.SGD(params, momentum=0.0)
        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)
    
    if vis:
        from utils.visualization import render_scene_sync
        objs, scene, cam_top, cam_front, render_proc, render_q = render_scene_sync(
            assets, name, scene=None, objs=[], cam_top=None, cam_front=None, init=True, data_path=glb_paths
        )
    # if other_assets:
    #     recaculate_bbox(other_assets)

    best_loss = float('inf')
    best_assets = None
    best_grad_dict = None
    recaculate_bbox(assets)
    wall_assignments = solver.build_wall_assignments() if solver is not None else None
    walls = getattr(solver, "wall", None) if solver is not None else None
    if z_locks:
        with torch.no_grad():
            for key, z_val in z_locks.items():
                if key in assets:
                    pos_ref = _asset_pos_ref(assets[key])
                    if pos_ref is not None:
                        pos_ref[2] = z_val
    door = None
    if door:
        recaculate_bbox(door)
    for step in tqdm(range(first_iter, iterations + 1), desc="optimization progress"):
        optimizer.zero_grad()
        loss_semantic = semantic_loss(solver, assets)
        loss_physics = physics_loss(assets, room_bound, door, wall_assignments=wall_assignments, walls=walls)
        loss_project = project_loss(assets, region_center, x_min=room_bound[0], x_max=room_bound[1], y_min=room_bound[2], y_max=room_bound[3], z_min=z_min, bounds_by_key=bounds_by_key, weight=2.0)
        loss = loss_semantic + loss_project + loss_physics * 2.0
        loss.backward()

        curr_loss_val = loss.item()
        if curr_loss_val < best_loss:
            if verbose:
                print(f"[Best Update] step {step}: loss improved {best_loss:.6f} -> {curr_loss_val:.6f}")
            best_loss = curr_loss_val
            # best_loss_list = [loss_semantic.detach().clone(), loss_project.detach().clone(), loss_physics.detach().clone()]

            # save parameters snapshot (detach + clone)
            best_assets = {k: _snapshot_asset(v) for k, v in assets.items()}

            # save gradients snapshot
            grad_dict = {}
            for obj_idx, asset in assets.items():
                pos_ref = _asset_pos_ref(asset)
                rot_ref = _asset_rot_ref(asset)
                if pos_ref is None:
                    pos_grad = None
                else:
                    pos_grad = pos_ref.grad.clone().detach() if pos_ref.grad is not None else torch.zeros_like(pos_ref)
                if rot_ref is None:
                    rot_grad = None
                else:
                    rot_grad = rot_ref.grad.clone().detach() if rot_ref.grad is not None else torch.zeros_like(rot_ref)
                grad_dict[obj_idx] = {'pos': pos_grad, 'phy': rot_grad}
            best_grad_dict = grad_dict
            # print(grad_dict)
        # update parameters
        optimizer.step()
        if z_locks:
            with torch.no_grad():
                for key, z_val in z_locks.items():
                    if key in assets:
                        pos_ref = _asset_pos_ref(assets[key])
                        if pos_ref is not None:
                            pos_ref[2] = z_val
        scheduler.step()
        recaculate_bbox(assets)
        if vis_cb is not None and vis_every is not None:
            if step % vis_every == 0 or step == iterations:
                vis_cb(step, assets)
        if step % 50 == 0 or step == iterations:
            tqdm.write(f"Step {step} - Current Loss: {loss.item():.7f}")
            if verbose:
                print(f"Step {step}, Loss Item: {loss.item()}, loss_semantic: {loss_semantic.item()}, loss_physics: {loss_physics.item()}")
                plot(step, assets)

        if step % 50 == 0 and step != iterations and vis:
            _, scene, cam_top, cam_front, _, _ = render_scene_sync(
                assets, name, scene, objs, cam_top, cam_front, init=False, render_queue=render_q, data_path=glb_paths
            )
    if vis:
        render_scene_sync(assets, name, scene, objs, cam_top, cam_front, stop=True, init=False, render_queue=render_q, data_path=glb_paths)
        if render_proc is not None:
            render_proc.join()
        del sys.modules["utils.visualization"]
    print("SGD_loss:", best_loss)
    return best_assets, best_grad_dict, best_loss, solver
