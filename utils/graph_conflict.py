"""
Graph-based constraint conflict detection (rule engine for SceneGraph / ConstraintSolver).

Canonical names (this repo): distance, place_align, align_with, point_towards, against, surround
Small / shelf: place_align_small, against_edge, point_to_edge, center, next_to
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from utils.tool import Point, parse_dsl_fixed_points

# Canonical constraint sets (system_prompts/constraints.txt)
POSITION_CONSTRAINT_TYPES = frozenset({
    "distance",
    "place_align",
    "place_align_small",
    "against",
    "against_edge",
    "surround",
    "center",
    "on",
    "above",
    "next_to",
})

ORIENTATION_CONSTRAINT_TYPES = frozenset({
    "align_with",
    "point_towards",
    "against",
    "against_edge",
    "point_to_edge",
    "place_align",
    "place_align_small",
    "surround",
    "next_to",
})

DIRECTION_EDGE_TYPES = frozenset({"place_align", "place_align_small"})

# Map legacy / GlobalLayout names to this repo's canonical names.
LEGACY_TO_CANONICAL = {
    "near": "distance",
    "against_wall": "against",
    "align_wall": "align_with",
    "side_of": "place_align",
    "center_align": "align_with",
}

WALL_TO_DIRECTION = {
    "west_wall": "right",
    "east_wall": "left",
    "south_wall": "up",
    "north_wall": "down",
    "left_wall": "left",
    "right_wall": "right",
    "front_wall": "down",
    "back_wall": "up",
}

WALL_TO_AXIS = {
    "west_wall": 1,
    "east_wall": 1,
    "south_wall": 0,
    "north_wall": 0,
    "left_wall": 1,
    "right_wall": 1,
    "front_wall": 0,
    "back_wall": 0,
}

# Max share of open-surface edge length that against_edge objects may occupy (per region).
EDGE_OCCUPANCY_RATIO = 0.85
# Cross-edge against_edge layouts are risky when both perpendicular edge pairs
# are nearly saturated on the same small open surface.
CROSS_EDGE_OCCUPANCY_RATIO = 0.75
CROSS_EDGE_MIN_COUNT_PER_AXIS = 2

EDGE_TO_BBOX_AXIS = {
    "left": 1,
    "right": 1,
    "front": 0,
    "back": 0,
}

SIDE_OF_DIRECTION_TO_ANGLE = {
    "right": 0.0,
    "up": 90.0,
    "left": 180.0,
    "down": -90.0,
    "front": 0.0,
    "back": 180.0,
}


def _region_edge_length(region_bound, edge: str) -> Optional[float]:
    if region_bound is None or len(region_bound) < 4:
        return None
    xmin, xmax, ymin, ymax = region_bound[:4]
    if edge in ("left", "right"):
        return float(ymax - ymin)
    if edge in ("front", "back"):
        return float(xmax - xmin)
    return None


def _region_width_height(region_bound) -> tuple[Optional[float], Optional[float]]:
    if region_bound is None or len(region_bound) < 4:
        return None, None
    xmin, xmax, ymin, ymax = region_bound[:4]
    return float(xmax - xmin), float(ymax - ymin)


def normalize_constraint_type(name: str) -> str:
    if name is None:
        return name
    return LEGACY_TO_CANONICAL.get(name, name)


def angle_from_side_of_direction(direction: str) -> float:
    if isinstance(direction, (int, float)):
        return float(direction)
    if not isinstance(direction, str):
        return 0.0
    key = direction.strip().lower()
    return SIDE_OF_DIRECTION_TO_ANGLE.get(key, 0.0)


def side_of_direction_from_angle(angle: float) -> str:
    ang = float(angle)
    ang = (ang + 180.0) % 360.0 - 180.0
    candidates = [(0.0, "right"), (90.0, "up"), (-90.0, "down"), (180.0, "left"), (-180.0, "left")]
    best = min(candidates, key=lambda x: abs(ang - x[0]))
    return best[1]


def _graph_edge_first_extra_arg(extra_args) -> Any:
    if not extra_args:
        return None
    try:
        return extra_args[0]
    except (TypeError, IndexError, KeyError):
        return None


def _edge_type(data: dict) -> str:
    return normalize_constraint_type(data.get("type"))


def _place_align_direction_from_edge(data: dict) -> Optional[str]:
    raw = _graph_edge_first_extra_arg(data.get("extra_args"))
    if raw is None:
        return None
    if isinstance(raw, str):
        d = raw.strip().lower()
        if d in PLACE_ALIGN_DIRECTIONS:
            return d
        return d
    try:
        return side_of_direction_from_angle(float(raw))
    except (TypeError, ValueError):
        return None


PLACE_ALIGN_DIRECTIONS = frozenset({"up", "down", "left", "right", "front", "back"})


def _side_of_direction_from_edge(data: dict) -> Optional[str]:
    return _place_align_direction_from_edge(data)


def _against_edge_direction_from_edge(data: dict) -> Optional[str]:
    extra = data.get("extra_args") or ()
    for val in reversed(extra):
        if isinstance(val, str):
            key = val.strip().lower()
            if key in PLACE_ALIGN_DIRECTIONS:
                return key
    raw = _graph_edge_first_extra_arg(extra)
    if isinstance(raw, str):
        return raw.strip().lower()
    return None


def _is_scoped_asset_node(node, init_assets: dict) -> bool:
    if isinstance(node, (int, np.integer)):
        return True
    return node in init_assets


FIXED_POINT_OCCUPANCY_MARGIN = 0.15  # meters
FIXED_POINT_ALIGN_MAX_DISTANCE = 0.10  # meters
FIXED_POINT_ALIGN_COLLISION_TOL = 0.15  # meters

WALL_TO_ROT_RAD = {
    "west_wall": 0.0,
    "east_wall": math.pi,
    "south_wall": math.pi / 2.0,
    "north_wall": 3.0 * math.pi / 2.0,
    "left_wall": 0.0,
    "right_wall": math.pi,
    "front_wall": math.pi / 2.0,
    "back_wall": 3.0 * math.pi / 2.0,
}


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _asset_pose_xy(asset) -> Optional[Tuple[float, float, float, float, float]]:
    """Return (cx, cy, rot_rad, width, depth) for an placed asset, or None."""
    pos = getattr(asset, "pos", None)
    rot = getattr(asset, "rot", None)
    bbox = getattr(asset, "bbox", None)
    if pos is None or rot is None or bbox is None:
        return None
    cx = _coerce_float(pos[0] if hasattr(pos, "__getitem__") else None)
    cy = _coerce_float(pos[1] if hasattr(pos, "__getitem__") else None)
    rot_rad = _coerce_float(rot[0] if hasattr(rot, "__getitem__") else rot)
    width = _coerce_float(bbox[0] if hasattr(bbox, "__getitem__") else None)
    depth = _coerce_float(bbox[1] if hasattr(bbox, "__getitem__") else None)
    if None in (cx, cy, rot_rad, width, depth):
        return None
    scale = getattr(asset, "scale", 1.0)
    if isinstance(scale, (list, tuple)):
        scale = scale[0]
    scale_f = _coerce_float(scale)
    if scale_f is not None:
        width *= scale_f
        depth *= scale_f
    return cx, cy, rot_rad, width, depth


def _asset_bbox_xy(asset_meta) -> Optional[Tuple[float, float]]:
    if asset_meta is None:
        return None
    bbox = asset_meta.get("bbox") if isinstance(asset_meta, dict) else getattr(asset_meta, "bbox", None)
    if bbox is None:
        return None
    width = _coerce_float(bbox[0] if hasattr(bbox, "__getitem__") else None)
    depth = _coerce_float(bbox[1] if hasattr(bbox, "__getitem__") else None)
    if width is None or depth is None:
        return None
    scale = asset_meta.get("scale", 1.0) if isinstance(asset_meta, dict) else getattr(asset_meta, "scale", 1.0)
    if isinstance(scale, (list, tuple)):
        scale = scale[0]
    scale_f = _coerce_float(scale)
    if scale_f is not None:
        width *= scale_f
        depth *= scale_f
    return width, depth


def _bbox_long_half_radius(asset_meta) -> float:
    """Half of the longer XY bbox side — placement disk radius at a fixed point."""
    dims = _asset_bbox_xy(asset_meta)
    if dims is None:
        return 0.0
    return max(dims[0], dims[1]) / 2.0


def _wall_align_angle_rad(wall_name: str, rel) -> Optional[float]:
    if wall_name not in WALL_TO_ROT_RAD:
        return None
    angle = WALL_TO_ROT_RAD[wall_name]
    offset = None
    if rel is not None and getattr(rel, "params", None):
        offset = rel.params.get("arg0")
    offset_f = _coerce_float(offset)
    if offset_f is not None:
        angle += math.radians(offset_f)
    return angle


def _distance_relation_max_value(rel) -> Optional[float]:
    if rel is None or getattr(rel, "params", None) is None:
        return None
    if "arg1" in rel.params:
        return _coerce_float(rel.params.get("arg1"))
    return _coerce_float(rel.params.get("arg0"))


def _obb_corners_xy(cx: float, cy: float, rot_rad: float, width: float, depth: float) -> np.ndarray:
    local = np.array(
        [
            [-width / 2.0, -depth / 2.0],
            [-width / 2.0, depth / 2.0],
            [width / 2.0, depth / 2.0],
            [width / 2.0, -depth / 2.0],
        ],
        dtype=float,
    )
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return local @ rot.T + np.array([cx, cy], dtype=float)


def _obb_overlap_xy(corners_a: np.ndarray, corners_b: np.ndarray, margin: float = 0.0) -> bool:
    axes = []
    for corners in (corners_a, corners_b):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            norm = np.linalg.norm(edge)
            if norm <= 1e-8:
                continue
            axes.append(np.array([-edge[1], edge[0]], dtype=float) / norm)
    for axis in axes:
        proj_a = corners_a @ axis
        proj_b = corners_b @ axis
        if proj_a.max() < proj_b.min() - margin or proj_b.max() < proj_a.min() - margin:
            return False
    return True


def _obb_overlap_depth_xy(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    min_depth = float("inf")
    axes = []
    for corners in (corners_a, corners_b):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            norm = np.linalg.norm(edge)
            if norm <= 1e-8:
                continue
            axes.append(np.array([-edge[1], edge[0]], dtype=float) / norm)
    if not axes:
        return 0.0
    for axis in axes:
        proj_a = corners_a @ axis
        proj_b = corners_b @ axis
        depth = min(proj_a.max(), proj_b.max()) - max(proj_a.min(), proj_b.min())
        if depth <= 0.0:
            return 0.0
        min_depth = min(min_depth, float(depth))
    return 0.0 if min_depth == float("inf") else min_depth


def _asset_obb_corners(asset) -> Optional[np.ndarray]:
    pose = _asset_pose_xy(asset)
    if pose is None:
        return None
    cx, cy, rot_rad, width, depth = pose
    return _obb_corners_xy(cx, cy, rot_rad, width, depth)


def _min_distance_xy_to_oriented_box(
    px: float,
    py: float,
    cx: float,
    cy: float,
    rot_rad: float,
    width: float,
    depth: float,
) -> float:
    dx, dy = px - cx, py - cy
    c, s = math.cos(-rot_rad), math.sin(-rot_rad)
    local_x = dx * c - dy * s
    local_y = dx * s + dy * c
    clamp_x = max(-width / 2.0, min(width / 2.0, local_x))
    clamp_y = max(-depth / 2.0, min(depth / 2.0, local_y))
    return math.hypot(local_x - clamp_x, local_y - clamp_y)


def _fixed_point_blocked_by_existing(
    px: float,
    py: float,
    placement_radius: float,
    asset,
    *,
    margin: float = FIXED_POINT_OCCUPANCY_MARGIN,
) -> bool:
    """
    True when a disk centered at the fixed point (radius = longer bbox side / 2)
    overlaps an existing asset OBB.
    """
    pose = _asset_pose_xy(asset)
    if pose is None:
        return False
    cx, cy, rot_rad, width, depth = pose
    dist = _min_distance_xy_to_oriented_box(px, py, cx, cy, rot_rad, width, depth)
    return dist < placement_radius + margin


def _point_inside_asset_xy(px: float, py: float, asset, *, margin: float = FIXED_POINT_OCCUPANCY_MARGIN) -> bool:
    """Legacy point-in-OBB check (center only)."""
    return _fixed_point_blocked_by_existing(px, py, 0.0, asset, margin=margin)


def _resolve_relation_fixed_point(rel, fixed_point_map: Dict[str, Point]) -> Optional[Point]:
    dst = rel.dst
    if isinstance(dst, Point):
        return dst
    if isinstance(dst, str):
        if dst.endswith("_wall") or dst.endswith("_edge"):
            return None
        return fixed_point_map.get(dst)
    return None


def _format_fixed_point_label(rel, point: Point) -> str:
    if isinstance(rel.dst, str):
        return rel.dst
    return f"Point({point.x:.3f}, {point.y:.3f}, {point.z:.3f})"


def _detect_occupied_fixed_point_conflicts(
    edge_center: nx.MultiDiGraph,
    *,
    fixed_point_map: Dict[str, Point],
    init_assets: Optional[dict],
    existing_assets: Optional[dict],
    existing_srcs: Optional[Set],
    active_srcs: Optional[Set],
    id_to_key: dict,
    verbose: bool = True,
    log_lines: Optional[List[str]] = None,
    logged_conflict_keys: Optional[set] = None,
) -> List[dict]:
    """
    Remove distance-to-fixed-point constraints when placing the src asset at the
    anchor (disk radius = longer bbox side / 2) would overlap an existing asset.
    """
    if log_lines is None:
        log_lines = []
    if logged_conflict_keys is None:
        logged_conflict_keys = set()
    if not existing_assets:
        return []

    scope_existing = set(existing_srcs or ())
    placed = {
        k: v for k, v in existing_assets.items()
        if k in scope_existing and _asset_pose_xy(v) is not None
    }
    if not placed:
        return []

    init_assets = init_assets or {}
    conflicts = []
    for src, dst, _key, data in edge_center.edges(keys=True, data=True):
        rel = data.get("relation")
        if rel is None or normalize_constraint_type(rel.name) != "distance":
            continue
        if active_srcs is not None and rel.src not in active_srcs:
            continue
        if scope_existing and rel.src in scope_existing:
            continue

        point = _resolve_relation_fixed_point(rel, fixed_point_map)
        if point is None:
            continue

        src_key = id_to_key.get(rel.src, rel.src)
        src_meta = init_assets.get(src_key) or init_assets.get(rel.src)
        placement_radius = _bbox_long_half_radius(src_meta)
        if placement_radius <= 0:
            continue

        px, py = float(point.x), float(point.y)
        occupier = None
        for ex_key, ex_asset in placed.items():
            if _fixed_point_blocked_by_existing(px, py, placement_radius, ex_asset):
                occupier = ex_key
                break
        if occupier is None:
            continue

        ckey = ("occupied_fixed_point", rel.src, rel.dst, occupier, round(px, 3), round(py, 3))
        if ckey in logged_conflict_keys:
            continue
        logged_conflict_keys.add(ckey)

        src_name = id_to_key.get(rel.src, rel.src)
        dst_label = _format_fixed_point_label(rel, point)
        conflicts.append({
            "type": "occupied_fixed_point",
            "edge": (rel.src, rel.dst, "distance"),
            "occupier": occupier,
            "point": (px, py, float(point.z)),
            "placement_radius": placement_radius,
        })
        if verbose:
            log_lines.append(
                f"[LogicError] occupied_fixed_point: remove distance({src_name}, {dst_label}): "
                f"Point({px:.3f}, {py:.3f}) + r={placement_radius:.3f}m overlaps existing {occupier}; "
                f"redefine a free spot in ImageA"
            )
    return conflicts


def _detect_fixed_point_align_collision_conflicts(
    edge_center: nx.MultiDiGraph,
    *,
    fixed_point_map: Dict[str, Point],
    init_assets: Optional[dict],
    existing_assets: Optional[dict],
    existing_srcs: Optional[Set],
    active_srcs: Optional[Set],
    id_to_key: dict,
    verbose: bool = True,
    log_lines: Optional[List[str]] = None,
    logged_conflict_keys: Optional[set] = None,
) -> List[dict]:
    """
    If distance(src, fixed Point, ..., <=0.1) and align_with(src, wall) together
    fully determine src pose, test that predicted OBB against fixed context.
    Remove align_with (not distance) when this deterministic pose collides.
    """
    if log_lines is None:
        log_lines = []
    if logged_conflict_keys is None:
        logged_conflict_keys = set()

    init_assets = init_assets or {}
    distance_by_src = {}
    align_by_src = {}
    for _src, _dst, _key, data in edge_center.edges(keys=True, data=True):
        rel = data.get("relation")
        if rel is None:
            continue
        rel_type = normalize_constraint_type(rel.name)
        if rel_type == "distance":
            if active_srcs is not None and rel.src not in active_srcs:
                continue
            if existing_srcs and rel.src in existing_srcs:
                continue
            point = _resolve_relation_fixed_point(rel, fixed_point_map)
            max_dist = _distance_relation_max_value(rel)
            if point is not None and max_dist is not None and max_dist <= FIXED_POINT_ALIGN_MAX_DISTANCE:
                distance_by_src[rel.src] = (rel, point, max_dist)
        elif rel_type == "align_with" and isinstance(rel.dst, str) and rel.dst in WALL_TO_ROT_RAD:
            if active_srcs is not None and rel.src not in active_srcs:
                continue
            if existing_srcs and rel.src in existing_srcs:
                continue
            align_by_src[rel.src] = rel

    candidates = []
    for src, (dist_rel, point, max_dist) in distance_by_src.items():
        align_rel = align_by_src.get(src)
        if align_rel is None:
            continue
        src_key = id_to_key.get(src, src)
        src_meta = init_assets.get(src_key) or init_assets.get(src)
        dims = _asset_bbox_xy(src_meta)
        if dims is None:
            continue
        rot_rad = _wall_align_angle_rad(str(align_rel.dst), align_rel)
        if rot_rad is None:
            continue
        width, depth = dims
        corners = _obb_corners_xy(
            float(point.x),
            float(point.y),
            rot_rad,
            width,
            depth,
        )
        candidates.append({
            "src": src,
            "src_key": src_key,
            "distance_rel": dist_rel,
            "align_rel": align_rel,
            "point": point,
            "max_dist": max_dist,
            "corners": corners,
        })

    if not candidates:
        return []

    conflicts = []
    scope_existing = set(existing_srcs or ())
    placed_existing = {
        k: v for k, v in (existing_assets or {}).items()
        if (not scope_existing or k in scope_existing)
    }

    for cand in candidates:
        occupier = None
        for ex_key, ex_asset in placed_existing.items():
            ex_corners = _asset_obb_corners(ex_asset)
            if ex_corners is not None and _obb_overlap_depth_xy(cand["corners"], ex_corners) > FIXED_POINT_ALIGN_COLLISION_TOL:
                occupier = ex_key
                break
        if occupier is None:
            continue
        _append_fixed_point_align_conflict(
            conflicts,
            cand,
            occupier,
            id_to_key,
            verbose=verbose,
            log_lines=log_lines,
            logged_conflict_keys=logged_conflict_keys,
        )

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if _obb_overlap_depth_xy(a["corners"], b["corners"]) <= FIXED_POINT_ALIGN_COLLISION_TOL:
                continue
            _append_fixed_point_align_conflict(
                conflicts,
                a,
                b["src_key"],
                id_to_key,
                verbose=verbose,
                log_lines=log_lines,
                logged_conflict_keys=logged_conflict_keys,
            )
            _append_fixed_point_align_conflict(
                conflicts,
                b,
                a["src_key"],
                id_to_key,
                verbose=verbose,
                log_lines=log_lines,
                logged_conflict_keys=logged_conflict_keys,
            )
    return conflicts


def _append_fixed_point_align_conflict(
    conflicts: List[dict],
    cand: dict,
    occupier,
    id_to_key: dict,
    *,
    verbose: bool,
    log_lines: List[str],
    logged_conflict_keys: set,
) -> None:
    align_rel = cand["align_rel"]
    point = cand["point"]
    ckey = (
        "fixed_point_align_collision",
        align_rel.src,
        align_rel.dst,
        occupier,
        round(float(point.x), 3),
        round(float(point.y), 3),
    )
    if ckey in logged_conflict_keys:
        return
    logged_conflict_keys.add(ckey)
    conflicts.append({
        "type": "fixed_point_align_collision",
        "edge": (align_rel.src, align_rel.dst, "align_with"),
        "occupier": occupier,
        "point": (float(point.x), float(point.y), float(point.z)),
    })
    if verbose:
        src_name = id_to_key.get(align_rel.src, align_rel.src)
        wall_name = id_to_key.get(align_rel.dst, align_rel.dst)
        log_lines.append(
            f"[LogicError] fixed_point_align_collision: remove align_with({src_name}, \"{wall_name}\"): "
            f"fixed Point({float(point.x):.3f}, {float(point.y):.3f}) with this wall alignment collides with {occupier}; "
            f"try aligning {src_name} with a perpendicular wall"
        )


def _detect_global_wall_conflicts(
    G: nx.MultiDiGraph,
    init_assets: dict,
    id_to_key: dict,
    wall_bound,
    *,
    verbose: bool = True,
    log_lines: Optional[List[str]] = None,
    logged_conflict_keys: Optional[set] = None,
    active_srcs: Optional[Set] = None,
    first_pass: bool = True,
) -> List[dict]:
    """Aggregate against-wall occupancy on the full scene graph; auto-remove one against edge per wall."""
    if log_lines is None:
        log_lines = []
    if logged_conflict_keys is None:
        logged_conflict_keys = set()
    if wall_bound is None:
        return []

    conflicts = []
    wall_occupy = defaultdict(float)
    wall_edge_entries: Dict[Any, List[tuple]] = defaultdict(list)

    for u, v, d in G.edges(data=True):
        if _edge_type(d) != "against" or str(v) not in WALL_TO_AXIS:
            continue
        axis = WALL_TO_AXIS[str(v)]
        asset_key = id_to_key.get(u, u)
        if asset_key not in init_assets:
            continue
        bbox = init_assets[asset_key].get("bbox", [0, 0, 0])
        contrib = float(bbox[axis]) if len(bbox) > axis else 0.0
        wall_occupy[v] += contrib
        wall_edge_entries[v].append((u, v, contrib))

    for wall_name, total_len in wall_occupy.items():
        axis = WALL_TO_AXIS.get(str(wall_name))
        bound = wall_bound[axis] if axis is not None else None
        if bound is None or total_len <= 0.8 * bound:
            continue
        ckey = ("aggregate_wall_conflict", wall_name)
        if ckey in logged_conflict_keys:
            continue
        logged_conflict_keys.add(ckey)

        candidates = list(wall_edge_entries.get(wall_name, []))
        if active_srcs is not None:
            scoped = [
                (u, v, c) for u, v, c in candidates
                if u in active_srcs or id_to_key.get(u, u) in active_srcs
            ]
            if scoped:
                candidates = scoped
        if not candidates:
            continue

        pick_list = list(reversed(candidates)) if first_pass else candidates
        u, v, _ = pick_list[0]
        u_name = id_to_key.get(u, u)
        conflicts.append({
            "type": "aggregate_wall_conflict",
            "wall": wall_name,
            "edge": (u, v, "against"),
            "reason": (
                f"Total occupied length {total_len:.3f} exceeds "
                f"bound {bound:.3f}"
            ),
        })
        if verbose:
            log_lines.append(
                f"[LogicError] aggregate_wall_conflict: {wall_name} "
                f"exceeds {bound:.3f}, remove against: {u_name}"
            )
    return conflicts


def detect_conflicts_in_graph(
    G: nx.MultiDiGraph,
    groups: List[set],
    init_assets: dict,
    id_to_key: dict,
    bound=None,
    *,
    verbose: bool = True,
    first_pass: bool = True,
    small_ids: Optional[Set] = None,
    log_lines: Optional[List[str]] = None,
    region_bounds: Optional[Dict[str, tuple]] = None,
    global_graph: Optional[nx.MultiDiGraph] = None,
    global_init_assets: Optional[dict] = None,
    active_srcs: Optional[Set] = None,
) -> Tuple[List[dict], dict]:
    """
    Detect structural conflicts on constraint graph G.
    Per-group logic/semantic checks run on subgraphs; wall-length aggregate uses
    ``global_graph`` across the whole scene when provided (ablation-aligned).
    Returns (conflicts, wall_occupy).
    """
    if log_lines is None:
        log_lines = []
    if small_ids is None:
        small_ids = set()

    wall_bound = None
    if bound is not None and len(bound) >= 4:
        wall_bound = [bound[1] - bound[0], bound[3] - bound[2]]

    all_conflicts = []
    wall_occupy = defaultdict(float)
    global_logged = set()
    wall_assets = global_init_assets if global_init_assets is not None else init_assets
    wall_graph = global_graph if global_graph is not None else G
    all_conflicts.extend(
        _detect_global_wall_conflicts(
            wall_graph,
            wall_assets,
            id_to_key,
            wall_bound,
            verbose=verbose,
            log_lines=log_lines,
            logged_conflict_keys=global_logged,
            active_srcs=active_srcs,
            first_pass=first_pass,
        )
    )

    for i, group_nodes in enumerate(groups):
        logged_conflict_keys = set()
        subG = G.copy()
        for node in list(subG.nodes):
            if _is_scoped_asset_node(node, init_assets) and node not in group_nodes:
                subG.remove_node(node)

        node_order = None
        try:
            if group_nodes:
                center = max(group_nodes, key=lambda n: subG.out_degree(n) if n in subG else -1)
                visited = []
                if center in subG:
                    seen = {center}
                    queue = [center]
                    subG_undir = subG.to_undirected()
                    while queue:
                        cur = queue.pop(0)
                        if isinstance(cur, (int, np.integer)):
                            visited.append(cur)
                        for nb in subG_undir.neighbors(cur):
                            if nb not in seen:
                                seen.add(nb)
                                queue.append(nb)
                remaining = [n for n in group_nodes if n not in set(visited)]
                node_order = visited + remaining
        except Exception:
            node_order = None

        conflicts, wall_occupy = _detect_conflicts_one(
            subG,
            wall_occupy,
            logged_conflict_keys,
            id_to_key,
            init_assets,
            wall_bound,
            small_ids,
            first_pass,
            verbose,
            log_lines,
            node_order=node_order,
            region_bounds=region_bounds,
        )
        if conflicts:
            if verbose:
                print(f"[Group {i}] Detected {len(conflicts)} conflicts.")
            all_conflicts.extend(conflicts)
        elif verbose:
            print(f"[Group {i}] OK — no conflicts.")

    return all_conflicts, wall_occupy


def _detect_conflicts_one(
    G,
    wall_occupy,
    logged_conflict_keys,
    id_to_key,
    init_assets,
    wall_bound,
    small_ids,
    first_pass,
    verbose,
    log_lines,
    node_order=None,
    region_bounds=None,
):
    conflicts = []

    def _edge_items_between(a, b):
        items = []
        if G.has_edge(a, b):
            for key, data in G.get_edge_data(a, b).items():
                items.append((a, b, key, data))
        if G.has_edge(b, a):
            for key, data in G.get_edge_data(b, a).items():
                items.append((b, a, key, data))
        return items

    def _collect_out_edge_types(node):
        return [_edge_type(d) for _, _, d in G.out_edges(node, data=True)]

    def _get_against_walls(node):
        walls = []
        for _, v, d in G.out_edges(node, data=True):
            if _edge_type(d) == "against":
                walls.append(v)
        return set(walls)

    # 1) side_of direction cycles
    direction_edges = []
    for u, v, d in G.edges(data=True):
        et = _edge_type(d)
        if et in DIRECTION_EDGE_TYPES:
            direction_edges.append((u, v, et))
    if small_ids:
        direction_edges = [
            (u, v, t) for (u, v, t) in direction_edges
            if u not in small_ids and v not in small_ids
        ]
    G_dir = nx.DiGraph()
    for u, v, et in direction_edges:
        G_dir.add_edge(u, v, type=et)
    try:
        for cyc in nx.simple_cycles(G_dir):
            if len(cyc) > 1:
                u, v = cyc[-1], cyc[0]
                etype = G_dir.edges[u, v]["type"]
                conflicts.append({"type": "direction_cycle", "edge": (u, v, etype)})
                if verbose:
                    u_name = id_to_key.get(u, u)
                    v_name = id_to_key.get(v, v)
                    log_lines.append(
                        f"[LogicError] Direction cycle detected: "
                        f"{' → '.join(map(str, cyc))}, removing edge {u_name}->{v_name}"
                    )
    except Exception:
        pass

    # 2) distance inconsistency
    dist_edges = []
    for u, v, d in G.edges(data=True):
        if _edge_type(d) != "distance":
            continue
        extra = d.get("extra_args") or ()
        dval = float(extra[0]) if len(extra) > 0 and extra[0] is not None else None
        dist_edges.append((u, v, dval))
    if small_ids:
        dist_edges = [
            (u, v, dval) for (u, v, dval) in dist_edges
            if u not in small_ids and v not in small_ids
        ]
    if dist_edges:
        Gd = nx.Graph()
        for a, b, dval in dist_edges:
            if dval is not None:
                Gd.add_edge(a, b, weight=dval)
        for a, b, dval in dist_edges:
            if a in Gd.nodes and b in Gd.nodes:
                try:
                    sp = nx.shortest_path_length(Gd, a, b, weight="weight")
                    if dval is not None and abs(sp - dval) > 1e-3:
                        conflicts.append({
                            "type": "distance_inconsistent",
                            "edge": (a, b, "distance"),
                        })
                        if verbose:
                            a_name = id_to_key.get(a, a)
                            b_name = id_to_key.get(b, b)
                            log_lines.append(
                                f"[LogicError] distance({a_name},{b_name})={dval}, "
                                f"but path={sp}"
                            )
                except nx.NetworkXNoPath:
                    continue

    wall_to_direction = WALL_TO_DIRECTION
    wall_to_axis = WALL_TO_AXIS

    node_iter = node_order if node_order else list(G.nodes)
    out_edges_map = {}
    place_in_edges_map = {}
    against_dirs_map = {}
    place_dirs_map = {}

    for u in node_iter:
        out_edges = [(u, v, k, d) for u, v, k, d in G.edges(u, keys=True, data=True)]
        in_edges = [(u2, v2, k2, d2) for u2, v2, k2, d2 in G.in_edges(u, keys=True, data=True)]
        out_edges_map[u] = out_edges

        wall_edges = []
        for _, v, _, d in out_edges:
            t = _edge_type(d)
            direction = None
            if t in ("place_align", "place_align_small"):
                direction = _place_align_direction_from_edge(d)
            elif t == "against":
                direction = wall_to_direction.get(str(v), None)
            elif t == "against_edge":
                direction = _against_edge_direction_from_edge(d)
            wall_edges.append((v, t, direction))

        against_dirs_map[u] = [
            dr for _, t, dr in wall_edges if t in ("against", "against_edge") and dr is not None
        ]

        place_in_edges = []
        for src, dst, k, d in in_edges:
            if _edge_type(d) not in ("place_align", "place_align_small"):
                continue
            if src in small_ids:
                continue
            direction = _side_of_direction_from_edge(d)
            if direction is not None:
                place_in_edges.append((src, dst, k, d, direction))
        place_in_edges_map[u] = place_in_edges
        place_dirs_map[u] = [direction for _, _, _, _, direction in place_in_edges]

    # surround fully fixes orientation for each src in src_list; no other rot constraints on those src.
    rot_priority = {"surround": 0, "against": 1, "align_with": 2, "point_towards": 3}
    rot_types = set(rot_priority.keys())

    for u in node_iter:
        if u in small_ids:
            continue
        out_edges = out_edges_map.get(u, [])
        if not out_edges:
            continue

        # rotation de-duplication: surround > against > align_with > point_towards
        rot_type_set = {_edge_type(d) for _, _, _, d in out_edges if _edge_type(d) in rot_types}
        if len(rot_type_set) > 1:
            keep_type = min(rot_type_set, key=lambda t: rot_priority[t])
            edge_iter = reversed(out_edges) if first_pass else out_edges
            for _, v, k, d in edge_iter:
                edge_type = _edge_type(d)
                if edge_type in rot_types and edge_type != keep_type:
                    conflicts.append({
                        "type": "exclusive_rot_conflict",
                        "edge": (u, v, edge_type),
                        "reason": f"duplicate rot constraints, keep {keep_type}",
                    })
                    # against + align_with(wall): remove align_with silently (Genesis-aligned).
                    silent_against_align_wall = (
                        keep_type == "against"
                        and edge_type == "align_with"
                        and str(v) in WALL_TO_AXIS
                    )
                    if verbose and not silent_against_align_wall:
                        u_name = id_to_key.get(u, u)
                        v_name = id_to_key.get(v, v)
                        log_lines.append(
                            f"[LogicError] Deplicate_rot_constraints: {u_name}-{v_name}, "
                            f"remove {edge_type}, keep {keep_type}"
                        )
                    break

        # multiple against targets
        against_targets = [v for _, v, _, d in out_edges if _edge_type(d) == "against"]
        if len(set(against_targets)) > 1:
            scan = list(reversed(out_edges)) if first_pass else out_edges
            for _, v, k, d in scan:
                if _edge_type(d) == "against":
                    conflicts.append({
                        "type": "multi_wall_conflict",
                        "edge": (u, v, _edge_type(d)),
                        "reason": "multiple against to different walls",
                    })
                    if verbose:
                        u_name = id_to_key.get(u, u)
                        v_name = id_to_key.get(v, v)
                        log_lines.append(
                            f"[LogicError] multi_wall_conflict: {u_name} -> {v_name}, remove against"
                        )
                    break

        # multiple side_of with different directions (like against_edge in GraphLayout)
        pa_dirs = []
        for _, v, _, d in out_edges:
            et = _edge_type(d)
            if et in ("place_align", "place_align_small"):
                dr = _place_align_direction_from_edge(d)
                if dr is not None:
                    pa_dirs.append(dr)
        if len(set(pa_dirs)) > 1:
            scan = list(reversed(out_edges)) if first_pass else out_edges
            for _, v, k, d in scan:
                et = _edge_type(d)
                if et in ("place_align", "place_align_small"):
                    conflicts.append({
                        "type": "multi_edge_conflict",
                        "edge": (u, v, et),
                        "reason": "multiple place_align to different directions",
                    })
                    if verbose:
                        u_name = id_to_key.get(u, u)
                        v_name = id_to_key.get(v, v)
                        log_lines.append(
                            f"[LogicError] multi_edge_conflict: {u_name} -> {v_name}, remove place_align"
                        )
                    break

        # against_edge multi-edge conflict (small objects on surfaces)
        against_edge_dirs = []
        for _, _, _, d in out_edges:
            if _edge_type(d) == "against_edge":
                dr = _against_edge_direction_from_edge(d)
                if dr is not None:
                    against_edge_dirs.append(dr)
        if len(set(against_edge_dirs)) > 1:
            scan = list(reversed(out_edges)) if first_pass else out_edges
            for _, v, k, d in scan:
                if _edge_type(d) == "against_edge":
                    conflicts.append({
                        "type": "multi_edge_conflict",
                        "edge": (u, v, "against_edge"),
                        "reason": "multiple against_edge to different edges",
                    })
                    if verbose:
                        u_name = id_to_key.get(u, u)
                        v_name = id_to_key.get(v, v)
                        log_lines.append(
                            f"[LogicError] multi_edge_conflict: {u_name} -> {v_name}, remove against_edge"
                        )
                    break

        # incoming place_align vs outgoing against direction clash
        place_dirs = place_dirs_map.get(u, [])
        against_dirs = against_dirs_map.get(u, [])
        if place_dirs and against_dirs:
            overlap = set(place_dirs) & set(against_dirs)
            if overlap:
                for src, dst, k, d, direction in reversed(place_in_edges_map.get(u, [])):
                    if direction in overlap:
                        conflicts.append({
                            "type": "direction_conflict",
                            "edge": (src, dst, _edge_type(d)),
                            "reason": (
                                f"direction mismatch: place_align {place_dirs} vs against {against_dirs}"
                            ),
                        })
                        if verbose:
                            u_name = id_to_key.get(src, src)
                            v_name = id_to_key.get(dst, dst)
                            log_lines.append(
                                f"[LogicError] direction_conflict: remove place_align: "
                                f"{u_name}-{v_name}:{place_dirs}"
                            )
                        break

    # aggregate open-surface edge occupancy (against_edge on region edges)
    if region_bounds:
        edge_occupy = defaultdict(float)
        edge_entries = defaultdict(list)
        for u, v, d in G.edges(data=True):
            if _edge_type(d) != "against_edge":
                continue
            edge_name = _against_edge_direction_from_edge(d)
            region_key = str(v)
            region_bound = region_bounds.get(region_key)
            if edge_name not in EDGE_TO_BBOX_AXIS or region_bound is None:
                continue
            asset_key = id_to_key.get(u)
            if asset_key not in init_assets:
                continue
            bbox = init_assets[asset_key].get("bbox", [0, 0, 0])
            axis = EDGE_TO_BBOX_AXIS[edge_name]
            occupied_len = float(bbox[axis]) if len(bbox) > axis else 0.0
            edge_occupy[(region_key, edge_name)] += occupied_len
            edge_entries[region_key].append({
                "node": u,
                "region_node": v,
                "edge": edge_name,
                "occupied_len": occupied_len,
                "bbox": bbox,
            })
        for (region_key, edge_name), total_len in edge_occupy.items():
            edge_len = _region_edge_length(region_bounds.get(region_key), edge_name)
            bound = edge_len * EDGE_OCCUPANCY_RATIO if edge_len is not None else None
            if bound is not None and total_len > bound:
                conflicts.append({
                    "type": "aggregate_edge_conflict",
                    "region": region_key,
                    "edge": edge_name,
                    "reason": (
                        f"Total occupied length {total_len:.3f} exceeds "
                        f"bound {bound:.3f} ({EDGE_OCCUPANCY_RATIO:.0%} of edge)"
                    ),
                })
                if verbose:
                    log_lines.append(
                        f"[LogicError] aggregate_edge_conflict: {region_key}:{edge_name} "
                        f"exceeds {bound:.3f}"
                    )

        for region_key, entries in edge_entries.items():
            region_bound = region_bounds.get(region_key)
            width, height = _region_width_height(region_bound)
            if width is None or height is None or width <= 0 or height <= 0:
                continue
            fb_entries = [e for e in entries if e["edge"] in ("front", "back")]
            lr_entries = [e for e in entries if e["edge"] in ("left", "right")]
            if (
                len(fb_entries) < CROSS_EDGE_MIN_COUNT_PER_AXIS
                or len(lr_entries) < CROSS_EDGE_MIN_COUNT_PER_AXIS
            ):
                continue
            fb_occupy = sum(float(e["occupied_len"]) for e in fb_entries)
            lr_occupy = sum(float(e["occupied_len"]) for e in lr_entries)
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
            keep_edges = "left/right" if remove_axis == "fb" else "front/back"
            for entry in remove_entries:
                conflicts.append({
                    "type": "aggregate_cross_edge_conflict",
                    "edge": (entry["node"], entry["region_node"], "against_edge"),
                    "region": region_key,
                    "reason": (
                        f"front/back occupancy {fb_occupy:.3f}/{width:.3f} "
                        f"({fb_ratio:.2f}) and left/right occupancy "
                        f"{lr_occupy:.3f}/{height:.3f} ({lr_ratio:.2f}) are both high; "
                        f"remove {remove_edges} against_edge constraints first and "
                        f"redistribute toward {keep_edges}."
                    ),
                })
                if verbose:
                    u_name = id_to_key.get(entry["node"], entry["node"])
                    log_lines.append(
                        f"[LogicError] aggregate_cross_edge_conflict: region_{region_key} "
                        f"front/back={fb_ratio:.2f}, left/right={lr_ratio:.2f}; "
                        f"remove against_edge on shorter edge pair ({remove_edges}): {u_name}"
                    )

    # semantic group checks (graph-only, aligned with GraphLayout)
    semantic_groups = {
        "bedroom": {
            "keywords": ["bed", "nightstand", "night_stand", "bedside_table", "lamp"],
            "pair_rules": [
                (
                    lambda ka, kb: (
                        ("nightstand" in ka or "bedside_table" in ka or "night_stand" in ka)
                        and "bed" in kb
                    ),
                    ["against"],
                    None,
                    "same_against_wall",
                ),
            ],
        },
        "desk_area": {
            "keywords": ["desk", "chair", "lamp", "bookshelf"],
            "pair_rules": [
                (
                    lambda ka, kb: "chair" in ka and "desk" in kb,
                    ["distance"],
                    (0.0, 1.0),
                    None,
                ),
            ],
        },
        "entertainment": {
            "keywords": ["sofa", "tv", "coffee_table", "armchair", "tv_stand"],
            "pair_rules": [
                (
                    lambda ka, kb: "sofa" in ka and "tv" in kb,
                    ["place_align"],
                    None,
                    None,
                ),
            ],
        },
        "dining_area": {
            "keywords": ["dining_table", "dining_chair", "sideboard"],
            "pair_rules": [
                (
                    lambda ka, kb: ka.startswith("dining_table"),
                    [],
                    None,
                    "not_against_wall",
                ),
            ],
        },
    }

    satisfied_nodes = set()
    nodes = list(G.nodes)
    for i in range(len(nodes)):
        a = nodes[i]
        if a not in id_to_key or a in small_ids:
            continue
        key_a = id_to_key[a]
        if a in satisfied_nodes:
            continue

        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if b not in id_to_key or b in small_ids:
                continue
            key_b = id_to_key.get(b, "")

            for group_name, group in semantic_groups.items():
                keywords = group["keywords"]
                if not (
                    any(k in key_a for k in keywords)
                    and any(k in key_b for k in keywords)
                ):
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

                        relation_ok = False
                        for rtype in required_types:
                            for _, _, _, data in _edge_items_between(a2, b2):
                                if _edge_type(data) != rtype:
                                    continue
                                if allowed_range is not None:
                                    val = _graph_edge_first_extra_arg(data.get("extra_args"))
                                    if val is None:
                                        continue
                                    min_r, max_r = allowed_range
                                    if min_r <= float(val) <= max_r:
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

                        if special == "same_against_wall":
                            walls_a = _get_against_walls(a2)
                            walls_b = _get_against_walls(b2)
                            if not walls_a or not walls_b or not (walls_a & walls_b):
                                reason = (
                                    f"{key_a2} and {key_b2} are expected to be against the same wall"
                                )
                                ckey = ("semantic_conflict", (a2, b2), reason)
                                if ckey not in logged_conflict_keys:
                                    logged_conflict_keys.add(ckey)
                                    conflicts.append({
                                        "type": "semantic_conflict",
                                        "nodes": (a2, b2),
                                        "reason": reason,
                                    })
                                    if verbose:
                                        log_lines.append(
                                            f"[SemanticCommon] {key_a2} vs {key_b2} "
                                            f"not against same wall"
                                        )
                            continue

                        if special == "not_against_wall":
                            walls = _get_against_walls(a2)
                            if walls:
                                reason = f"{key_a2} should not be against wall"
                                ckey = ("semantic_conflict", (a2, walls), reason)
                                if ckey not in logged_conflict_keys:
                                    logged_conflict_keys.add(ckey)
                                    conflicts.append({
                                        "type": "semantic_conflict",
                                        "nodes": (a2, walls),
                                        "reason": reason,
                                    })
                                    if verbose:
                                        log_lines.append(
                                            f"[SemanticCommon] {key_a2} should not be against wall"
                                        )
                            continue

                        if required_types:
                            valid = False
                            for rtype in required_types:
                                for _, _, _, data in _edge_items_between(a2, b2):
                                    if _edge_type(data) != rtype:
                                        continue
                                    if allowed_range is not None:
                                        val = _graph_edge_first_extra_arg(data.get("extra_args"))
                                        if val is None:
                                            continue
                                        min_r, max_r = allowed_range
                                        if min_r <= float(val) <= max_r:
                                            valid = True
                                            break
                                    else:
                                        valid = True
                                        break
                                if valid:
                                    break
                            if not valid:
                                reason = (
                                    f"{key_a2} vs {key_b2} missing/invalid relation {required_types}"
                                )
                                ckey = ("semantic_conflict", (a2, b2), reason)
                                if ckey not in logged_conflict_keys:
                                    logged_conflict_keys.add(ckey)
                                    conflicts.append({
                                        "type": "semantic_conflict",
                                        "nodes": (a2, b2),
                                        "reason": reason,
                                    })
                                    if verbose:
                                        log_lines.append(
                                            f"[SemanticCommon] {key_a2} vs {key_b2} "
                                            f"missing/invalid relation {required_types}"
                                        )
                    except Exception as e:
                        if verbose:
                            log_lines.append(
                                f"[SemanticCheckError] group={group_name}, "
                                f"pair=({key_a},{key_b}), err={e}"
                            )
                        continue

    return conflicts, wall_occupy


def _remove_one_conflict_edge(
    G: nx.MultiDiGraph,
    u,
    v,
    edge_type: str,
    remove_constraint_cb,
    *,
    verbose: bool = True,
) -> bool:
    edge_type = normalize_constraint_type(edge_type)
    edges_to_remove = [
        (key, data)
        for key, data in G.get_edge_data(u, v, default={}).items()
        if _edge_type(data) == edge_type
    ]
    if not edges_to_remove:
        return False
    k, _ = edges_to_remove[0]
    G.remove_edge(u, v, key=k)
    if remove_constraint_cb is not None:
        remove_constraint_cb(u, v, edge_type)
    if verbose:
        print(f"[Auto-Fix] Removed edge: {u} → {v} ({edge_type})")
    return True


def detect_and_repair_graph(
    solver,
    G: nx.MultiDiGraph,
    groups: List[set],
    init_assets: dict,
    id_to_key: dict,
    bound=None,
    *,
    verbose: bool = True,
    first_pass: bool = True,
    max_repair_rounds: int = 32,
    remove_constraint_fn=None,
) -> Tuple[nx.MultiDiGraph, List, List, List[str]]:
    """
    Full detect loop: find conflicts, remove conflicting edges, repeat.

    remove_constraint_fn: callable(solver, u, v, edge_type) — defaults to no-op.
    """
    removed_edges = []
    log_lines = []
    small_ids = _collect_small_asset_ids(init_assets, solver)
    if remove_constraint_fn is None:
        remove_cb = None
    else:
        remove_cb = lambda u, v, t: remove_constraint_fn(solver, u, v, t)

    for _ in range(max_repair_rounds):
        conflicts, _ = detect_conflicts_in_graph(
            G,
            groups,
            init_assets,
            id_to_key,
            bound,
            verbose=verbose,
            first_pass=first_pass,
            small_ids=small_ids,
            log_lines=log_lines,
        )
        edge_conflicts = [c for c in conflicts if "edge" in c]
        if not edge_conflicts:
            if verbose and not conflicts:
                print("[OK] No structural conflicts detected.")
            break

        edge_removed = False
        for c in edge_conflicts:
            u, v, edge_type = c["edge"]
            if _remove_one_conflict_edge(G, u, v, edge_type, remove_cb, verbose=verbose):
                removed_edges.append((u, v, normalize_constraint_type(edge_type)))
                edge_removed = True
        if not edge_removed:
            break

    low_outdegree_nodes = _check_outgoing_completeness(
        G, id_to_key, log_lines, verbose=verbose
    )
    return G, removed_edges, low_outdegree_nodes, log_lines


def _collect_small_asset_ids(init_assets, solver) -> Set:
    if not getattr(solver, "skip_small_conflicts", False):
        return set()
    threshold = float(getattr(solver, "small_area_threshold", 0.0))
    small_ids = set()
    for key, asset in (init_assets or {}).items():
        bbox = asset.get("bbox")
        if bbox is None:
            continue
        try:
            w, l = float(bbox[0]), float(bbox[1])
        except Exception:
            continue
        scale = asset.get("scale", 1.0)
        try:
            scale_f = float(scale[0]) if isinstance(scale, (list, tuple)) else float(scale)
        except Exception:
            scale_f = 1.0
        if w * l * (scale_f ** 2) <= threshold:
            aid = asset.get("id")
            if aid is not None:
                small_ids.add(aid)
    return small_ids


def _check_outgoing_completeness(G, id_to_key, log_lines, verbose=True) -> List:
    low = []
    if getattr(G, "disable_constraint_completeness", False):
        return low
    for node in G.nodes():
        out_types = [
            _edge_type(d) for _, _, d in G.out_edges(node, data=True)
        ]
        if "against" in out_types:
            continue
        has_pos = any(t in POSITION_CONSTRAINT_TYPES for t in out_types)
        has_orient = any(t in ORIENTATION_CONSTRAINT_TYPES for t in out_types)
        if node not in id_to_key:
            continue
        name = id_to_key[node]
        if not has_pos or not has_orient:
            low.append(node)
            parts = []
            if not has_pos:
                parts.append("missing position-type outgoing constraint")
            if not has_orient:
                parts.append("missing orientation-type outgoing constraint")
            msg = f"[ConstraintCompleteness] {name}: {'; '.join(parts)}"
            log_lines.append(msg)
            if verbose:
                print(msg)
    return low


def _relation_extra_args(rel) -> tuple:
    args = []
    i = 0
    while f"arg{i}" in rel.params:
        args.append(rel.params[f"arg{i}"])
        i += 1
    return tuple(args)


def build_graph_from_edge_center(edge_center: nx.MultiDiGraph) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for u, v, key, data in edge_center.edges(keys=True, data=True):
        rel = data.get("relation")
        if rel is None:
            continue
        G.add_edge(
            u,
            v,
            key=key,
            type=normalize_constraint_type(rel.name),
            fn=rel.name,
            extra_args=_relation_extra_args(rel),
            relation=rel,
            kwargs={},
        )
    return G


def collect_small_node_names(
    nodes: Optional[dict],
    threshold: float = 0.10,
) -> Set:
    small = set()
    if not nodes:
        return small
    for name, obj in nodes.items():
        bbox = getattr(obj, "bbox", None)
        if bbox is None:
            continue
        try:
            w, l = float(bbox[0]), float(bbox[1])
        except Exception:
            continue
        if w * l <= threshold:
            small.add(name)
    return small


def _remove_conflict_from_graphs(
    edge_center: nx.MultiDiGraph,
    G: nx.MultiDiGraph,
    u,
    v,
    edge_type: str,
) -> Optional[Any]:
    edge_type = normalize_constraint_type(edge_type)
    rel = None
    for key, data in list(G.get_edge_data(u, v, default={}).items()):
        if _edge_type(data) == edge_type:
            rel = data.get("relation")
            G.remove_edge(u, v, key=key)
            break
    if rel is None:
        return None
    for key, data in list(edge_center.get_edge_data(rel.src, rel.dst, default={}).items()):
        if data.get("relation") == rel:
            edge_center.remove_edge(rel.src, rel.dst, key=key)
            break
    return rel


def detect_scene_graph_conflicts(
    edge_center: nx.MultiDiGraph,
    *,
    nodes: Optional[dict] = None,
    room_bound=None,
    groups: Optional[List[set]] = None,
    active_srcs: Optional[Set] = None,
    existing_srcs: Optional[Set] = None,
    existing_assets: Optional[dict] = None,
    dsl_code: Optional[str] = None,
    fixed_point_map: Optional[Dict[str, Point]] = None,
    skip_small_conflicts: bool = True,
    small_area_threshold: float = 0.10,
    verbose: bool = True,
    first_pass: bool = True,
    max_repair_rounds: int = 32,
    region_bounds: Optional[Dict[str, tuple]] = None,
) -> Tuple[list, list, list]:
    """
    Rule-based conflict detection on SceneGraph.edge_center (Relation on edges).
    Per-region groups detect local logic/semantic conflicts; wall-length aggregate
    uses the full edge_center graph and all known assets (ablation-aligned).
    Returns (removed_relations, low_outdegree_nodes, log_lines).
    """
    removed: List[Any] = []
    log_lines: List[str] = []

    init_assets_all = {}
    id_to_key = {}
    if nodes:
        for name, obj in nodes.items():
            bbox = getattr(obj, "bbox", None)
            init_assets_all[name] = {
                "id": name,
                "bbox": list(bbox) if bbox is not None else [0, 0, 0],
            }
            id_to_key[name] = name

    scoped_assets = dict(init_assets_all)
    if active_srcs is not None:
        scope = set(active_srcs)
        if existing_srcs:
            scope |= set(existing_srcs)
        scoped_assets = {k: v for k, v in init_assets_all.items() if k in scope}

    small_ids: Set = set()
    if skip_small_conflicts:
        small_ids = collect_small_node_names(nodes, small_area_threshold)

    if groups is None:
        asset_nodes = set(nodes.keys()) if nodes else set()
        sub_nodes = {n for n in edge_center.nodes if n in asset_nodes}
        if active_srcs is not None:
            scope = set(active_srcs)
            if existing_srcs:
                scope |= set(existing_srcs)
            sub_nodes = {n for n in sub_nodes if n in scope}
        if sub_nodes:
            undirected = edge_center.subgraph(sub_nodes).to_undirected()
            groups = list(nx.connected_components(undirected))
        else:
            groups = [set(edge_center.nodes)]

    G = build_graph_from_edge_center(edge_center)
    G_full = G
    resolved_fixed_points = dict(fixed_point_map or {})
    resolved_fixed_points.update(parse_dsl_fixed_points(dsl_code or ""))
    occupied_logged: set = set()
    fixed_align_logged: set = set()

    def _active_edge(u, v, data) -> bool:
        if active_srcs is None:
            return True
        rel = data.get("relation")
        if rel is None:
            return True
        if rel.name == "surround" and isinstance(rel.src, list):
            return any(s in active_srcs for s in rel.src)
        if rel.src in active_srcs:
            return True
        if isinstance(rel.dst, str) and rel.dst in active_srcs:
            return True
        return False

    for _ in range(max_repair_rounds):
        occupied_conflicts = _detect_occupied_fixed_point_conflicts(
            edge_center,
            fixed_point_map=resolved_fixed_points,
            init_assets=init_assets_all,
            existing_assets=existing_assets,
            existing_srcs=existing_srcs,
            active_srcs=active_srcs,
            id_to_key=id_to_key,
            verbose=verbose,
            log_lines=log_lines,
            logged_conflict_keys=occupied_logged,
        )
        fixed_align_conflicts = _detect_fixed_point_align_collision_conflicts(
            edge_center,
            fixed_point_map=resolved_fixed_points,
            init_assets=init_assets_all,
            existing_assets=existing_assets,
            existing_srcs=existing_srcs,
            active_srcs=active_srcs,
            id_to_key=id_to_key,
            verbose=verbose,
            log_lines=log_lines,
            logged_conflict_keys=fixed_align_logged,
        )

        conflicts, _ = detect_conflicts_in_graph(
            G,
            groups,
            scoped_assets,
            id_to_key,
            room_bound,
            verbose=verbose,
            first_pass=first_pass,
            small_ids=small_ids,
            log_lines=log_lines,
            region_bounds=region_bounds,
            global_graph=G_full,
            global_init_assets=init_assets_all,
            active_srcs=active_srcs,
        )
        edge_conflicts = occupied_conflicts + fixed_align_conflicts + [c for c in conflicts if "edge" in c]
        if not edge_conflicts:
            break

        edge_removed = False
        for c in edge_conflicts:
            u, v, edge_type = c["edge"]
            edge_data = G.get_edge_data(u, v, default={})
            if not any(_active_edge(u, v, d) for d in edge_data.values()):
                continue
            rel = _remove_conflict_from_graphs(edge_center, G, u, v, edge_type)
            if rel is not None:
                removed.append(rel)
                edge_removed = True
                if verbose:
                    print(
                        f"[Auto-Fix] Removed edge: {u} → {v} "
                        f"({normalize_constraint_type(edge_type)})"
                    )
        if not edge_removed:
            break

    low = _check_outgoing_completeness(G, id_to_key, log_lines, verbose=verbose)
    if nodes:
        low = [n for n in low if n in nodes]
    if active_srcs is not None:
        low = [n for n in low if n in active_srcs]
    return removed, low, log_lines
