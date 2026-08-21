"""
Task asset parsing and layout post-processing utilities.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

UNIT_SCALE = 1.0


def normalize_label(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return str(name).strip().lower().replace(" ", "_").replace("/", "_")


def _category_count(meta: Any) -> int:
    if isinstance(meta, dict):
        return int(meta.get("count", 1))
    if isinstance(meta, (int, float)):
        return int(meta)
    return 1


def summarize_floor_categories(categories: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Return {label: count} for floor asset categories from ``assets_cate``."""
    if not categories:
        return {}
    return {str(name): _category_count(meta) for name, meta in categories.items()}


def format_asset_categories_for_prompt(categories: Optional[Dict[str, Any]]) -> str:
    """Format floor asset label + count for get_areas / get_objects prompts."""
    if not categories:
        return "None (infer floor objects from room description; no fixed asset list provided)"
    lines = []
    for name in sorted(categories.keys()):
        count = _category_count(categories[name])
        lines.append(f"- {name}: count={count}")
    return "\n".join(lines)


def default_test_asset_dir() -> str:
    return os.path.join(genesis_data_root(), "test_asset_dir")


def normalize_assets_cate_entry(meta: Any) -> Dict[str, Any]:
    """Canonical ``assets_cate`` value: ``{"count": N}`` only."""
    if isinstance(meta, dict):
        return {"count": int(meta.get("count", 1))}
    if isinstance(meta, (int, float)):
        return {"count": int(meta)}
    return {"count": 1}


def build_assets_cate_from_task_assets(
    task: Dict[str, Any],
    *,
    test_asset_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse ``task["assets"]`` and return a floor-only
    ``assets_cate`` dict. Works for both run_objaverse and run.py (3D-FUTURE
    index still uses the same test_asset_dir GLBs).
    """
    if not task.get("assets"):
        return {}
    base = test_asset_dir or default_test_asset_dir()
    groups = parse_task_assets(task, base_dir=base)
    floor = groups.get("assets_categories_floor") or {}
    return {str(label): {"count": _category_count(meta)} for label, meta in floor.items()}


def get_floor_categories_from_task(
    task: Dict[str, Any],
    *,
    test_asset_dir: Optional[str] = None,
    prefer_json: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Floor object types + counts for GPT ``categories`` prompt.

    - ``prefer_json=True`` (default): use precomputed ``task["assets_cate"]``.
    - Otherwise, or when JSON field is missing, parse ``task["assets"]``.
    """
    if prefer_json and task.get("assets_cate"):
        return {
            str(k): normalize_assets_cate_entry(v)
            for k, v in task["assets_cate"].items()
        }
    built = build_assets_cate_from_task_assets(task, test_asset_dir=test_asset_dir)
    if built:
        return built
    if task.get("assets_cate"):
        return {
            str(k): normalize_assets_cate_entry(v)
            for k, v in task["assets_cate"].items()
        }
    return None


def genesis_data_root() -> str:
    env_root = os.environ.get("GENESIS_VLM2_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    return os.getcwd()


def resolve_glb_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    """Resolve relative dataset paths from the project root (or GENESIS_VLM2_ROOT)."""
    if not path:
        return None
    path = str(path).strip()
    root = base_dir or genesis_data_root()
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(root, path.lstrip("/")))


def resolve_mesh_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    """Normalize task / FAISS paths to an existing mesh file (.glb / .obj)."""
    if not path:
        return None
    path = str(path).strip()
    if os.path.isdir(path):
        name = os.path.basename(path.rstrip(os.sep))
        for cand in (
            os.path.join(path, f"{name}.glb"),
            os.path.join(path, f"{name}.obj"),
        ):
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        for fname in sorted(os.listdir(path)):
            if fname.endswith((".glb", ".obj")):
                return os.path.abspath(os.path.join(path, fname))
    resolved = resolve_glb_path(path, base_dir=base_dir)
    if resolved and os.path.isfile(resolved):
        return resolved
    return resolved


def resolve_task_asset_path(uid: str, base_dir: str = "test_asset_dir") -> str:
    glb_path = os.path.join(base_dir, uid, f"{uid}.glb")
    resolved = resolve_mesh_path(glb_path, base_dir=base_dir)
    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(f"[TaskAssets] Missing asset file: {glb_path}")
    return resolved


def is_rug_label(label: Optional[str]) -> bool:
    n = normalize_label(label) or ""
    return any(k in n for k in ("rug", "mat", "carpet"))


def is_wall_mount_label(label: Optional[str]) -> bool:
    n = normalize_label(label) or ""
    if not n:
        return False
    if any(
        k in n
        for k in (
            "wall_lamp",
            "wall_light",
            "wall_sconce",
            "sconce",
            "chalkboard",
            "chalk_board",
            "blackboard",
            "wall_mounted",
            "wall_mount",
            "painting",
            "wall_art",
            "wall_mirror",
        )
    ):
        return True
    if "wall" in n and any(k in n for k in ("lamp", "light", "mount", "art", "painting", "sconce")):
        return True
    return False


def split_floor_and_wall_objects(
    objects_in_areas: Dict[str, Any],
    objects_extra: Optional[Dict[str, Any]],
    asset_groups: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split GPT floor plan into floor-only and wall-mount-only structures."""
    wall_labels = set(asset_groups.get("labels_wall") or [])
    wall_paths = asset_groups.get("assets_path_wall") or {}

    extra_by_name: Dict[str, List[Dict]] = {}
    if objects_extra:
        for area in objects_extra.get("areas", []):
            mounts = area.get("wall_mount") or []
            if mounts:
                extra_by_name[area.get("area_name", "")] = mounts

    floor_areas: List[Dict] = []
    wall_areas: List[Dict] = []

    for area in objects_in_areas.get("areas", []):
        area_name = area.get("area_name", "")
        floor_objs: List[Dict] = []
        wall_objs: List[Dict] = []
        seen_wall: Set[str] = set()

        for obj in extra_by_name.get(area_name, []):
            name = normalize_label(obj.get("name")) or ""
            if not name:
                continue
            wall_objs.append({**obj, "name": name})
            seen_wall.add(name)

        for obj in area.get("objects", []):
            name = normalize_label(obj.get("name")) or ""
            if not name or is_rug_label(name):
                continue
            is_wall = (
                name in wall_labels
                or is_wall_mount_label(name)
                or resolve_path_for_label(name, wall_paths) is not None
            )
            if is_wall:
                if name not in seen_wall:
                    wall_objs.append({**obj, "name": name})
                    seen_wall.add(name)
            else:
                floor_objs.append({**obj, "name": name})

        floor_areas.append({"area_name": area_name, "objects": floor_objs})
        if wall_objs:
            wall_areas.append({"area_name": area_name, "objects": wall_objs})

    return {"areas": floor_areas}, {"areas": wall_areas}


def build_task_asset_paths(asset_groups: Dict[str, Any]) -> Dict[str, str]:
    """Merge task asset path maps (floor / wall / ceiling / rug)."""
    merged: Dict[str, str] = {}
    for key in (
        "assets_path_floor",
        "assets_path_wall",
        "assets_path_ceiling",
        "assets_path_rug",
    ):
        merged.update(asset_groups.get(key) or {})
    return merged


def resolve_path_for_label(label: str, paths: Dict[str, str]) -> Optional[str]:
    """Map GPT object name to a task asset glb path; tolerant to minor naming drift."""
    if not paths:
        return None
    n = normalize_label(label.replace(" ", "_"))
    if n in paths:
        return paths[n]

    # exact key after normalize
    norm_paths = {normalize_label(k): v for k, v in paths.items()}
    if n in norm_paths:
        return norm_paths[n]

    alias_groups = (
        {"nightstand", "night_stand", "bedside_table", "night_stand_table"},
        {"rug", "area_rug", "carpet", "floor_mat", "mat"},
        {"chalkboard", "blackboard", "chalk_board"},
        {"wall_lamp", "sconce", "wall_light"},
    )
    for group in alias_groups:
        if n in group:
            for candidate in group:
                if candidate in norm_paths:
                    return norm_paths[candidate]
            for key, val in norm_paths.items():
                if any(a in key or key in a for a in group):
                    return val

    for key, val in norm_paths.items():
        if n in key or key in n:
            return val
    return None


def parse_task_assets(
    task: Dict[str, Any],
    base_dir: str = "test_asset_dir",
) -> Dict[str, Any]:
    """
    Split task['assets'] into floor / rug / wall / ceiling / small groups (main-test.py logic).
    Returns dict with categories, path maps, and labels per group.
    """
    empty = {
        "assets_categories_floor": {},
        "assets_categories_rug": {},
        "assets_categories_wall": {},
        "assets_categories_extra": None,
        "assets_path_floor": {},
        "assets_path_rug": {},
        "assets_path_wall": {},
        "assets_path_ceiling": {},
        "assets_path_small": {},
        "labels_floor": [],
        "labels_rug": [],
        "allowed_small_labels": None,
        "assets_path_small_alias": {},
        "rug_descriptions": {},
    }
    if "assets" not in task:
        if "assets_cate" in task:
            empty["assets_categories_floor"] = task["assets_cate"]
            empty["assets_categories_extra"] = task["assets_cate"]
        return empty

    uid_buff: Dict[str, int] = {}
    assets_categories_floor: Dict[str, Dict] = {}
    assets_path_floor: Dict[str, str] = {}
    assets_categories_wall: Dict[str, Dict] = {}
    assets_path_wall: Dict[str, str] = {}
    assets_categories_ceiling: Dict[str, Dict] = {}
    assets_path_ceiling: Dict[str, str] = {}
    assets_categories_rug: Dict[str, Dict] = {}
    assets_path_rug: Dict[str, str] = {}
    assets_categories_small: Dict[str, Dict] = {}
    assets_path_small: Dict[str, str] = {}
    rug_descriptions: Dict[str, str] = {}
    labels_floor: List[str] = []
    labels_wall: List[str] = []
    labels_ceiling: List[str] = []
    labels_rug: List[str] = []
    labels_small: List[str] = []

    def add_to_group(label_name, count, bbox, uid, labels_list, cats_dict, path_dict):
        base_label = normalize_label(label_name)
        t_i = 0
        new_label = base_label
        while new_label in labels_list and count <= 1:
            new_label = f"{base_label}{t_i}"
            t_i += 1
        labels_list.append(new_label)
        cats_dict[new_label] = {"count": count, "boundingBox": bbox}
        path_dict[new_label] = resolve_task_asset_path(uid, base_dir)

    for uid_key in task["assets"]:
        uid = uid_key.split("-")[0]
        if uid in uid_buff:
            uid_buff[uid] += 1
            count = uid_buff[uid]
        else:
            uid_buff[uid] = 1
            count = 1

        data_json_path = os.path.join(base_dir, uid, "data.json")
        with open(data_json_path, "r", encoding="utf-8") as f:
            asset_json = json.load(f)
        label_ = normalize_label(asset_json["annotations"]["category"])
        bbox = asset_json["assetMetadata"]["boundingBox"]
        on_floor = asset_json["annotations"].get("onFloor", False)
        on_object = asset_json["annotations"].get("onObject", False)
        on_ceiling = asset_json["annotations"].get("onCeiling", False)
        on_wall = asset_json["annotations"].get("onWall", False)

        is_small = on_object and (
            "cushion" in label_
            or "pillow" in label_
            or "candelabrum" in label_
            or "plant" in label_
            or "vase" in label_
            or "bottle" in label_
            or "cup" in label_
            or "glass" in label_
            or "bowl" in label_
            or "clock" in label_
            or "lamp" in label_
            or label_ == "book"
            or "computer" in label_
            or "laptop" in label_
            or "monitor" in label_
            or "keyboard" in label_
            or "mouse" in label_
        )
        is_ceiling = on_ceiling
        is_wall = on_wall and not is_small and not is_ceiling and not on_floor
        is_rug = ("rug" in label_) or ("mat" in label_) or ("carpet" in label_)
        is_floor = on_floor and not (is_small or is_ceiling or is_wall or is_rug)

        if is_floor:
            add_to_group(label_, count, bbox, uid, labels_floor, assets_categories_floor, assets_path_floor)
        if is_wall:
            add_to_group(label_, count, bbox, uid, labels_wall, assets_categories_wall, assets_path_wall)
        if is_ceiling:
            add_to_group(label_, count, bbox, uid, labels_ceiling, assets_categories_ceiling, assets_path_ceiling)
        if is_rug:
            add_to_group(label_, count, bbox, uid, labels_rug, assets_categories_rug, assets_path_rug)
            rug_descriptions[labels_rug[-1]] = asset_json["annotations"].get("description", label_)
        if is_small:
            add_to_group(label_, count, bbox, uid, labels_small, assets_categories_small, assets_path_small)

    assets_categories_extra: Dict[str, Dict] = {}
    assets_categories_extra.update(assets_categories_wall)
    assets_categories_extra.update(assets_categories_ceiling)
    assets_categories_extra.update(assets_categories_rug)

    assets_path_small_alias = dict(assets_path_small)
    allowed_small_labels = None
    if assets_path_small:
        if "book" in assets_path_small:
            assets_path_small_alias["standing_book"] = assets_path_small["book"]
            assets_path_small_alias["flat_book"] = assets_path_small["book"]
        allowed_small_labels = set(assets_categories_small.keys())
        if "book" in allowed_small_labels:
            allowed_small_labels.update(["standing_book", "flat_book"])

    return {
        "assets_categories_floor": assets_categories_floor,
        "assets_categories_rug": assets_categories_rug,
        "assets_categories_wall": assets_categories_wall,
        "assets_categories_extra": assets_categories_extra or None,
        "assets_path_floor": assets_path_floor,
        "assets_path_rug": assets_path_rug,
        "assets_path_wall": assets_path_wall,
        "assets_path_ceiling": assets_path_ceiling,
        "assets_path_small": assets_path_small,
        "labels_floor": labels_floor,
        "labels_rug": labels_rug,
        "labels_wall": labels_wall,
        "allowed_small_labels": allowed_small_labels,
        "assets_path_small_alias": assets_path_small_alias,
        "rug_descriptions": rug_descriptions,
    }


def _to_float_list(v) -> List[float]:
    try:
        import torch
        if torch.is_tensor(v):
            return v.detach().cpu().numpy().tolist()
    except ImportError:
        pass
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


def world_corners(asset: Dict) -> np.ndarray:
    pos = _to_float_list(asset["pos"])
    bbox = _to_float_list(asset["bbox"])
    angle = _to_float_list(asset["phy"])[0]
    dx = bbox[0] / 2.0
    dy = bbox[1] / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    return local @ rot.T + np.array([pos[0], pos[1]], dtype=np.float32)


def point_in_convex_quad(pt, quad, eps: float = 1e-6) -> bool:
    signs = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        signs.append(cross)
    return all(s >= -eps for s in signs) or all(s <= eps for s in signs)


def compute_area_centers(vis_assets: Dict, num_areas: int, room_center: List[float]) -> List[List[float]]:
    sums = {i: np.zeros(2, dtype=np.float32) for i in range(num_areas)}
    counts = {i: 0 for i in range(num_areas)}
    for asset in vis_assets.values():
        rid = asset.get("region_idx")
        if rid is None or rid not in sums:
            continue
        pos = _to_float_list(asset["pos"])
        sums[rid] += np.array([pos[0], pos[1]], dtype=np.float32)
        counts[rid] += 1
    centers = []
    for i in range(num_areas):
        if counts[i] > 0:
            centers.append((sums[i] / counts[i]).tolist())
        else:
            centers.append([room_center[0], room_center[1]])
    return centers


def asset_xy_aabb(asset: Dict) -> Tuple[float, float, float, float]:
    """Axis-aligned XY bounds from oriented footprint corners."""
    corners = world_corners(asset)
    return (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    )


def xy_aabb_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    eps: float = 1e-3,
) -> bool:
    """True if two XY AABBs (minx, miny, maxx, maxy) intersect."""
    amin_x, amin_y, amax_x, amax_y = a
    bmin_x, bmin_y, bmax_x, bmax_y = b
    return not (
        amax_x < bmin_x - eps
        or bmax_x < amin_x - eps
        or amax_y < bmin_y - eps
        or bmax_y < amin_y - eps
    )


def asset_top_z(asset: Dict) -> float:
    pos = _to_float_list(asset["pos"])
    bbox = _to_float_list(asset["bbox"])
    return float(pos[2]) + float(bbox[2])


def set_asset_pos_z(asset: Dict, z: float) -> None:
    z = float(z)
    try:
        import torch
        if torch.is_tensor(asset.get("pos")):
            asset["pos"][2] = torch.tensor(z, dtype=asset["pos"].dtype, device=asset["pos"].device)
            return
    except ImportError:
        pass
    pos = _to_float_list(asset["pos"])
    pos[2] = z
    asset["pos"] = pos


def _asset_overlaps_rug_xy(asset: Dict, rug: Dict, eps: float = 1e-3) -> bool:
    """True when asset footprint overlaps rug in XY (AABB or corner inside rug quad)."""
    if xy_aabb_overlap(asset_xy_aabb(asset), asset_xy_aabb(rug), eps=eps):
        return True
    rug_quad = world_corners(rug)
    return any(point_in_convex_quad(c, rug_quad, eps=eps) for c in world_corners(asset))


def lift_objects_on_rugs(
    vis_assets: Dict,
    rug_assets: Dict,
    rug_keys: Optional[set] = None,
    *,
    skip_keys: Optional[set] = None,
    eps: float = 1e-3,
) -> set:
    """
    Keep rugs at their layout height; raise floor objects that overlap a rug in XY
    so their bottom (pos.z) sits on the rug top surface.
    Returns keys of lifted floor objects.
    """
    if not rug_assets:
        return set()

    rug_keys = rug_keys or set(rug_assets.keys())
    skip_keys = skip_keys or set()
    lifted_keys: Set[str] = set()

    for floor_key, floor_asset in vis_assets.items():
        if floor_key in rug_keys or floor_key in rug_assets or floor_key in skip_keys:
            continue

        old_z = _to_float_list(floor_asset["pos"])[2]
        target_z = old_z
        on_rugs: List[str] = []

        for rug_key, rug in rug_assets.items():
            if not _asset_overlaps_rug_xy(floor_asset, rug, eps=eps):
                continue
            rug_top = asset_top_z(rug)
            if rug_top > target_z + eps:
                target_z = rug_top
                on_rugs.append(rug_key)

        if on_rugs and target_z > old_z + eps:
            set_asset_pos_z(floor_asset, target_z)
            lifted_keys.add(floor_key)
            print(
                f"[Rug] Lift '{floor_key}' z: {old_z:.4f} -> {target_z:.4f} "
                f"(on rug {on_rugs})"
            )

    return lifted_keys


def lift_rugs_over_floor_xy_overlap(
    vis_assets: Dict,
    rug_assets: Dict,
    rug_keys: Optional[set] = None,
    *,
    eps: float = 1e-3,
) -> set:
    """Backward-compatible alias: lifts objects on rugs, not the rug itself."""
    return lift_objects_on_rugs(vis_assets, rug_assets, rug_keys, eps=eps)


def apply_rug_on_top_height(
    vis_assets: Dict,
    rug_assets: Dict,
    *,
    skip_keys: Optional[set] = None,
) -> None:
    """After floor layout: lift objects whose footprint lies on a rug."""
    lift_objects_on_rugs(vis_assets, rug_assets, skip_keys=skip_keys)
