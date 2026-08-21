"""Mesh instances for SDF-based COL metrics (aligned with GenesisVLM2)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

# Max overlap depth before COL counts as collision (0.5 cm).
COLLISION_TOLERANCE_FIXED = 0.01
# Nudge one mesh by this distance; if overlap drops below threshold, treat as grazing contact.
COLLISION_RESOLVE_SHIFT = 0.005


def _to_float_scalar(x, default: float = 0.0) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _yaw_degrees_from_asset(asset: Union[dict, Any]) -> float:
    """Return yaw in degrees. GenesisVLM2 stores phy in degrees; Object.rot is radians."""
    if isinstance(asset, dict):
        if "phy" in asset:
            return _to_float_scalar(asset["phy"], 0.0)
        if "rot" in asset:
            return float(np.rad2deg(_to_float_scalar(asset["rot"], 0.0)))
        return 0.0
    if getattr(asset, "degree_rot", None) is not None:
        return _to_float_scalar(asset.degree_rot, 0.0)
    rot = getattr(asset, "rot", 0.0)
    return float(np.rad2deg(_to_float_scalar(rot, 0.0)))


def _pos_from_asset(asset: Union[dict, Any]) -> np.ndarray:
    if isinstance(asset, dict):
        pos = asset.get("pos", [0.0, 0.0, 0.0])
    else:
        pos = getattr(asset, "pos", [0.0, 0.0, 0.0])
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    return arr


def _scale_from_asset(asset: Union[dict, Any], default: float = 1.0) -> float:
    if isinstance(asset, dict):
        return _to_float_scalar(asset.get("scale", default), default)
    return _to_float_scalar(getattr(asset, "scale", default), default)


def layout_to_metrics_assets(assets: dict) -> dict:
    """Convert Object or dict layouts to metrics/render dicts (phy in degrees)."""
    out = {}
    for key, asset in assets.items():
        if isinstance(asset, dict):
            entry = dict(asset)
            yaw = _yaw_degrees_from_asset(entry)
            entry["phy"] = np.asarray([yaw], dtype=np.float64)
            if "pos" in entry:
                entry["pos"] = np.asarray(entry["pos"], dtype=np.float64).reshape(-1)
                if entry["pos"].size >= 3:
                    entry["pos"] = entry["pos"].copy()
                    entry["pos"][2] = 0.0
            if "corners" in entry and entry["corners"] is not None:
                entry["corners"] = np.asarray(entry["corners"], dtype=np.float64)
            out[key] = entry
            continue
        pos = _pos_from_asset(asset)
        yaw = _yaw_degrees_from_asset(asset)
        corners = getattr(asset, "corners", None)
        if corners is None:
            corners = np.zeros((4, 2), dtype=np.float64)
        else:
            corners = np.asarray(corners, dtype=np.float64)
        bbox = getattr(asset, "bbox", None)
        if bbox is None:
            bbox = np.zeros(3, dtype=np.float64)
        else:
            bbox = np.asarray(bbox, dtype=np.float64)
        out[key] = {
            "pos": pos,
            "phy": np.asarray([yaw], dtype=np.float64),
            "corners": corners,
            "bbox": bbox,
            "scale": _scale_from_asset(asset),
        }
        if out[key]["pos"].size >= 3:
            out[key]["pos"] = out[key]["pos"].copy()
            out[key]["pos"][2] = 0.0
    return out


def _iter_merged_regions(merged: Union[list, dict]) -> Iterable[tuple[int, dict]]:
    if isinstance(merged, dict):
        def _sort_key(k):
            try:
                return int(k)
            except (TypeError, ValueError):
                return k
        for k in sorted(merged.keys(), key=_sort_key):
            try:
                idx = int(k)
            except (TypeError, ValueError):
                idx = k
            yield idx, merged[k]
    else:
        for idx, region in enumerate(merged):
            yield idx, region


def _item_glb_path_and_scale(item) -> tuple[str, float] | tuple[None, None]:
    """Support GenesisVLM2 tuple items and GraphLayout run.py dict items."""
    if isinstance(item, dict):
        path = item.get("name")
        scale = item.get("scale")
    elif isinstance(item, (list, tuple)) and len(item) >= 4:
        path = item[1]
        scale = item[3]
    else:
        return None, None
    if path is None or scale is None:
        return None, None
    return str(path), float(scale)


def build_all_mesh_instances_for_col(
    vis_assets,
    ids_floor,
    result_small,
    merged_small,
    glb_render,
) -> List[Dict]:
    """
    Build world-space mesh instances for SDF COL.
    Pose logic follows utils/visualization.py render_scene_fin.
    """
    from utils.visualization import auto_align_pos_bottom, auto_align_pos_bottom_center_ea

    instances = []
    labels_idx = {}
    parent_world = {}

    for name_full, asset in vis_assets.items():
        n = name_full.rsplit("_", 1)[0]
        mesh_path = glb_render.get(n)
        if not mesh_path:
            continue
        scale = _scale_from_asset(asset)
        pos = _pos_from_asset(asset)
        pos[2] = 0.0
        yaw = _yaw_degrees_from_asset(asset)
        z_shift = float(auto_align_pos_bottom(mesh_path, pos, scale))
        instances.append(
            {
                "name": n,
                "label": n,
                "object_id": f"floor::{name_full}",
                "mesh_path": mesh_path,
                "scale": scale,
                "position": [float(pos[0]), float(pos[1]), float(pos[2] + z_shift)],
                "euler_deg": [90.0, 0.0, 90.0 + float(yaw)],
            }
        )
        if n in ids_floor:
            labels_idx.setdefault(n, []).append(name_full)
        parent_world[name_full] = {
            "pos": pos,
            "phy": yaw,
            "scale": scale,
            "parent_z_shift": z_shift,
        }

    for base_label in ids_floor:
        ea_result = result_small.get(base_label, {})
        merged = merged_small.get(base_label, [])
        if merged is None:
            continue
        for region_idx, region_dict in _iter_merged_regions(merged):
            assets_region = region_dict.get("items", {})
            pos_dict = ea_result.get(str(region_idx), {})
            for name_, placement in pos_dict.items():
                item_name = name_.rsplit("_", 1)[0]
                if item_name not in assets_region:
                    continue
                glb_path, local_scale = _item_glb_path_and_scale(assets_region[item_name])
                if glb_path is None:
                    continue
                for l_idx in labels_idx.get(base_label, []):
                    p = parent_world.get(l_idx)
                    if p is None:
                        continue
                    big_scale = float(p["scale"])
                    big_yaw = float(p["phy"])
                    parent_z_shift = float(p.get("parent_z_shift", 0.0))
                    scale = big_scale * local_scale
                    if_book = item_name == "standing_book"
                    pos_local = auto_align_pos_bottom_center_ea(
                        glb_path,
                        placement,
                        scale,
                        parent_z_shift,
                        big_scale=big_scale,
                        if_book=if_book,
                    )
                    pos_local = np.array([-pos_local[0], -pos_local[1], pos_local[2]], dtype=np.float64)
                    r_big = R.from_euler("XYZ", (0.0, 0.0, 90.0 + big_yaw), degrees=True)
                    big_pos = np.asarray(p["pos"], dtype=np.float64).reshape(-1)
                    if big_pos.size < 3:
                        big_pos = np.pad(big_pos, (0, 3 - big_pos.size))
                    big_pos[2] = 0.0
                    pos_world = r_big.apply(pos_local) + big_pos
                    if if_book:
                        euler_local = (180.0, 0.0, -90.0)
                        r_local = R.from_euler("XYZ", euler_local, degrees=True)
                        r_world = r_big * r_local
                        euler_deg = list(r_world.as_euler("XYZ", degrees=True))
                    else:
                        euler_deg = [90.0, 0.0, 90.0 + float(big_yaw)]
                    instances.append(
                        {
                            "name": item_name,
                            "label": item_name,
                            "object_id": f"small::{base_label}::{region_idx}::{name_}",
                            "parent_floor_id": f"floor::{l_idx}",
                            "mesh_path": glb_path,
                            "scale": scale,
                            "position": [
                                float(pos_world[0]),
                                float(pos_world[1]),
                                float(pos_world[2]),
                            ],
                            "euler_deg": [
                                float(euler_deg[0]),
                                float(euler_deg[1]),
                                float(euler_deg[2]),
                            ],
                        }
                    )
    return instances
