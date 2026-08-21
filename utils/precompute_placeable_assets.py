"""
Pre-filter assets with at least one placeable horizontal support surface (SDF box probe).

Run from repo root:
  python utils/precompute_placeable_assets.py --dry-run
  python utils/precompute_placeable_assets.py --limit_per_dataset 2

Output layout (default --output_dir assets_feature/placeable):
  assets_feature/placeable/placeable_assets.json   # loadable index (grouped by dataset)
  assets_feature/placeable/{dataset}/{asset_id}/support_regions_3d_{top,front,angled}.png
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

_SUPPORT_UTILS = None
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_NAMES = ("3d_front", "hssd", "test_asset_dir", "objaverse")
VIEW_NAMES = ("top", "front", "angled")


def _ensure_repo_root_on_path() -> str:
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return _REPO_ROOT


def _get_support_utils():
    global _SUPPORT_UTILS
    if _SUPPORT_UTILS is not None:
        return _SUPPORT_UTILS

    _ensure_repo_root_on_path()
    from utils.find_plane import (
        center_mesh_on_bottom_surface,
        extract_support_surfaces,
        load_mesh,
        segment_all_horizontal_surfaces,
        visualize_support_regions_3d,
    )

    _SUPPORT_UTILS = (
        center_mesh_on_bottom_surface,
        extract_support_surfaces,
        load_mesh,
        segment_all_horizontal_surfaces,
        visualize_support_regions_3d,
    )
    return _SUPPORT_UTILS


@dataclass
class FilterConfig:
    sdf_threshold: float
    lift_after_fail: float
    shrink_after_fail: float
    min_cube_side: float
    sample_points: int
    min_region_area: float
    clearances_thresh: float


def _normalize_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def _rel_path(path: str, base_dir: str) -> str:
    return os.path.relpath(_normalize_path(path), _normalize_path(base_dir))


def _asset_id_from_path(dataset: str, mesh_path: str) -> str:
    norm = _normalize_path(mesh_path)
    if dataset == "test_asset_dir":
        return os.path.basename(os.path.dirname(norm))
    if dataset == "3d_front":
        return os.path.basename(os.path.dirname(norm))
    return os.path.splitext(os.path.basename(norm))[0]


def _rotate_obj_like_find_plane(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    angle_rad = np.pi / 2
    rot_matrix = R.from_euler("x", angle_rad).as_matrix()
    tf = np.vstack([np.hstack([rot_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]])
    mesh.apply_transform(tf)
    return mesh


def _scene_to_mesh(loaded) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            return loaded.to_geometry()
        return loaded.dump(concatenate=True)
    return loaded


def load_mesh_for_support(mesh_path: str) -> trimesh.Trimesh:
    center_mesh_on_bottom_surface, _, load_mesh, _, _ = _get_support_utils()
    mesh = load_mesh(mesh_path).copy()

    mesh = center_mesh_on_bottom_surface(mesh)
    return mesh


def _sample_surface(mesh: trimesh.Trimesh, sample_points: int) -> np.ndarray:
    if len(mesh.faces) <= 0:
        return np.empty((0, 3), dtype=np.float64)
    pts, _ = trimesh.sample.sample_surface(mesh, count=max(64, int(sample_points)))
    return pts


def _pair_penetration(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, sample_points: int) -> Optional[float]:
    pts_a = _sample_surface(mesh_a, sample_points)
    pts_b = _sample_surface(mesh_b, sample_points)
    try:
        s_ab = trimesh.proximity.signed_distance(mesh_b, pts_a) if len(pts_a) else np.array([])
        s_ba = trimesh.proximity.signed_distance(mesh_a, pts_b) if len(pts_b) else np.array([])
    except Exception:
        return None
    pen_ab = float(np.max(s_ab)) if s_ab.size else 0.0
    pen_ba = float(np.max(s_ba)) if s_ba.size else 0.0
    return max(pen_ab, pen_ba)


def _make_probe_box(
    center_xy: Tuple[float, float],
    support_height: float,
    width: float,
    depth: float,
    height: float,
    lift: float,
) -> trimesh.Trimesh:
    center = np.array(
        [center_xy[0], center_xy[1], support_height + float(lift) + float(height) / 2.0],
        dtype=np.float64,
    )
    probe = trimesh.creation.box(
        extents=np.array([width, depth, height], dtype=np.float64)
    )
    probe.apply_translation(center)
    return probe


def _candidate_from_region(region, shrink_xy: float, min_side: float) -> Optional[Dict]:
    poly = region.get("polygon", None)
    if poly is None or poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    width = float(maxx - minx) - 2.0 * float(shrink_xy)
    depth = float(maxy - miny) - 2.0 * float(shrink_xy)
    if width <= 0 or depth <= 0:
        return None

    side = min(width, depth)
    if side < float(min_side):
        return None

    return {
        "center_xy": (float((minx + maxx) / 2.0), float((miny + maxy) / 2.0)),
        "support_height": float(region["support_height"]),
        "width": float(width),
        "depth": float(depth),
        "side": float(side),
        "region_area": float(poly.area),
        "clearance": float(region.get("clearance", 0.0)),
    }


def _region_to_json(region, region_index: int) -> Dict:
    poly = region.get("polygon")
    bounds = list(poly.bounds) if poly is not None and not poly.is_empty else []
    centroid = region.get("centroid_xy")
    if centroid is None and poly is not None and not poly.is_empty:
        centroid = (float(poly.centroid.x), float(poly.centroid.y))
    elif centroid is not None:
        centroid = (float(centroid[0]), float(centroid[1]))
    return {
        "region_index": int(region_index),
        "support_height": float(region.get("support_height", 0.0)),
        "clearance": float(region.get("clearance", 0.0)),
        "area": float(poly.area) if poly is not None and not poly.is_empty else 0.0,
        "bounds_xy": bounds,
        "centroid_xy": centroid,
    }


def evaluate_asset(mesh_path: str, cfg: FilterConfig) -> Tuple[Dict, trimesh.Trimesh, List[Dict]]:
    _, extract_support_surfaces, _, segment_all_horizontal_surfaces, _ = _get_support_utils()
    mesh = load_mesh_for_support(mesh_path)
    support_h, support_v = extract_support_surfaces(mesh)
    regions = segment_all_horizontal_surfaces(
        mesh,
        support_h,
        support_v,
        clearance_thresh=cfg.clearances_thresh,
    )

    checked = []
    accepted = []
    accepted_regions: List[Dict] = []
    for region_idx, region in enumerate(regions):
        base = _candidate_from_region(region, shrink_xy=0.0, min_side=cfg.min_cube_side)
        if base is None or base["region_area"] < cfg.min_region_area:
            continue

        cube_base = _make_probe_box(
            center_xy=base["center_xy"],
            support_height=base["support_height"],
            width=base["width"],
            depth=base["depth"],
            height=base["side"],
            lift=0.0,
        )
        pen_base = _pair_penetration(cube_base, mesh, sample_points=cfg.sample_points)
        pass_base = pen_base is not None and pen_base <= cfg.sdf_threshold

        relaxed = _candidate_from_region(
            region,
            shrink_xy=cfg.shrink_after_fail,
            min_side=cfg.min_cube_side,
        )
        pen_relaxed = None
        pass_relaxed = False
        if (not pass_base) and (relaxed is not None):
            cube_relaxed = _make_probe_box(
                center_xy=relaxed["center_xy"],
                support_height=relaxed["support_height"],
                width=relaxed["width"],
                depth=relaxed["depth"],
                height=relaxed["side"],
                lift=cfg.lift_after_fail,
            )
            pen_relaxed = _pair_penetration(cube_relaxed, mesh, sample_points=cfg.sample_points)
            pass_relaxed = pen_relaxed is not None and pen_relaxed <= cfg.sdf_threshold

        checked_entry = {
            "region_index": int(region_idx),
            "cube_side": float(base["side"]),
            "probe_width": float(base["width"]),
            "probe_depth": float(base["depth"]),
            "support_height": float(base["support_height"]),
            "clearance": float(base["clearance"]),
            "penetration_base": None if pen_base is None else float(pen_base),
            "penetration_relaxed": None if pen_relaxed is None else float(pen_relaxed),
            "pass_base": bool(pass_base),
            "pass_relaxed": bool(pass_relaxed),
            "accept": bool(pass_base or pass_relaxed),
        }
        checked.append(checked_entry)
        if checked_entry["accept"]:
            accepted.append(checked_entry)
            accepted_regions.append(region)

    entry = {
        "mesh_path": _normalize_path(mesh_path),
        "checked_surface_count": len(checked),
        "valid_surface_count": len(accepted),
        "is_placeable": len(accepted) > 0,
        "accepted_surfaces": accepted,
        "accepted_regions": [_region_to_json(r, a["region_index"]) for r, a in zip(accepted_regions, accepted)],
    }
    return entry, mesh, accepted_regions


def save_placeable_viz(
    mesh: trimesh.Trimesh,
    accepted_regions: List[Dict],
    viz_dir: str,
    output_dir: str,
) -> Dict[str, str]:
    """Multi-view PNGs for filtered support regions (same views as find_plane)."""
    if not accepted_regions:
        return {}

    os.makedirs(viz_dir, exist_ok=True)
    _, _, _, _, visualize_support_regions_3d = _get_support_utils()
    import matplotlib

    matplotlib.use("Agg")
    visualize_support_regions_3d(mesh, accepted_regions, viz_dir)

    rel_viz_dir = _rel_path(viz_dir, output_dir)
    images = {}
    for view in VIEW_NAMES:
        fname = f"support_regions_3d_{view}.png"
        abs_path = os.path.join(viz_dir, fname)
        if os.path.isfile(abs_path):
            images[view] = os.path.join(rel_viz_dir, fname).replace("\\", "/")
    return images


def _empty_datasets_block() -> Dict[str, Dict]:
    return {name: {"assets": [], "valid_asset_paths": [], "count": 0} for name in DATASET_NAMES}


def _iter_3d_front_assets(root: str) -> Iterable[str]:
    if not root or (not os.path.isdir(root)):
        return []
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        if "raw_model.obj" in filenames:
            paths.append(os.path.join(dirpath, "raw_model.obj"))
    return paths


def _iter_mesh_assets(root: str, *, skip_support_surface: bool = True) -> Iterable[str]:
    if not root or (not os.path.isdir(root)):
        return []
    mesh_ext = {".glb", ".gltf", ".obj"}
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            lower = filename.lower()
            if skip_support_surface and "supportsurface" in lower:
                continue
            if os.path.splitext(filename)[1].lower() in mesh_ext:
                paths.append(os.path.join(dirpath, filename))
    return paths


def _iter_hssd_assets(root: str) -> Iterable[str]:
    if not root or (not os.path.isdir(root)):
        return []
    objects_root = os.path.join(root, "objects")
    search_root = objects_root if os.path.isdir(objects_root) else root
    return _iter_mesh_assets(search_root, skip_support_surface=True)


def _iter_test_asset_dir(root: str) -> Iterable[str]:
    if not root or (not os.path.isdir(root)):
        return []
    paths: List[str] = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        glb = os.path.join(sub, f"{name}.glb")
        if os.path.isfile(glb):
            paths.append(glb)
            continue
        for fn in os.listdir(sub):
            if fn.lower().endswith(".glb"):
                paths.append(os.path.join(sub, fn))
                break
    return paths


def _collect_assets(
    threed_front_root: str,
    hssd_root: str,
    test_asset_dir_root: str,
    objaverse_root: str = "",
    datasets: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    want = set(datasets) if datasets else set(DATASET_NAMES)
    assets: List[Tuple[str, str]] = []
    if "3d_front" in want:
        for p in _iter_3d_front_assets(threed_front_root):
            assets.append(("3d_front", p))
    if "hssd" in want:
        for p in _iter_hssd_assets(hssd_root):
            assets.append(("hssd", p))
    if "test_asset_dir" in want:
        for p in _iter_test_asset_dir(test_asset_dir_root):
            assets.append(("test_asset_dir", p))
    if "objaverse" in want and objaverse_root:
        for p in _iter_mesh_assets(objaverse_root, skip_support_surface=True):
            assets.append(("objaverse", p))
    return assets


def _processed_paths_for_dataset(existing: Optional[Dict], dataset: str) -> set:
    if not existing:
        return set()
    block = existing.get("datasets", {}).get(dataset, {})
    paths = set()
    for entry in block.get("assets", []):
        if isinstance(entry, dict) and entry.get("mesh_path"):
            paths.add(_normalize_path(entry["mesh_path"]))
    return paths


def _merge_dataset_blocks(
    datasets_out: Dict[str, Dict],
    existing: Optional[Dict],
    processed_datasets: set,
) -> Dict[str, Dict]:
    if not existing:
        return datasets_out
    merged = _empty_datasets_block()
    for name in DATASET_NAMES:
        if name in processed_datasets:
            merged[name] = datasets_out[name]
        elif name in existing.get("datasets", {}):
            merged[name] = existing["datasets"][name]
        else:
            merged[name] = datasets_out[name]
    return merged


def _finalize_dataset_blocks(datasets_out: Dict[str, Dict]) -> None:
    for name in DATASET_NAMES:
        block = datasets_out[name]
        block["valid_asset_paths"] = sorted(
            {e["mesh_path"] for e in block["assets"] if e.get("is_placeable")}
        )
        block["count"] = len(block["assets"])
        block["valid_count"] = len(block["valid_asset_paths"])


def _process_asset_worker(payload: Tuple) -> Tuple[str, object]:
    dataset, mesh_path, cfg_dict, output_dir, save_viz = payload
    cfg = FilterConfig(**cfg_dict)
    try:
        entry, mesh, accepted_regions = evaluate_asset(mesh_path, cfg)
        asset_id = _asset_id_from_path(dataset, mesh_path)
        entry["dataset"] = dataset
        entry["asset_id"] = asset_id

        if entry["is_placeable"] and save_viz and accepted_regions:
            viz_dir = os.path.join(output_dir, dataset, asset_id)
            entry["viz_dir"] = _rel_path(viz_dir, output_dir).replace("\\", "/")
            entry["viz_images"] = save_placeable_viz(
                mesh, accepted_regions, viz_dir, output_dir
            )
        else:
            entry["viz_dir"] = ""
            entry["viz_images"] = {}

        return ("ok", entry)
    except Exception as exc:
        return (
            "error",
            {
                "dataset": dataset,
                "mesh_path": _normalize_path(mesh_path),
                "error": str(exc),
            },
        )


def load_placeable_assets(json_path: str) -> Dict:
    """Load placeable_assets.json written by this script."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_placeable_asset_set(json_path: str) -> Optional[set]:
    """Return normalized mesh paths that passed filtering (for find_assets-style whitelists)."""
    if not json_path or not os.path.isfile(json_path):
        return None
    data = load_placeable_assets(json_path)
    allowed = set()
    for p in data.get("valid_asset_paths", []):
        if isinstance(p, str) and p:
            allowed.add(_normalize_path(p))
    for ds_block in data.get("datasets", {}).values():
        if not isinstance(ds_block, dict):
            continue
        for p in ds_block.get("valid_asset_paths", []):
            if isinstance(p, str) and p:
                allowed.add(_normalize_path(p))
    for entry in data.get("assets", []):
        if isinstance(entry, dict) and entry.get("is_placeable") and entry.get("mesh_path"):
            allowed.add(_normalize_path(entry["mesh_path"]))
    return allowed if allowed else set()


def build_placeable_asset_lookup(data: Dict) -> Dict[str, Dict]:
    """Map mesh path / asset_id / basename -> precomputed asset entry."""
    lookup: Dict[str, Dict] = {}

    def _register(key: str, entry: Dict) -> None:
        if key and key not in lookup:
            lookup[key] = entry

    def _add_entry(entry: Dict) -> None:
        if not isinstance(entry, dict):
            return
        mesh_path = entry.get("mesh_path", "")
        asset_id = entry.get("asset_id", "")
        if mesh_path:
            norm = _normalize_path(mesh_path)
            _register(norm, entry)
            _register(os.path.basename(norm), entry)
            parent = os.path.basename(os.path.dirname(norm))
            if parent:
                _register(parent, entry)
        if asset_id:
            _register(str(asset_id), entry)

    for block in data.get("datasets", {}).values():
        if not isinstance(block, dict):
            continue
        for entry in block.get("assets", []):
            _add_entry(entry)
    for entry in data.get("assets", []):
        _add_entry(entry)
    return lookup


def lookup_placeable_entry(lookup: Dict[str, Dict], mesh_path: str) -> Optional[Dict]:
    if not lookup or not mesh_path:
        return None
    norm = _normalize_path(mesh_path)
    candidates = [
        norm,
        os.path.basename(norm),
        os.path.splitext(os.path.basename(norm))[0],
        os.path.basename(os.path.dirname(norm)),
    ]
    for key in candidates:
        if key in lookup:
            return lookup[key]
    base = os.path.basename(norm)
    for key, entry in lookup.items():
        if key == base or norm.endswith(key) or key.endswith(base):
            return entry
    return None


def accepted_region_indices(entry: Dict) -> List[int]:
    indices: List[int] = []
    for surf in entry.get("accepted_surfaces", []):
        if not surf.get("accept", False):
            continue
        indices.append(int(surf["region_index"]))
    if not indices:
        for reg in entry.get("accepted_regions", []):
            indices.append(int(reg["region_index"]))
    return sorted(set(indices))


def support_regions_from_placeable_entry(
    mesh: trimesh.Trimesh,
    entry: Dict,
    *,
    clearances_thresh: float = 0.0,
) -> List[Dict]:
    """
    Re-segment mesh and keep only regions listed in precomputed placeable_assets.json.
    Geometry (polygon) comes from find_plane; which regions to use comes from JSON.
    """
    indices = accepted_region_indices(entry)
    if not indices:
        return []

    _, extract_support_surfaces, _, segment_all_horizontal_surfaces, _ = _get_support_utils()
    support_h, support_v = extract_support_surfaces(mesh)
    all_regions = segment_all_horizontal_surfaces(
        mesh,
        support_h,
        support_v,
        clearance_thresh=clearances_thresh,
    )
    picked: List[Dict] = []
    for idx in indices:
        if 0 <= idx < len(all_regions):
            picked.append(all_regions[idx])
    return picked


def stage_placeable_viz_for_gpt(
    placeable_root: str,
    entry: Dict,
    exp_dir: str,
) -> bool:
    """Copy precomputed support_regions_3d_*.png into the run output folder for GPT."""
    viz_images = entry.get("viz_images") or {}
    if not viz_images:
        return False
    os.makedirs(exp_dir, exist_ok=True)
    copied = 0
    for view, rel in viz_images.items():
        src = rel if os.path.isabs(rel) else os.path.join(placeable_root, rel)
        dst = os.path.join(exp_dir, f"support_regions_3d_{view}.png")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
    return copied > 0


def run_filter(
    threed_front_root: str,
    hssd_root: str,
    test_asset_dir_root: str,
    output_dir: str,
    cfg: FilterConfig,
    limit_per_dataset: int = -1,
    objaverse_root: str = "",
    save_viz: bool = True,
    datasets: Optional[List[str]] = None,
    workers: int = 1,
    resume: bool = False,
) -> Dict:
    output_dir = _normalize_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "placeable_assets.json")
    processed_datasets = set(datasets) if datasets else set(DATASET_NAMES)

    existing = None
    if os.path.isfile(output_json):
        existing = load_placeable_assets(output_json)

    assets = _collect_assets(
        threed_front_root,
        hssd_root,
        test_asset_dir_root,
        objaverse_root=objaverse_root,
        datasets=datasets,
    )

    if limit_per_dataset > 0:
        limited: List[Tuple[str, str]] = []
        counters: Dict[str, int] = {}
        for dataset, path in assets:
            counters.setdefault(dataset, 0)
            if counters[dataset] >= limit_per_dataset:
                continue
            counters[dataset] += 1
            limited.append((dataset, path))
        assets = limited

    if resume:
        skipped = 0
        filtered: List[Tuple[str, str]] = []
        for dataset, path in assets:
            done = _processed_paths_for_dataset(existing, dataset)
            norm = _normalize_path(path)
            if norm in done:
                skipped += 1
                continue
            filtered.append((dataset, path))
        if skipped:
            print(f"[Resume] skipped {skipped} assets already in {output_json}")
        assets = filtered

    datasets_out = _empty_datasets_block()
    if resume and existing:
        for name in processed_datasets:
            if name in existing.get("datasets", {}):
                datasets_out[name] = {
                    "assets": list(existing["datasets"][name].get("assets", [])),
                    "valid_asset_paths": list(
                        existing["datasets"][name].get("valid_asset_paths", [])
                    ),
                    "count": int(existing["datasets"][name].get("count", 0)),
                    "valid_count": int(existing["datasets"][name].get("valid_count", 0)),
                }

    all_results: List[Dict] = []
    valid_asset_paths: List[str] = []
    errors = []
    cfg_dict = asdict(cfg)
    workers = max(1, int(workers))

    def _ingest_result(status: str, payload: object) -> None:
        nonlocal all_results, valid_asset_paths, errors
        if status == "error":
            errors.append(payload)
            return
        entry = payload
        dataset = entry["dataset"]
        datasets_out[dataset]["assets"].append(entry)
        all_results.append(entry)
        if entry.get("is_placeable"):
            valid_asset_paths.append(entry["mesh_path"])

    if workers == 1:
        for dataset, mesh_path in tqdm(assets, desc="Filter placeable assets"):
            status, payload = _process_asset_worker(
                (dataset, mesh_path, cfg_dict, output_dir, save_viz)
            )
            _ingest_result(status, payload)
    else:
        tasks = [
            (dataset, mesh_path, cfg_dict, output_dir, save_viz)
            for dataset, mesh_path in assets
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_asset_worker, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Filter placeable assets"):
                status, payload = fut.result()
                _ingest_result(status, payload)

    datasets_out = _merge_dataset_blocks(datasets_out, existing, processed_datasets)
    _finalize_dataset_blocks(datasets_out)

    all_results = []
    valid_asset_paths = []
    for name in DATASET_NAMES:
        for entry in datasets_out[name]["assets"]:
            all_results.append(entry)
            if entry.get("is_placeable"):
                valid_asset_paths.append(entry["mesh_path"])

    output = {
        "version": 1,
        "output_dir": output_dir,
        "config": {
            "sdf_threshold": cfg.sdf_threshold,
            "lift_after_fail": cfg.lift_after_fail,
            "shrink_after_fail": cfg.shrink_after_fail,
            "min_cube_side": cfg.min_cube_side,
            "sample_points": cfg.sample_points,
            "min_region_area": cfg.min_region_area,
            "clearances_thresh": cfg.clearances_thresh,
            "threed_front_root": _normalize_path(threed_front_root) if threed_front_root else "",
            "hssd_root": _normalize_path(hssd_root) if hssd_root else "",
            "test_asset_dir_root": _normalize_path(test_asset_dir_root) if test_asset_dir_root else "",
            "objaverse_root": _normalize_path(objaverse_root) if objaverse_root else "",
            "save_viz": bool(save_viz),
            "workers": workers,
            "datasets": sorted(processed_datasets),
        },
        "summary": {
            "total_assets": len(all_results),
            "valid_assets": len(valid_asset_paths),
            "invalid_assets": len(all_results) - len(valid_asset_paths),
            "errors": len(errors),
            "viz_saved": sum(1 for e in all_results if e.get("viz_images")),
        },
        "datasets": datasets_out,
        "valid_asset_paths": sorted(set(valid_asset_paths)),
        "assets": all_results,
        "errors": errors,
    }

    if os.path.isfile(output_json):
        backup_json = output_json + ".bak"
        try:
            shutil.copy2(output_json, backup_json)
        except Exception as e:
            print(f"[Warn] Failed to backup existing placeable JSON: {e}")
    fd, tmp_json = tempfile.mkstemp(
        prefix=".placeable_assets.",
        suffix=".json.tmp",
        dir=output_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_json, output_json)
    except Exception:
        try:
            os.remove(tmp_json)
        except OSError:
            pass
        raise
    output["json_path"] = output_json
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute placeable assets: SDF filter + per-dataset JSON + multi-view viz."
    )
    parser.add_argument(
        "--threed_front_root",
        type=str,
        default="./3D-FUTURE-model",
    )
    parser.add_argument(
        "--hssd_root",
        type=str,
        default="./hssd-models",
    )
    parser.add_argument(
        "--test_asset_dir_root",
        type=str,
        default="./test_asset_dir",
    )
    parser.add_argument(
        "--objaverse_root",
        type=str,
        default="./objaverse",
        help="Optional extra mesh root (leave empty to skip).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets_feature/placeable",
        help="Root for placeable_assets.json and {dataset}/{asset_id}/*.png",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Deprecated: if set, parent dir is used as --output_dir.",
    )
    parser.add_argument("--sdf_threshold", type=float, default=0.01)
    parser.add_argument("--lift_after_fail", type=float, default=0.01)
    parser.add_argument("--shrink_after_fail", type=float, default=0.01)
    parser.add_argument("--min_cube_side", type=float, default=0.05)
    parser.add_argument("--sample_points", type=int, default=256)
    parser.add_argument("--min_region_area", type=float, default=0.02)
    parser.add_argument(
        "--clearances_thresh",
        type=float,
        default=0.0,
        help="Min clearance passed to support-region segmentation.",
    )
    parser.add_argument(
        "--limit_per_dataset",
        type=int,
        default=-1,
        help="Debug only: limit number of assets evaluated per dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan mesh paths and print counts; do not run SDF filtering.",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Skip multi-view PNG export for placeable assets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=None,
        help="Only process these datasets (others kept from existing JSON if present).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel worker processes (default: cpu_count - 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip mesh paths already present in output JSON for selected datasets.",
    )
    return parser.parse_args()


def _print_scan_summary(
    threed_front_root: str,
    hssd_root: str,
    test_asset_dir_root: str,
    objaverse_root: str = "",
    datasets: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    assets = _collect_assets(
        threed_front_root,
        hssd_root,
        test_asset_dir_root,
        objaverse_root=objaverse_root,
        datasets=datasets,
    )
    counts: Dict[str, int] = {}
    for dataset, _ in assets:
        counts[dataset] = counts.get(dataset, 0) + 1
    print("[Scan] mesh counts per dataset:")
    for name in sorted(counts):
        print(f"  {name}: {counts[name]}")
    print(f"[Scan] total: {len(assets)}")
    return assets


@dataclass
class PlaceableContext:
    enabled: bool = False
    lookup: Dict[str, Dict] = field(default_factory=dict)
    clearance_thresh: float = 0.0
    root: str = ""


def init_placeable_context(
    placeable_json: str = "assets_feature/placeable/placeable_assets.json",
    placeable_dir: str = "assets_feature/placeable",
) -> PlaceableContext:
    """Load placeable_assets.json for runtime small-object placement (optional)."""
    placeable_root = os.path.abspath(placeable_dir)
    placeable_json = os.path.abspath(placeable_json)
    lookup: Dict[str, Dict] = {}
    clearance_thresh = 0.0
    enabled = False
    if os.path.isfile(placeable_json):
        placeable_data = load_placeable_assets(placeable_json)
        lookup = build_placeable_asset_lookup(placeable_data)
        clearance_thresh = float(placeable_data.get("config", {}).get("clearances_thresh", 0.0))
        if os.path.isdir(placeable_data.get("output_dir", "")):
            placeable_root = os.path.abspath(placeable_data["output_dir"])
        enabled = True
        print(f"[Placeable] Loaded {len(lookup)} entries from {placeable_json}")
    else:
        print(
            f"[Placeable] {placeable_json} not found; "
            "using HSM / find_plane segmentation for support regions"
        )
    return PlaceableContext(
        enabled=enabled,
        lookup=lookup,
        clearance_thresh=clearance_thresh,
        root=placeable_root,
    )


def resolve_parent_support_regions(
    mesh_path: str,
    ctx: PlaceableContext,
    *,
    output_dir: Optional[str] = None,
    label: Optional[str] = None,
) -> Optional[Tuple[trimesh.Trimesh, List[Dict], Optional[Dict]]]:
    """
    Return (mesh, support_regions, placeable_entry) or None if the parent should be skipped.

    When ``ctx.enabled``, only assets listed as placeable in the precomputed index are kept.
    Otherwise fall back to ``find_plane`` segmentation (legacy GraphLayout).
    """
    from utils.find_plane import (
        center_mesh_on_bottom_surface,
        extract_support_surfaces,
        load_mesh,
        segment_all_horizontal_surfaces,
        visualize_support_regions_3d,
        visualize_support_surfaces,
    )

    tag = label or mesh_path

    if ctx.enabled:
        placeable_entry = lookup_placeable_entry(ctx.lookup, mesh_path)
        if placeable_entry is None:
            print(f"[Skip] {tag}: mesh not in placeable_assets.json ({mesh_path})")
            return None
        if not placeable_entry.get("is_placeable", False):
            print(f"[Skip] {tag}: precomputed as not placeable (no valid top surface)")
            return None

        mesh = load_mesh(mesh_path)
        mesh.process()
        mesh = center_mesh_on_bottom_surface(mesh)
        support_regions = support_regions_from_placeable_entry(
            mesh,
            placeable_entry,
            clearances_thresh=ctx.clearance_thresh,
        )
        if not support_regions:
            print(f"[Skip] {tag}: accepted region indices missing after segmentation")
            return None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if not stage_placeable_viz_for_gpt(ctx.root, placeable_entry, output_dir):
                visualize_support_regions_3d(mesh, support_regions, output_dir)
        print(
            f"Loaded {len(support_regions)} placeable region(s) "
            f"(indices={placeable_entry.get('accepted_surfaces', [])})"
        )
        return mesh, support_regions, placeable_entry

    mesh = load_mesh(mesh_path)
    mesh.process()
    mesh = center_mesh_on_bottom_surface(mesh)
    support_h, support_v = extract_support_surfaces(mesh)
    support_regions = segment_all_horizontal_surfaces(mesh, support_h, support_v)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, surf in enumerate(support_h):
            print(
                f"Horizontal surface {i}: {len(surf['faces'])} faces, "
                f"center height = {surf['height']:.2f} m"
            )
        visualize_support_surfaces(mesh, support_h, filename=f"{output_dir}/support_h.png")
        visualize_support_surfaces(mesh, support_v, filename=f"{output_dir}/support_v.png")
        visualize_support_regions_3d(mesh, support_regions, output_dir)

    print(f"Segmented support regions: {len(support_regions)}")
    return mesh, support_regions, None


if __name__ == "__main__":
    _ensure_repo_root_on_path()
    args = parse_args()
    if args.dry_run:
        _print_scan_summary(
            args.threed_front_root,
            args.hssd_root,
            args.test_asset_dir_root,
            objaverse_root=args.objaverse_root,
            datasets=args.datasets,
        )
        raise SystemExit(0)

    output_dir = args.output_dir
    if args.output_json:
        output_dir = os.path.dirname(args.output_json) or output_dir

    cfg = FilterConfig(
        sdf_threshold=float(args.sdf_threshold),
        lift_after_fail=float(args.lift_after_fail),
        shrink_after_fail=float(args.shrink_after_fail),
        min_cube_side=float(args.min_cube_side),
        sample_points=int(args.sample_points),
        min_region_area=float(args.min_region_area),
        clearances_thresh=float(args.clearances_thresh),
    )
    out = run_filter(
        threed_front_root=args.threed_front_root,
        hssd_root=args.hssd_root,
        test_asset_dir_root=args.test_asset_dir_root,
        output_dir=output_dir,
        cfg=cfg,
        limit_per_dataset=int(args.limit_per_dataset),
        objaverse_root=args.objaverse_root,
        save_viz=not args.no_viz,
        datasets=args.datasets,
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    print(
        f"[Done] total={out['summary']['total_assets']} "
        f"valid={out['summary']['valid_assets']} "
        f"viz={out['summary']['viz_saved']} "
        f"errors={out['summary']['errors']}"
    )
    print(f"  JSON -> {out['json_path']}")
    print(f"  PNGs -> {out['output_dir']}/{{dataset}}/{{asset_id}}/support_regions_3d_*.png")
