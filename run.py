from utils.find_plane import load_mesh, center_mesh_on_bottom_surface, extract_support_surfaces, \
    visualize_support_surfaces, segment_all_horizontal_surfaces, visualize_support_regions_3d
from utils.precompute_placeable_assets import init_placeable_context, resolve_parent_support_regions
from utils.ea import run_regions_ccea, run_floor_regions_ccea, ccea_gd_layout_optimization, \
    visualize_3d_layout, freeze_assets, build_fixed_anchors #, visualize_3d_region
from utils.ea_small import run_all_regions_ea
from main import generate_obj_simple, generate_assets_from_merged, merge_items_into_regions as merge_small_items_into_regions
from utils.find_assets import encode_assets, encode_assets_3d_future, load_index_and_assets, \
    extract_queries_from_json, match_text_queries, extract_queries_on_floor
from utils.gpt import GPT
from utils.draw import plot_floorplan_with_doors_windows
from utils.visualization import render_scene
from utils.labeling import rendering_views
from utils.optimization import Scene, ConstraintSolver, recaculate_bbox, recaculate_bbox_np, _snapshot_asset
from utils.node import Object, ObjectSet, Wall, Door
from utils.scene_graph import update_graph
from multiprocessing import Process
import argparse
import torch
import re
import ast
import json
import os
import trimesh
import random
import numpy as np
import sys
from typing import List, Any, Union, Optional
import copy
from metrics import get_metrics, save_metrics_results
from utils.mesh_col import (
    build_all_mesh_instances_for_col,
    layout_to_metrics_assets,
    COLLISION_TOLERANCE_FIXED,
    COLLISION_RESOLVE_SHIFT,
)
from utils.task_assets import (
    format_asset_categories_for_prompt,
    get_floor_categories_from_task,
    resolve_mesh_path,
    summarize_floor_categories,
)
from utils.tool import get_mesh_bbox_dimensions, clean_pattern


def generate_assets(merged: dict[str, Any], scene: Scene):
    def _layer_to_depth(layer):
        if layer is None:
            return None
        if isinstance(layer, (int, float)):
            return int(layer)
        if isinstance(layer, str):
            if layer.isdigit():
                return int(layer)
            if layer.lower().startswith("floor"):
                return 0
            mapping = {
                "wall": 2,
                "ceiling": 3,
            }
            return mapping.get(layer.lower())
        return None

    def _get_count(count, thickness=None):
        if count == "unlimited":
            count = max(1, int(shelf_length / thickness))
        elif type(count) == str:
            count = int(count)
        return count

    for idx, region in merged.items():
        items = region.get("items", {})
        idx = str(idx)
        if "polygon" in region.keys():
            shelf_length = region["polygon"].bounds[2] - region["polygon"].bounds[0]

        assets: dict[str, Union[Object, ObjectSet]] = {}
        for item_name, item_info in items.items():
            if not item_info:
                continue
            id = item_info["id"]
            glb_path = item_info["name"]
            count = item_info["count"]
            scale = item_info["scale"]
            vertical = item_info["vertical"]
            depth_from_layer = _layer_to_depth(item_info.get("layer"))
            support_parent = item_info.get("support")
            bbox = get_mesh_bbox_dimensions(glb_path, vertical, scale)
            if bbox is None:
                continue

            count = _get_count(count, bbox[0])
            for i in range(count):
                key = item_name + f"_{i}"
                obj = Object(key, id, bbox, scale, "single", "asset")
                obj.layer = item_info.get("layer")
                if depth_from_layer is not None:
                    obj.depth = depth_from_layer
                if support_parent:
                    obj.parent = support_parent
                assets[key] = obj

        scene._init_assets(idx, assets)


def _normalize_floor_objects(objects_in_areas: dict[str, Any], strip_extra_objects: bool = False):
    def _ensure_dims(obj: dict, default_height: float = 0.02):
        dims = obj.get("dimensions")
        if not isinstance(dims, list):
            return
        if len(dims) == 2:
            dims = [dims[0], dims[1], default_height]
        elif len(dims) == 1:
            dims = [dims[0], dims[0], default_height]
        elif len(dims) == 0:
            dims = [1.0, 1.0, default_height]
        obj["dimensions"] = dims

    def _norm_name(value: Any):
        if not isinstance(value, str):
            return value
        return value.lower().replace(" ", "_").replace("-", "_")

    for area in objects_in_areas.get("areas", []):
        floor_objs = list(area.get("objects") or area.get("floor_objects") or [])
        area["objects"] = floor_objs
        area["floor_objects"] = floor_objs

        for obj in area.get("objects", []):
            obj["name"] = _norm_name(obj.get("name"))
            obj.setdefault("vertical", False)
            obj.setdefault("layer", "floor")
            _ensure_dims(obj)

        if strip_extra_objects:
            rug_names = {
                r.get("name")
                for r in area.get("rug_objects", [])
                if r.get("name")
            }
            floor_objs = [
                o for o in area.get("objects", [])
                if o.get("layer") not in ("floor_l1", "rug")
                and o.get("support") not in rug_names
            ]
            area["objects"] = floor_objs
            area["floor_objects"] = floor_objs
            area["wall_objects"] = []
            area["ceiling_objects"] = []
            area["rug_objects"] = []
            area.pop("asset_sets", None)


def build_category_objects(objects_in_areas: dict[str, Any], category: str) -> dict[str, Any]:
    key_map = {
        "rug": "rug_objects",
        "wall": "wall_objects",
        "ceiling": "ceiling_objects",
    }
    src_key = key_map.get(category)
    if not src_key:
        raise ValueError(f"Unknown category: {category}")
    areas_out = []
    for area in objects_in_areas.get("areas", []):
        objs = copy.deepcopy(area.get(src_key, []))
        if category == "rug":
            # Tag rugs for downstream matching logic (e.g., allow up-scaling).
            for o in objs:
                o["layer"] = "rug"
        elif category in {"wall", "ceiling"}:
            for o in objs:
                o.setdefault("layer", category)
        areas_out.append({
            "area_name": area.get("area_name", ""),
            # Feed as floor_objects so _normalize_floor_objects populates area["floor_objects"] correctly.
            "floor_objects": objs,
            "wall_objects": [],
            "ceiling_objects": [],
            "rug_objects": [],
            "asset_sets": [],
        })
    return {"areas": areas_out}


def filter_dsl_by_src(dsl_code: str, allowed_srcs: set[str]) -> str:
    if not dsl_code.strip():
        return ""
    lines = dsl_code.splitlines()
    try:
        tree = ast.parse(dsl_code)
    except SyntaxError:
        return ""
    kept_lines = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not call.args:
            continue
        src_arg = call.args[0]
        srcs = []
        if isinstance(src_arg, ast.Name):
            srcs = [src_arg.id]
        elif isinstance(src_arg, ast.List):
            names = []
            valid = True
            for elt in src_arg.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
                else:
                    valid = False
                    break
            if valid:
                srcs = names
        if not srcs:
            continue
        if all(s in allowed_srcs for s in srcs):
            line_idx = getattr(node, "lineno", None)
            if line_idx is not None and 1 <= line_idx <= len(lines):
                kept_lines.append(lines[line_idx - 1].strip())
    return "\n".join([l for l in kept_lines if l])


def _dst_name(node):
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_point(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Point"

def _is_point_3d(node):
    if not _is_point(node):
        return False
    if len(node.args) >= 3:
        return True
    kw_names = {
        kw.arg
        for kw in getattr(node, "keywords", [])
        if isinstance(kw, ast.keyword) and isinstance(kw.arg, str)
    }
    if not ({"z", "height"} & kw_names):
        return False
    if len(node.args) >= 2:
        return True
    return {"x", "y"} <= kw_names


def _get_call_arg(call: ast.Call, index: int, kw_names: tuple[str, ...] = ()):
    if len(call.args) > index:
        return call.args[index]
    for kw in call.keywords:
        if isinstance(kw, ast.keyword) and kw.arg in kw_names:
            return kw.value
    return None


def filter_dsl_for_category(
    dsl_code: str,
    allowed_srcs: set[str],
    category: str,
    allowed_dst_names: Optional[set[str]] = None,
) -> str:
    if not dsl_code.strip():
        return ""
    lines = dsl_code.splitlines()
    try:
        tree = ast.parse(dsl_code)
    except SyntaxError:
        return ""
    kept_lines = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not call.args:
            continue
        src_arg = call.args[0]
        srcs = []
        if isinstance(src_arg, ast.Name):
            srcs = [src_arg.id]
        elif isinstance(src_arg, ast.List):
            names = []
            valid = True
            for elt in src_arg.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
                else:
                    valid = False
                    break
            if valid:
                srcs = names
        if not srcs or not all(s in allowed_srcs for s in srcs):
            continue

        rel = call.func.id if isinstance(call.func, ast.Name) else ""
        dst_node = _get_call_arg(call, 1, ("dst", "to", "target"))

        if category == "wall":
            # Wall objects: ONLY allow against(src, <wall>, height)
            if rel == "against":
                dst_name = _dst_name(dst_node)
                if not dst_name or not dst_name.endswith("_wall"):
                    continue
                has_height_kw = any(isinstance(kw, ast.keyword) and kw.arg == "height" for kw in call.keywords)
            elif rel == "distance":
                if dst_node is None:
                    continue
                dst_name = _dst_name(dst_node)
            if len(call.args) < 3 and not has_height_kw:
                continue

        elif category == "rug":
            # Rugs: only align_with to a wall, and distance to Point or wall.
            if rel == "align_with" or rel == "against":
                dst_name = _dst_name(dst_node)
                if not dst_name or not dst_name.endswith("_wall"):
                    continue
            elif rel == "distance":
                if dst_node is None:
                    continue
                dst_name = _dst_name(dst_node)
            else:
                continue

        elif category == "ceiling":
            # Ceiling: ONLY above or distance; distance must target Point(x, y, z)
            if rel == "above":
                dst_name = _dst_name(dst_node)
                if dst_node is None or not (dst_name or _is_point(dst_node)):
                    continue
                has_height_kw = any(isinstance(kw, ast.keyword) and kw.arg == "height" for kw in call.keywords)
                if len(call.args) < 3 and not has_height_kw:
                    continue
            elif rel == "distance":
                if dst_node is None:
                    continue
                if not _is_point_3d(dst_node):
                    continue
            else:
                continue

        else:
            continue

        line_idx = getattr(node, "lineno", None)
        if line_idx is not None and 1 <= line_idx <= len(lines):
            kept_lines.append(lines[line_idx - 1].strip())

    return "\n".join([l for l in kept_lines if l])


def build_assets_from_merged(merged: dict[str, Any]) -> dict[str, dict[str, Union[Object, ObjectSet]]]:
    def _layer_to_depth(layer):
        if layer is None:
            return None
        if isinstance(layer, (int, float)):
            return int(layer)
        if isinstance(layer, str):
            if layer.isdigit():
                return int(layer)
            if layer.lower().startswith("floor"):
                return 0
            mapping = {
                "floor_l0": 0,
                "floor_l1": 0,
                "furniture_l1": 2,
                "furniture_l2": 3,
                "furniture_l3": 4,
                "wall": 2,
                "ceiling": 3,
            }
            return mapping.get(layer.lower())
        return None

    def _get_count(count, thickness=None):
        if count == "unlimited":
            count = max(1, int(shelf_length / thickness))
        elif type(count) == str:
            count = int(count)
        return count

    assets_by_region: dict[str, dict[str, Union[Object, ObjectSet]]] = {}
    for idx, region in merged.items():
        items = region.get("items", {})
        idx = str(idx)
        if "polygon" in region.keys():
            shelf_length = region["polygon"].bounds[2] - region["polygon"].bounds[0]
        assets: dict[str, Union[Object, ObjectSet]] = {}
        for item_name, item_info in items.items():
            if not item_info:
                continue
            id = item_info["id"]
            glb_path = item_info["name"]
            count = item_info["count"]
            scale = item_info["scale"]
            vertical = item_info["vertical"]
            depth_from_layer = _layer_to_depth(item_info.get("layer"))
            support_parent = item_info.get("support")
            bbox = get_mesh_bbox_dimensions(glb_path, vertical, scale)
            if bbox is None:
                continue

            count = _get_count(count, bbox[0])
            for i in range(count):
                key = item_name + f"_{i}"
                obj = Object(key, id, bbox, scale, "single", "asset")
                obj.layer = item_info.get("layer")
                if depth_from_layer is not None:
                    obj.depth = depth_from_layer
                if support_parent:
                    obj.support = support_parent
                assets[key] = obj

        assets_by_region[idx] = assets
    return assets_by_region


def build_assets_by_region(scene: Scene, assets: dict[str, Union[Object, ObjectSet]]) -> dict[str, dict[str, Union[Object, ObjectSet]]]:
    out: dict[str, dict[str, Union[Object, ObjectSet]]] = {}
    for region_idx, region in scene.regions.items():
        region_assets = region.get("assets", {})
        out[region_idx] = {k: assets[k] for k in region_assets.keys() if k in assets}
    return out


def update_render_maps_from_merged(merged: dict[str, Any], glb_render: dict, scales_render: Optional[dict] = None, labels_render: Optional[list] = None) -> list[str]:
    added_labels: list[str] = []
    for region in merged.values():
        items = region.get("items", {})
        for item, info in items.items():
            key = item.replace(' ', '_').replace('-', '_').lower()
            if key in glb_render and glb_render[key] != info["name"]:
                print(f"[Warning] glb_render overwrite for '{key}': {glb_render[key]} -> {info['name']}")
            glb_render[key] = info["name"]
            if scales_render is not None:
                scales_render[key] = info["scale"]
            if labels_render is not None and key not in labels_render:
                labels_render.append(key)
                added_labels.append(key)
    return added_labels


def merge_items_into_regions(region_list: dict[str, Any], match_results: List[dict], top_k: int):
    for region in region_list.values():
        region["items"] = {}
    print(match_results)
    def _unique_item_key(items: dict, base_name: str) -> str:
        if base_name not in items:
            return base_name
        suffix = 1
        while f"{base_name}_{suffix}" in items:
            suffix += 1
        return f"{base_name}_{suffix}"

    for match in match_results:
        idx = match["id"]
        if match["matches"]:
            if 0 <= int(idx) < len(region_list):
                # paired: [(glb, score, ...), ...]
                sorted_matched = sorted(
                    match["matches"],
                    key=lambda x: float(x[1]),
                    reverse=True
                )
                top_pairs = sorted_matched[:top_k]
                chosen_match = random.choice(top_pairs)

                chosen_name = chosen_match[0]
                if len(chosen_match) > 3:
                    chosen_scale = chosen_match[3]
                else:
                    chosen_scale = 1.0
                base_name = match["item"].replace(' ', '_').replace('-', '_').lower()
                name = _unique_item_key(region_list[idx]["items"], base_name)
                region_list[idx]["items"][name] = {
                    "id": match["id"],
                    "name": chosen_name,
                    "count": match["count"],
                    "scale": chosen_scale,
                    "vertical": match["vertical"],
                    "layer": match.get("layer"),
                    "support": match.get("support"),
                    "location": match.get("location"),
                }
            else:
                region_list[idx]["items"] = {}
                print(f"[Warning] ID {idx} not found in region list.")
        else:
            region_list[idx]["items"] = {}

    return region_list


def get_region_bound(support_regions):
    bound = {}
    for i, region in enumerate(support_regions):
        poly = region['polygon']
        z = region['clearance']
        minx, miny, maxx, maxy = poly.bounds
        shelf_length = maxx - minx
        shelf_width = maxy - miny
        bound[i] = [shelf_length, shelf_width, z]
    return bound

def generate_walls(min_x, max_x, min_y, max_y):
    # Generate wall dict from floor boundary
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    wall: dict[str, Wall] = {}

    wall["south_wall"] = Wall(
        key="south_wall",
        id=0,
        pos=[center_x, min_y, 0.],
        rot=[90*torch.pi/180],
        length=max_x,
        group="single",
        otype="edge")
    
    wall["east_wall"] = Wall(
        key="east_wall",
        id=1,
        pos=[max_x, center_y, 0.],
        rot=[180*torch.pi/180],
        length=max_y,
        group="single",
        otype="edge")
    
    wall["north_wall"] = Wall(
        key="north_wall",
        id=2,
        pos=[center_x, max_y, 0.],
        rot=[270*torch.pi/180],
        length=max_x,
        group="single",
        otype="edge")
    
    wall["west_wall"] = Wall(
        key="west_wall",
        id=3,
        pos=[min_x, center_y, 0.],
        rot=[0.],
        length=max_y,
        group="single",
        otype="edge")

    return wall


def generate_obj(assets_region, region_idx, obj_idx=0, name_buff={}):
    # Generate obj dict from ea_results and floor_regions
    objs_dict = {}
    for asset in assets_region:
        bbox = torch.tensor([asset['width'], asset['length'], asset['height']]).float()
        name = asset['name'].replace(' ', '_').replace('-', '_').lower()
        if name not in name_buff:
            name_buff[name] = 0
        else:
            name_buff[name] += 1
        idx = name_buff[name]
        objs_dict[f"{name}_{idx}"] = {
            'region_idx': region_idx,
            'bbox': bbox,
            'scale': asset['scale'],
            'id': obj_idx
        }
        obj_idx += 1
    return objs_dict, name_buff


def generate_door(door_loc):
    center = door_loc['center']
    wall_dir = [[90*torch.pi/180], [180*torch.pi/180], [270*torch.pi/180], [0.]]
    shift_dir = [[0., 0.4], [-0.4, 0.], [0., -0.4], [0.4, 0.]]  # south, east, north, west
    phy = wall_dir[door_loc["wall_id"]]

    door: dict[str, Door] = {}
    door["door"] = Door(
        key="door",
        id=0,
        pos=[center[0] + shift_dir[door_loc["wall_id"]][0], center[1] + shift_dir[door_loc["wall_id"]][1], 0.],
        rot=phy,
        bbox=[0.5, 0.5, 1.0],
        group="single",
        otype="door",
        center=center,
        wall_id=door_loc.get("wall_id"),
        hinge=door_loc.get("hinge"),
    )
    
    return door

def _door_forbidden_polygon(door_loc, room_bound):
    if not door_loc or room_bound is None:
        return None
    center = np.array(door_loc["center"], dtype=float)
    hinge = door_loc.get("hinge", "right")
    wid = int(door_loc.get("wall_id", 0))
    # Wall endpoints based on wall_id ordering: 0 south, 1 east, 2 north, 3 west
    xmin, xmax, ymin, ymax = room_bound
    if wid == 0:
        p1 = np.array([xmin, ymin]); p2 = np.array([xmax, ymin])
    elif wid == 1:
        p1 = np.array([xmax, ymin]); p2 = np.array([xmax, ymax])
    elif wid == 2:
        p1 = np.array([xmax, ymax]); p2 = np.array([xmin, ymax])
    else:
        p1 = np.array([xmin, ymax]); p2 = np.array([xmin, ymin])
    wall_dir = p2 - p1
    n = np.linalg.norm(wall_dir)
    if n < 1e-8:
        return None
    wall_dir = wall_dir / n
    # inward normal by wall id (0 south, 1 east, 2 north, 3 west)
    if wid == 0:
        n_in = np.array([0.0, 1.0])
    elif wid == 1:
        n_in = np.array([-1.0, 0.0])
    elif wid == 2:
        n_in = np.array([0.0, -1.0])
    else:
        n_in = np.array([1.0, 0.0])
    half = 0.8 / 2.0
    if hinge == "left":
        hinge_pt = center - wall_dir * half
    else:
        hinge_pt = center + wall_dir * half
    # green wedge: square of door width projected inward from hinge
    corner1 = hinge_pt
    corner2 = hinge_pt + wall_dir * 0.8
    corner3 = hinge_pt + wall_dir * 0.8 + n_in * 0.8
    corner4 = hinge_pt + n_in * 0.8
    return [tuple(corner1), tuple(corner2), tuple(corner3), tuple(corner4)]

def split_focused_and_other_areas(areas, focused_id):
    """
    Split areas into focused area and other areas.
    """
    area_data1 = None
    area_data2 = []

    for idx, area in enumerate(areas):
        if idx == focused_id:
            area_data1 = area
        else:
            area_data2.append(area["area_name"])

    return area_data1, area_data2

def detach_tensor(assets):
    for asset in assets.values():
        for key, value in list(asset.items()):
            if isinstance(value, torch.Tensor):
                asset[key] = value.detach().cpu()
    return assets


def detach_small_results(result_small):
    for region_layouts in result_small.values():
        for assets in region_layouts.values():
            detach_tensor(assets)
    return result_small

def detach_layout_assets(layouts):
    for asset in layouts.values():
        asset.pos = asset.pos.detach().cpu().numpy() if isinstance(asset.pos, torch.Tensor) else asset.pos
        asset.rot = asset.rot.detach().cpu().numpy() if isinstance(asset.rot, torch.Tensor) else asset.rot
        asset.degree_rot = asset.rot/np.pi*180
        asset.corners = asset.corners.detach().cpu().numpy() if isinstance(asset.corners, torch.Tensor) else asset.corners
        asset.bbox = asset.bbox.detach().cpu().numpy() if isinstance(asset.bbox, torch.Tensor) else asset.bbox

def _corners_from_bbox(bbox, rot_deg=0.0):
    w, l = float(bbox[0]), float(bbox[1])
    corners = np.array([[-w/2, -l/2], [-w/2, l/2], [w/2, -l/2], [w/2, l/2]], dtype=np.float32)
    if rot_deg:
        theta = np.deg2rad(rot_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        corners = corners @ rot.T
    return corners

def collect_floor_object_placements(scene, areas_plan, layouts):
    areas_out = []
    for idx, area in enumerate(areas_plan.get("areas", [])):
        area_name = area.get("area_name", f"area_{idx}")
        assets_region = scene.regions[str(idx)].get("assets", {})
        objs = []
        for key, obj in assets_region.items():
            if key not in layouts:
                continue
            asset = layouts[key]
            pos = asset.pos.detach().cpu().numpy() if isinstance(asset.pos, torch.Tensor) else np.array(asset.pos)
            rot = asset.rot.detach().cpu().numpy() if isinstance(asset.rot, torch.Tensor) else np.array(asset.rot)
            bbox = asset.bbox.detach().cpu().numpy() if isinstance(asset.bbox, torch.Tensor) else np.array(asset.bbox)
            objs.append({
                "name": key,
                "pos": [float(pos[0]), float(pos[1])],
                "rot": float(rot[0]) if np.ndim(rot) > 0 else float(rot),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2])],
            })
        areas_out.append({"area_name": area_name, "objects": objs})
    return {"areas": areas_out}

    return {"areas": areas_out}


def update_scene(scene, walls, scene_graph):
    # Build per-depth solvers and assets
    def _get_rel_depth(rel, depth_map):
        if rel.name == "surround" and isinstance(rel.src, list):
            depths = [depth_map.get(s) for s in rel.src if s in depth_map]
            return max(depths) if depths else None
        return depth_map.get(rel.src)

    scene.layers_by_depth = {}
    depth_map_all: dict[str, int] = build_depth_map_from_scene(scene)
    depth_dsl_global: dict[int, list[str]] = {}
    for rel in scene_graph.edges:
        d = _get_rel_depth(rel, depth_map_all)
        if d is None:
            continue
        depth_dsl_global.setdefault(int(d), []).append(scene_graph._relation_to_dsl(rel))

    depth_assets_global: dict[int, dict[str, Any]] = {}
    all_assets: dict[str, Any] = {}
    for region in scene.regions.values():
        assets_region = region.get("assets", {})
        all_assets.update(assets_region)
        for name, obj in assets_region.items():
            d = depth_map_all.get(name, 0)
            depth_assets_global.setdefault(int(d), {})[name] = obj

    for depth, lines in depth_dsl_global.items():
        layer_assets = depth_assets_global.get(depth, {})
        layer_code = "\n".join(lines).strip()
        solver = ConstraintSolver(layer_code, walls)
        solver.set_asset_fallback(all_assets)
        centrality = scene_graph.centrality or scene_graph.compute_centrality()
        solver.update_constraint_weights(centrality)
        scene._init_global_layer(depth, solver, layer_assets)


def apply_depths_to_scene(scene, depth_map: dict[str, int]):
    for region in scene.regions.values():
        assets = region.get("assets", {})
        for name, obj in assets.items():
            if name in depth_map:
                obj.depth = depth_map[name]
            else:
                obj.depth = 0

def build_depth_map_from_scene(scene) -> dict[str, int]:
    depth_map: dict[str, int] = {}
    for region in scene.regions.values():
        assets = region.get("assets", {})
        for name, obj in assets.items():
            depth = int(getattr(obj, "depth", 0) or 0)
            depth_map[name] = depth
    return depth_map

def get_bound(areas_plan, wall_h, min_x, min_y, max_x, max_y):
    room_bound = (min_x, max_x, min_y, max_y)
    regions_bound = {str(i): [max_x - min_x, max_y - min_y, wall_h] for i in range(len(areas_plan["areas"]))}
    return room_bound, regions_bound

def get_walls(task, room_type, floor_vertices, areas_plan):
    wall_h = task["boundary"]["wall_height"]
    floor_regions: dict[str, Any] = {}
    for i in range(len(areas_plan["areas"])):
        floor_regions[str(i)] = {
            'clearance': wall_h,
            'support_height' : 0.0,
            'solver': None  # will be updated later
        }
    min_x, min_y = np.array(floor_vertices).min(axis=0)[:2]
    max_x, max_y = np.array(floor_vertices).max(axis=0)[:2]
    room_bound, regions_bound = get_bound(areas_plan, wall_h, min_x, min_y, max_x, max_y)
    scene = Scene(room_type, floor_regions, room_bound)
    print("floor_regions: ", floor_regions)

    # Generate walls
    walls = generate_walls(min_x, max_x, min_y, max_y)
    return wall_h, floor_regions, room_bound, regions_bound, scene, walls

def floor_plan(task, output_dir, floor_xy, gpt_api):
    floor_plan_path = f"{output_dir}/floor_plan.txt"
    if not os.path.exists(floor_plan_path):
        content_system, content_user = gpt_api.create_door(task, floor_xy)
        json_str = gpt_api(content_system, content_user)
        with open(floor_plan_path, "w+") as f:
            f.write(json_str)
    with open(floor_plan_path, "r") as f:
        json_str = f.read()
    json_str = re.search(r"```json\n(.+?)```", json_str, re.DOTALL).group(1)
    floor_plan = json.loads(json_str)
    
    door_location, window_locations = floor_plan['door_location'], floor_plan['window_locations']
    plot_floorplan_with_doors_windows(floor_xy, door_location, window_locations, out_path=f"{output_dir}/floor_plan.png")
    print(f"Floor Plan saved to {output_dir}/floor_plan.png")
    return door_location, window_locations

def area_plan(task, output_dir, room_type, floor_xy, gpt_api, assets_categories=None):
    area_path = f"{output_dir}/areas.txt"
    if not os.path.exists(area_path):
        content_system, content_user = gpt_api.get_areas(
            task, floor_xy, output_dir, room_type, assets_categories
        )
        json_str = gpt_api(content_system, content_user)
        with open(area_path, "w+") as f:
            f.write(json_str)
    with open(area_path, "r") as f:
        json_str = f.read()
    json_str = re.search(r"```json\n(.+?)```", json_str, re.DOTALL).group(1)
    areas_plan = json.loads(json_str)
    return areas_plan

def floor_objects(task, output_dir, room_type, gpt_api, areas_plan, assets_categories=None):
    floor_objects_path = f"{output_dir}/floor_objects.txt"
    if not os.path.exists(floor_objects_path):
        content_system, content_user = gpt_api.get_objects(
            task, areas_plan, output_dir, room_type, assets_categories
        )
        json_str = gpt_api(content_system, content_user)
        with open(floor_objects_path, "w+") as f:
            f.write(json_str)
    with open(floor_objects_path, "r") as f:
        json_str = f.read()
    json_str = re.search(r"```json\n(.+?)```", json_str, re.DOTALL).group(1)
    objects_in_areas = json.loads(json_str)
    _normalize_floor_objects(objects_in_areas, strip_extra_objects=True)
    return objects_in_areas

def _parse_reject_response(text: str) -> Optional[dict]:
    try:
        json_str = re.search(r"```json\n(.+?)```", text, re.DOTALL).group(1)
        return json.loads(json_str)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return None


def _compute_generated_scale(bbox, dims):
    if not bbox or not dims:
        return 1.0
    l_d, w_d = max(dims[0], dims[1]), min(dims[0], dims[1])
    l_n, w_n = max(bbox[0], bbox[1]), min(bbox[0], bbox[1])
    if l_n <= 0 or w_n <= 0:
        return 1.0
    scale = min(l_d / l_n, w_d / w_n)
    if bbox[2] > 0 and dims[2] > 0:
        scale = min(scale, dims[2] / bbox[2])
    return scale


def _collect_support_parents(objects_in_areas: dict[str, Any]) -> set[str]:
    support_parents = set()
    for area in objects_in_areas.get("areas", []):
        for obj in area.get("objects", []):
            support = obj.get("support")
            if support:
                support_parents.add(str(support))
    return support_parents


def _is_size_mismatch(bbox, dims, ratio_hi=1.6, ratio_lo=0.6, height_hi=2.0, height_lo=0.5):
    if not bbox or not dims:
        return False
    try:
        l_d, w_d = max(dims[0], dims[1]), min(dims[0], dims[1])
        l_n, w_n = max(bbox[0], bbox[1]), min(bbox[0], bbox[1])
        if l_d <= 0 or w_d <= 0 or l_n <= 0 or w_n <= 0:
            return False
        r1 = l_n / l_d
        r2 = w_n / w_d
        if r1 > ratio_hi or r1 < ratio_lo or r2 > ratio_hi or r2 < ratio_lo:
            return True
        if len(dims) > 2 and len(bbox) > 2 and dims[2] > 0 and bbox[2] > 0:
            rh = bbox[2] / dims[2]
            if rh > height_hi or rh < height_lo:
                return True
    except Exception:
        return False
    return False


def asset_match(
    task,
    floor_regions,
    regions_bound,
    objects_in_areas
):
    root_dir = "3D-FUTURE-model"
    encode_dir = "assets_feature"
    dataset = "3d_future"
    if not os.path.exists(os.path.join(encode_dir, "faiss.index")):
        encode_assets_3d_future(root_dir, encode_dir)
    _, filenames, index = load_index_and_assets(encode_dir)

    queries = extract_queries_on_floor(objects_in_areas, None)
    results = match_text_queries(queries, None, index, filenames, task, top_k=10)
    merged = merge_items_into_regions(floor_regions, results, top_k=1)
    return merged

def save_for_render(merged):
    glb_render, scales_render, labels_render = {}, {}, []
    for region_idx in range(len(merged)):
        for item, info in merged[str(region_idx)]['items'].items():
            item = item.replace(' ', '_').replace('-', '_').lower()
            labels_render.append(item)
            glb_render[item] = info["name"]
            scales_render[item] = info["scale"]
            print(f"  {item}:  name: {info['name']} -> scale: {info['scale']}")
    return glb_render, scales_render, labels_render


def build_merged_from_selected_assets(
    task: dict[str, Any],
    floor_regions: dict[str, Any],
    objects_in_areas: dict[str, Any],
) -> Optional[dict[str, Any]]:
    selected = task.get("selected_floor_assets") or {}
    assets_list = task.get("assets_list") or {}
    if not selected and not assets_list:
        return None

    merged = copy.deepcopy(floor_regions)
    for region in merged.values():
        region["items"] = {}

    def _norm_label(value: Any) -> str:
        return str(value).lower().replace(" ", "_").replace("-", "_")

    def _selected_path(label: str) -> Optional[str]:
        info = selected.get(label)
        if isinstance(info, str):
            return info
        if isinstance(info, dict):
            path = info.get("path") or info.get("name")
            return str(path) if path else None
        path = assets_list.get(label)
        if isinstance(path, str):
            return path
        return None

    def _floor_asset_scale(mesh_path: str, obj: dict[str, Any]) -> float:
        bbox = get_mesh_bbox_dimensions(mesh_path, bool(obj.get("vertical", False)))
        dims = obj.get("dimensions") or []
        if not bbox or not isinstance(dims, (list, tuple)) or len(dims) < 2:
            return 1.0
        length, width = bbox[0], bbox[1]
        if max(length, width) <= 1e-8:
            return 1.0
        return min(max(dims[1], dims[0]) / max(length, width), 1.0)

    added = 0
    for region_idx, area in enumerate(objects_in_areas.get("areas", [])):
        region_key = str(region_idx)
        if region_key not in merged:
            continue
        for obj in area.get("floor_objects") or area.get("objects") or []:
            label = _norm_label(obj.get("name"))
            selected_path = _selected_path(label)
            if not selected_path:
                print(f"[AssetSelect][Skip] no selected asset for '{label}'")
                continue
            mesh_path = resolve_mesh_path(selected_path)
            if not mesh_path or not os.path.exists(mesh_path):
                print(f"[AssetSelect][Skip] selected asset for '{label}' is missing: {selected_path}")
                continue
            merged[region_key]["items"][label] = {
                "id": region_key,
                "name": mesh_path,
                "count": int(obj.get("amount", obj.get("count", 1)) or 1),
                "scale": _floor_asset_scale(mesh_path, obj),
                "vertical": bool(obj.get("vertical", False)),
                "layer": obj.get("layer", "floor"),
                "support": obj.get("support"),
                "location": obj.get("location"),
            }
            added += 1

    if added == 0:
        print("[AssetSelect][Warning] selected_floor_assets/assets_list provided but no usable assets were added.")
    else:
        print(f"[AssetSelect] using preselected floor assets: {added} labels")
    return merged

def update_scene_region(scene, walls, scene_graph, region_idx: int, existing_assets=None):
    """Build solver/layer for one region (floor pipeline, single layer)."""
    region_key = str(region_idx)
    scene.layers_by_depth = {}
    region_assets = scene.regions[region_key].get("assets", {})
    if not region_assets:
        return
    active_keys = set(region_assets.keys())
    existing_assets = existing_assets or {}
    available_keys = active_keys | set(existing_assets.keys())

    def _endpoint_available(name):
        if not isinstance(name, str):
            return True
        if name.endswith("_wall") or name.endswith("_edge"):
            return True
        return name in available_keys

    def _rel_applies(rel):
        srcs = rel.src if isinstance(rel.src, list) else [rel.src]
        if not all(_endpoint_available(s) for s in srcs):
            return False
        return _endpoint_available(rel.dst)

    lines = [
        scene_graph._relation_to_dsl(rel)
        for rel in scene_graph.edges
        if _rel_applies(rel)
    ]
    layer_code = "\n".join(lines).strip()
    solver = ConstraintSolver(layer_code, walls)
    fallback = dict(region_assets)
    if existing_assets:
        fallback.update(build_fixed_anchors(existing_assets))
        fallback.update(existing_assets)
    solver.set_asset_fallback(fallback)
    centrality = scene_graph.compute_centrality(active_srcs=available_keys)
    solver.update_constraint_weights(centrality, use_weight=True, ea_asset_keys=active_keys)
    scene._init_global_layer(0, solver, dict(region_assets))


def get_region_code(scene_graph, region_log_dir, region_srcs):
    os.makedirs(region_log_dir, exist_ok=True)
    active_srcs = set(region_srcs)
    current_lines = []
    for rel in scene_graph.edges:
        if rel.name == "surround":
            if any(src in active_srcs for src in rel.src):
                current_lines.append(scene_graph._relation_to_dsl(rel))
        elif rel.src in active_srcs:
            current_lines.append(scene_graph._relation_to_dsl(rel))
    current_code = "\n".join(current_lines).strip()
    return current_code, active_srcs


def run_floor_layout_pipeline(
    task,
    output_dir,
    gpt_api,
    areas_plan,
    scene,
    walls,
    walls_input,
    door,
    door_forbid,
    room_bound,
    glb_render,
    floor_xy,
    door_location,
    window_locations,
    vis_cfg=None,
):
    """
    Ablation-aligned floor layout: per region get_graph -> refine_graph -> CCEA,
    accumulating placed assets from earlier regions as fixed fallback.
    """
    log_dir = f"{output_dir}/refine_logs"
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "summary.jsonl")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    scene_graph = None
    current_codes: list[str] = []
    vis_assets: dict[str, Any] = {}
    n_regions = len(areas_plan["areas"])

    for region_idx in range(n_regions):
        region_key = str(region_idx)
        assets_region = scene.regions[region_key].get("assets", {})
        if not assets_region:
            print(f"[Region {region_idx}] No assets, skipping.")
            continue

        print(f"\n========== Region {region_idx}: constraints ==========")
        current_codes, scene_graph = get_graph(
            task,
            output_dir,
            gpt_api,
            areas_plan,
            scene,
            walls,
            walls_input,
            region_idx=region_idx,
            prior_codes=current_codes,
            scene_graph=scene_graph,
        )

        if scene_graph is None:
            continue

        if region_idx == 0:
            scene_graph.add_walls(walls, room_bound)
            if door_forbid:
                scene_graph.add_forbidden_polygon(door_forbid)

        region_log_dir = os.path.join(log_dir, f"region_{region_idx}")
        region_srcs = set(assets_region.keys())
        current_code, active_srcs = get_region_code(scene_graph, region_log_dir, region_srcs)
        if active_srcs:
            print(f"[Region {region_idx}] Refining ({len(active_srcs)} assets)...")
            print(f"Current code:\n{current_code}")
            scene_graph.refine_graph(
                region_log_dir,
                gpt_api,
                walls_input,
                task,
                current_code,
                output_dir=output_dir,
                active_srcs=active_srcs,
                existing_srcs=set(vis_assets.keys()),
                existing_assets=vis_assets,
                depth_label=f"region_{region_idx}",
            )

        scene_graph.clean_group()
        scene_graph.compute_centrality()
        update_scene_region(scene, walls, scene_graph, region_idx, existing_assets=vis_assets)

        print(f"[Region {region_idx}] Running CCEA...")
        region_layout = run_floor_regions_ccea(
            door,
            walls,
            scene,
            scene_graph,
            glb_render,
            vis_assets=vis_assets,
            graph=True,
            use_weight=True,
            if_gd=True,
            output_dir=output_dir,
            vis_cfg=vis_cfg,
        )

        for key, asset in region_layout.items():
            vis_assets[key] = _snapshot_asset(asset)
            if key in assets_region:
                assets_region[key] = asset

        plot_floorplan_with_doors_windows(
            floor_xy,
            door_location,
            window_locations,
            areas=None,
            result=vis_assets,
            out_path=f"{output_dir}/ccea/floor_plan.png",
        )

    return vis_assets, scene_graph, current_codes


def run_category_layout_pipeline(
    category: str,
    task,
    output_dir,
    gpt_api,
    areas_plan,
    room_type,
    walls,
    walls_input,
    door,
    door_forbid,
    room_bound,
    glb_render,
    category_assets_by_region: dict[str, dict[str, Union[Object, ObjectSet]]],
    fixed_assets_by_region: dict[str, dict[str, Union[Object, ObjectSet]]],
    fixed_assets_all: dict[str, Union[Object, ObjectSet]],
    render_images: Optional[list] = None,
    vis_cfg=None,
):
    """
    Per-region conflict refine + CCEA for wall / ceiling / rug layers.
    Mirrors ``run_floor_layout_pipeline`` but uses category-specific constraints.
    """
    if not any(category_assets_by_region.values()):
        return {}

    floor_regions = {
        str(i): {"clearance": 0.0, "support_height": 0.0, "solver": None}
        for i in range(len(areas_plan["areas"]))
    }
    scene_cat = Scene(room_type, floor_regions, room_bound)
    for region_key, assets in category_assets_by_region.items():
        if assets:
            scene_cat._init_assets(region_key, assets)

    log_dir = f"{output_dir}/refine_logs_{category}"
    os.makedirs(log_dir, exist_ok=True)

    _, scene_graph = get_graph_for_category(
        task,
        output_dir,
        gpt_api,
        areas_plan,
        walls,
        walls_input,
        category,
        category_assets_by_region,
        fixed_assets_by_region,
        render_images=render_images,
    )
    if scene_graph is None:
        return {}

    scene_graph.add_walls(walls, room_bound)
    if door_forbid:
        scene_graph.add_forbidden_polygon(door_forbid)

    vis_assets = {k: _snapshot_asset(v) for k, v in fixed_assets_all.items()}
    category_layouts: dict[str, Union[Object, ObjectSet]] = {}

    for region_idx in range(len(areas_plan["areas"])):
        region_key = str(region_idx)
        assets_region = category_assets_by_region.get(region_key, {})
        if not assets_region:
            continue

        region_log_dir = os.path.join(log_dir, f"region_{region_idx}")
        region_srcs = set(assets_region.keys())
        current_code, active_srcs = get_region_code(scene_graph, region_log_dir, region_srcs)
        if active_srcs:
            print(f"[{category} region {region_idx}] Refining ({len(active_srcs)} assets)...")
            scene_graph.refine_graph(
                region_log_dir,
                gpt_api,
                walls_input,
                task,
                current_code,
                output_dir=output_dir,
                active_srcs=active_srcs,
                existing_srcs=set(vis_assets.keys()),
                existing_assets=vis_assets,
                depth_label=f"{category}_region_{region_idx}",
            )

        scene_graph.clean_group()
        scene_graph.compute_centrality()
        update_scene_region(scene_cat, walls, scene_graph, region_idx, existing_assets=vis_assets)

        print(f"[{category} region {region_idx}] Running CCEA...")
        region_layout = run_floor_regions_ccea(
            door,
            walls,
            scene_cat,
            scene_graph,
            glb_render,
            vis_assets=vis_assets,
            graph=True,
            use_weight=True,
            if_gd=True,
            output_dir=output_dir,
            vis_cfg=vis_cfg,
        )

        for key, asset in region_layout.items():
            if key in assets_region:
                category_layouts[key] = asset
                vis_assets[key] = _snapshot_asset(asset)
                assets_region[key] = asset

    return category_layouts


def get_graph(
    task,
    output_dir,
    gpt_api,
    areas_plan,
    scene,
    walls,
    walls_input,
    region_idx: Optional[int] = None,
    prior_codes: Optional[list[str]] = None,
    scene_graph=None,
):
    current_codes = list(prior_codes or [])
    indices = (
        [region_idx]
        if region_idx is not None
        else list(range(len(areas_plan["areas"])))
    )
    for idx in indices:
        constraint_path = f"{output_dir}/constraints_{idx}.txt"
        assets_region = scene.regions[str(idx)]["assets"]
        if not assets_region:
            continue
        if not os.path.exists(constraint_path):
            obj_input = {k: {'bbox': v.bbox.numpy().tolist()} for k, v in assets_region.items()}
            content_system, content_user = gpt_api.define_optim_func(
                walls_input,
                obj_input,
                task,
                current_codes,
                areas_plan['areas'][idx],
                output_dir,
            )
            all_constraints = gpt_api(content_system, content_user)
            with open(constraint_path, "w+") as f:
                f.write(all_constraints)
        with open(constraint_path, "r") as f:
            all_constraints = f.read()
        dsl_code = clean_pattern(all_constraints)
        current_codes.append(dsl_code)
        print(F"DSL constraints (region {idx}):\n{dsl_code}")

        scene_graph = update_graph(scene_graph, assets_region, dsl_code, str(idx))
    if scene_graph is not None:
        print(f"Scene graph nodes: {len(scene_graph.nodes)}, edges: {len(scene_graph.edges)}")
    return current_codes, scene_graph


def get_graph_for_category(
    task,
    output_dir,
    gpt_api,
    areas_plan,
    walls,
    walls_input,
    category: str,
    category_assets_by_region: dict[str, dict[str, Union[Object, ObjectSet]]],
    fixed_assets_by_region: dict[str, dict[str, Union[Object, ObjectSet]]],
    dst_allowlist_by_region: Optional[dict[str, set[str]]] = None,
    render_images: Optional[list[tuple[str, str]]] = None,
):
    def _rewrite_wall_constraints_against_only(dsl: str) -> str:
        if not dsl.strip():
            return ""
        try:
            tree = ast.parse(dsl)
        except SyntaxError:
            return ""
        try:
            min_x = float(walls["west_wall"].pos[0])
            max_x = float(walls["east_wall"].pos[0])
            min_y = float(walls["south_wall"].pos[1])
            max_y = float(walls["north_wall"].pos[1])
        except Exception:
            return ""

        def _pt(node):
            if not _is_point(node):
                return None
            vals = []
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)):
                    vals.append(float(a.value))
                elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant):
                    vals.append(-float(a.operand.value))
                else:
                    return None
            if len(vals) != 3:
                return None
            return vals  # x,y,z

        def _nearest_wall_key(x: float, y: float) -> str:
            candidates = [
                ("south_wall", abs(y - min_y)),
                ("north_wall", abs(max_y - y)),
                ("west_wall", abs(x - min_x)),
                ("east_wall", abs(max_x - x)),
            ]
            return sorted(candidates, key=lambda t: t[1])[0][0]

        out_lines: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Name):
                continue
            fn = call.func.id
            if not call.args:
                continue
            if fn == "against":
                # keep as-is (will be filtered later)
                line_idx = getattr(node, "lineno", None)
                if line_idx is not None and 1 <= line_idx <= len(dsl.splitlines()):
                    out_lines.append(dsl.splitlines()[line_idx - 1].strip())
                continue
            if fn != "distance" or len(call.args) < 2:
                continue
            src = call.args[0]
            dst = call.args[1]
            if not isinstance(src, ast.Name):
                continue
            pt = _pt(dst)
            if pt is None:
                continue
            x, y, z = pt
            wall_key = _nearest_wall_key(x, y)
            out_lines.append(f"against({src.id}, {wall_key}, {float(z):.2f})")
        return "\n".join([l for l in out_lines if l])

    current_codes = []
    scene_graph = None
    dsl_by_region: dict[str, str] = {}
    constraint_path = f"{output_dir}/{category}_constraints.txt"
    for region_idx in range(len(areas_plan['areas'])):
        region_key = str(region_idx)
        assets_region = category_assets_by_region.get(region_key, {})
        if not assets_region:
            continue
        fixed_region = fixed_assets_by_region.get(region_key, {})
        # Always regenerate category constraints; older cached empty files are a common failure mode.
        regen_constraints = True

        # For rugs/wall objects, we filter dst types strictly; avoid passing extra fixed assets that would
        # tempt the model into emitting filtered-out constraints.
        if category in {"wall", "rug"}:
            assets_for_prompt = dict(assets_region)
        else:
            assets_for_prompt = {**assets_region, **fixed_region}

        obj_input = {}
        for k, v in assets_for_prompt.items():
            bbox = v.bbox.detach().cpu().numpy().tolist() if isinstance(v.bbox, torch.Tensor) else v.bbox
            obj_input[k] = {"bbox": bbox}

        # Use category-specific prompts.
        user_prompt_name = f"{category}_constraints"
        system_prompt_name = f"{category}_constraints"

        all_constraints = ""
        if regen_constraints:
            content_system, content_user = gpt_api.define_optim_func(
                walls_input,
                obj_input,
                task,
                current_codes,
                areas_plan['areas'][region_idx],
                output_dir,
                extra_rules="",
                render_images=render_images,
                user_prompt_name=user_prompt_name,
                system_prompt_name=system_prompt_name,
            )
            all_constraints = gpt_api(content_system, content_user) or ""
        dsl_code = clean_pattern(all_constraints)
        if category == "wall":
            dsl_code = _rewrite_wall_constraints_against_only(dsl_code)
        if category in {"wall", "rug", "ceiling"}:
            dsl_code = filter_dsl_for_category(dsl_code, set(assets_region.keys()), category, None)
        else:
            dsl_code = filter_dsl_by_src(dsl_code, set(assets_region.keys()))
        dsl_by_region[region_key] = dsl_code
        current_codes.append(dsl_code)
        print(F"[{category}] DSL constraints:\n{dsl_code}")

        assets_for_graph = {**fixed_region, **assets_region}
        scene_graph = update_graph(scene_graph, assets_for_graph, dsl_code, str(region_idx))

    # Persist a single constraints file per category (one section per region).
    with open(constraint_path, "w+", encoding="utf-8") as f:
        for region_idx in range(len(areas_plan.get("areas", []))):
            region_key = str(region_idx)
            area_name = areas_plan.get("areas", [])[region_idx].get("area_name", f"area_{region_idx}")
            code = dsl_by_region.get(region_key, "")
            f.write(f"# region {region_key}: {area_name}\n")
            if code.strip():
                f.write(code.strip() + "\n")
            f.write("\n")
    return current_codes, scene_graph

def get_depth_code(scene_graph, log_dir, depth_map, depth):
    active_srcs = {n for n, d in depth_map.items() if d == depth}
    depth_log_dir = os.path.join(log_dir, f"depth_{depth}")
    os.makedirs(depth_log_dir, exist_ok=True)
    current_lines = []
    for rel in scene_graph.edges:
        if rel.name == "surround":
            if any(src in active_srcs for src in rel.src):
                current_lines.append(scene_graph._relation_to_dsl(rel))
        else:
            if rel.src in active_srcs:
                current_lines.append(scene_graph._relation_to_dsl(rel))
    current_code = "\n".join(current_lines).strip()
    return current_code, active_srcs, depth_log_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json_file", help="Path to scene JSON file", default="benchmark_tasks/dining_room/dining_room_0.json")
    parser.add_argument("--gpt_api_key", type=str, default="",
                        help="GPT API key to use. If not specified, will use value found from config file.")
    parser.add_argument("--gpt_version", type=str, default="gpt-4.1",
                        help="GPT version to use.")
    parser.add_argument("--img_path", type=str, default="image.jpg",
                        help="Image path.")
    parser.add_argument("--output_root", type=str, default="results_areas/",
                        help="Output path.")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Run index within a GPT seed, e.g. 1/2/3 -> scene_seed1234_run1.")
    parser.add_argument("--gpt_seed", type=int, default=1234,
                        help="OpenAI API seed for GPT calls (1111, 1234, 2026 for batch eval).")
    parser.add_argument("--exp_name", type=str, default="test_full",
                        help="experiment name.")
    parser.add_argument("--verbose", action='store_true', help="verbose or not")
    parser.add_argument("--layered_opt", action="store_true",
                        help="Enable layer-by-layer optimization based on asset.depth")
    parser.add_argument("--vis_opt_video", action="store_true",
                        help="Enable CCEA/GD optimization 2D visualization and video export.")
    parser.add_argument("--vis_ea_every", type=int, default=1,
                        help="EA visualization interval (generations).")
    parser.add_argument("--vis_gd_every", type=int, default=20,
                        help="GD visualization interval (steps).")
    parser.add_argument("--vis_fps", type=int, default=10,
                        help="FPS for optimization videos.")
    parser.add_argument(
        "--placeable_json",
        type=str,
        default="assets_feature/placeable/placeable_assets.json",
        help="Precomputed placeable asset index (from utils/precompute_placeable_assets.py).",
    )
    parser.add_argument(
        "--placeable_dir",
        type=str,
        default="assets_feature/placeable",
        help="Root dir with placeable_assets.json and per-asset viz PNGs.",
    )
    parser.add_argument(
        "--test_asset_dir",
        type=str,
        default=None,
        help="Objaverse asset root for parsing task assets (default: GenesisVLM2/test_asset_dir).",
    )
    args = parser.parse_args()

    print("""
    ===================================================
    =========== Step 1: Create Assets Group ===========
    ===================================================
    """)
    with open(args.scene_json_file, 'r') as f:
        task = json.load(f)
    from utils.batch_eval import resolve_scene_output_dir
    output_dir = str(resolve_scene_output_dir(
        args.output_root,
        args.scene_json_file,
        gpt_seed=args.gpt_seed,
        run_tag=args.run_tag,
    ))
    room_type = args.scene_json_file.split('/')[-2]
    os.makedirs(output_dir, exist_ok=True)
    print(f"[run.py] output_dir={output_dir} gpt_seed={args.gpt_seed} run_tag={args.run_tag or '(none)'}")

    floor_vertices = task["boundary"]["floor_vertices"]
    floor_xy = [(float(v[0]), float(v[1])) for v in floor_vertices]
    print("floor_xy:", floor_xy)

    assets_categories = get_floor_categories_from_task(
        task,
        test_asset_dir=args.test_asset_dir,
        prefer_json=True,
    )
    floor_category_counts = summarize_floor_categories(assets_categories)
    print("[run.py] floor_objects (label -> count):")
    for label, count in sorted(floor_category_counts.items()):
        print(f"  {label}: {count}")
    if assets_categories:
        print("[run.py] categories prompt:\n" + format_asset_categories_for_prompt(assets_categories))
    
    gpt_api = GPT(args)
    metrics_api = GPT(args, if_test=True)
    door_location, window_locations = floor_plan(task, output_dir, floor_xy, gpt_api)
    areas_plan = area_plan(task, output_dir, room_type, floor_xy, gpt_api, assets_categories)
    wall_h, floor_regions, room_bound, regions_bound, scene, walls = get_walls(task, room_type, floor_vertices, areas_plan)
    door = generate_door(door_location)
    vis_cfg = None
    if args.vis_opt_video:
        vis_cfg = {
            "enabled": True,
            "output_dir": output_dir,
            "floor_xy": floor_xy,
            "door_location": door_location,
            "window_locations": window_locations,
            "ea_every": args.vis_ea_every,
            "gd_every": args.vis_gd_every,
            "fps": args.vis_fps,
            "include_fixed": True,
        }

    objects_in_areas = floor_objects(
        task, output_dir, room_type, gpt_api, areas_plan, assets_categories
    )
    base_objects_in_areas = objects_in_areas
    print("objects_in_areas:", objects_in_areas)
    print("""
    ===================================================
    ============== Step 2: Asset Matching =============
    ===================================================
    """)
    # Matching uses the description to ensure label and style,
    # calculates the MSE of the bounding box to ensure size.
    # Here is 3D-Front dataset
    merged = build_merged_from_selected_assets(task, floor_regions, base_objects_in_areas)
    if merged is None:
        merged = asset_match(task, floor_regions, regions_bound, base_objects_in_areas)
    generate_assets(merged, scene)

    # Save for render:
    glb_render, scales_render, labels_render = save_for_render(merged)
    os.makedirs(f"{output_dir}/ccea", exist_ok=True)
    with open(f"{output_dir}/ccea/glb_render.json", "w") as f:
        json.dump(glb_render, f, indent=2)
    with open(f"{output_dir}/ccea/scales_render.json", "w") as f:
        json.dump(scales_render, f, indent=2)
    rendering_views(labels_render, glb_render, scales_render, floor_xy)

    print("""
    ===================================================
    =========== Step 3: Layout Optimization ===========
    ===================================================
    """)
    os.makedirs(f"{output_dir}/ccea", exist_ok=True)
    walls_input = {k: {'length': v.length, 'rot': v.rot[0].item()} for k, v in walls.items()}
    door_forbid = _door_forbidden_polygon(door_location, room_bound)

    layouts, scene_graph, current_codes = run_floor_layout_pipeline(
        task,
        output_dir,
        gpt_api,
        areas_plan,
        scene,
        walls,
        walls_input,
        door,
        door_forbid,
        room_bound,
        glb_render,
        floor_xy,
        door_location,
        window_locations,
        vis_cfg=vis_cfg,
    )
    vis_assets = layouts
    fixed_assets_all = {k: _snapshot_asset(v) for k, v in layouts.items()}

    layout_full = {}
    for obj in layouts.values():
        obj.get_assets(layout_full)
    for asset in layout_full.values():
        try:
            if asset.pos is not None and len(asset.pos) >= 3:
                asset.pos[2] = 0.0
        except Exception:
            pass
    recaculate_bbox_np(layout_full)

    for obj in layouts.values():
        print(f"{obj.key}: pos = {obj.pos}, phy = {obj.rot}, bbox = {obj.bbox}")

    layout_floor = dict(layout_full)
    door_windows = {"door_location": door_location, "window_locations": window_locations}
    from utils.visualization import render_scene
    p1 = Process(target=render_scene, args=(layout_full, f'{output_dir}/ccea/layout_floor', floor_xy, door_windows), kwargs={"scene": None, "objs": [], "cam_top": None, "cam_front": None, "stop": True, "init": True, "data_path": glb_render, "if_seg": False})
    p1.start()
    p1.join()
    sys.modules.pop("utils.visualization", None)
    
    # layout_floor_json = {}
    # for k, v in layout_floor.items():
    #     pos = v.pos.detach().cpu().numpy().tolist() if isinstance(v.pos, torch.Tensor) else list(v.pos)
    #     rot = v.rot.detach().cpu().numpy().tolist() if isinstance(v.rot, torch.Tensor) else list(v.rot)
    #     bbox = v.bbox.detach().cpu().numpy().tolist() if isinstance(v.bbox, torch.Tensor) else list(v.bbox)
    #     corners = v.corners.detach().cpu().numpy().tolist() if isinstance(v.corners, torch.Tensor) else list(v.corners)
    #     layout_floor_json[k] = {"pos": pos, "rot": rot, "bbox": bbox, "corners": corners}
    # with open(f"{output_dir}/ccea/layout_floor.json", "w") as f:
    #     json.dump(layout_floor_json, f, indent=2)
    
    render_images = []
    floor_top = f"{output_dir}/ccea/layout_floor_top.png"
    floor_side = f"{output_dir}/ccea/layout_floor_side.png"
    if os.path.exists(floor_top):
        render_images.append(("Floor Top", floor_top))
    if os.path.exists(floor_side):
        render_images.append(("Floor Side", floor_side))
    floor_plan_layout_coords = f"{output_dir}/ccea/floor_plan_layout_coords.png"
    try:
        plot_floorplan_with_doors_windows(
            floor_xy,
            door_location,
            window_locations,
            areas=None,
            result=layout_floor,
            out_path=floor_plan_layout_coords,
        )
        if os.path.exists(floor_plan_layout_coords):
            render_images.append(("Floorplan (coords + layout)", floor_plan_layout_coords))
    except Exception as e:
        print(f"[Warning] Failed to draw floorplan coords/layout image: {e}")

    print(layout_full)
    for asset in layout_full.values():
        print(f"asset_key={asset.key}, asset_pos={asset.pos}, asset_rot={asset.rot}")
    plot_floorplan_with_doors_windows(
        floor_xy,
        door_location,
        window_locations,
        areas=None,
        result=layouts,
        out_path=f"{output_dir}/ccea/floor_plan_final.png",
    )
    door_windows = {"door_location": door_location, "window_locations": window_locations}
    from utils.visualization import render_scene
    p1 = Process(target=render_scene, args=(layout_full, f'{output_dir}/ccea/layout', floor_xy, door_windows), kwargs={"scene": None, "objs": [], "cam_top": None, "cam_front": None, "stop": True, "init": True, "data_path": glb_render, "if_seg": False})
    p1.start()
    p1.join()
    sys.modules.pop("utils.visualization", None)


    content_system, content_user = gpt_api.find_big_object(objects_in_areas, task, output_dir)
    output = gpt_api(content_system, content_user)
    with open(f"{output_dir}/base_objects.txt", "w+") as f:
        f.write(output)
    with open(f"{output_dir}/base_objects.txt", "r") as f:
        output = f.read()
    json_str_1 = re.search(r"```json\n(.+?)```", output, re.DOTALL).group(1)
    json_data_1 = json.loads(json_str_1)
    print(json_str_1)
    ids_floor = []
    result_small = {}
    merged_small = {}
    print(glb_render)
    # ai2thorhab dataset
    # root_dir = "small_assets/ai2thorhab-uncompressed/assets"
    # img_dir = "small_assets/render"
    # dataset = "ai2thorhub"

    # hssd dataset
    root_dir = "./hssd-models/objects"
    img_dir = "./hssd_render/objects"
    encode_dir = "assets_feature_hssd"
    dataset = "hssd"
    if not os.path.exists(os.path.join(encode_dir, "faiss.index")):
        encode_assets(root_dir, img_dir, encode_dir)
    embeddings, filenames, index = load_index_and_assets(encode_dir)
    placeable_ctx = init_placeable_context(
        getattr(args, "placeable_json", "assets_feature/placeable/placeable_assets.json"),
        getattr(args, "placeable_dir", "assets_feature/placeable"),
    )
    for big_obj in json_data_1["list"]:
        ### for small assets:
        label = big_obj["object"].lower().replace(' ', '_').replace('-', '_')
        mesh_path = glb_render.get(label)
        if not mesh_path:
            print(
                f"[SmallAssets][Skip] parent object '{label}' was selected for small-object placement "
                "but has no mesh in glb_render."
            )
            continue
        print("""
        ===================================================
        =========== Step 1: Load Placeable Surfaces =========
        ===================================================
        """)

        resolved = resolve_parent_support_regions(
            mesh_path,
            placeable_ctx,
            output_dir=f"{output_dir}/{label}",
            label=label,
        )
        if resolved is None:
            continue
        mesh, support_regions, _placeable_entry = resolved
        print(mesh_path)
        print(support_regions)

        print("""
        ===================================================
        ============ Step 2: Asset Assignment =============
        ===================================================
        """)
        content_system, content_user = gpt_api.get_small_assets(f"{output_dir}/{label}", big_obj, output_dir)
        json_str = gpt_api(content_system, content_user)
        with open(f"{output_dir}/{label}/small_assets.txt", "w+") as f:
            f.write(json_str)
        with open(f"{output_dir}/{label}/small_assets.txt", "r") as f:
            json_str = f.read()

        print("GPT output", json_str)
        m_small = re.search(r"```json\n(.+?)```", json_str, re.DOTALL)
        if not m_small:
            print(f"[Warning] No JSON block in small_assets for {label}, skipping.")
            continue
        json_data_r = json.loads(m_small.group(1))
        for reg in json_data_r["regions"]:
            for key_ in reg["item"]:
                key_ = key_.lower().replace(" ", "_").replace("-", "_")
        rid_list = [r['id'] for r in json_data_r['regions']]
        for rid, region in enumerate(support_regions):
            if rid not in rid_list:
                json_data_r['regions'].append({"id": rid, "item": {}, "description": None, "xy_utilization": 0.0})
        utilization_map = {r["id"]: r.get("xy_utilization", 0.0) for r in json_data_r["regions"]}
        open_surface_id = []
        for rid, region in enumerate(support_regions):
            region["utilization"] = utilization_map.get(rid, 0.0)
            if float(region["clearance"]) == 1.0:
                open_surface_id.append(str(rid))
        print(utilization_map)
        print(open_surface_id)
            

        print("""
        ===================================================
        ============== Step 3: Asset Matching =============
        ===================================================
        """)
        queries = extract_queries_from_json(json_data_r)
        bound = get_region_bound(support_regions)
        print("bound:", bound)
        if not bound:
            continue
        print(queries)
        results = match_text_queries(queries, bound, index, filenames, dataset, top_k=15)
        merged = merge_small_items_into_regions(support_regions, results, top_k=1)
        print("result: ", merged)
        if all(len(region['items']) == 0 for region in merged):
            continue

        for i, region in enumerate(merged):
            print(f"Region {i}:")
            if not region['items']:
                print("  No items assigned.")
                continue
            for item in region['items']:
                if not region['items'][item]:
                    print(f"  {item}: No match found.")
                    continue
                id_small, name, count, scale, z_axis, center = region['items'][item]
                print(f"  {item}: {name}, scale={scale}, center={center}")

        print("""
        ===================================================
        =========== Step 4: Layout Optimization ===========
        ===================================================
        """)

        assets = generate_assets_from_merged(merged, dataset)

        obj_descriptions_all = {}
        open_region = {}
        name_buff_ = {}
        for r in json_data_r['regions']:
            idx = r['id']
            str_idx = str(idx)
            if str_idx not in open_surface_id:
                continue
            obj_descriptions, name_buff_ = generate_obj_simple(assets[str_idx], str_idx, name_buff_)
            if not obj_descriptions:
                continue
            obj_descriptions_all[str_idx] = obj_descriptions
            open_region[str_idx] = r

        json_data_1 = None
        if obj_descriptions_all:
            content_system, content_user = gpt_api.define_small_optim_func(obj_descriptions_all, open_region, output_dir)
            gpt_output = gpt_api(content_system, content_user)
            gpt_output = gpt_output.replace("\\n\\", "\\n")
            with open(f"{output_dir}/{label}/small_constraints.txt", "w+") as f:
                f.write(gpt_output)
            with open(f"{output_dir}/{label}/small_constraints.txt", "r") as f:
                gpt_output = f.read()
            # print(gpt_output)
            json_str_1 = re.search(r"```json\n(.+?)```", gpt_output, re.DOTALL).group(1)
            try:
                json_data_1 = json.loads(json_str_1)
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse JSON for {label}: {e}")
                json_data_1 = None
            print(json_data_1)
        print(open_region)
        ea_result, open_region_idx = run_all_regions_ea(
            support_regions,
            assets,
            obj_descriptions_all,
            gpt_api,
            json_data_1,
            open_region,
            output_dir=output_dir,
            label=label,
        )
        if all(len(v) == 0 for v in ea_result.values()):
            continue
        
        ids_floor.append(label)
        print("""
        ===================================================
        ========== Step 5: Layout Visualization ===========
        ===================================================
        """)
        visualize_3d_layout(mesh, ea_result, support_regions, f"{output_dir}/{label}")
        
        # from utils.visualization import visualize_ea_layout_from_paths
        # visualize_ea_layout_from_paths(mesh_path, merged, ea_result, dataset, save_path=f"{output_dir}/{label}/genesis_rendered.png")
        # sys.modules.pop("utils.visualization", None)
        
        result_small[label] = ea_result
        merged_small[label] = merged
        
        for idx, assets_res in ea_result.items():
            recaculate_bbox(assets_res)

    # Ours 
    from utils.visualization import render_scene_fin
    recaculate_bbox_np(vis_assets)
    detach_layout_assets(vis_assets)
    detach_small_results(result_small)
    vis_assets_metrics = layout_to_metrics_assets(vis_assets)
    p1_1 = Process(target=render_scene_fin, args=(vis_assets_metrics, door_windows, 'ccea/layout', ids_floor, result_small, merged_small, glb_render, floor_xy, output_dir, False))
    p1_1.start()
    p1_1.join()
    p1_2 = Process(target=render_scene_fin, args=(vis_assets_metrics, door_windows, 'ccea/layout', ids_floor, result_small, merged_small, glb_render, floor_xy, output_dir, True))
    p1_2.start()
    p1_2.join()
    sys.modules.pop("utils.visualization", None)
    all_mesh_instances = build_all_mesh_instances_for_col(
        vis_assets=vis_assets_metrics,
        ids_floor=ids_floor,
        result_small=result_small,
        merged_small=merged_small,
        glb_render=glb_render,
    )
    nav, col, oob, pop, pos, rot, ovr = get_metrics(
        floor_vertices,
        vis_assets_metrics,
        task,
        metrics_api,
        [f'{output_dir}/ccea/layout_top.png', f'{output_dir}/ccea/layout_side.png'],
        oob_tolerance=1.5e-2,
        mesh_instances=all_mesh_instances,
        sdf_collision_threshold=COLLISION_TOLERANCE_FIXED,
        sdf_sample_points=256,
        sdf_resolve_shift=COLLISION_RESOLVE_SHIFT,
    )
    room_area = (room_bound[1] - room_bound[0]) * (room_bound[3] - room_bound[2])
    oob = oob / room_area if room_area > 0 else oob
    save_metrics_results(
        f"{output_dir}/ccea/metrics_results",
        nav=nav,
        col=col,
        oob=oob,
        pop=pop,
        pos=pos,
        rot=rot,
        ovr=ovr,
        vis_assets=vis_assets_metrics,
        mesh_instances=all_mesh_instances,
        room_vertices=floor_vertices,
        oob_tolerance=1.5e-2,
        sdf_collision_threshold=COLLISION_TOLERANCE_FIXED,
        sdf_sample_points=256,
        sdf_resolve_shift=COLLISION_RESOLVE_SHIFT,
        count=len(vis_assets_metrics),
    )
