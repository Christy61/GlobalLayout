import json
import numpy as np
from shapely import Polygon, MultiPolygon, Point
from typing import List, Dict, Optional
from pydantic import BaseModel
from torch import tensor as tensor
from utils.gpt import GPT
import re
import trimesh
from utils.mesh_sdf_utils import is_collision, pair_penetration, repair_mesh_for_sdf

class Metrics(BaseModel):
    NAV: float
    COL: float
    OOB: float
    POS: float | None
    ROT: float | None
    OVR: float | None

def sort_corners_ccw(corners: np.ndarray) -> np.ndarray:
    center = corners.mean(axis=0)
    print("center", center)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    order = np.argsort(angles)
    return corners[order]

def get_metrics_small(
        room_vertices,
        assets,
) -> Metrics:
    print("room_vertices", room_vertices)
    room_vertices = sort_corners_ccw(np.array(room_vertices))
    room_polygon = Polygon(room_vertices)
    asset_polygons: List[Polygon] = []
    for asset in assets.values():
        corners = asset['corners'] + asset['pos'][:2]
        corners = sort_corners_ccw(corners)
        asset_polygons.append(Polygon(corners))
    return 0.0


def get_metrics_cfib(
        room_vertices,
        assets
) -> Metrics:
    print("room_vertices", room_vertices)
    room_vertices = sort_corners_ccw(np.array(room_vertices))
    room_polygon = Polygon(room_vertices)
    asset_polygons: List[Polygon] = []
    for asset in assets.values():
        corners = asset['corners'] + asset['pos'][:2]
        corners = sort_corners_ccw(corners)
        asset_polygons.append(Polygon(corners))
    return (
        COL(room_polygon=room_polygon, asset_polygons=asset_polygons),
        OOB(room_polygon=room_polygon, asset_polygons=asset_polygons),
    )


def get_metrics(
        room_vertices,
        assets,
        task,
        gpt_api,
        image_path, # 这里可能要加主视图
        oob_tolerance: float = 0.005,
        mesh_instances: Optional[List[Dict]] = None,
        sdf_collision_threshold: float = 0.005,
        sdf_sample_points: int = 256,
        sdf_resolve_shift: float = 0.005,
) -> Metrics:
    room_vertices = sort_corners_ccw(np.array(room_vertices))
    room_polygon = Polygon(room_vertices)
    
    asset_polygons: List[Polygon] = []
    for asset in assets.values():
        corners = asset['corners'] + asset['pos'][:2]
        corners = sort_corners_ccw(corners)
        asset_polygons.append(Polygon(corners))
    pos_score, rot_score, ovr_score = None, None, None
    if image_path:
        pos_res = query_VLM("eval_prompt/position_metric.txt", task, gpt_api, image_path)
        rot_res = query_VLM("eval_prompt/rotation_metric.txt", task, gpt_api, image_path)
        ovr_res = query_VLM("eval_prompt/overall_metric.txt", task, gpt_api, image_path)
        pos_score = pos_res[0] if pos_res else None
        rot_score = rot_res[0] if rot_res else None
        ovr_score = ovr_res[0] if ovr_res else None
    col = 0.0
    if mesh_instances is not None:
        col = COL_mesh_sdf(
            mesh_instances,
            sdf_collision_threshold=sdf_collision_threshold,
            sample_points=sdf_sample_points,
            sdf_resolve_shift=sdf_resolve_shift,
        )
    return (
        NAV(room_polygon=room_polygon, asset_polygons=asset_polygons),
        col,
        OOB(room_polygon=room_polygon, asset_polygons=asset_polygons, tolerance=oob_tolerance),
        POP(room_polygon=room_polygon, asset_polygons=asset_polygons),
        pos_score,
        rot_score,
        ovr_score,
    ) 
    

def NAV(room_polygon: Polygon, asset_polygons: List[Polygon]) -> float:    
    union_polygon = asset_polygons[0]
    for asset_polygon in asset_polygons[1:]:
        union_polygon = union_polygon.union(asset_polygon)
    
    difference = room_polygon.difference(union_polygon)

    if type(difference) == Polygon:
        return 1.0
    
    total = 0.
    largest = 0.
    for geom in difference.geoms:
        largest = max(largest, geom.area)
        total += geom.area
    return float("inf") if total == 0. else largest / total

def COL(room_polygon: Polygon, asset_polygons: List[Polygon]) -> float:
    cnt = len(asset_polygons)
    if cnt == 0:
        return 0.0
    tolerance = 0.001
    collided = set()
    for i in range(cnt):
        for j in range(i + 1, cnt):
            inter_area = asset_polygons[i].intersection(asset_polygons[j]).area
            if inter_area > tolerance:
                collided.add(i)
                collided.add(j)

    return len(collided) / cnt


def COL_mesh_sdf(
    mesh_instances: List[Dict],
    sdf_collision_threshold: float = 0.005,
    sample_points: int = 256,
    sdf_resolve_shift: float = 0.005,
    return_details: bool = False,
) -> float:
    """
    Collision ratio based on mesh SDF penetration (GenesisVLM2).

    sdf_collision_threshold: max allowed overlap before COL (m). Default 0.005 = 0.5 cm.
      - overlap depth <  0.5 cm -> not colliding (touching / slight overlap OK)
      - overlap depth >= 0.5 cm -> colliding (after optional resolve check)

    sdf_resolve_shift: if a 0.5 cm translation separates the pair below threshold,
      downgrade to non-collision (filters grazing / numerical contact).
    """
    cnt = len(mesh_instances)
    if cnt == 0:
        return 0.0

    mesh_cache = {}
    world_meshes = []
    surface_samples = []
    aabbs = []
    sdf_stats = {"signed": 0, "unsigned": 0, "failed": 0, "non_watertight": 0}

    for inst in mesh_instances:
        mesh_path = str(inst["mesh_path"])
        scale = float(inst.get("scale", 1.0))
        key = (mesh_path, scale)
        if key not in mesh_cache:
            loaded = trimesh.load(mesh_path, force="scene")
            if isinstance(loaded, trimesh.Scene):
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            mesh = mesh.copy()
            mesh.apply_scale(scale)
            mesh, watertight = repair_mesh_for_sdf(mesh, label=mesh_path)
            if not watertight:
                sdf_stats["non_watertight"] += 1
            mesh_cache[key] = mesh
        base_mesh = mesh_cache[key].copy()

        tf = trimesh.transformations.euler_matrix(
            np.deg2rad(float(inst["euler_deg"][0])),
            np.deg2rad(float(inst["euler_deg"][1])),
            np.deg2rad(float(inst["euler_deg"][2])),
            axes="sxyz",
        )
        tf[:3, 3] = np.asarray(inst["position"], dtype=np.float64)[:3]
        base_mesh.apply_transform(tf)
        world_meshes.append(base_mesh)
        aabbs.append(base_mesh.bounds)

        if len(base_mesh.faces) > 0:
            pts, _ = trimesh.sample.sample_surface(base_mesh, count=max(64, int(sample_points)))
            surface_samples.append(pts)
        else:
            surface_samples.append(np.empty((0, 3), dtype=np.float64))

    collided = set()
    collided_pairs = set()

    def _is_parent_child_pair(inst_a: Dict, inst_b: Dict) -> bool:
        """Skip SDF between a small object and its support parent (resting contact)."""
        id_a = str(inst_a.get("object_id", ""))
        id_b = str(inst_b.get("object_id", ""))
        if id_a.startswith("floor::") and id_b.startswith("small::"):
            floor_id, small_id = id_a, id_b
            small_inst, floor_inst = inst_b, inst_a
        elif id_b.startswith("floor::") and id_a.startswith("small::"):
            floor_id, small_id = id_b, id_a
            small_inst, floor_inst = inst_a, inst_b
        else:
            return False
        parent_floor_id = small_inst.get("parent_floor_id")
        if parent_floor_id:
            return parent_floor_id == floor_id
        # Fallback: same base label, e.g. floor::table_0 vs small::table::...
        floor_key = floor_id.split("floor::", 1)[-1]
        floor_label = floor_key.rsplit("_", 1)[0]
        small_parts = small_id.split("::")
        return len(small_parts) >= 2 and small_parts[1] == floor_label

    def _pair_penetration(mesh_i, pts_i, mesh_j, pts_j, pair_label=""):
        pen, method = pair_penetration(
            mesh_i,
            pts_i,
            mesh_j,
            pts_j,
            sdf_collision_threshold=sdf_collision_threshold,
            pair_label=pair_label,
        )
        sdf_stats[method] = sdf_stats.get(method, 0) + 1
        return pen

    def _aabb_separated(min_a, max_a, min_b, max_b, eps: float = 1e-9) -> bool:
        return bool(np.any(max_a <= min_b + eps) or np.any(max_b <= min_a + eps))

    def _aabb_overlap_depth(min_a, max_a, min_b, max_b) -> float:
        overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
        return float(np.min(overlap))

    def _aabb_collides(min_a, max_a, min_b, max_b, eps: float = 1e-9) -> bool:
        return _aabb_overlap_depth(min_a, max_a, min_b, max_b) > eps

    for i in range(cnt):
        min_i, max_i = aabbs[i]
        for j in range(i + 1, cnt):
            min_j, max_j = aabbs[j]
            if _aabb_separated(min_i, max_i, min_j, max_j):
                continue

            pair_label = f"{mesh_instances[i].get('object_id', i)} vs {mesh_instances[j].get('object_id', j)}"
            penetration = _pair_penetration(
                world_meshes[i], surface_samples[i],
                world_meshes[j], surface_samples[j],
                pair_label=pair_label,
            )

            if not is_collision(penetration, sdf_collision_threshold):
                continue
            if not _aabb_collides(min_i, max_i, min_j, max_j):
                continue

            c_i = world_meshes[i].centroid
            c_j = world_meshes[j].centroid
            v = np.asarray(c_j - c_i, dtype=np.float64)
            n = np.linalg.norm(v)
            if n < 1e-9:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                n = 1.0
            u = v / n
            shift = float(max(sdf_resolve_shift, 0.0))
            deltas = [
                shift * u,
                -shift * u,
                np.array([shift, 0.0, 0.0], dtype=np.float64),
                np.array([-shift, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, shift, 0.0], dtype=np.float64),
                np.array([0.0, -shift, 0.0], dtype=np.float64),
            ]

            still_collide = True
            for d in deltas:
                min_j_shift = min_j + d
                max_j_shift = max_j + d
                if not _aabb_collides(min_i, max_i, min_j_shift, max_j_shift):
                    still_collide = False
                    break
                mesh_j_shift = world_meshes[j].copy()
                mesh_j_shift.apply_translation(d)
                pts_j_shift = surface_samples[j] + d if len(surface_samples[j]) else surface_samples[j]
                pen_shift = _pair_penetration(
                    world_meshes[i], surface_samples[i],
                    mesh_j_shift, pts_j_shift,
                )
                if not is_collision(pen_shift, sdf_collision_threshold):
                    still_collide = False
                    break

            if still_collide:
                collided.add(i)
                collided.add(j)
                collided_pairs.add((i, j))

    ratio = len(collided) / cnt
    if sdf_stats["unsigned"] or sdf_stats["failed"] or sdf_stats["non_watertight"]:
        print(
            "[SDF] query stats:",
            f"signed={sdf_stats.get('signed', 0)}",
            f"unsigned_fallback={sdf_stats.get('unsigned', 0)}",
            f"failed={sdf_stats.get('failed', 0)}",
            f"non_watertight_meshes={sdf_stats.get('non_watertight', 0)}",
        )
    if not return_details:
        return ratio

    collided_objects = []
    for idx in sorted(collided):
        inst = mesh_instances[idx]
        collided_objects.append(
            {
                "index": int(idx),
                "id": inst.get("object_id", inst.get("name", f"obj_{idx}")),
                "label": inst.get("label", inst.get("name", f"obj_{idx}")),
            }
        )
    collided_pair_details = []
    for i, j in sorted(collided_pairs):
        a = mesh_instances[i]
        b = mesh_instances[j]
        collided_pair_details.append(
            {
                "a": {
                    "index": int(i),
                    "id": a.get("object_id", a.get("name", f"obj_{i}")),
                    "label": a.get("label", a.get("name", f"obj_{i}")),
                },
                "b": {
                    "index": int(j),
                    "id": b.get("object_id", b.get("name", f"obj_{j}")),
                    "label": b.get("label", b.get("name", f"obj_{j}")),
                },
            }
        )
    return ratio, collided_objects, collided_pair_details

    # cnt = len(asset_polygons)
    # collision_area = 0.

    # for i in range(cnt):
    #     for j in range(i + 1, cnt):
    #         collision_area += asset_polygons[i].intersection(asset_polygons[j]).area

    # return collision_area

def _iter_geometry_coords(geom):
    def _xy(coord):
        if len(coord) < 2:
            return None
        return float(coord[0]), float(coord[1])

    if geom.is_empty:
        return
    geom_type = geom.geom_type
    if geom_type == "Point":
        yield (geom.x, geom.y)
    elif geom_type in {"LineString", "LinearRing"}:
        for coord in geom.coords:
            xy = _xy(coord)
            if xy is not None:
                yield xy
    elif geom_type == "Polygon":
        for coord in geom.exterior.coords:
            xy = _xy(coord)
            if xy is not None:
                yield xy
        for interior in geom.interiors:
            for coord in interior.coords:
                xy = _xy(coord)
                if xy is not None:
                    yield xy
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_geometry_coords(part)


def max_oob_distance(room_polygon: Polygon, asset_polygon: Polygon) -> float:
    outside = asset_polygon.difference(room_polygon)
    if outside.is_empty:
        return 0.0
    max_dist = 0.0
    for x, y in _iter_geometry_coords(outside):
        max_dist = max(max_dist, float(room_polygon.distance(Point(x, y))))
    return max_dist


def OOB(room_polygon: Polygon, asset_polygons: List[Polygon], tolerance: float = 0.005) -> float:
    cnt = len(asset_polygons)
    if cnt == 0:
        return 0.0

    oob_assets = 0
    for asset in asset_polygons:
        if max_oob_distance(room_polygon, asset) > tolerance:
            oob_assets += 1

    return oob_assets / cnt

    # oob_area = 0.
    # for asset_polygon in asset_polygons:
    #     oob_area += asset_polygon.difference(room_polygon).area
    
    # return oob_area


def OOB_details(
    room_polygon: Polygon,
    asset_polygons: List[Polygon],
    asset_meta: List[Dict],
    tolerance: float = 0.005,
) -> List[Dict]:
    """Return metadata for assets whose footprint exceeds the room boundary."""
    details = []
    for idx, asset in enumerate(asset_polygons):
        oob_area = asset.difference(room_polygon).area
        oob_distance = max_oob_distance(room_polygon, asset)
        if oob_distance > tolerance:
            meta = asset_meta[idx] if idx < len(asset_meta) else {}
            details.append(
                {
                    "index": int(idx),
                    "id": meta.get("id", f"obj_{idx}"),
                    "label": meta.get("label", f"obj_{idx}"),
                    "key": meta.get("key", meta.get("id", f"obj_{idx}")),
                    "oob_area": float(oob_area),
                    "oob_distance": float(oob_distance),
                }
            )
    return details


def build_floor_asset_polygons_and_meta(assets: dict) -> tuple[List[Polygon], List[Dict]]:
    """Build 2D footprints and id/label metadata for floor assets (OOB / violation logging)."""
    asset_polygons: List[Polygon] = []
    asset_meta: List[Dict] = []
    for name_full, asset in assets.items():
        corners = asset["corners"] + asset["pos"][:2]
        corners = sort_corners_ccw(corners)
        asset_polygons.append(Polygon(corners))
        label = str(name_full).rsplit("_", 1)[0]
        asset_meta.append(
            {
                "id": f"floor::{name_full}",
                "label": label,
                "key": str(name_full),
            }
        )
    return asset_polygons, asset_meta


def collect_collision_violations(
    mesh_instances: Optional[List[Dict]],
    *,
    sdf_collision_threshold: float = 0.005,
    sdf_sample_points: int = 256,
    sdf_resolve_shift: float = 0.005,
) -> tuple[float, List[Dict], List[Dict]]:
    if not mesh_instances:
        return 0.0, [], []
    return COL_mesh_sdf(
        mesh_instances,
        sdf_collision_threshold=sdf_collision_threshold,
        sample_points=sdf_sample_points,
        sdf_resolve_shift=sdf_resolve_shift,
        return_details=True,
    )


def collect_oob_violations(
    room_vertices,
    assets: dict,
    *,
    oob_tolerance: float = 0.005,
) -> List[Dict]:
    room_vertices = sort_corners_ccw(np.array(room_vertices))
    room_polygon = Polygon(room_vertices)
    asset_polygons, asset_meta = build_floor_asset_polygons_and_meta(assets)
    return OOB_details(
        room_polygon=room_polygon,
        asset_polygons=asset_polygons,
        asset_meta=asset_meta,
        tolerance=oob_tolerance,
    )


def save_metrics_results(
    output_prefix: str,
    *,
    nav: float,
    col: float,
    oob: float,
    pop: float,
    pos,
    rot,
    ovr,
    vis_assets: dict,
    mesh_instances: Optional[List[Dict]] = None,
    room_vertices=None,
    oob_tolerance: float = 0.005,
    sdf_collision_threshold: float = 0.005,
    sdf_sample_points: int = 256,
    sdf_resolve_shift: float = 0.005,
    count: Optional[int] = None,
    extra_json: Optional[Dict] = None,
) -> Dict:
    """
    Write metrics_results.txt and metrics_results.json, including colliding / OOB object lists.
    ``output_prefix`` is the path without extension, e.g. ``{output_dir}/ccea/metrics_results``.
    """
    col_ratio, col_objects, col_pairs = collect_collision_violations(
        mesh_instances,
        sdf_collision_threshold=sdf_collision_threshold,
        sdf_sample_points=sdf_sample_points,
        sdf_resolve_shift=sdf_resolve_shift,
    )
    # Keep reported COL consistent with caller (may match get_metrics); log details from recomputation.
    _ = col_ratio

    oob_objects: List[Dict] = []
    if room_vertices is not None and vis_assets:
        oob_objects = collect_oob_violations(
            room_vertices,
            vis_assets,
            oob_tolerance=oob_tolerance,
        )

    txt_path = f"{output_prefix}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"NAV: {nav}\n")
        f.write(f"COL: {col}\n")
        f.write(f"OOB: {oob}\n")
        f.write(f"POP: {pop}\n")
        if count is not None:
            f.write(f"COUNT: {count}\n")
        f.write(f"POS: {pos}\n")
        f.write(f"ROT: {rot}\n")
        f.write(f"OVR: {ovr}\n")
        f.write("COL_OBJECTS:\n")
        for item in col_objects:
            f.write(f"- id={item['id']}, label={item['label']}\n")
        f.write("COL_PAIRS:\n")
        for pair in col_pairs:
            f.write(
                f"- {pair['a']['id']}({pair['a']['label']}) <-> "
                f"{pair['b']['id']}({pair['b']['label']})\n"
            )
        f.write("OOB_OBJECTS:\n")
        for item in oob_objects:
            f.write(
                f"- id={item['id']}, label={item['label']}, "
                f"oob_distance={item['oob_distance']:.6f}, "
                f"oob_area={item['oob_area']:.6f}\n"
            )

    metrics_payload = {
        "NAV": nav,
        "COL": col,
        "OOB": oob,
        "POP": pop,
        "POS": pos,
        "ROT": rot,
        "OVR": ovr,
        "COL_OBJECTS": col_objects,
        "COL_PAIRS": col_pairs,
        "OOB_OBJECTS": oob_objects,
    }
    if count is not None:
        metrics_payload["COUNT"] = count

    json_path = f"{output_prefix}.json"
    json_root: Dict = {"metrics": metrics_payload}
    if extra_json:
        json_root.update(extra_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_root, f, indent=2)

    if col_objects:
        print("[Metrics][COL] collided objects:")
        for item in col_objects:
            print(f"  - id={item['id']}, label={item['label']}")
    if oob_objects:
        print("[Metrics][OOB] out-of-bound objects:")
        for item in oob_objects:
            print(
                f"  - id={item['id']}, label={item['label']}, "
                f"oob_distance={item['oob_distance']:.6f}, oob_area={item['oob_area']:.6f}"
            )

    return metrics_payload

def POP(room_polygon: Polygon, asset_polygons: List[Polygon]) -> float:
    union_polygon = asset_polygons[0]
    for asset_polygon in asset_polygons[1:]:
        union_polygon = union_polygon.union(asset_polygon)
    
    difference = union_polygon.intersection(room_polygon)
    total = 0.
    if type(difference) == Polygon:
        total = difference.area
    else: 
        total = 0.
        for geom in difference.geoms:
            total += geom.area
    
    return total / room_polygon.area

def query_VLM(
        instruction_file,
        task,
        gpt_api,
        image_path):
    
    if not image_path:
        return None
    with open(instruction_file, "r", encoding="utf-8") as f:
        prompting_text_user = f.read().format(layout_criteria=task["layout_criteria"])
    prompting_text_system = "You are an interior designer."
    text_dict_system = {
        "type": "text",
        "text": prompting_text_system
    }
    content_system = [text_dict_system]
    
    imageA = gpt_api.encode_image(image_path[0])
    imageB = gpt_api.encode_image(image_path[1])
    content_user = [
        {
            "type": "text",
            "text": prompting_text_user
        },
        {
            "type": "text",
            "text": "top-down view:\n"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpg;base64,{imageA}" 
            }
        },
        {
            "type": "text",
            "text": "side view:\n"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpg;base64,{imageB}" 
            }
        }
    ]
    response = gpt_api(content_system, content_user)
    print(response)
    if not response:
        return None
    lines = response.split("\n")
    for line in lines:
        if "### my final rating is" in line:
            try:
                value_str = line.split(":")[1].strip()
                value_str = re.sub(r"[\*\_]", "", value_str)
                return float(value_str), response
            except Exception as err:
                print(err)
                return None, response
    return None, response

if __name__ == "__main__":
    assets = {0: {'id': 23, 'pos': tensor([0.4824, 0.7936, 0.0000], requires_grad=True), 'phy': tensor([0.0042], requires_grad=True), 'bbox': tensor([0.6414, 1.2368, 0.7599], requires_grad=True), 'scale': 1.0, 'description': 'Reception Desk', 'region_idx': 0, 'corners': tensor([[-0.3233, -0.6170],
        [-0.3181,  0.6197],
        [ 0.3181, -0.6197],
        [ 0.3233,  0.6170]])}, 1: {'id': 3, 'pos': tensor([2.4658, 6.3524, 0.0000], requires_grad=True), 'phy': tensor([-1.5534], requires_grad=True), 'bbox': tensor([1.0489, 1.0489, 0.3519], requires_grad=True), 'scale': 1.0, 'description': 'Dining Table', 'region_idx': 0, 'corners': tensor([[ 0.5152, -0.5335],
        [-0.5335, -0.5153],
        [ 0.5335,  0.5153],
        [-0.5152,  0.5335]])}, 2: {'id': 4, 'pos': tensor([3.5226, 6.3251, 0.0000], requires_grad=True), 'phy': tensor([4.7134], requires_grad=True), 'bbox': tensor([1.0489, 1.0489, 0.3519], requires_grad=True), 'scale': 1.0, 'description': 'Dining Table', 'region_idx': 0, 'corners': tensor([[ 0.5239, -0.5250],
        [-0.5249, -0.5239],
        [ 0.5249,  0.5239],
        [-0.5239,  0.5250]])}, 3: {'id': 5, 'pos': tensor([2.4774, 2.6808, 0.0000], requires_grad=True), 'phy': tensor([1.5670], requires_grad=True), 'bbox': tensor([1.0489, 1.0489, 0.3519], requires_grad=True), 'scale': 1.0, 'description': 'Dining Table', 'region_idx': 0, 'corners': tensor([[-0.5264,  0.5224],
        [ 0.5224,  0.5264],
        [-0.5224, -0.5264],
        [ 0.5264, -0.5224]])}}
    nav, col, oob, pop = get_metrics([(0.0, 0.0), (6.0, 0.0), (6.0, 8.0), (0.0, 8.0)], assets)
    with open("metrics_results.txt", "w") as f:
        f.write(f"NAV: {nav}\n")
        f.write(f"COL: {col}\n")
        f.write(f"OOB: {oob}\n")
        f.write(f"POP: {pop}\n")
