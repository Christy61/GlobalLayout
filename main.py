from utils.find_plane import load_mesh, center_mesh_on_bottom_surface, extract_support_surfaces, \
    visualize_support_surfaces, segment_all_horizontal_surfaces, visualize_support_regions_3d
from utils.precompute_placeable_assets import init_placeable_context, resolve_parent_support_regions
from utils.ea import ccea_gd_layout_optimization, \
    visualize_3d_layout #, visualize_3d_region
from utils.gpt import GPT
from utils.find_assets import encode_assets, encode_assets_3d_future, load_index_and_assets, \
    extract_queries_from_json, match_text_queries, extract_queries_on_floor
from utils.draw import plot_floorplan_with_doors_windows
from utils.labeling import rendering_views
from utils.optimization import recaculate_bbox
from multiprocessing import Process
import argparse
import clip
import torch
import re
import json
import os
import trimesh
import random
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
from metrics import get_metrics
from utils.tool import extract_name_from_path


def get_mesh_bbox_dimensions(glb_path, scale):
    """
    给定 GLB 文件路径，返回其 axis-aligned bbox 的 dx, dy, dz
    """
    try:
        scene = trimesh.load(glb_path, force='scene')
        mesh = scene.dump(concatenate=True)
        mesh.apply_scale(scale)
        bbox = mesh.bounds  # shape: (2, 3)
        dx, dy, dz = bbox[1] - bbox[0]
        
        # trimesh: X (→), Y (↑), Z (depth)
        # genesis: X (→), Y (→ depth), Z (↑)
        length = dx
        width = dz   # 原来 Z 是 depth（→ 转到 Genesis 的 Y）
        height = dy  # 原来 Y 是 up（→ 转到 Genesis 的 Z）

        return [length, width, height]
    except Exception as e:
        print(f"[Error] Failed to load {glb_path}: {e}")
        return None

def generate_assets_from_merged(merged, dataset):
    assets = {}
    for idx, region in enumerate(merged):
        items = region.get('items', {})
        idx = str(idx)
        if 'polygon' in region.keys():
            shelf_length = region['polygon'].bounds[2] - region['polygon'].bounds[0]
            # shelf_width = region['polygon'].bounds[3] - region['polygon'].bounds[1]
        assets[idx] = []
        if not items:
            continue
        for item_name in items:
            if not items[item_name]:
                continue
            id, img_path, count, scale, z_axis, center = items[item_name]
            glb_path = extract_name_from_path(img_path, dataset)
            dims = get_mesh_bbox_dimensions(glb_path, scale)
            if dims is None:
                continue
            length, width, height = dims  # (x, y, z) order

            if item_name == 'standing_book':
                book_thickness = height  # 把 X 方向视为 thickness
                if count == "unlimited":
                    count = max(1, int(shelf_length / book_thickness))
                elif type(count) == str:
                    count = int(count)
                for i in range(count):
                    assets[idx].append({
                        "id": id,
                        "name": "standing_book",
                        "length": height,
                        "width": length,
                        "height": width,
                        "z-axis": z_axis,
                        "center": center,
                        "scale": scale
                    })
            else:
                if item_name == 'flat_book':
                    z_axis = False
                if count == "unlimited":
                    count = max(1, int(shelf_length / length))
                elif type(count) == str:
                    count = int(count)
                for i in range(count):
                    assets[idx].append({
                        "id": id,
                        "name": item_name,
                        "length": length,
                        "width": width,
                        "height": height,
                        "z-axis": z_axis,
                        "center": center,
                        "scale": scale
                    })

    return assets

def find_best_matches(text_query, asset_embeddings, asset_filenames, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model, _ = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([text_query]).to(model.visual.device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    similarities = (asset_embeddings @ text_embedding.T).squeeze()
    top_indices = similarities.topk(top_k).indices

    return [(asset_filenames[i], similarities[i].item()) for i in top_indices]

def merge_items_into_regions(region_list, match_results, top_k=1):
    """
    Args:
        region_list: list of dicts, original A (without id)
        match_results: list of dicts, each with 'id', 'item', 'matches'

    Returns:
        Modified region_list with 'item' and 'matches' inserted for matching ids
    """
    for region in region_list:
        region["items"] = {}
    for match in match_results:
        idx = match["id"]
        if not (0 <= idx < len(region_list)):
            print(f"[Warning] ID {idx} not found in region list; skipping item={match.get('item')!r}.")
            continue
        if match["matches"]:
            matches_with_index = list(enumerate(match["matches"]))  # [(idx, (glb_name, score)), ...]
            sorted_matches = sorted(
                matches_with_index, 
                key=lambda x: float(x[1][1]), 
                reverse=True
            )
            top_indices = [i for i in range(len(sorted_matches))][:top_k]
            chosen_idx = random.choice(top_indices)
            chosen_name = match["matches"][chosen_idx][0]
            chosen_scale = match["scale"][chosen_idx]
            name = match["item"].replace(' ', '_').replace('-', '_').lower()
            region_list[idx]["items"][name] = (match["id"], chosen_name, match["count"], chosen_scale, match["z-axis"], match["center"])
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


def get_bound_by_group(json_data, wall_h):
    bound = {}
    for i, region in enumerate(json_data["list"]):
        bound_xy = region["bounds"]
        xs = [pt[0] for pt in bound_xy]
        ys = [pt[1] for pt in bound_xy]
        shelf_length = max(xs) - min(xs)
        shelf_width = max(ys) - min(ys)
        bound[i] = [shelf_length, shelf_width, wall_h]
    return bound


def generate_floor(min_x, max_x, min_y, max_y):
    # ====== Generate wall and obj dicts ======
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    wall = {}
    # front wall (y=min_y, center along x)
    wall['front_wall'] = {
        'pos': torch.tensor([center_x, min_y, 0.]).float(),
        'phy': torch.tensor([90*torch.pi/180]).float(),  # face +y (up)
        'description': 'facing +y'
    }
    # back wall (y=max_y, center along x)
    wall['back_wall'] = {
        'pos': torch.tensor([center_x, max_y, 0.]).float(),
        'phy': torch.tensor([270*torch.pi/180]).float(),  # face -y (down)
        'description': 'facing -y'
    }
    # left wall (x=min_x, center along y)
    wall['left_wall'] = {
        'pos': torch.tensor([min_x, center_y, 0.]).float(),
        'phy': torch.tensor([0.]).float(),  # face +x (right)
        'description': 'facing +x'
    }
    # right wall (x=max_x, center along y)
    wall['right_wall'] = {
        'pos': torch.tensor([max_x, center_y, 0.]).float(),
        'phy': torch.tensor([180*torch.pi/180]).float(),  # face -x (left)
        'description': 'facing -x'
    }

    return wall

def generate_obj(assets_region, region_idx, obj_idx=0, name_buff={}):
    # Generate obj dict from ea_results and floor_region
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


def generate_obj_simple(assets_region, region_idx, name_buff=None):
    # Generate obj dict from ea_results and floor_region
    objs_dict = {}
    if name_buff is None:
        name_buff = {}
    i = 0
    if not assets_region:
        return objs_dict, name_buff
    for asset in assets_region:
        bbox = torch.tensor([asset['width'], asset['length'], asset['height']]).float()
        name = asset['name'].replace(' ', '_').replace('-', '_').lower()
        if name=="standing_book":
            continue
        if name not in name_buff:
            name_buff[name] = 0
        elif asset['z-axis']==True:
            continue
        else:
            name_buff[name] += 1
        idx = name_buff[name]
        objs_dict[f"{name}_{idx}"] = {
            'region_idx': str(region_idx),
            'bbox': bbox,
            'scale': asset['scale'],
            'id': i
        }
        i += 1
    return objs_dict, name_buff

def generate_door(door_loc):
    door = {}
    center = door_loc['center']
    wall_dir = [[90*torch.pi/180], [180*torch.pi/180], [270*torch.pi/180], [0.]]
    shift_dir = [[0., 0.4], [-0.4, 0.], [0., -0.4], [0.4, 0.]]  # front, right, back, left
    phy = wall_dir[door_loc["wall_id"]]
    door['door'] = {
        'pos': torch.tensor([center[0] + shift_dir[door_loc["wall_id"]][0], center[1] + shift_dir[door_loc["wall_id"]][1], 0.]).float(),
        'phy': torch.tensor(phy).float(),  # face +y (up)
        'bbox': torch.tensor([0.4, 0.4, 1.0]).float(),
    }
    return door

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

def tensor_to_np(assets):
    for asset in assets.values():
        pos_array = asset['pos'].detach().cpu().numpy()
        asset['pos'] = pos_array

        pos_array = (asset['phy']/torch.pi*180).detach().cpu().numpy()
        asset['phy'] = pos_array

        corners_array = asset['corners'].detach().cpu().numpy()
        asset['corners'] = corners_array
    return assets

def detach_tensor(assets):
    for asset in assets.values():
        pos_array = asset['pos'].detach().cpu()
        asset['pos'] = pos_array

        pos_array = asset['phy'].detach().cpu()
        asset['phy'] = pos_array

        corners_array = asset['corners'].detach().cpu()
        asset['corners'] = corners_array
    return assets

# def trans_global(assets):
#     for asset in assets.values():
#         gx, gy = asset["pos"][0].detach().cpu(), asset["pos"][1].detach().cpu()
#         asset["pos"][0] = gx
#         asset["pos"][1] = gy
#     return assets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json_file", help="Path to scene JSON file", default="benchmark_tasks/bedroom/bedroom_0.json")
    parser.add_argument("--gpt_api_key", type=str, default="",
                        help="GPT API key to use. If not specified, will use value found from config file.")
    parser.add_argument("--gpt_version", type=str, default="gpt-4.1",
                        help="GPT version to use.")
    parser.add_argument("--img_path", type=str, default="image.jpg",
                        help="Image path.")
    parser.add_argument("--output_root", type=str, default="results_areas/",
                        help="Output path.")
    parser.add_argument("--exp_name", type=str, default="test_full",
                        help="experiment name.")
    parser.add_argument("--verbose", action='store_true', help="verbose or not")
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
    args = parser.parse_args()

    print("""
    ===================================================
    =========== Step 1: Create Assets Group ===========
    ===================================================
    """)
    with open(args.scene_json_file, 'r') as f:
        task = json.load(f)
    output_dir = args.output_root + args.scene_json_file.split('/')[-1].split('.')[0]
    room_type = args.scene_json_file.split('/')[-2]
    os.makedirs(output_dir, exist_ok=True)

    floor_vertices = task["boundary"]["floor_vertices"]
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    floor_mesh = trimesh.Trimesh(vertices=floor_vertices, faces=faces)

    floor_xy = [(float(v[0]), float(v[1])) for v in floor_vertices]
    print(floor_xy)
    horiz_info = {
        "height": 0.0,
        "normal": [0, 0, 1]
    }
    gpt_api = GPT(args)
    metrics_api = GPT(args, if_test=True)
    content_system, content_user = gpt_api.create_door(task, floor_xy)
    json_str = gpt_api(content_system, content_user)
    print(json_str)
    with open(f"{output_dir}/floor_plan.txt", "w+") as f:
        f.write(json_str)
    with open(f"{output_dir}/floor_plan.txt", "r") as f:
        json_str = f.read()
    json_str = re.search(r"```json\n(.+?)```", json_str, re.DOTALL).group(1)
    floor_plan = json.loads(json_str)
    
    door_location = floor_plan['door_location']
    window_locations = floor_plan['window_locations']
    plot_floorplan_with_doors_windows(floor_xy, door_location, window_locations, out_path=f"{output_dir}/floor_plan.png")
    print(f"Saved to {output_dir}/floor_plan.png")

    content_system, content_user = gpt_api.get_areas(task, floor_xy, output_dir, room_type)
    json_str = gpt_api(content_system, content_user)
    with open(f"{output_dir}/areas.txt", "w+") as f:
        f.write(json_str)
    with open(f"{output_dir}/areas.txt", "r") as f:
        json_str = f.read()
    print(json_str)
    json_str = re.search(r"```json\n(.+?)```", json_str, re.DOTALL).group(1)
    areas_plan = json.loads(json_str)
    
    # 这里是3d front数据集，也可以替换成hssd数据集
    root_dir = "3D-FUTURE-model"
    encode_dir = "assets_feature"
    dataset = "3d_future"
    # root_dir = "hssd-models/objects"
    # img_dir = "hssd_render/objects"
    # encode_dir = "assets_feature_hssd"
    # dataset = "hssd"
    if not os.path.exists(os.path.join(encode_dir, "faiss.index")):
        encode_assets_3d_future(root_dir, encode_dir)
    # if not os.path.exists(os.path.join(encode_dir, "faiss.index")):
    #     encode_assets(root_dir, img_dir, encode_dir)
    wall_h = task["boundary"]["wall_height"]
    floor_region = []
    for i in range(len(areas_plan["areas"])):
        region = {
            'clearance': wall_h,
            'support_height' : 0.0
        }
        floor_region.append(region)
    print(floor_region)
    
    glb_render = {}
    scales_render = {}
    labels_render = []
    floor_vertices_np = np.array(floor_vertices)
    min_x, min_y = floor_vertices_np.min(axis=0)[:2]
    max_x, max_y = floor_vertices_np.max(axis=0)[:2]
    wall = generate_floor(min_x, max_x, min_y, max_y)
    room_bound = (min_x, max_x, min_y, max_y)

    content_system, content_user = gpt_api.get_objects(task, areas_plan, output_dir, room_type)
    json_str = gpt_api(content_system, content_user)
    print(json_str)
    with open(f"{output_dir}/floor_objects.txt", "w+") as f:
        f.write(json_str)
    with open(f"{output_dir}/floor_objects.txt", "r") as f:
        json_str = f.read()
    json_str = re.search(r"```json\n(.+?)```", json_str, re.DOTALL).group(1)
    objects_in_areas = json.loads(json_str)
    for area in objects_in_areas["areas"]:
        for obj in area["objects"]:
            # 转为小写并替换空格为下划线
            obj["name"] = obj["name"].lower().replace(" ", "_").replace("-", "_")
    print("""
    ===================================================
    ============== Step 2: Asset Matching =============
    ===================================================
    """)
    # 这里匹配使用description来保证风格，另外计算和bbox的mse来保证大小
    embeddings, filenames, index = load_index_and_assets(encode_dir)
    queries = extract_queries_on_floor(objects_in_areas)
    # if no region bound, set bound to None
    results = match_text_queries(queries, None, index, filenames, dataset, top_k=10)
    merged = merge_items_into_regions(floor_region, results, top_k=1)
    print("result: ", merged)

    print("""
    ===================================================
    =========== Step 3: Layout Optimization ===========
    ===================================================
    """)
    assets = generate_assets_from_merged(merged, dataset)
    print(assets)

    for region_idx in range(len(merged)):
        for item, (id, name, count, scale, z_axis, center) in merged[region_idx]['items'].items():
            item = item.replace(' ', '_').replace('-', '_').lower()
            labels_render.append(item)
            glb_render[item] = extract_name_from_path(name, dataset)
            scales_render[item] = scale
            print(f"  {item}:  name: {name} -> scale: {scale}")
    rendering_views(labels_render, glb_render, scales_render, floor_xy)

    # CCEA + init + use_weight
    layouts = {}
    vis_assets = {}
    obj_descriptions_all = {}
    init = True
    last_solver = None
    name_buff = {}
    os.makedirs(f"{output_dir}/ccea", exist_ok=True)
    plot_floorplan_with_doors_windows(floor_xy, door_location, window_locations, out_path=f"{output_dir}/ccea/floor_plan.png")
    door = generate_door(door_location)
    for region_idx in range(len(areas_plan['areas'])):
        objects_current = objects_in_areas["areas"][region_idx]
        obj_descriptions, name_buff = generate_obj(assets[str(region_idx)], region_idx, len(vis_assets), name_buff)
        obj_input = {k: {'bbox': v['bbox']} for k, v in obj_descriptions.items()}
        content_system, content_user = gpt_api.define_optim_func(wall, obj_input, task, "", areas_plan['areas'][region_idx])
        all_constraints = gpt_api(content_system, content_user)
        with open(f"{output_dir}/constraints_{region_idx}.txt", "w+") as f:
            f.write(all_constraints)
        with open(f"{output_dir}/constraints_{region_idx}.txt", "r") as f:
            all_constraints = f.read()
        # print(all_constraints)
        code_text_match = re.search(r"```python\n(.+?)```", all_constraints, re.DOTALL)
        code_text = code_text_match.group(1) if code_text_match else ""
        code_text = re.sub(r"solver\s*=\s*ConstraintSolver\(\)\s*\n?", "", code_text)
        init_assets_dict, init_solver, wall = load_assets_and_constraints(code_text, wall, obj_descriptions, assets_add=vis_assets, solver=last_solver)
        layouts[str(region_idx)] = run_floor_regions_ccea(init_assets_dict, door, init_solver, room_bound, code_text, wall, glb_render, gpt_api, True, True, True, ccea_gd_layout_optimization)
        layouts[str(region_idx)] = detach_tensor(layouts[str(region_idx)])
        last_solver = init_solver
        from utils.visualization import render_scene
        vis_assets = layouts[str(region_idx)].copy()
        if init:
            init = False
            obj_descriptions_all = obj_descriptions.copy()
        else:
            obj_descriptions_all.update(obj_descriptions)
        init = False
        plot_floorplan_with_doors_windows(floor_xy, door_location, window_locations, areas=None, result=vis_assets, out_path=f"{output_dir}/ccea/floor_plan.png")
    vis_assets = tensor_to_np(vis_assets)
    p = Process(target=render_scene, args=(vis_assets, f'{output_dir}/ccea/layout', floor_xy, floor_plan, None, [], None, None, True, True, glb_render))
    p.start()
    p.join()
    del sys.modules["utils_areas.visualization"]

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
    layouts_all = copy.deepcopy(vis_assets)
    print(glb_render)
    # ai2thorhab dataset
    # root_dir = "small_assets/ai2thorhab-uncompressed/assets"
    # img_dir = "small_assets/render"
    # dataset = "ai2thorhub"

    # hssd dataset
    root_dir = "hssd-models/objects"
    img_dir = "hssd_render/objects"
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
        mesh_path = glb_render[label]
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
        json_str = re.search(r"```json\n(.+?)```", json_str, re.DOTALL).group(1)
        json_data_r = json.loads(json_str)
        for reg in json_data_r["regions"]:
            for key_ in reg["item"]:
                # 转为小写并替换空格为下划线
                key_ = key_.lower().replace(" ", "_").replace("-", "_")
        utilization_map = {r['id']: r['xy_utilization'] for r in json_data_r['regions']}
        open_surface_id = []
        print(utilization_map)
        for rid, region in enumerate(support_regions):
            region['utilization'] = utilization_map.get(rid, 0)
            if float(region["clearance"]) == 1.0:
                open_surface_id.append(str(rid))
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
        results = match_text_queries(queries, bound, index, filenames, dataset, top_k=5)
        merged = merge_items_into_regions(support_regions, results, top_k=1)
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
                print(f"  {item}: {name}, scale={scale}")

        print("""
        ===================================================
        =========== Step 4: Layout Optimization ===========
        ===================================================
        """)

        assets = generate_assets_from_merged(merged, dataset)

        # 这里要改，生成一个字典，对应每个区域的constraint，但是这里不需要每个区域都有。输入可以是各个区域的物体，这个家具的名字，这样的。还要输入上面整体大家具的render图，以便生成正确的位置关系。
        obj_descriptions_all = {}
        open_region = {}

        for idx in utilization_map:
            str_idx = str(idx)
            if str_idx not in open_surface_id:
                continue
            obj_descriptions = generate_obj_simple(assets[str_idx], str_idx)
            if not obj_descriptions:
                continue
            obj_descriptions_all[str_idx] = obj_descriptions
            for region in json_data_r['regions']:
                if region['id'] == idx:
                    open_region[str_idx] = region
                    break

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
        
        from utils.visualization import visualize_ea_layout_from_paths
        visualize_ea_layout_from_paths(mesh_path, merged, ea_result, dataset, save_path=f"{output_dir}/{label}/genesis_rendered.png")
        del sys.modules["utils_areas.visualization"]
        
        result_small[label] = ea_result
        merged_small[label] = merged
        # for idx in open_region_idx:
        #     res = trans_global(ea_result[idx])
        #     layouts_all.update(res)
    from utils.visualization import render_scene_fin
    render_scene_fin(vis_assets, floor_plan, 'final', ids_floor, result_small, merged_small, glb_render, floor_xy, output_dir)

    # 这里layout要加上小物体的？怎么加，就单独的吗？
    nav, col, oob, pop, pos, rot, ovr = get_metrics(floor_vertices, layouts_all, task, metrics_api, f'{output_dir}/final_top.png')   
    with open(f"{output_dir}/metrics_results.txt", "w") as f:
        f.write(f"NAV: {nav}\n")
        f.write(f"COL: {col}\n")
        f.write(f"OOB: {oob}\n")
        f.write(f"POP: {pop}\n")

        f.write(f"POS: {pos}\n")
        f.write(f"ROT: {rot}\n")
        f.write(f"OVR: {ovr}\n")
