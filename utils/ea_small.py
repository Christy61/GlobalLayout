"""Small-object EA: shelf fill + open-surface CCEA+GD (ablation-aligned)."""
import copy
import random
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from copy import deepcopy
from tqdm import tqdm
from shapely.geometry import Polygon
import networkx as nx
from deap import base, creator, tools
from scipy.spatial.distance import cdist
from utils.optimization import recaculate_bbox, recaculate_bbox_w_remove, ensure_grad, cal_initial_loss_group, optimize_pose_region
from utils.small_conflict import update_solver_s, build_group_center_graph
from utils.small_solver import load_assets_and_constraints_s
from utils.ea import preprocess_shelf_region, postprocess_layout

_EA_ABLATION_MODE = "baseline"

def oriented_half_extents(asset, theta_deg):
    w, l = float(asset['bbox'][0]), float(asset['bbox'][1])
    theta_norm = theta_deg % 180
    nearest = round(float(theta_norm)/90)*90
    if int(nearest) % 180 == 0:
        return w/2.0, l/2.0
    else:
        return l/2.0, w/2.0
        
def sample_new_group_center(group_bbox, group_center, rb):
    """在房间内采样一个新的群体中心位置"""
    bbox_width = group_bbox[0]
    bbox_height = group_bbox[1]
    xmin, xmax, ymin, ymax = rb
    min_x = xmin + bbox_width / 2
    max_x = xmax - bbox_width / 2
    min_y = ymin + bbox_height / 2
    max_y = ymax - bbox_height / 2
    
    if min_x >= max_x or min_y >= max_y:
        return group_center
    
    # 随机偏移
    offset = np.random.normal(0., 0.5, size=(2))
    new_x = group_center[0] + offset[0]
    new_y = group_center[1] + offset[1]

    # 检查 x 是否越界
    if new_x < min_x:
        new_x = min_x + abs(offset[0])  # 反向偏移
    elif new_x > max_x:
        new_x = max_x - abs(offset[0])  # 反向偏移

    # 检查 y 是否越界
    if new_y < min_y:
        new_y = min_y + abs(offset[1])
    elif new_y > max_y:
        new_y = max_y - abs(offset[1])
    new_x = np.clip(new_x, min_x, max_x)
    new_y = np.clip(new_y, min_y, max_y)
    return [new_x, new_y]

def apply_translation(existing_assets, translation_vector):
    """将平移向量应用到所有物体"""
    shifted_existing_assets = {}

    for idx, key in enumerate(existing_assets):
        asset = existing_assets[key].copy()
        x = asset['pos'][0].item() if hasattr(asset['pos'][0], 'item') else asset['pos'][0]
        y = asset['pos'][1].item() if hasattr(asset['pos'][1], 'item') else asset['pos'][1]
        new_x = x + translation_vector[0]
        new_y = y + translation_vector[1]
        with torch.no_grad():
            asset['pos'][:2] = torch.tensor([new_x, new_y], dtype=asset['pos'].dtype, device=asset['pos'].device)
        shifted_existing_assets[key] = asset
    
    return shifted_existing_assets

def calculate_group_bbox_and_center(existing_assets):
    """计算整个群体的边界框和中心位置（假设theta=0）"""
    all_min_x, all_min_y = float('inf'), float('inf')
    all_max_x, all_max_y = float('-inf'), float('-inf')
    
    for key in existing_assets:
        asset = existing_assets[key]
        x = asset['pos'][0].item() if hasattr(asset['pos'][0], 'item') else asset['pos'][0]
        y = asset['pos'][1].item() if hasattr(asset['pos'][1], 'item') else asset['pos'][1]
        r = asset['phy'][0] * 180 / torch.pi if isinstance(asset['phy'][0], float) else (asset['phy'][0] * 180 / torch.pi).item()
        # 假设theta=0计算半尺寸
        half_extents = oriented_half_extents(asset, r)
        half_x, half_y = half_extents[0], half_extents[1]
        
        # 更新边界框
        all_min_x = min(all_min_x, x - half_x)
        all_min_y = min(all_min_y, y - half_y)
        all_max_x = max(all_max_x, x + half_x)
        all_max_y = max(all_max_y, y + half_y)
    
    # 计算群体中心
    group_center = [(all_min_x + all_max_x) / 2, (all_min_y + all_max_y) / 2]
    group_bbox = [all_max_x-all_min_x, all_max_y-all_min_y]
    
    return group_bbox, group_center

def shift_init(existing_assets, rb):
    if not existing_assets:
        return {}
    group_bbox, group_center = calculate_group_bbox_and_center(existing_assets)
    new_center = sample_new_group_center(group_bbox, group_center, rb)
    translation_vector = np.array(new_center) - np.array(group_center)
    shifted_existing_assets = apply_translation(existing_assets, translation_vector)
    return shifted_existing_assets

def normalize_pose_vectors(vectors):
    """Normalize rotation part so that distance is comparable."""
    normed = vectors.copy()
    for i in range(3, vectors.shape[1], 4):  # 每 4 维的第 4 个是角度
        normed[:, i] = normed[:, i] / 180.0
    return normed

def ea_fill_shelf(
    assets, region_idx, shelf_length, shelf_width, shelf_z, clearance,
    target_utilization=0.6, generations=30, pop_size=50,
    spacing_choices=[0.0, 0.025, 0.05, 0.1, 0.2]
):
    region_bound = (0.0, shelf_length, 0.0, shelf_width)
    spacing_choices = [x * shelf_length for x in spacing_choices]
    num_assets = len(assets)
    num_spacing = len(spacing_choices)
    # Inset along shelf width (depth in layout) so the first scan row is not flush at y=0.
    shelf_plane_y_inset_m = 0.05

    # 每个物体间隙都可独立选择，最左边也有spacing
    creator_key_fitness = "FitnessMulti"
    creator_key_ind = "Individual"
    # 防止重复创建
    if not hasattr(creator, creator_key_fitness):
        creator.create(creator_key_fitness, base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, creator_key_ind):
        creator.create(creator_key_ind, list, fitness=getattr(creator, creator_key_fitness))

    toolbox = base.Toolbox()

    def generate_individual():
        order = random.sample(range(num_assets), num_assets)
        # 每个物体之间的spacing index, 以及最左边的spacing index
        spacing_idxs = [random.randint(0, num_spacing - 1) for _ in range(num_assets + 1)]
        return creator.Individual(order + spacing_idxs)

    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def adaptive_rate(gen, max_gen, start_rate, end_rate):
        return start_rate - (start_rate - end_rate) * (gen / max_gen)

    def is_collision(pos, asset, placed, spacing):
        x1, y1 = pos
        l1, w1 = asset["length"], asset["width"]
        for item in placed:
            x2, y2 = item["x"], item["y"]
            l2, w2 = item["length"], item["width"]
            if not (x1 + l1 + spacing <= x2 or x2 + l2 + spacing <= x1 or
                    y1 + w1 + spacing <= y2 or y2 + w2 + spacing <= y1):
                return True
        return False

    def find_valid_position(asset, placed, spacing, shelf_z, clearance, x_start):
        step = 0.02
        if asset.get("z-axis"):
            for item in placed[::-1]:
                if item['name'] == asset['name']:
                    top_z = item["z"] + item["height"]
                    if top_z + asset["height"] >= shelf_z + clearance:
                        continue
                    pos = (item["x"], item["y"], top_z)
                    return pos

        x = x_start
        while x + asset["length"] <= shelf_length + 1e-8:
            y = float(shelf_plane_y_inset_m)
            while y + asset["width"] <= shelf_width + 1e-8:
                if asset["width"] > shelf_width or asset["length"] > shelf_length:
                    continue
                if not is_collision((x, y), asset, placed, spacing):
                    return x, y, shelf_z
                y += step
            x += step
        return None

    def fitness(individual, target_utilization, shelf_z):
        order = individual[:num_assets]
        spacing_idxs = individual[num_assets:]
        spacings = [spacing_choices[idx] for idx in spacing_idxs]  # len = num_assets+1

        placed = []
        area_used = 0.0
        z_penalty = 0.0
        center_penalty = 0.0
        x_cursor = spacings[0]  # 最左边的spacing

        for i, idx in enumerate(order):
            asset = assets[idx]
            spacing = spacings[i + 1] if i < num_assets - 1 else 0.0  # 最后一个物体右侧不加spacing
            pos = find_valid_position(asset, placed, 0.0, shelf_z, clearance, x_cursor)
            if pos:
                x, y, z = pos
                placed.append({
                    "name": asset["name"],
                    "x": x + asset["length"]/2,
                    "y": shelf_width - y - asset["width"]/2,
                    "z": z,
                    "length": asset["length"],
                    "width": asset["width"],
                    "height": asset.get("height", 0.05),
                    "scale": asset["scale"]
                })
                area_used += asset["length"] * asset["width"]
                if asset["center"] == True:
                    actual_center = x + asset["length"] / 2
                    expected_center = shelf_length / 2
                    deviation = abs(actual_center - expected_center) / shelf_length
                    center_penalty += deviation
                x_cursor = x + asset["length"] + spacing
            else:
                # 放不下直接break
                break

        # for i, item_i in enumerate(placed):
        #     for j, item_j in enumerate(placed):
        #         if i == j:
        #             continue
        #         top_i = item_i["z"] + item_i["height"]
        #         top_j = item_j["z"] + item_j["height"]
        #         if item_i["y"] > item_j["y"] and top_i > top_j:
        #             xi1, xi2 = item_i["x"], item_i["x"] + item_i["length"]
        #             xj1, xj2 = item_j["x"], item_j["x"] + item_j["length"]
        #             overlap = max(0, min(xi2, xj2) - max(xi1, xj1))
        #             if overlap > 0:
        #                 z_penalty += (top_i - top_j + 0.1) * overlap

        utilization_factor = area_used / (shelf_length * shelf_width)
        if utilization_factor - target_utilization >= 0.0 and (utilization_factor - target_utilization < 0.1):
            utilization_loss = 0.0
        else:
            utilization_loss = abs(utilization_factor - target_utilization)
        penalty = center_penalty
        return (utilization_loss, penalty), placed

    def eval_individual(individual):
        score, _ = fitness(individual, target_utilization, shelf_z)
        return score

    def trans(layout):
        result = {}
        name_buffer = {}
        for idx, item in enumerate(layout):
            n_idx = name_buffer.get(item["name"], 0) + 1
            pos = torch.tensor([item["x"], item["y"], item["z"]], dtype=torch.float32)
            bbox = torch.tensor([item["width"], item["length"], item["height"]], dtype=torch.float32)
            result[f"{item['name']}_{n_idx}"] = {
                "pos": pos,
                "phy": torch.tensor([0.0]),
                "scale": item["scale"],
                "region_idx": 0,
                "bbox": bbox,
                "id": idx
            }
            name_buffer[item["name"]] = n_idx
        return result
    
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    best_layout = None
    best_score = float("inf")

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    for gen in tqdm(range(generations)):
        if num_assets > 1:
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            cxpb = adaptive_rate(gen, generations, start_rate=0.8, end_rate=0.3)
            mutpb = adaptive_rate(gen, generations, start_rate=0.4, end_rate=0.1)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    # 只交叉排列部分
                    toolbox.mate(child1[:num_assets], child2[:num_assets])
                    # spacing部分也可以交叉
                    if random.random() < 0.4:
                        idx = random.randint(0, num_assets)
                        child1[num_assets + idx], child2[num_assets + idx] = child2[num_assets + idx], child1[num_assets + idx]
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant[:num_assets])
                    # spacing部分独立变异
                    for i in range(num_assets + 1):
                        if random.random() < 0.6:
                            mutant[num_assets + i] = random.randint(0, num_spacing - 1)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

        top_ind = tools.selBest(population, 1)[0]
        top_score, top_layout = fitness(top_ind, target_utilization, shelf_z)

        if top_score[0] < best_score:
            best_score = top_score[0]
            best_layout = top_layout
    layout = trans(best_layout)
    layout_all = recaculate_bbox_w_remove(layout, region_bound)
    return layout_all

def ea_fill_open_region(
    init_assets_dict, 
    obj_descriptions,
    region_i,
    shelf_length, 
    shelf_width, 
    shelf_z, 
    clearance,
    target_utilization, 
    gpt_api,
    code,
    output_dir=None,
    label=None,
    area_info=None,
    rotation_choices=[0, 90, 180, 270],
    generations=51, 
    pop_size=50,
    top_k=5,
    optimize_every_n=10,
    graph=True,
    visualize=False,
    use_ea=True,
    use_feasible_region=False,
    mutate_use_feasible=False,
    solver_flags=None,
):
    grid_size = 0.015
    region_bound = (0.0, shelf_length, 0.0, shelf_width)
    xmin, xmax, ymin, ymax = region_bound

    def snap_to_grid(v):
        return round(round(v / grid_size) * grid_size, 6)

    def build_region_grid_points():
        # 以grid对齐生成所有货架区域内网格点
        gx_min = snap_to_grid(0.0)
        gx_max = snap_to_grid(shelf_length)
        gy_min = snap_to_grid(0.0)
        gy_max = snap_to_grid(shelf_width)
        xs = []
        x = gx_min
        while x <= gx_max + 1e-9:
            xs.append(round(x, 6))
            x += grid_size
        ys = []
        y = gy_min
        while y <= gy_max + 1e-9:
            ys.append(round(y, 6))
            y += grid_size
        return [(xv, yv) for xv in xs for yv in ys]

    def oriented_half_extents(asset, theta_deg):
        # 以物体宽长与scale得到外接矩形半尺寸；90/270交换宽长
        s = float(asset.get('scale', 1.0))
        if "width" in asset:
            w = float(asset['width']) * s
            l = float(asset['length']) * s
        else:
            w = float(asset['bbox'][0]) * s
            l = float(asset['bbox'][1]) * s
        # 这里起始是朝向front不是right。
        theta_norm = theta_deg % 180  # 先折叠到 [0, 180)
        nearest = theta_norm / 90 * 90
        if int(nearest) % 180 == 0:
            return l / 2.0, w / 2.0
        else:
            return w / 2.0, l / 2.0

    def clamp_xy_to_region(x, y, hx, hy, rb):
        xmin, xmax, ymin, ymax = rb
        # 裁剪到 [边界 + 半尺寸, 边界 - 半尺寸]
        min_x = xmin + hx
        max_x = xmax - hx
        min_y = ymin + hy
        max_y = ymax - hy
        x = min(max(min_x, x), max_x)
        y = min(max(min_y, y), max_y)
        return snap_to_grid(x), snap_to_grid(y)
    

    room_grid_points = build_region_grid_points()
    region_info_list = []

    for asset_idx, asset in enumerate(init_assets_dict):
        region_info_list.append((region_i, asset_idx, asset))
    
    stack_counts = {}
    stack_templates = {}
    base_region_info_list = []

    def is_stackable_asset(asset):
        name = str(asset.get("name", "")).lower().replace(" ", "_").replace("-", "_")
        if name.startswith("flat_") or name == "flat_book":
            return False
        return bool(asset.get("z-axis", False)) or name == "standing_book"

    for region_idx, asset_idx, asset in region_info_list:
        if is_stackable_asset(asset):
            nm = asset['name']
            stack_counts[nm] = stack_counts.get(nm, 0) + 1
            if nm not in stack_templates:
                stack_templates[nm] = asset
                base_region_info_list.append((region_idx, asset_idx, asset))  # 仅保留第一件作为底部代表
            # 其他同类实例不进入基因，稍后在 post-process 中展开
        else:
            base_region_info_list.append((region_idx, asset_idx, asset))
    
    num_assets_total = len(base_region_info_list)
    if_placed = [False] * num_assets_total

    # def generate_individual():
    #     ind = []
    #     last_position_by_name = {}
    #     for i, (region_idx, _, asset) in enumerate(base_region_info_list):
    #         rots = list(rotation_choices)
    #         random.shuffle(rots)

    #         picked = None
    #         picked_theta = None

    #         if asset['name'] == "standing_book":  # standing_book
    #             # xmin, xmax, ymin, ymax = region_bound
    #             # hx, hy = oriented_half_extents(asset, 0)
    #             # if asset["name"] in last_position_by_name:
    #             #     y_fixed = last_position_by_name[asset["name"]][1]
    #             # else:
    #             #     y_fixed = ymax - hy
    #             # valid_xs = [p[0] for p in room_grid_points if xmin + hx <= p[0] <= xmax - hx]
    #             # if not valid_xs:
    #             #     continue
    #             # picked_x = random.choice(valid_xs)
    #             # picked_theta = 0
    #             # ind.extend([picked_x, y_fixed, picked_theta])
    #             # if not asset["name"] in last_position_by_name:
    #             #     last_position_by_name[asset["name"]] = (picked_x, y_fixed, picked_theta)
    #             # if_placed[i] = True
    #             pass

    #         elif not asset.get("z-axis", False):  
    #             for theta in rots:
    #                 hx, hy = oriented_half_extents(asset, theta)
    #                 xmin, xmax, ymin, ymax = region_bound
    #                 valid_points = [
    #                     p for p in room_grid_points
    #                     if (xmin + hx <= p[0] <= xmax - hx) and (ymin + hy <= p[1] <= ymax - hy)
    #                 ]
    #                 if not valid_points:
    #                     continue
    #                 picked = random.choice(valid_points)
    #                 picked_theta = theta
    #                 break

    #             if picked:
    #                 if_placed[i] = True
    #                 ind.extend([picked[0], picked[1], picked_theta])
    #                 last_position_by_name[asset["name"]] = (picked[0], picked[1], picked_theta)

    #         else:
    #             # -------- 堆叠物体，继承同类的坐标 --------
    #             if asset["name"] in last_position_by_name:
    #                 # x, y, theta = last_position_by_name[asset["name"]]
    #                 # ind.extend([x, y, theta])
    #                 pass
    #             else:
    #                 # 第一次出现的同类，还是得先正常放置
    #                 for theta in rots:
    #                     hx, hy = oriented_half_extents(asset, theta)
    #                     xmin, xmax, ymin, ymax = region_bound
    #                     valid_points = [
    #                         p for p in room_grid_points
    #                         if (xmin + hx <= p[0] <= xmax - hx) and (ymin + hy <= p[1] <= ymax - hy)
    #                     ]
    #                     if not valid_points:
    #                         continue
    #                     picked = random.choice(valid_points)
    #                     picked_theta = theta
    #                     break

    #                 if picked:
    #                     if_placed[i] = True
    #                     ind.extend([picked[0], picked[1], picked_theta])
    #                     last_position_by_name[asset["name"]] = (picked[0], picked[1], picked_theta)

    #     return ind

    def generate_individual(init_assets, region_bound, seed, center_assets=None, center_nodes=None, groups=None, id_to_key=None, if_center=False):
        np.random.seed(seed)
        rotation_choices = [i*30 for i in range(12)]
        individual = []
        xmin, xmax, ymin, ymax = region_bound
        xs = np.arange(xmin, xmax + 1e-6, grid_size)
        ys = np.arange(ymin, ymax + 1e-6, grid_size)
        room_grid_points = [(x, y) for x in xs for y in ys]
        if (not if_center) and (center_assets is not None):
            refer_center = {j: center_nodes[i] for i, group in enumerate(groups) for j in group}
            for key, asset in init_assets.items():
                theta_deg = np.random.choice(rotation_choices)
                hx, hy = oriented_half_extents(asset, theta_deg)
                center_id = refer_center[asset['id']]
                refer_center_asset = center_assets[id_to_key[center_id]]
                corners = refer_center_asset['corners'].detach()
                xmin_c, xmax_c = float(corners.min(axis=0).values[0] + refer_center_asset['pos'][0]), \
                    float(corners.max(axis=0).values[0] + refer_center_asset['pos'][0])
                ymin_c, ymax_c = float(corners.min(axis=0).values[1] + refer_center_asset['pos'][1]), \
                    float(corners.max(axis=0).values[1] + refer_center_asset['pos'][1])

                margin = 0.3
                outer_xmin = xmin_c - margin
                outer_xmax = xmax_c + margin
                outer_ymin = ymin_c - margin
                outer_ymax = ymax_c + margin

                # 在 bbox 周围四个方向上任选一个区段采样
                side = np.random.choice(["left", "right", "up", "down"])
                if side == "left":
                    x = np.random.uniform(outer_xmin - hx, xmin_c - hx)
                    y = np.random.uniform(outer_ymin, outer_ymax)
                elif side == "right":
                    x = np.random.uniform(xmax_c + hx, outer_xmax + hx)
                    y = np.random.uniform(outer_ymin, outer_ymax)
                elif side == "up":
                    x = np.random.uniform(outer_xmin, outer_xmax)
                    y = np.random.uniform(ymax_c + hy, outer_ymax + hy)
                else:  # down
                    x = np.random.uniform(outer_xmin, outer_xmax)
                    y = np.random.uniform(outer_ymin - hy, ymin_c - hy)
                x, y = clamp_xy_to_region(x, y, hx, hy, region_bound)
                z = 0.0
                individual.extend([x, y, z, theta_deg])
        else:
            for key, asset in init_assets.items():
                theta_deg = np.random.choice(rotation_choices)
                hx, hy = oriented_half_extents(asset, theta_deg)
                if room_grid_points:
                    valid_points = [
                        p for p in room_grid_points
                        if (xmin + hx <= p[0] <= xmax - hx) and (ymin + hy <= p[1] <= ymax - hy)
                    ]
                    if valid_points:
                        x, y = valid_points[np.random.choice(len(valid_points))]
                    else:
                        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                        x, y = clamp_xy_to_region(cx, cy, hx, hy, region_bound)
                z = 0.0
                individual.extend([x, y, z, theta_deg])

        return individual

    def decode_individual(individual, init_assets):
        result = {}
        i = 0
        for key, asset in init_assets.items():
            x = individual[4*i]
            y = individual[4*i + 1]
            z = shelf_z
            pos = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True)
            r = torch.tensor([individual[4*i + 3]*torch.pi/180], dtype=torch.float32, requires_grad=True)
            bbox = torch.tensor(asset['bbox'], dtype=torch.float32, requires_grad=False)
            scale = asset['scale']
            result[key] = {
                'id': asset['id'],
                'pos': pos,
                'phy': r,
                'bbox': bbox,
                'scale': scale,
                'region_idx': asset['region_idx']
            }
            i += 1
        return result
    
    def encode_individual(asset_dict):
        individual = []
        for obj_idx in asset_dict.keys():
            asset = asset_dict[obj_idx]
            x = asset['pos'][0] if isinstance(asset['pos'][0], float) else asset['pos'][0].item()
            y = asset['pos'][1] if isinstance(asset['pos'][1], float) else asset['pos'][1].item()
            z = asset['pos'][2] if isinstance(asset['pos'][2], float) else asset['pos'][2].item()
            r = asset['phy'][0] * 180 / torch.pi if isinstance(asset['phy'][0], float) else (asset['phy'][0] * 180 / torch.pi).item()
            individual.extend([x, y, z, r])
        return individual

    def apply_repulsion(offspring, groups_all=None, alpha=0.005):
        """
        群体排斥力避免塌缩（只对 mask_all 指定的 asset 生效）
        
        Args:
            offspring: (λ, D)，种群个体矩阵
            groups_all: list[int] or np.ndarray，表示需要计算排斥的 asset 索引（基于每个 asset 的4个参数起点）
            alpha: float，排斥强度
        """
        xs = offspring[:, 0::4]
        ys = offspring[:, 1::4]
        pos = np.stack([xs, ys], axis=-1)  # shape: (λ, n_assets, 2)

        if groups_all is not None:
            groups_all = np.array(groups_all, dtype=int)
            pos_masked = pos[:, groups_all, :]  # 只取被 mask 的 asset
        else:
            pos_masked = pos

        # --- pairwise 差分 (λ, λ, n_masked_assets, 2)
        diffs = pos_masked[:, None, :, :] - pos_masked[None, :, :, :]

        # --- 距离平方并避免除0
        dist2 = np.sum(diffs ** 2, axis=-1, keepdims=True) + 1e-6

        # --- 归一化方向并加权（距离越近，排斥越强）
        repel = np.sum(diffs / dist2, axis=1)  # (λ, n_masked_assets, 2)

        # --- 更新被 mask 的 asset 的位置
        pos[:, groups_all, :] = pos[:, groups_all, :] + alpha * repel

        # --- 写回 offspring
        offspring[:, 0::4] = pos[:, :, 0]
        offspring[:, 1::4] = pos[:, :, 1]

        return offspring


    def recombine_and_mutate(assets, parents, parent_fitness, mask_all, w_l, n_offspring, 
                            seed, sigma_pos=[0.3, 0.3], sigma_rot=0.5, repel_alpha=0.01, 
                            swap_probs=0.5, group_mut_prob=0.10, mask_group=None, groups_all=None,
                            idx_swap=None, center_map=None, cg_weight=0.20, d_max=1.0, d_min=0.5):
        """
        ESGD Recombination + Mutation step (vectorized)
        ------------------------------------------------
        parents: np.array of shape (μ, D)
        parent_fitness: list of fitness (lower is better)
        returns: np.array of shape (λ, D)
        """
        np.random.seed(seed)
        parents = np.array(parents)
        n_assets = len(assets)
        # --- Normalize fitness -> probability for parent selection ---
        fitness_arr = np.array(parent_fitness)
        if idx_swap == None:
            idx_swap = [i for i in range(n_assets)]
        
        if fitness_arr.std() < 1e-4:
            probs = np.ones_like(fitness_arr) / len(fitness_arr)
        else:
            beta = 1.0 / fitness_arr.std()
            probs = np.exp(-beta * (fitness_arr - fitness_arr.min()))
            probs /= probs.sum()

        # -------- Step 1. 多中心采样 --------
        parent_indices = np.random.choice(len(parents), size=n_offspring, p=probs)
        offspring = parents[parent_indices]
        
        if len(idx_swap) >= 2:
            for i in range(n_offspring):
                if np.random.rand() < swap_probs:
                    i1, i2 = np.random.choice(len(idx_swap), size=2, replace=False)
                    idx1, idx2 = idx_swap[i1], idx_swap[i2]
                    temp = offspring[i, idx1*4:idx1*4+4].copy()
                    offspring[i, idx1*4:idx1*4+4] = offspring[i, idx2*4:idx2*4+4]
                    offspring[i, idx2*4:idx2*4+4] = temp

        # -------- Step 2. 加噪声变异 --------
        D = offspring.shape[1]
        noise_x = np.random.normal(0.0, sigma_pos[0], size=(n_offspring, D//4))
        noise_y = np.random.normal(0.0, sigma_pos[1], size=(n_offspring, D//4))
        noise_rot = np.random.normal(0.0, sigma_rot, size=(n_offspring, D//4))
        delta_xs = np.where(mask_all[0::4], noise_x, 0.)
        delta_ys = np.where(mask_all[1::4], noise_y, 0.)
        offspring[:, 3::4] += np.where(mask_all[3::4], noise_rot, 0.)
        
        # --- Group-wise mutation (10% probability) ---
        if mask_group is not None:
            np.random.seed(seed+10)
            trigger = np.random.rand(n_offspring) < group_mut_prob
            if np.any(trigger):
                delta = np.random.uniform(
                low=[-xmax*0.05, -ymin*0.05], 
                high=[xmax*0.05, ymax*0.05], 
                size=(np.sum(trigger), 2)
                )
                group_indices = np.where(mask_group.reshape(n_assets, 4)[:, 0])[0]
                delta_xs[trigger][:, group_indices] += delta[:, [0]]
                delta_ys[trigger][:, group_indices] += delta[:, [1]]

        offspring[:, 0::4] += delta_xs
        offspring[:, 1::4] += delta_ys
        # --- Center-guided mutation (members pulled toward their group center) ---
        # offspring shape: (n_offspring, D)
        # build pos array (λ, n_assets, 2)
        xs = offspring[:, 0::4]   # (λ, n_assets)
        ys = offspring[:, 1::4]   # (λ, n_assets)
        pos = np.stack([xs, ys], axis=-1)  # (λ, n_assets, 2)
        if center_map is not None and len(center_map) > 0:
            members = np.array(list(center_map.keys()), dtype=int)
            centers = np.array([center_map[m] for m in members], dtype=int)
            pos_members = pos[:, members, :]
            pos_centers = pos[:, centers, :]

            bbox_half = w_l[centers] / 2.0  # (n_members, 2)
            radius_scale = np.random.uniform(0.8, 1.2, size=(n_offspring, len(members), 1))
            angles = np.random.uniform(0, 2*np.pi, size=(n_offspring, len(members), 1))
            offset = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
            target_pos = pos_centers + offset * (bbox_half[None, :, :] * radius_scale + d_min)

            vec = target_pos - pos_members
            pos[:, members, :] += cg_weight * vec
            offspring[:, 0::4] = pos[:, :, 0]
            offspring[:, 1::4] = pos[:, :, 1]

        # -------- Step 3. 排斥调整 --------
        offspring = apply_repulsion(offspring, groups_all, alpha=repel_alpha)

        # --- Compute half extents for all individuals in the population ---
        thetas = offspring[:, 3::4]
        theta_norm = thetas % 180
        nearest = np.round(theta_norm / 90) * 90
        is_0_or_180 = (nearest % 180 == 0)
        half_extents = np.zeros((is_0_or_180.shape[0], w_l.shape[0], 2))  # Ensure correct shape
        half_extents[:, :, 0] = np.where(is_0_or_180, w_l[:, 0] / 2.0, w_l[:, 1] / 2.0)  # hx
        half_extents[:, :, 1] = np.where(is_0_or_180, w_l[:, 1] / 2.0, w_l[:, 0] / 2.0)  # hy

        # --- Clamp to room bounds ---
        offspring[:, 0::4] = np.clip(offspring[:, 0::4], xmin + half_extents[:, :, 0], xmax - half_extents[:, :, 0])
        offspring[:, 1::4] = np.clip(offspring[:, 1::4], ymin + half_extents[:, :, 1], ymax - half_extents[:, :, 1])
        offspring[:, 3::4] = thetas

        return offspring
    
    # -------- fitness & GD --------
    def fitness(assets, solver, groups, weight_semantic=1.0, weight_physics=1.0, weight_project=0.5):
        loss_g, loss = cal_initial_loss_group(assets, solver, groups, region_bound, other_assets=None, region_center=None, z_min=0.0, weight_semantic=weight_semantic, weight_physics=weight_physics, weight_project=weight_project)
        return loss_g, loss
    
    def gradient_descent(iters, obj, solver, i, weight=0.2, vis=False):
        assets_pose, grad_dict, loss_opt, _ = optimize_pose_region(
            iters, f"full_{i}", obj, solver, region_bound, None, weight, z_min=shelf_z, glb_paths=None, vis=vis, door=None, pure_gd=True
        )
        return assets_pose, grad_dict, loss_opt

    def select_group_by_loss(loss_g, tau=1.0, p_min=0.05):
        loss_g = loss_g.detach().numpy()
        mn, mx = loss_g.min(), loss_g.max()
        if mx - mn < 1e-12:
            norm_loss = np.ones_like(loss_g) * 0.5
        else:
            norm_loss = (loss_g - mn) / (mx - mn)
        
        exp_scores = np.exp(norm_loss / tau)
        probs = exp_scores / exp_scores.sum()
        
        probs = np.clip(probs, p_min, 1.0)
        probs = probs / probs.sum()
        selected_idx = int(np.random.choice(len(loss_g), p=probs))
        return selected_idx
    
    def build_group_center_graph(G, groups, centrality, obj_new):
        """
        构建一个新的图 G_centered，只保留每个 group 的中心节点。
        如果 group 内部有多个高中心率节点，则分为多个 subgroup。
        保留原图的方向和边属性。
        """
        # --- 1. 找出每个 group 的中心节点或子组划分 ---
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

            # 计算中心率
            scores = {k: centrality.get(k, 0) for k in group_keys}
            high_nodes = [k for k, v in scores.items() if v > 2]

            # 情况 2: group 内物体较多且多个节点中心率高
            if len(group_keys) > 4 and len(high_nodes) >= 2:
                for node in high_nodes:
                    center_nodes[i] = node
                    # 子组：该中心节点 + 与它距离较近的节点
                    sub_members = []
                    for k in group_keys:
                        if k == node:
                            sub_members.append(k)
                        elif G.has_edge(node, k) or G.has_edge(k, node):
                            sub_members.append(k)
                    new_groups.append(set(sub_members))
                    i += 1 
                continue

            # 情况 3: 其他情况，只选中心率最高的节点
            node = max(scores, key=scores.get)
            center_nodes[i] = node
            new_groups.append(set(group_keys))
            i += 1

        # --- 2. 保留中心节点的有向多重子图 ---
        center_node_set = set(center_nodes.values())
        G_centered = G.copy()
        nodes_to_remove = [n for n in obj_new_ids if n not in center_node_set]
        G_centered.remove_nodes_from(nodes_to_remove)

        return G_centered, center_nodes, new_groups, nodes_to_remove

    def expand_stacks(result_base):
        result = {k: v for k, v in result_base.items()}
        next_idx = (len(result) + 1) if result else 0

        bases_by_name = {}
        for k, v in result_base.items():
            nm = k.rsplit('_', 1)[0]
            if nm in stack_counts: 
                bases_by_name[nm] = v

        for name, total in stack_counts.items():
            if total <= 1:
                continue
            if name not in bases_by_name:
                continue

            base_v = bases_by_name[name]
            if isinstance(base_v["bbox"], torch.Tensor):
                h = float(base_v["bbox"][2].item())
            else:
                h = float(base_v["bbox"][2])

            x0 = float(base_v["pos"][0].item())
            y0 = float(base_v["pos"][1].item())
            z0 = float(base_v["pos"][2].item())
            phy0 = base_v["phy"].clone() if hasattr(base_v["phy"], "clone") else torch.tensor(base_v["phy"])
            bbox0 = base_v["bbox"].clone() if hasattr(base_v["bbox"], "clone") else torch.tensor(base_v["bbox"])

            # 已有 1 件（底部代表），这里补 total-1 件
            for i in range(1, total):
                zi = z0 + i * h
                # 判断是否超出净高：顶部 = 底部 z + 高度
                if zi + h > clearance:
                    break  # 超高，停止堆叠
                pos = torch.tensor([x0, y0, zi]).float()
                result[f"{name}_{i}"] = {
                    "id": next_idx,
                    "pos": pos,
                    "phy": phy0.clone() if hasattr(phy0, "clone") else torch.tensor(phy0),
                    "bbox": bbox0.clone() if hasattr(bbox0, "clone") else torch.tensor(bbox0),
                    "scale": base_v["scale"],
                    "region_idx": base_v["region_idx"],
                }
                next_idx += 1
        return result
    
    gpt_init = False
    use_weight = True
    if_gd = True
    if_visualize = bool(visualize) and output_dir is not None

    def run_ea(G_ea, ea_assets, existing_assets, refer_center, center_nodes, groups, id_to_key, seed, center_assets, if_center=False, use_ea=True):
        # -----------------------------
        # Step 1. Initialization
        # -----------------------------
        if not ea_assets:
            return {}
        bound_m = np.array([xmax, ymax])
        population_mu = []
        mu = 8
        lam = pop_size - mu
        rho = mu
        m = mu
        sigma_pos = bound_m * 0.15
        asset_id_order = [asset["id"] for asset in ea_assets.values()]
        id_to_slot = {asset_id: slot for slot, asset_id in enumerate(asset_id_order)}
        asset_ids_present = set(asset_id_order)
        groups_all = [asset_id for asset_id in asset_id_order if asset_id not in existing_assets]

        weight = 0.10
        swap_probs = 0.10
        sigma_rot = 5.0
        gd_step = 300
        group_mut_prob = 0.15
        cg_weight = 0.0
        # existing_assets = ensure_grad(existing_assets)
        if gpt_init:
            init_count = max(mu // 5, 2)
            population_mu.extend([encode_individual(ea_assets) for _ in range(init_count)])
            population_mu.extend([generate_individual(ea_assets, region_bound, seed*i, center_assets, center_nodes, groups, id_to_key, if_center=if_center) for i in range(mu - init_count)])
        else:
            population_mu = [generate_individual(ea_assets, region_bound, seed*i, center_assets, center_nodes, groups, id_to_key, if_center=if_center) for i in range(mu)]
        
        fitness_mu = []
        assets_mu = {}
        solver_mu = {}
        loss_group = {}
        
        center_ids = set([v['id'] for v in obj_descriptions.values() if v['id'] in center_nodes.values()])
        anchor_ids = set([cid for cid in center_ids if cid in asset_ids_present])
        movable_ids = [asset_id for asset_id in groups_all if asset_id not in anchor_ids]
        movable_slots = [id_to_slot[asset_id] for asset_id in movable_ids if asset_id in id_to_slot]
        refer_center = {
            k: v for k, v in refer_center.items()
            if k in id_to_slot and v in id_to_slot
        }
        refer_center_slots = {
            id_to_slot[k]: id_to_slot[v] for k, v in refer_center.items()
        }
        repel_alpha = 0.03 / max(len(movable_slots), 1)
        for i, ind in enumerate(population_mu):
            # evaluate each group individual in full context
            assets_mu[i] = decode_individual(ind, ea_assets)
            existing_assets = ensure_grad(existing_assets)
            assets_mu[i].update(existing_assets)
            # else:
            #     shifted_existing_assets = shift_init(existing_assets, room_bound)
            #     shifted_existing_assets = ensure_grad(shifted_existing_assets)
            #     assets_mu[i].update(shifted_existing_assets)
            population_mu[i] = encode_individual(assets_mu[i])
            solver_mu[i] = copy.deepcopy(init_solver)
            if use_weight:
                solver_mu[i].update_constraints(G_ea, assets_mu[i], ea_assets, id_to_key)
            else:
                solver_mu[i].update_constraints_woweight(G_ea, assets_mu[i], id_to_key)
            loss_g, loss = fitness(
                assets_mu[i], solver_mu[i], groups,
                weight_semantic=1.0, weight_physics=1.0
            )
            loss_group[i] = loss_g
            fitness_mu.append(loss.item())
        
        fitness_mu = np.array(fitness_mu)
        if not use_ea:
            best_idx = int(np.argsort(fitness_mu)[0])
            best_assets = assets_mu[best_idx]
            optimized_assets, _, _ = gradient_descent(
                gd_step, best_assets, solver_mu[best_idx], best_idx, weight=0.3, vis=False
            )
            recaculate_bbox_w_remove(optimized_assets, region_bound)
            return optimized_assets
        print(f"[Init] mu={mu}, rho={rho}, lambda={lam}, m={m}")
        print(f"[Init] Evaluated {mu} parents, mean fitness = {float(fitness_mu.mean()):.4f}")

        total_generations = generations
        prev_mean_fitness = None
        stagnation_counter = 0
        stagnation_eps = 1e-3
        stagnation_k = 3
        gd_step_hybrid = max(1, gd_step // 4)
        gd_step_refine = gd_step
        ablation_wo_ea = _EA_ABLATION_MODE == "wo_ea"
        ablation_wo_gd = _EA_ABLATION_MODE == "wo_gd"

        def _shift_offspring_group(offspring_list, prob=0.3):
            if not offspring_list or not movable_slots:
                return offspring_list, 0
            shifted = np.array(offspring_list, dtype=float, copy=True)
            moved = 0
            sx = max(0.02, float(sigma_pos[0]) * 0.4)
            sy = max(0.02, float(sigma_pos[1]) * 0.4)
            for idx in range(len(shifted)):
                if np.random.rand() >= prob:
                    continue
                dx = np.random.uniform(-sx, sx)
                dy = np.random.uniform(-sy, sy)
                for slot in movable_slots:
                    shifted[idx, slot * 4] += dx
                    shifted[idx, slot * 4 + 1] += dy
                moved += 1
            shifted[:, 0::4] = np.clip(shifted[:, 0::4], xmin, xmax)
            shifted[:, 1::4] = np.clip(shifted[:, 1::4], ymin, ymax)
            return shifted.tolist(), moved

        for gen in range(total_generations):
            current_mean_fitness = float(fitness_mu.mean())
            if prev_mean_fitness is not None:
                if abs(current_mean_fitness - prev_mean_fitness) < stagnation_eps:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
            prev_mean_fitness = current_mean_fitness

            if gen < 5:
                swap_probs = 1.0 * (0.8 ** gen)
                sigma_pos_stage = sigma_pos
                sigma_rot_stage = sigma_rot
                elite_keep_fix = mu
                elite_keep = mu
                gd_step_n = 0
                do_gd = False
            elif gen < total_generations - 1:
                swap_probs = 0.2 * (0.95 ** (gen - 3))
                sigma_pos_stage = [sigma_pos[0] * 1.5, sigma_pos[1] * 1.5]
                sigma_rot_stage = sigma_rot * 2
                elite_keep_fix = mu - 1
                elite_keep = mu
                gd_step_n = gd_step_hybrid
                do_gd = False
            else:
                swap_probs = 0.0
                sigma_pos_stage = [0.0, 0.0]
                sigma_rot_stage = 0.0
                elite_keep_fix = mu - 2
                elite_keep = mu
                gd_step_n = gd_step_refine
                do_gd = True

            if (not ablation_wo_ea) and gen < total_generations - 1:
                if not if_center:
                    swap_slots_run = []
                    mask_group = np.zeros(len(population_mu[0]), dtype=bool)
                    m_idxs = [select_group_by_loss(loss_group[idx]) for idx in range(mu)]
                    group_probabilities = np.array(
                        [m_idxs.count(i) for i in range(len(groups))], dtype=float
                    )
                    group_probabilities = group_probabilities / max(group_probabilities.sum(), 1e-8)
                    if len(ea_assets) <= 10:
                        mutate_group_idx = int(np.random.choice(len(groups), p=group_probabilities))
                        for obj_id in groups[mutate_group_idx]:
                            if obj_id in anchor_ids:
                                continue
                            slot = id_to_slot.get(obj_id)
                            if slot is None:
                                continue
                            swap_slots_run.append(slot)
                            mask_group[slot * 4:slot * 4 + 2] = True
                            mask_group[slot * 4 + 3:slot * 4 + 4] = True
                else:
                    mask_group = None
                    swap_slots_run = [
                        id_to_slot[cid] for cid in center_ids if cid in id_to_slot
                    ]

                mask_all = np.zeros(len(population_mu[0]), dtype=bool)
                for asset_id, slot in id_to_slot.items():
                    if asset_id in anchor_ids:
                        continue
                    mask_all[slot * 4:slot * 4 + 2] = True
                    mask_all[slot * 4 + 3:slot * 4 + 4] = True

                w_l = np.zeros((len(ea_assets), 2))
                for j, asset in enumerate(ea_assets.values()):
                    w_l[j] = [float(asset['bbox'][0]), float(asset['bbox'][1])]

                offspring_lambda = recombine_and_mutate(
                    ea_assets,
                    np.array(population_mu),
                    fitness_mu, mask_all, w_l,
                    lam,
                    seed * 100 + gen * 10,
                    sigma_pos=sigma_pos_stage,
                    sigma_rot=sigma_rot_stage,
                    repel_alpha=repel_alpha,
                    swap_probs=swap_probs,
                    group_mut_prob=group_mut_prob,
                    mask_group=mask_group,
                    groups_all=movable_slots,
                    idx_swap=swap_slots_run,
                    center_map=refer_center_slots,
                    cg_weight=cg_weight,
                ).tolist()

                if (gen >= 5) and (stagnation_counter >= stagnation_k):
                    offspring_lambda, moved_count = _shift_offspring_group(offspring_lambda, prob=0.3)
                    print(
                        f"[Gen {gen+1}] stagnation={stagnation_counter}, "
                        f"group_shift moved {moved_count}/{len(offspring_lambda)} offspring"
                    )

                fitness_lambda = []
                assets = {}
                solver = {}
                loss_group_all = loss_group.copy()
                for i_, ind in enumerate(offspring_lambda):
                    i = i_ + mu
                    assets[i] = decode_individual(ind, ea_assets)
                    solver[i] = copy.deepcopy(init_solver)
                    if use_weight:
                        solver[i].update_constraints(G_ea, assets[i], ea_assets, id_to_key)
                    else:
                        solver[i].update_constraints_woweight(G_ea, assets[i], id_to_key)
                    loss_g, loss = fitness(
                        assets[i], solver[i], groups,
                        weight_semantic=1.0, weight_physics=1.0,
                    )
                    loss_group_all[i] = loss_g
                    fitness_lambda.append(loss.item())
                fitness_lambda = np.array(fitness_lambda)

                population_all = population_mu + offspring_lambda
                fitness_all = np.concatenate([fitness_mu, fitness_lambda])

                elite_idx = np.argsort(fitness_all)[:elite_keep_fix]
                if fitness_all[elite_idx[0]] < 0.01:
                    ind_best = population_all[elite_idx[0]]
                    best_assets = decode_individual(ind_best, ea_assets)
                    recaculate_bbox(best_assets)
                    print(f"Early stopping as best fitness is {fitness_all[elite_idx[0]]}")
                    return best_assets

                remaining_idx = np.setdiff1d(np.arange(len(fitness_all)), elite_idx)
                candidate_size = min(len(remaining_idx), mu * 3)
                candidate_idx = remaining_idx[np.argsort(fitness_all[remaining_idx])[:candidate_size]]
                candidate_pop = np.array([population_all[i] for i in candidate_idx])
                candidate_fit = fitness_all[candidate_idx]

                selected_idx = list(elite_idx)
                selected_vectors = np.array([population_all[i] for i in selected_idx])
                num_needed = elite_keep - len(selected_idx)
                diversity_weight = 0.2
                for _ in range(num_needed):
                    if len(candidate_pop) == 0:
                        break
                    if len(selected_vectors) == 0:
                        dist_mean = np.ones(len(candidate_pop)) * 1e3
                    else:
                        candidate_pop_norm = normalize_pose_vectors(candidate_pop)
                        selected_vectors_norm = normalize_pose_vectors(selected_vectors)
                        dist = cdist(candidate_pop_norm, selected_vectors_norm)
                        dist_mean = dist.mean(axis=1)
                    scores = (1 - diversity_weight) * (candidate_fit / (candidate_fit.min() + 1e-6)) \
                        + diversity_weight * (1.0 / (dist_mean + 1e-6))
                    pick_idx = np.argmin(scores)
                    selected_vectors = np.vstack([selected_vectors, candidate_pop[pick_idx]])
                    selected_idx.append(candidate_idx[pick_idx])
                    candidate_pop = np.delete(candidate_pop, pick_idx, axis=0)
                    candidate_fit = np.delete(candidate_fit, pick_idx, axis=0)
                    candidate_idx = np.delete(candidate_idx, pick_idx)

                population_mu = [population_all[i] for i in selected_idx]
                fitness_mu = fitness_all[selected_idx]
                loss_group = {i: loss_group_all[j] for i, j in enumerate(selected_idx)}

                sigma_pos *= 0.97
                sigma_rot *= 0.95
                group_mut_prob *= 0.95
                cg_weight *= 0.98
                repel_alpha *= 0.98

            if (not ablation_wo_gd) and if_gd and (not if_center) and do_gd and gd_step_n > 0:
                new_elites = []
                for idx, ind_mu in enumerate(population_mu):
                    assets_mu[idx] = decode_individual(ind_mu, ea_assets)
                    solver_mu[idx] = copy.deepcopy(init_solver)
                    if use_weight:
                        solver_mu[idx].update_constraints(G_ea, assets_mu[idx], ea_assets, id_to_key)
                    else:
                        solver_mu[idx].update_constraints_woweight(G_ea, assets_mu[idx], id_to_key)
                    loss_orig = fitness_mu[idx]
                    optimized_assets, _, loss_opt = gradient_descent(
                        gd_step_n,
                        assets_mu[idx],
                        solver_mu[idx],
                        idx, weight=0.3, vis=False,
                    )
                    opt_solver = copy.deepcopy(init_solver)
                    if use_weight:
                        opt_solver.update_constraints(G_ea, optimized_assets, ea_assets, id_to_key)
                    else:
                        opt_solver.update_constraints_woweight(G_ea, optimized_assets, id_to_key)
                    loss_g_opt, loss_opt_ea = fitness(
                        optimized_assets, opt_solver, groups,
                        weight_semantic=1.0, weight_physics=1.0,
                    )
                    if loss_opt_ea.item() <= loss_orig.item():
                        fitness_mu[idx] = loss_opt_ea.item()
                        loss_group[idx] = loss_g_opt
                        assets_mu[idx] = optimized_assets
                        new_elites.append(encode_individual(optimized_assets))
                    else:
                        new_elites.append(ind_mu)
                population_mu = new_elites

            print(f"[Gen {gen+1}] best = {fitness_mu.min():.4f}, mean = {fitness_mu.mean():.4f}")
        best_idx = np.argsort(fitness_mu)[0]
        ind_best = population_mu[best_idx]
        best_assets = decode_individual(ind_best, ea_assets)
        # new_assets = recaculate_bbox_w_remove(best_assets, region_bound)
        print("best_score is:", fitness_mu[best_idx])
        return best_assets
    
    if graph:
        init_assets, init_solver, _ = load_assets_and_constraints_s(code, None, obj_descriptions, region_idx=region_i, region=(0.0, shelf_length, 0.0, shelf_width))
        seed = 1234
        region_bounds = [0.0, shelf_length, 0.0, shelf_width]
        obj_list = {str(region_i): obj_descriptions}
        refine_solver = gpt_api is not None and output_dir is not None and label is not None
        if refine_solver:
            G, groups, centrality = update_solver_s(
                init_assets,
                init_solver,
                region_bound,
                use_weight,
                gpt_api=gpt_api,
                current_code=code,
                areas=area_info,
                output_dir=output_dir,
                label=label,
                wall=None,
                task=None,
                existing_assets=None,
                region_idx=region_i,
                use_small_reflection=True,
                region_bounds=region_bounds,
                obj_list=obj_list,
                open_surface_area_check=True,
            )
        else:
            G, groups, centrality = update_solver_s(init_assets, init_solver, region_bound, use_weight)
        if not init_assets:
            return {}
        G_centered, center_nodes, groups, nodes_to_remove = build_group_center_graph(G, groups, centrality, obj_descriptions)
        refer_center = {j: center_nodes[i] for i, group in enumerate(groups) for j in group}
        id_to_key = {v['id']: k for k, v in init_assets.items()}
        """
        Stage 1:
        """
        best_assets = run_ea(
            G, init_assets, {}, refer_center, center_nodes, groups, id_to_key, seed, None, False,
            use_ea=use_ea,
        )

    # best_full_assets = expand_stacks(best_assets)
    return best_assets
def run_ea_for_region_open(region, assets, obj_descriptions, idx, gpt_api, code, ea_func, output_dir=None, label=None, area_info=None):

    # 预处理支持面
    preprocess_shelf_region(region)
    shelf_polygon = region["local_polygon"]
    shelf_length = shelf_polygon.bounds[2] - shelf_polygon.bounds[0]
    shelf_width = shelf_polygon.bounds[3] - shelf_polygon.bounds[1]
    shelf_z = region.get("support_height", 0.0)

    try:
        layouts = ea_func(
            deepcopy(assets),
            obj_descriptions,
            idx,
            shelf_length=shelf_length,
            shelf_width=shelf_width,
            shelf_z=shelf_z,
            clearance=region["clearance"],
            target_utilization=region["utilization"],
            gpt_api=gpt_api,
            code=code,
            output_dir=output_dir,
            label=label,
            area_info=area_info,
            generations=51,
            pop_size=50,
            top_k=5,
            optimize_every_n=300,
            use_ea=True,
        )
    except Exception as exc:
        print(
            f"[run_ea_for_region_open] constraint/EA open-surface path failed (idx={idx!r}): {exc}; "
            "falling back to closed-shelf EA."
        )
        return run_ea_for_region_close(region, assets, idx, ea_fill_shelf)
    print(layouts)
    return postprocess_layout(layouts, region)

def run_ea_for_region_close(region, assets, idx, ea_func):
    # 预处理支持面
    preprocess_shelf_region(region)
    shelf_polygon = region["local_polygon"]
    shelf_length = shelf_polygon.bounds[2] - shelf_polygon.bounds[0]
    shelf_width = shelf_polygon.bounds[3] - shelf_polygon.bounds[1]
    shelf_z = region.get("support_height", 0.0)

    # 调用 EA 放置
    layouts = ea_func(
        deepcopy(assets),
        idx,
        shelf_length=shelf_length,
        shelf_width=shelf_width,
        shelf_z=shelf_z,
        clearance=region["clearance"],
        target_utilization=region["utilization"],
        generations=30,
        pop_size=50,
    )
    return postprocess_layout(layouts, region)
    
def _constraint_entry_matches_region(c_id, region_index: int) -> bool:
    """GPT JSON may use int or str ids; ``enumerate`` gives int region index."""

    try:
        return int(c_id) == int(region_index)
    except (TypeError, ValueError):
        return str(c_id) == str(region_index)


def run_all_regions_ea(regions_json, assets, obj_descriptions_all, gpt_api, code, open_region, output_dir=None, label=None, area_info=None):
    all_layout = {}
    open_region_idx = []
    for idx, region in enumerate(regions_json):
        str_idx = str(idx)
        if code and code.get('constraints'):
            code_region = next(
                (c["code"] for c in code["constraints"] if _constraint_entry_matches_region(c.get("id"), idx)),
                None,
            )
            if code_region and obj_descriptions_all.get(str_idx):
                code_region = code_region.replace("\\n", "\n")
                layout = run_ea_for_region_open(
                    region,
                    assets[str_idx],
                    obj_descriptions_all[str_idx],
                    str_idx,
                    gpt_api,
                    code_region,
                    ea_fill_open_region,
                    output_dir=output_dir,
                    label=label,
                    area_info=area_info,
                )
                open_region_idx.append(str_idx)
            else:
                layout = run_ea_for_region_close(region, assets[str_idx], str_idx, ea_fill_shelf)
        else:
            layout = run_ea_for_region_close(region, assets[str_idx], str_idx, ea_fill_shelf)
        print(layout)
        all_layout[str_idx] = layout
        
    print(all_layout)
    return all_layout, open_region_idx
