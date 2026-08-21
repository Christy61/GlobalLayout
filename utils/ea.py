import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Any, Union, Optional
import numpy as np
import random
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
from copy import deepcopy
import matplotlib.cm as cm
import torch
import re
import networkx as nx
import imageio.v2 as imageio
from utils.geometry import compute_rotated_corners_np
from utils.optimization import recaculate_bbox, recaculate_bbox_w_remove, ensure_grad, \
    optimize_pose, cal_initial_loss_group, cal_initial_loss, build_parent_bounds, build_parent_z_locks, build_constraint_z_locks, _snapshot_asset, \
    build_ea_constraint_graph, prepare_solver_for_eval
from utils.scene_graph import SceneGraph
from utils.node import Wall, Door, Object, ObjectSet, FixedObject
from utils.optimization import Scene, ConstraintSolver
# from utils.draw import plot_floorplan_with_doors_windows
import copy
from deap import base, creator, tools
from scipy.spatial.distance import cdist


def normalize_pose_vectors(vectors):
    """Normalize rotation part so that distance is comparable."""
    normed = vectors.copy()
    for i in range(3, vectors.shape[1], 4):  # 每 4 维的第 4 个是角度
        normed[:, i] = normed[:, i] / 180.0
    return normed


def _oriented_half_extents_object(asset, theta_deg):
    w, l = float(asset.bbox[0]), float(asset.bbox[1])
    theta_norm = theta_deg % 180
    nearest = round(float(theta_norm) / 90) * 90
    if int(nearest) % 180 == 0:
        return w / 2.0, l / 2.0
    return l / 2.0, w / 2.0


def sample_new_group_center(group_bbox, group_center, rb):
    bbox_width = group_bbox[0]
    bbox_height = group_bbox[1]
    xmin, xmax, ymin, ymax = rb
    min_x = xmin + bbox_width / 2
    max_x = xmax - bbox_width / 2
    min_y = ymin + bbox_height / 2
    max_y = ymax - bbox_height / 2
    if min_x >= max_x or min_y >= max_y:
        return group_center
    offset = np.random.normal(0.0, 1.0, size=(2))
    new_x = group_center[0] + offset[0]
    new_y = group_center[1] + offset[1]
    if new_x < min_x:
        new_x = min_x + abs(offset[0])
    elif new_x > max_x:
        new_x = max_x - abs(offset[0])
    if new_y < min_y:
        new_y = min_y + abs(offset[1])
    elif new_y > max_y:
        new_y = max_y - abs(offset[1])
    new_x = np.clip(new_x, min_x, max_x)
    new_y = np.clip(new_y, min_y, max_y)
    return [new_x, new_y]


def apply_translation_objects(existing_assets, translation_vector):
    shifted_existing_assets = {}
    for key, asset in existing_assets.items():
        asset = copy.deepcopy(asset)
        pos = asset.pos
        x = float(pos[0].item() if torch.is_tensor(pos[0]) else pos[0])
        y = float(pos[1].item() if torch.is_tensor(pos[1]) else pos[1])
        z = float(pos[2].item() if torch.is_tensor(pos[2]) else pos[2])
        new_x = x + translation_vector[0]
        new_y = y + translation_vector[1]
        with torch.no_grad():
            asset.pos = torch.tensor(
                [new_x, new_y, z],
                dtype=pos.dtype if torch.is_tensor(pos) else torch.float32,
                device=pos.device if torch.is_tensor(pos) else None,
            )
        shifted_existing_assets[key] = asset
    return shifted_existing_assets


def calculate_group_bbox_and_center_objects(existing_assets):
    all_min_x, all_min_y = float("inf"), float("inf")
    all_max_x, all_max_y = float("-inf"), float("-inf")
    for asset in existing_assets.values():
        pos = asset.pos
        x = float(pos[0].item() if torch.is_tensor(pos[0]) else pos[0])
        y = float(pos[1].item() if torch.is_tensor(pos[1]) else pos[1])
        rot = asset.rot
        r = rot[0] * 180 / torch.pi
        r = float(r.item() if torch.is_tensor(r) else r)
        half_x, half_y = _oriented_half_extents_object(asset, r)
        all_min_x = min(all_min_x, x - half_x)
        all_min_y = min(all_min_y, y - half_y)
        all_max_x = max(all_max_x, x + half_x)
        all_max_y = max(all_max_y, y + half_y)
    group_center = [(all_min_x + all_max_x) / 2, (all_min_y + all_max_y) / 2]
    group_bbox = [all_max_x - all_min_x, all_max_y - all_min_y]
    return group_bbox, group_center


def compute_group_xy_bounds_objects(assets_group):
    g_min_x, g_min_y = float("inf"), float("inf")
    g_max_x, g_max_y = float("-inf"), float("-inf")
    for asset in assets_group.values():
        pos = asset.pos
        x = float(pos[0].item() if torch.is_tensor(pos[0]) else pos[0])
        y = float(pos[1].item() if torch.is_tensor(pos[1]) else pos[1])
        rot = asset.rot
        theta = rot[0].item() if torch.is_tensor(rot[0]) else float(rot[0])
        bx = float(asset.bbox[0].item() if torch.is_tensor(asset.bbox[0]) else asset.bbox[0])
        by = float(asset.bbox[1].item() if torch.is_tensor(asset.bbox[1]) else asset.bbox[1])
        corners = compute_rotated_corners_np(np.array([theta]), np.array([bx, by]))
        corners = corners + np.array([x, y], dtype=np.float64)
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
        g_min_x = min(g_min_x, float(min_xy[0]))
        g_min_y = min(g_min_y, float(min_xy[1]))
        g_max_x = max(g_max_x, float(max_xy[0]))
        g_max_y = max(g_max_y, float(max_xy[1]))
    return g_min_x, g_max_x, g_min_y, g_max_y


def shift_init_objects(existing_assets, rb):
    if not existing_assets:
        return {}
    group_bbox, group_center = calculate_group_bbox_and_center_objects(existing_assets)
    new_center = sample_new_group_center(group_bbox, group_center, rb)
    translation_vector = np.array(new_center) - np.array(group_center)
    return apply_translation_objects(existing_assets, translation_vector)


def shift_existing_group_safe_objects(existing_assets, room_bound):
    if not existing_assets:
        return existing_assets
    candidate_shifted = shift_init_objects(existing_assets, room_bound)
    if not candidate_shifted:
        return existing_assets
    anchor_key = next(iter(existing_assets.keys()))
    p0 = existing_assets[anchor_key].pos
    p1 = candidate_shifted[anchor_key].pos
    x0 = float(p0[0].item() if torch.is_tensor(p0[0]) else p0[0])
    y0 = float(p0[1].item() if torch.is_tensor(p0[1]) else p0[1])
    x1 = float(p1[0].item() if torch.is_tensor(p1[0]) else p1[0])
    y1 = float(p1[1].item() if torch.is_tensor(p1[1]) else p1[1])
    dx, dy = x1 - x0, y1 - y0
    xmin, xmax, ymin, ymax = room_bound
    g_min_x, g_max_x, g_min_y, g_max_y = compute_group_xy_bounds_objects(existing_assets)
    dx_min = xmin - g_min_x
    dx_max = xmax - g_max_x
    dy_min = ymin - g_min_y
    dy_max = ymax - g_max_y
    dx = float(np.clip(dx, dx_min, dx_max))
    dy = float(np.clip(dy, dy_min, dy_max))
    return apply_translation_objects(existing_assets, np.array([dx, dy], dtype=np.float64))


def extract_existing_from_ind_objects(ind, chromosome_assets, existing_slots, slot_to_existing_key):
    extracted = {}
    for slot in existing_slots:
        key = slot_to_existing_key[slot]
        base_asset = chromosome_assets[key]
        x = float(ind[4 * slot])
        y = float(ind[4 * slot + 1])
        z = float(ind[4 * slot + 2])
        theta_deg = float(ind[4 * slot + 3])
        asset = _snapshot_asset(base_asset)
        asset.pos = torch.tensor([x, y, z], dtype=torch.float32)
        asset.rot = torch.tensor([theta_deg * np.pi / 180.0], dtype=torch.float32)
        extracted[key] = asset
    return extracted


def write_existing_to_ind_objects(ind, shifted_existing, existing_slots, slot_to_existing_key):
    for slot in existing_slots:
        key = slot_to_existing_key[slot]
        shifted_asset = shifted_existing.get(key)
        if shifted_asset is None:
            continue
        pos = shifted_asset.pos
        ind[4 * slot] = float(pos[0].item() if torch.is_tensor(pos[0]) else pos[0])
        ind[4 * slot + 1] = float(pos[1].item() if torch.is_tensor(pos[1]) else pos[1])
    return ind


def apply_existing_group_shift_to_offspring_objects(
    offspring_lambda,
    chromosome_assets,
    existing_slots,
    slot_to_existing_key,
    room_bound,
    seed,
    gen,
    prob=0.12,
):
    if len(offspring_lambda) == 0 or len(existing_slots) == 0:
        return offspring_lambda, 0
    rng = np.random.default_rng(seed * 1000 + gen)
    trigger_mask = rng.random(len(offspring_lambda)) < prob
    moved_count = int(trigger_mask.sum())
    if moved_count == 0:
        return offspring_lambda, 0
    for i_off, do_shift in enumerate(trigger_mask):
        if not do_shift:
            continue
        ind = offspring_lambda[i_off]
        existing_from_ind = extract_existing_from_ind_objects(
            ind, chromosome_assets, existing_slots, slot_to_existing_key
        )
        shifted_existing = shift_existing_group_safe_objects(existing_from_ind, room_bound)
        offspring_lambda[i_off] = write_existing_to_ind_objects(
            ind, shifted_existing, existing_slots, slot_to_existing_key
        )
    return offspring_lambda, moved_count


def _asset_to_draw(asset):
    if asset is None:
        return None
    if isinstance(asset, dict):
        pos = asset.get("pos")
        rot = asset.get("rot", asset.get("phy"))
        bbox = asset.get("bbox")
    else:
        pos = getattr(asset, "pos", None)
        rot = getattr(asset, "rot", None)
        bbox = getattr(asset, "bbox", None)
    if pos is None or bbox is None:
        return None
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    pos = np.array(pos, dtype=float).reshape(-1)
    if pos.shape[0] < 2:
        return None
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().numpy()
    bbox = np.array(bbox, dtype=float).reshape(-1)
    if bbox.shape[0] < 2:
        return None
    if rot is None:
        theta = 0.0
    else:
        if isinstance(rot, torch.Tensor):
            rot = rot.detach().cpu().numpy()
        rot = np.array(rot, dtype=float).reshape(-1)
        theta = float(rot[0]) if rot.size else 0.0
    w, h = float(bbox[0]), float(bbox[1])
    corners = compute_rotated_corners_np(np.array([theta]), np.array([w, h]))
    return {"pos": pos, "bbox": np.array([w, h]), "corners": corners}


def _build_draw_assets_map(assets):
    if not assets:
        return {}
    out = {}
    for key, asset in assets.items():
        draw_asset = _asset_to_draw(asset)
        if draw_asset is not None:
            out[key] = draw_asset
    return out


def _clone_assets_for_decode(assets):
    cloned = {}
    for key, asset in assets.items():
        if isinstance(asset, dict):
            cloned[key] = copy.deepcopy(asset)
        else:
            cloned[key] = _snapshot_asset(asset)
    return cloned


def render_2d_frame(
    floor_xy,
    door_location,
    window_locations,
    moving_assets,
    fixed_assets,
    out_path,
    title=None,
    include_fixed=True,
):
    from utils.draw import draw_door, draw_window, draw_grid, draw_assets, polygon_is_clockwise
    import matplotlib.patches as patches

    if not floor_xy:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    xs = [p[0] for p in floor_xy]
    ys = [p[1] for p in floor_xy]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 0.2

    fig, ax = plt.subplots(figsize=(6, 6))
    poly = patches.Polygon(floor_xy, closed=True, facecolor=(0.7, 0.85, 1.0, 0.15),
                           edgecolor="red", linewidth=1.5, zorder=2)
    ax.add_patch(poly)

    cw = polygon_is_clockwise(floor_xy)
    n = len(floor_xy)
    if door_location:
        wid = int(door_location.get("wall_id", 0))
        p1 = floor_xy[wid]
        p2 = floor_xy[(wid + 1) % n]
        draw_door(
            ax,
            p1,
            p2,
            door_location.get("center"),
            door_location.get("hinge", "right"),
            "inward",
            clockwise_poly=cw,
            zorder=5,
        )
    if window_locations:
        for win in window_locations:
            wid = int(win.get("wall_id", 0))
            p1 = floor_xy[wid]
            p2 = floor_xy[(wid + 1) % n]
            draw_window(ax, floor_xy, win.get("center"), zorder=5)

    draw_grid(ax, xmin, xmax, ymin, ymax)

    if include_fixed and fixed_assets:
        fixed_map = _build_draw_assets_map(fixed_assets)
        if fixed_map:
            draw_assets(ax, fixed_map, zorder=4)
    moving_map = _build_draw_assets_map(moving_assets)
    if moving_map:
        draw_assets(ax, moving_map, zorder=6)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

def freeze_assets(assets):
    for asset in assets.values():
        if isinstance(asset, dict):
            for k in ("pos", "phy", "rot"):
                if k in asset and isinstance(asset[k], torch.Tensor):
                    asset[k] = asset[k].detach()
                    asset[k].requires_grad_(False)
            continue
        if hasattr(asset, "pos") and isinstance(asset.pos, torch.Tensor):
            asset.pos = asset.pos.detach()
            asset.pos.requires_grad_(False)
        if hasattr(asset, "rot") and isinstance(asset.rot, torch.Tensor):
            asset.rot = asset.rot.detach()
            asset.rot.requires_grad_(False)
    return assets

def build_fixed_anchors(fixed_assets: dict[str, Union[Object, ObjectSet]]):
    anchors = {}
    for key, asset in fixed_assets.items():
        if getattr(asset, "pos", None) is None or getattr(asset, "rot", None) is None:
            continue
        pos = asset.pos.detach().clone() if isinstance(asset.pos, torch.Tensor) else torch.tensor(asset.pos, dtype=torch.float32)
        rot = asset.rot.detach().clone() if isinstance(asset.rot, torch.Tensor) else torch.tensor(asset.rot, dtype=torch.float32)
        anchor = FixedObject(key, pos, rot)
        if getattr(asset, "bbox", None) is not None:
            bbox = asset.bbox
            if isinstance(bbox, torch.Tensor):
                anchor.bbox = bbox.detach().clone()
            else:
                anchor.bbox = torch.tensor(bbox, dtype=torch.float32)
        anchors[key] = anchor
    return anchors


def merge_existing_assets(
    ea_assets: dict[str, Union[Object, ObjectSet]],
    prior_assets: Optional[dict[str, Union[Object, ObjectSet]]] = None,
    *,
    trainable_existing: bool = True,
):
    """Merge fixed / previously placed assets into the evaluation context (ablation ``update``).

    Existing assets participate in loss as fixed context by default.
    """
    if not prior_assets:
        return ea_assets
    merged = copy.deepcopy(ea_assets)
    for key, asset in prior_assets.items():
        merged[key] = copy.deepcopy(asset)
    existing_subset = {k: merged[k] for k in prior_assets if k in merged}
    if trainable_existing:
        ensure_grad(existing_subset)
    return merged


def write_back_existing_assets(
    existing_assets: dict[str, Union[Object, ObjectSet]],
    optimized_assets: dict[str, Union[Object, ObjectSet]],
    existing_keys: Optional[set[str]] = None,
):
    """Persist GD-optimized poses for existing assets (chromosome stays new-only)."""
    keys = existing_keys if existing_keys is not None else set(existing_assets.keys())
    for key in keys:
        if key in optimized_assets:
            existing_assets[key] = _snapshot_asset(optimized_assets[key])
    if existing_assets:
        freeze_assets({k: existing_assets[k] for k in keys if k in existing_assets})


def adaptive_generations(num_new_assets: int) -> int:
    """Scale EA rounds by the number of new assets in this EA run."""
    return 31 if num_new_assets < 10 else 51


def ccea_gd_layout_optimization(
    door: Door,
    wall: List[Wall],
    scene: Scene,
    scene_graph: SceneGraph,
    glb_paths: Dict[str, str],
    grid_size: float,
    generations: int,
    gd_step: int,
    pop_size: int,
    mu: int,
    graph: bool,
    use_weight: bool,
    if_gd: bool,
    use_group: bool,
    output_dir: Optional[str] = None,
    existing_assets: Optional[dict[str, Union[Object, ObjectSet]]] = None,
    vis_cfg: Optional[dict[str, Any]] = None,
):

    def snap_to_grid(v):
        return round(round(float(v / grid_size)) * grid_size, 6)

    def clamp_xy_to_room(x, y, hx, hy, rb):
        xmin, xmax, ymin, ymax = rb
        min_x = xmin + hx
        max_x = xmax - hx
        min_y = ymin + hy
        max_y = ymax - hy
        if min_x > max_x:
            x = (xmin+xmax)/2.0
        else:
            x = min(max(min_x, x), max_x)
        if min_y > max_y:
            y = (ymin+ymax)/2.0
        else:
            y = min(max(min_y, y), max_y)
        return snap_to_grid(x), snap_to_grid(y)

    def oriented_half_extents(asset, theta_deg):
        w, l = float(asset.bbox[0]), float(asset.bbox[1])
        theta_norm = theta_deg % 180
        nearest = round(float(theta_norm)/90)*90
        if int(nearest) % 180 == 0:
            return w/2.0, l/2.0
        else:
            return l/2.0, w/2.0
        
    def generate_individual(assets, room_bound, seed, z_locks=None, existing_assets=None):
        np.random.seed(seed)
        rotation_choices = [i * 30 for i in range(12)]
        individual = []
        xmin, xmax, ymin, ymax = room_bound
        xs = np.arange(xmin, xmax + 1e-6, grid_size)
        ys = np.arange(ymin, ymax + 1e-6, grid_size)
        room_grid_points = [(x, y) for x in xs for y in ys]
        existing_assets = existing_assets or {}

        def _pos_rot_deg(asset):
            pos = asset.pos
            rot = asset.rot
            x = float(pos[0].item() if torch.is_tensor(pos[0]) else pos[0])
            y = float(pos[1].item() if torch.is_tensor(pos[1]) else pos[1])
            z = float(pos[2].item() if torch.is_tensor(pos[2]) else pos[2])
            r = rot[0] * 180 / torch.pi
            theta_deg = float(r.item() if torch.is_tensor(r) else r)
            return x, y, z, theta_deg

        for key, asset in assets.items():
            if key in existing_assets:
                x, y, z, theta_deg = _pos_rot_deg(existing_assets[key])
                individual.extend([x, y, z, theta_deg])
                continue
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
                    x, y = clamp_xy_to_room(cx, cy, hx, hy, room_bound)
            else:
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                x, y = clamp_xy_to_room(cx, cy, hx, hy, room_bound)
            z = 0.0
            if z_locks and key in z_locks:
                z = z_locks[key]
            individual.extend([x, y, z, theta_deg])
        return individual

    def decode_individual(individual, assets):
        i = 0
        for key, asset in assets.items():
            x = individual[4*i]
            y = individual[4*i + 1]
            z = individual[4*i + 2]
            asset.pos = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True)
            asset.rot = torch.tensor([individual[4*i + 3]*torch.pi/180], dtype=torch.float32, requires_grad=True)
            i += 1
        return assets

    def encode_individual(asset_dict, asset_keys=None):
        keys = asset_keys if asset_keys is not None else list(asset_dict.keys())
        individual = []
        for key in keys:
            asset = asset_dict[key]
            pos = asset.pos
            rot = asset.rot
            x = pos[0].item() if torch.is_tensor(pos[0]) else float(pos[0])
            y = pos[1].item() if torch.is_tensor(pos[1]) else float(pos[1])
            z = pos[2].item() if torch.is_tensor(pos[2]) else float(pos[2])
            r = rot[0] * 180 / torch.pi
            r = r.item() if torch.is_tensor(r) else float(r)
            individual.extend([x, y, z, r])
        return individual

    def apply_repulsion(offspring, groups_all=None, alpha=0.005):
        """
        群体排斥力避免塌缩（只对 groups_all 指定的 asset 生效）

        Args:
            offspring: (λ, D)，种群个体矩阵
            groups_all: list[int] or np.ndarray，需要计算排斥的 asset 索引
            alpha: float，排斥强度
        """
        xs = offspring[:, 0::4]
        ys = offspring[:, 1::4]
        pos = np.stack([xs, ys], axis=-1)  # shape: (λ, n_assets, 2)

        if groups_all is not None:
            groups_all = np.array(groups_all, dtype=int)
            pos_masked = pos[:, groups_all, :]
        else:
            pos_masked = pos

        diffs = pos_masked[:, None, :, :] - pos_masked[None, :, :, :]
        dist2 = np.sum(diffs ** 2, axis=-1, keepdims=True) + 1e-6
        repel = np.sum(diffs / dist2, axis=1)

        pos[:, groups_all, :] = pos[:, groups_all, :] + alpha * repel

        offspring[:, 0::4] = pos[:, :, 0]
        offspring[:, 1::4] = pos[:, :, 1]

        return offspring

    def recombine_and_mutate(
        parents,
        parent_fitness,
        mask_all,
        w_l,
        n_offspring,
        seed,
        sigma_pos=(0.3, 0.3),
        sigma_rot=5.0,
        repel_alpha=0.05,
        swap_probs=0.3,
        group_mut_prob=0.10,
        mask_group=None,
        groups_all=None,
        idx_swap=None,
        center_map=None,
        cg_weight=0.20,
        d_min=0.5,
    ):
        np.random.seed(seed)
        parents = np.asarray(parents)
        n_assets = w_l.shape[0]
        D = parents.shape[1]

        if idx_swap is None:
            idx_swap = list(range(n_assets))

        fitness_arr = np.asarray(parent_fitness)
        if fitness_arr.std() < 1e-4:
            probs = np.ones_like(fitness_arr) / len(fitness_arr)
        else:
            beta = 1.0 / fitness_arr.std()
            probs = np.exp(-beta * (fitness_arr - fitness_arr.min()))
            probs /= probs.sum()

        parent_indices = np.random.choice(len(parents), size=n_offspring, p=probs)
        offspring = parents[parent_indices].copy()

        if len(idx_swap) >= 2:
            for i in range(n_offspring):
                if np.random.rand() < swap_probs:
                    i1, i2 = np.random.choice(len(idx_swap), size=2, replace=False)
                    idx1, idx2 = idx_swap[i1], idx_swap[i2]
                    temp = offspring[i, idx1 * 4:idx1 * 4 + 4].copy()
                    offspring[i, idx1 * 4:idx1 * 4 + 4] = offspring[i, idx2 * 4:idx2 * 4 + 4]
                    offspring[i, idx2 * 4:idx2 * 4 + 4] = temp

        noise_x = np.random.normal(0.0, sigma_pos[0], size=(n_offspring, n_assets))
        noise_y = np.random.normal(0.0, sigma_pos[1], size=(n_offspring, n_assets))
        noise_rot = np.random.normal(0.0, sigma_rot, size=(n_offspring, n_assets))
        delta_xs = np.where(mask_all[0::4], noise_x, 0.0)
        delta_ys = np.where(mask_all[1::4], noise_y, 0.0)
        offspring[:, 3::4] += np.where(mask_all[3::4], noise_rot, 0.0)

        if mask_group is not None and group_mut_prob > 0.0:
            trigger = np.random.rand(n_offspring) < group_mut_prob
            if np.any(trigger):
                delta = np.random.uniform(
                    low=[-w_l[:, 0].max() * 0.2, -w_l[:, 1].max() * 0.2],
                    high=[w_l[:, 0].max() * 0.2, w_l[:, 1].max() * 0.2],
                    size=(int(trigger.sum()), 2),
                )
                group_indices = np.where(mask_group.reshape(n_assets, 4)[:, 0])[0]
                delta_xs[trigger][:, group_indices] += delta[:, [0]]
                delta_ys[trigger][:, group_indices] += delta[:, [1]]

        offspring[:, 0::4] += delta_xs
        offspring[:, 1::4] += delta_ys

        xs = offspring[:, 0::4]
        ys = offspring[:, 1::4]
        pos = np.stack([xs, ys], axis=-1)
        if center_map is not None and len(center_map) > 0:
            members = np.array(list(center_map.keys()), dtype=int)
            centers = np.array([center_map[m] for m in members], dtype=int)
            pos_members = pos[:, members, :]
            pos_centers = pos[:, centers, :]
            bbox_half = w_l[centers] / 2.0
            radius_scale = np.random.uniform(0.8, 1.2, size=(n_offspring, len(members), 1))
            angles = np.random.uniform(0, 2 * np.pi, size=(n_offspring, len(members), 1))
            offset = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
            target_pos = pos_centers + offset * (bbox_half[None, :, :] * radius_scale + d_min)
            vec = target_pos - pos_members
            pos[:, members, :] += cg_weight * vec
            offspring[:, 0::4] = pos[:, :, 0]
            offspring[:, 1::4] = pos[:, :, 1]

        # Floor CCEA (ablation / Genesis): repulsion uses the same pool as swap.
        offspring = apply_repulsion(offspring, idx_swap, alpha=repel_alpha)

        thetas = offspring[:, 3::4]
        theta_norm = thetas % 180
        nearest = np.round(theta_norm / 90) * 90
        is_0_or_180 = (nearest % 180 == 0)
        half_extents = np.zeros((is_0_or_180.shape[0], w_l.shape[0], 2))
        half_extents[:, :, 0] = np.where(is_0_or_180, w_l[:, 0] / 2.0, w_l[:, 1] / 2.0)
        half_extents[:, :, 1] = np.where(is_0_or_180, w_l[:, 1] / 2.0, w_l[:, 0] / 2.0)
        offspring[:, 0::4] = np.clip(
            offspring[:, 0::4], xmin + half_extents[:, :, 0], xmax - half_extents[:, :, 0]
        )
        offspring[:, 1::4] = np.clip(
            offspring[:, 1::4], ymin + half_extents[:, :, 1], ymax - half_extents[:, :, 1]
        )
        offspring[:, 3::4] = thetas

        return offspring

    # -------- fitness & GD --------
    def fitness(assets, solver, groups=None, bounds_by_key=None, weight_semantic=1.0, weight_physics=1.0, weight_project=1.0):
        if groups:
            loss_g, total_loss = cal_initial_loss_group(
                assets, solver, groups, bound,
                other_assets=None, region_center=None, z_min=0.0,
                weight_semantic=weight_semantic, weight_physics=weight_physics, weight_project=weight_project,
                bounds_by_key=bounds_by_key,
            )
            return loss_g, total_loss
        total_loss = cal_initial_loss(
            assets, solver, bound,
            other_assets=None, region_center=None, z_min=0.0,
            weight_semantic=weight_semantic, weight_physics=weight_physics,
            bounds_by_key=bounds_by_key,
        )
        return None, total_loss
    
    def gradient_descent(iters, obj, solver, i, weight, vis=False, bounds_by_key=None, vis_cb=None, vis_every=None, z_locks=None):
        assets_pose, grad_dict, loss_opt, _ = optimize_pose(
            iters,
            f"full_{i}",
            obj,
            solver,
            bound,
            None,
            weight,
            z_min=0.0,
            glb_paths=glb_paths,
            vis=vis,
            door=door,
            bounds_by_key=bounds_by_key,
            vis_cb=vis_cb,
            vis_every=vis_every,
            z_locks=z_locks,
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
    

    def run_ea(
        ea_assets: dict[str, Union[Object, ObjectSet]],
        solver: ConstraintSolver,
        existing_assets: Optional[dict[str, Union[Object, ObjectSet]]] = None,
        context_assets: Optional[dict[str, Union[Object, ObjectSet]]] = None,
        bounds_by_key: Optional[dict[str, tuple[float, float, float, float]]] = None,
        z_locks: Optional[dict[str, float]] = None,
        if_ea: bool=True,
        seed: int=1234,
        groups_override: Optional[list[set[str]]] = None,
        center_nodes_override: Optional[dict[int, str]] = None,
        use_group: bool = True,
        use_weight: bool = True,
        vis_cfg: Optional[dict[str, Any]] = None,
        depth: Optional[int] = None,
        vis_tag: str = "main",
        generation_override: Optional[int] = None,
        do_final_gd: bool = True,
        gd_step_override: Optional[int] = None,
    ):
        # Chromosome encodes NEW assets only; existing (other regions / lower depth) are merged at eval.
        existing_assets = existing_assets or {}
        context_assets = context_assets or {}

        anchor_assets = {}
        if existing_assets:
            anchor_assets.update(build_fixed_anchors(existing_assets))
            anchor_assets.update(existing_assets)
        if context_assets:
            anchor_assets.update(build_fixed_anchors(context_assets))
            anchor_assets.update(context_assets)
        init_solver = copy.deepcopy(solver)
        init_solver.set_asset_fallback(None)
        ea_key_set = set(ea_assets.keys())
        G_ea = None
        if scene_graph is not None:
            G_ea, _ = build_ea_constraint_graph(
                scene_graph,
                ea_key_set,
                context_srcs=set(existing_assets.keys()) | set(context_assets.keys()),
                use_weight=use_weight,
            )
        elif not use_weight:
            init_solver.update_constraint_weights({}, use_weight=False)

        def solver_for_fitness(assets_eval: dict) -> ConstraintSolver:
            eval_solver = prepare_solver_for_eval(
                init_solver, G_ea, assets_eval, ea_key_set, use_weight=use_weight
            )
            eval_solver.set_asset_fallback(anchor_assets if anchor_assets else None)
            return eval_solver

        def merge_eval_assets(
            moving_assets: dict[str, Union[Object, ObjectSet]],
            prior_assets: Optional[dict[str, Union[Object, ObjectSet]]],
        ) -> dict[str, Union[Object, ObjectSet]]:
            merged = merge_existing_assets(
                moving_assets,
                prior_assets,
                trainable_existing=True,
            )
            if context_assets:
                merged = merge_existing_assets(
                    merged,
                    context_assets,
                    trainable_existing=False,
                )
                freeze_assets({k: merged[k] for k in context_assets if k in merged})
            return merged

        lam = pop_size - mu
        bound_m = np.array([xmax, ymax])
        asset_keys = list(ea_assets.keys())
        n_assets = len(asset_keys)
        sigma_pos = bound_m * 0.12
        sigma_rot = 5.0
        weight = 1.0
        gd_step_refine = (
            int(gd_step_override)
            if gd_step_override is not None
            else 200
        )
        gd_step_hybrid = 10
        group_mut_prob = 0.1
        cg_weight = 0.10
        key_to_idx = {k: i for i, k in enumerate(asset_keys)}
        w_l = np.zeros((n_assets, 2))
        for j, asset in enumerate(ea_assets.values()):
            w_l[j] = [float(asset.bbox[0]), float(asset.bbox[1])]
        groups_local = []
        vis_enabled = bool(vis_cfg and vis_cfg.get("enabled"))
        vis_frames = {}
        vis_root = None
        if use_group:
            source_groups = groups_override
            if source_groups is None and scene_graph is not None and getattr(scene_graph, "groups", None):
                source_groups = scene_graph.groups
            if source_groups:
                for group in source_groups:
                    if not group:
                        continue
                    inter = [k for k in group if k in key_to_idx]
                    if inter:
                        groups_local.append(set(inter))
        else:
            if asset_keys:
                groups_local = [set(asset_keys)]
        print(
            f"[EA][groups] n_assets={n_assets}, n_groups={len(groups_local)}, "
            f"groups={[sorted(group) for group in groups_local]}"
        )
        centrality_debug = {}
        if scene_graph is not None:
            centrality_debug = scene_graph.compute_centrality(active_srcs=set(asset_keys))
        centers_debug = []
        for group_idx, group in enumerate(groups_local):
            center_key = center_nodes_override.get(group_idx) if center_nodes_override else None
            centers_debug.append({
                "group_idx": group_idx,
                "center": center_key,
                "centrality": centrality_debug.get(center_key) if center_key is not None else None,
                "members": sorted(group),
            })
        print(f"[EA][centers] {centers_debug}")
        center_map = None
        if center_nodes_override and groups_local:
            center_map = {}
            for group_idx, group in enumerate(groups_local):
                center_key = center_nodes_override.get(group_idx)
                if center_key is None or center_key not in key_to_idx:
                    continue
                center_idx = key_to_idx[center_key]
                for member_key in group:
                    if member_key == center_key:
                        continue
                    member_idx = key_to_idx.get(member_key)
                    if member_idx is not None:
                        center_map[member_idx] = center_idx
            if not center_map:
                center_map = None
        if vis_enabled:
            depth_tag = depth if depth is not None else 0
            vis_root = os.path.join(
                vis_cfg.get("output_dir", "."),
                "ccea",
                "opt_vis",
                vis_tag,
                f"depth_{depth_tag}",
            )
            vis_frames = {i: [] for i in range(mu)}

        def _emit_frame(parent_idx, assets, tag):
            if not vis_enabled or vis_root is None:
                return
            frame_dir = os.path.join(vis_root, f"parent_{parent_idx}")
            os.makedirs(frame_dir, exist_ok=True)
            out_path = os.path.join(frame_dir, f"{tag}.png")
            include_fixed = bool(vis_cfg.get("include_fixed", True)) if vis_cfg else True
            fixed_context_for_vis = dict(existing_assets)
            fixed_context_for_vis.update(context_assets)
            render_2d_frame(
                vis_cfg.get("floor_xy"),
                vis_cfg.get("door_location"),
                vis_cfg.get("window_locations"),
                assets,
                fixed_context_for_vis,
                out_path,
                title=None,
                include_fixed=include_fixed,
            )
            vis_frames[parent_idx].append(out_path)
        population_mu = [
            generate_individual(ea_assets, bound, seed=seed * i, z_locks=z_locks, existing_assets=None)
            for i in range(mu)
        ]
        fitness_mu = []
        assets_mu = {}
        loss_group = {}
        groups_all = list(range(n_assets))
        repel_alpha = 0.15 / (len(groups_all) + 1e-8)
        for i, ind in enumerate(population_mu):
            assets_mu[i] = decode_individual(ind, copy.deepcopy(ea_assets))
            if i < mu - 2:
                prior_for_eval = existing_assets
            else:
                prior_for_eval = shift_existing_group_safe_objects(
                    copy.deepcopy(existing_assets), bound
                )
            assets_eval = merge_eval_assets(assets_mu[i], prior_for_eval)
            eval_solver = solver_for_fitness(assets_eval)
            loss_g, loss = fitness(
                assets_eval, eval_solver,
                groups=groups_local if groups_local else None,
                bounds_by_key=bounds_by_key,
                weight_semantic=1.0, weight_physics=1.0,
            )
            if loss_g is not None:
                loss_group[i] = loss_g
            fitness_mu.append(loss.item())
        
        fitness_mu = np.array(fitness_mu)
        print(f"[Init] mu={mu}, rho={mu}, lambda={lam}, m={mu}")
        print(f"[Init] Evaluated {mu} parents, mean fitness = {float(fitness_mu.mean()):.4f}")
        if vis_enabled:
            ea_every = int(vis_cfg.get("ea_every", 1)) if vis_cfg else 1
            if 0 % ea_every == 0:
                for idx in range(mu):
                    _emit_frame(idx, assets_mu[idx], f"gen_{0:04d}")

        total_generations = (
            int(generation_override)
            if generation_override is not None
            else adaptive_generations(n_assets)
        )
        print(f"[EA] total_generations={total_generations} (new_assets={n_assets})")
        prev_mean_fitness = None
        stagnation_counter = 0
        stagnation_eps = 0.03
        stagnation_k = 4
        existing_by_mu = None
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
                do_gd = False
                gd_step_n = 0
            elif gen < total_generations - 1:
                swap_probs = 0.2 * (0.95 ** (gen - 3))
                sigma_pos_stage = [sigma_pos[0] * 1.5, sigma_pos[1] * 1.5]
                sigma_rot_stage = sigma_rot * 2
                elite_keep_fix = mu - 1
                elite_keep = mu
                do_gd = False
                gd_step_n = 0
            else:
                swap_probs = 0.0
                sigma_pos_stage = [0.0, 0.0]
                sigma_rot_stage = 0.0
                elite_keep_fix = mu - 2
                elite_keep = mu
                do_gd = if_gd and do_final_gd
                gd_step_n = gd_step_refine

            if if_ea and gen < total_generations - 1:
                swap_ids_run = []
                mask_group = np.zeros(len(population_mu[0]), dtype=bool)
                if groups_local:
                    m_idxs = [select_group_by_loss(loss_group[idx]) for idx in range(mu)]
                    group_probabilities = np.array(
                        [m_idxs.count(i) for i in range(len(groups_local))], dtype=float
                    )
                    group_probabilities = group_probabilities / group_probabilities.sum()
                    mutate_group_idx = int(np.random.choice(len(groups_local), p=group_probabilities))
                    for member_key in groups_local[mutate_group_idx]:
                        if member_key not in key_to_idx:
                            continue
                        idx = key_to_idx[member_key]
                        swap_ids_run.append(idx)
                        mask_group[idx * 4:idx * 4 + 2] = True
                        mask_group[idx * 4 + 3] = True

                mask_all = np.zeros(len(population_mu[0]), dtype=bool)
                for idx in range(n_assets):
                    mask_all[idx * 4:idx * 4 + 2] = True
                    mask_all[idx * 4 + 3] = True

                offspring_lambda = recombine_and_mutate(
                    np.array(population_mu),
                    fitness_mu,
                    mask_all,
                    w_l,
                    lam,
                    seed * 100 + gen * 10,
                    sigma_pos=sigma_pos_stage,
                    sigma_rot=sigma_rot_stage,
                    repel_alpha=repel_alpha,
                    swap_probs=swap_probs,
                    group_mut_prob=group_mut_prob,
                    mask_group=mask_group,
                    groups_all=groups_all,
                    idx_swap=swap_ids_run,
                    center_map=center_map,
                    cg_weight=cg_weight,
                ).tolist()

                shift_rng = np.random.default_rng(seed * 1000 + gen)
                shift_enabled = gen >= 5 and stagnation_counter >= stagnation_k
                moved_count = 0

                fitness_lambda = []
                loss_group_all = loss_group.copy() if groups_local else None
                for i_, ind in enumerate(offspring_lambda):
                    i = i_ + mu
                    if shift_enabled and existing_assets and shift_rng.random() < 0.12:
                        prior_for_eval = shift_existing_group_safe_objects(
                            copy.deepcopy(existing_assets), bound
                        )
                        moved_count += 1
                    else:
                        prior_for_eval = existing_assets
                    assets_i = decode_individual(ind, copy.deepcopy(ea_assets))
                    assets_eval = merge_eval_assets(assets_i, prior_for_eval)
                    eval_solver = solver_for_fitness(assets_eval)
                    loss_g, loss = fitness(
                        assets_eval, eval_solver,
                        groups=groups_local if groups_local else None,
                        bounds_by_key=bounds_by_key,
                        weight_semantic=1.0, weight_physics=1.0,
                    )
                    if groups_local:
                        loss_group_all[i] = loss_g
                    fitness_lambda.append(loss.item())
                if shift_enabled and moved_count:
                    print(
                        f"[Gen {gen+1}] stagnation={stagnation_counter}, "
                        f"existing_group_shift moved {moved_count}/{len(offspring_lambda)} offspring"
                    )
                fitness_lambda = np.array(fitness_lambda)

                population_all = population_mu + offspring_lambda
                fitness_all = np.concatenate([fitness_mu, fitness_lambda])

                elite_idx = np.argsort(fitness_all)[:elite_keep_fix]
                early_stop_thr = 0.01
                if fitness_all[elite_idx[0]] < early_stop_thr:
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
                if groups_local:
                    loss_group = {i: loss_group_all[j] for i, j in enumerate(selected_idx)}

            if do_gd and gd_step_n > 0:
                ea_keys = list(ea_assets.keys())
                existing_keys = set(existing_assets.keys())
                new_elites = []
                new_existing_by_mu = []
                for idx, ind_mu in enumerate(population_mu):
                    assets_mu[idx] = decode_individual(ind_mu, ea_assets)
                    candidate_existing = copy.deepcopy(existing_assets)
                    assets_eval = merge_eval_assets(assets_mu[idx], candidate_existing)
                    eval_solver = solver_for_fitness(assets_eval)
                    loss_orig = fitness_mu[idx]
                    vis_cb = None
                    vis_every = None
                    if vis_enabled:
                        vis_every = int(vis_cfg.get("gd_every", 20)) if vis_cfg else 20
                        def _gd_cb(step, assets, parent_idx=idx):
                            _emit_frame(parent_idx, assets_mu[parent_idx], f"gd_{step:05d}")
                        vis_cb = _gd_cb
                    optimized_assets, _, loss_opt = gradient_descent(
                        gd_step_n,
                        assets_eval,
                        eval_solver,
                        idx, weight, vis=False,
                        bounds_by_key=bounds_by_key,
                        vis_cb=vis_cb,
                        vis_every=vis_every,
                        z_locks=z_locks,
                    )
                    if float(loss_opt) <= float(loss_orig):
                        fitness_mu[idx] = loss_opt
                        assets_mu[idx] = {k: optimized_assets[k] for k in ea_keys if k in optimized_assets}
                        write_back_existing_assets(candidate_existing, optimized_assets, existing_keys)
                        new_elites.append(encode_individual(optimized_assets, ea_keys))
                        new_existing_by_mu.append(candidate_existing)
                    else:
                        new_elites.append(ind_mu)
                        new_existing_by_mu.append(candidate_existing)
                population_mu = new_elites
                existing_by_mu = new_existing_by_mu

            sigma_pos *= 0.97
            sigma_rot *= 0.95
            group_mut_prob *= 0.95
            cg_weight *= 0.98
            repel_alpha *= 0.98
            print(f"[Gen {gen+1}] best = {fitness_mu.min():.4f}, mean = {fitness_mu.mean():.4f}")
            if vis_enabled:
                ea_every = int(vis_cfg.get("ea_every", 1)) if vis_cfg else 1
                if (gen + 1) % ea_every == 0:
                    for idx, ind_mu in enumerate(population_mu):
                        assets_vis = decode_individual(ind_mu, _clone_assets_for_decode(ea_assets))
                        _emit_frame(idx, assets_vis, f"gen_{gen+1:04d}")

        best_idx = int(np.argmin(fitness_mu))
        if existing_by_mu is not None and best_idx < len(existing_by_mu):
            selected_existing = existing_by_mu[best_idx]
            for key in existing_keys:
                if key in selected_existing:
                    existing_assets[key] = _snapshot_asset(selected_existing[key])
        best_assets = decode_individual(population_mu[best_idx], ea_assets)
        recaculate_bbox(best_assets)
        print("best_score is:", fitness_mu[best_idx])
        if vis_enabled and vis_frames and vis_root is not None:
            fps = int(vis_cfg.get("fps", 10)) if vis_cfg else 10
            for parent_idx, frame_paths in vis_frames.items():
                if not frame_paths:
                    continue
                out_video = os.path.join(vis_root, f"parent_{parent_idx}.mp4")
                frames = None
                try:
                    frames = [imageio.imread(p) for p in frame_paths]
                    max_h = max(f.shape[0] for f in frames)
                    max_w = max(f.shape[1] for f in frames)
                    max_c = max((f.shape[2] if f.ndim == 3 else 1) for f in frames)
                    padded = []
                    for f in frames:
                        if f.ndim == 2:
                            f = np.stack([f] * max_c, axis=-1)
                        h, w = f.shape[0], f.shape[1]
                        c = f.shape[2]
                        if c < max_c:
                            f = np.concatenate([f, np.ones((h, w, max_c - c), dtype=f.dtype) * 255], axis=2)
                        if h == max_h and w == max_w:
                            padded.append(f)
                            continue
                        canvas = np.ones((max_h, max_w, max_c), dtype=f.dtype) * 255
                        canvas[:h, :w, :max_c] = f[:h, :w, :max_c]
                        padded.append(canvas)
                    imageio.mimsave(out_video, padded, fps=fps, macro_block_size=1)
                except Exception as e:
                    gif_path = out_video.replace(".mp4", ".gif")
                    try:
                        if frames is None:
                            frames = [imageio.imread(p) for p in frame_paths]
                            max_h = max(f.shape[0] for f in frames)
                            max_w = max(f.shape[1] for f in frames)
                            max_c = max((f.shape[2] if f.ndim == 3 else 1) for f in frames)
                            padded = []
                            for f in frames:
                                if f.ndim == 2:
                                    f = np.stack([f] * max_c, axis=-1)
                                h, w = f.shape[0], f.shape[1]
                                c = f.shape[2]
                                if c < max_c:
                                    f = np.concatenate([f, np.ones((h, w, max_c - c), dtype=f.dtype) * 255], axis=2)
                                if h == max_h and w == max_w:
                                    padded.append(f)
                                    continue
                                canvas = np.ones((max_h, max_w, max_c), dtype=f.dtype) * 255
                                canvas[:h, :w, :max_c] = f[:h, :w, :max_c]
                                padded.append(canvas)
                            frames = padded
                        imageio.mimsave(gif_path, frames, fps=fps)
                    except Exception as e2:
                        print(f"[Warning] Failed to save video for parent {parent_idx}: {e2}")
        return best_assets

    # Sort for consistent ordering
    ea_assets_layers = {}
    gd_solver = {}
    for depth in scene.layers_by_depth.keys():
        ea_assets_layers[depth] = copy.deepcopy(scene.layers_by_depth[depth]["assets"])
        gd_solver[depth] = copy.deepcopy(scene.layers_by_depth[depth]["solver"])
    bound = scene.bound
    xmin, xmax, ymin, ymax = bound

    fixed_assets = {}
    cross_region_existing = existing_assets or {}
    for depth in sorted(scene.layers_by_depth.keys()):
        cur_assets = copy.deepcopy(ea_assets_layers[depth])
        depth_existing = {k: copy.deepcopy(v) for k, v in cross_region_existing.items()}
        depth_existing.update(fixed_assets)
        parent_refs = dict(depth_existing)
        bounds_by_key = build_parent_bounds(cur_assets, parent_refs)
        z_locks = build_parent_z_locks(cur_assets, parent_refs)
        z_locks.update(build_constraint_z_locks(cur_assets, gd_solver[depth], fixed_assets=parent_refs))
        center_nodes_override = None
        groups_override = None
        if use_group and scene_graph is not None:
            active_srcs = set(cur_assets.keys())
            center_nodes_override, groups_override = scene_graph.segment_graph(
                active_srcs=active_srcs, inplace=False
            )
        if len(cur_assets) >= 30 and center_nodes_override:
            center_keys = {
                key for key in center_nodes_override.values()
                if key in cur_assets
            }
            center_assets = {
                key: copy.deepcopy(cur_assets[key])
                for key in cur_assets
                if key in center_keys
            }
            remaining_assets = {
                key: copy.deepcopy(asset)
                for key, asset in cur_assets.items()
                if key not in center_keys
            }
            print(
                f"[EA][center-stage] n_assets={len(cur_assets)}, "
                f"centers={sorted(center_assets.keys())}, "
                "generations=31, gd_steps=200"
            )
            center_groups = [{key} for key in center_assets]
            optimized_centers = run_ea(
                center_assets,
                gd_solver[depth],
                existing_assets=depth_existing,
                bounds_by_key={
                    key: value for key, value in bounds_by_key.items()
                    if key in center_assets
                },
                z_locks={
                    key: value for key, value in z_locks.items()
                    if key in center_assets
                },
                if_ea=True,
                seed=1234,
                groups_override=center_groups,
                center_nodes_override=None,
                use_group=use_group,
                use_weight=use_weight,
                vis_cfg=vis_cfg,
                depth=depth,
                vis_tag="center",
                generation_override=31,
                do_final_gd=True,
                gd_step_override=200,
            )
            optimized_centers = {
                key: _snapshot_asset(asset)
                for key, asset in optimized_centers.items()
            }
            freeze_assets(optimized_centers)
            if remaining_assets:
                remaining_centers = None
                remaining_groups = None
                if use_group and scene_graph is not None:
                    remaining_centers, remaining_groups = scene_graph.segment_graph(
                        active_srcs=set(remaining_assets.keys()), inplace=False
                    )
                optimized_remaining = run_ea(
                    remaining_assets,
                    gd_solver[depth],
                    existing_assets=depth_existing,
                    context_assets=optimized_centers,
                    bounds_by_key={
                        key: value for key, value in bounds_by_key.items()
                        if key in remaining_assets
                    },
                    z_locks={
                        key: value for key, value in z_locks.items()
                        if key in remaining_assets
                    },
                    if_ea=True,
                    seed=1234,
                    groups_override=remaining_groups,
                    center_nodes_override=remaining_centers,
                    use_group=use_group,
                    use_weight=use_weight,
                    vis_cfg=vis_cfg,
                    depth=depth,
                    vis_tag="main",
                    gd_step_override=400,
                )
                best_assets = dict(optimized_centers)
                best_assets.update(optimized_remaining)
            else:
                best_assets = optimized_centers
        else:
            best_assets = run_ea(
                cur_assets,
                gd_solver[depth],
                existing_assets=depth_existing,
                bounds_by_key=bounds_by_key,
                z_locks=z_locks,
                if_ea=True,
                seed=1234,
                groups_override=groups_override,
                center_nodes_override=center_nodes_override,
                use_group=use_group,
                use_weight=use_weight,
                vis_cfg=vis_cfg,
                depth=depth,
                vis_tag="main",
            )
        for key, asset in depth_existing.items():
            if key in fixed_assets:
                fixed_assets[key] = _snapshot_asset(asset)
            if key in cross_region_existing:
                cross_region_existing[key] = _snapshot_asset(asset)
        fixed_assets.update(best_assets)
    return fixed_assets


def compute_local_axes(polygon):
    """
    将原始 polygon 设置为默认局部坐标系：
    X 正方向为水平右，Y 为竖直上
    """
    # 如果是 MultiPolygon，选择面积最大的一个
    if isinstance(polygon, MultiPolygon):
        if len(polygon.geoms) == 0:
            raise ValueError("Empty MultiPolygon passed to compute_local_axes")
        polygon = max(polygon.geoms, key=lambda p: p.area)

    if not isinstance(polygon, Polygon):
        raise TypeError(f"Expected Polygon, got {type(polygon)}")

    coords = np.array(polygon.exterior.coords)
    min_x, min_y = coords.min(axis=0)
    origin = np.array([min_x, min_y])
    x_axis = np.array([1.0, 0.0])
    y_axis = np.array([0.0, 1.0])
    return origin, x_axis, y_axis


def align_polygon(polygon, origin, x_axis, y_axis):
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    coords = np.array(polygon.exterior.coords)
    local_coords = coords - origin
    return Polygon(local_coords)

def apply_inverse_transform(x, y, shelf):
    origin = shelf["local_origin"]
    return origin[0] + x, origin[1] + y


def preprocess_shelf_region(shelf_region):
    origin, x_axis, y_axis = compute_local_axes(shelf_region["polygon"])
    shelf_region["local_origin"] = origin
    shelf_region["local_x_axis"] = x_axis
    shelf_region["local_y_axis"] = y_axis
    shelf_region["local_polygon"] = align_polygon(shelf_region["polygon"], origin, x_axis, y_axis)


def postprocess_layout(layout, shelf):
    for asset in layout.values():
        if isinstance(asset['pos'], torch.Tensor):
            pos_array = asset['pos'].detach().cpu()
        else:
            pos_array = torch.tensor(asset['pos'])
        asset['pos'] = pos_array
        gx, gy = apply_inverse_transform(pos_array[0], pos_array[1], shelf)
        asset["pos"][0] = gx
        asset["pos"][1] = gy
    return layout
     

def run_regions_ccea(
    door: Door,
    walls: List[Wall],
    scene: Scene,
    scene_graph: SceneGraph,
    glb_paths: Dict[str, str],
    graph=True,
    use_weight=True,
    if_gd=True,
    output_dir: Optional[str] = None,
    existing_assets: Optional[dict[str, Union[Object, ObjectSet]]] = None,
    vis_cfg: Optional[dict[str, Any]] = None,
):
    gd_step=200
    pop_size=50
    mu = 5
    grid_size = 0.125
    all_layout= ccea_gd_layout_optimization(
        door,
        walls,
        scene,
        scene_graph,
        glb_paths,
        grid_size=grid_size,
        generations=31,
        gd_step=gd_step,
        pop_size=pop_size,
        mu=mu,
        graph=graph,
        use_weight=use_weight,
        if_gd=if_gd,
        use_group=True,
        output_dir=output_dir,
        existing_assets=existing_assets,
        vis_cfg=vis_cfg,
    )
    return all_layout


def run_floor_regions_ccea(
    door: Door,
    walls: List[Wall],
    scene: Scene,
    scene_graph: SceneGraph,
    glb_paths: Dict[str, str],
    vis_assets: Optional[dict[str, Union[Object, ObjectSet]]] = None,
    graph: bool = True,
    use_weight: bool = True,
    if_gd: bool = True,
    output_dir: Optional[str] = None,
    vis_cfg: Optional[dict[str, Any]] = None,
):
    """
    Per-region floor CCEA (GenesisVLM2 ``run_floor_regions_ccea`` aligned).
    ``vis_assets`` = already-placed assets from prior regions (Genesis ``existing_assets``).
    """
    return run_regions_ccea(
        door,
        walls,
        scene,
        scene_graph,
        glb_paths,
        graph=graph,
        use_weight=use_weight,
        if_gd=if_gd,
        output_dir=output_dir,
        existing_assets=vis_assets,
        vis_cfg=vis_cfg,
    )


def visualize_3d_layout(mesh, placed, support_regions, exp_name, filename="full_layout"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 原始 mesh
    all_verts = [mesh.vertices[face] for face in mesh.faces]
    ax.add_collection3d(Poly3DCollection(all_verts, color=(0.7, 0.7, 0.7, 0.2), linewidths=0.2))
    
    for i, region in enumerate(support_regions):
        color = cm.tab20(i % 20)

        # 绘制每个放置的物体
        if placed[str(i)]:
            print(placed[str(i)])
            for key, asset in placed[str(i)].items():
                x0, y0, z0 = asset["pos"]
                dy, dx, dz = asset["bbox"]
                color = cm.tab20(i % 20)
                box = np.array([
                    [x0 - dx/2, y0 - dy/2, z0],
                    [x0 + dx/2, y0 - dy/2, z0],
                    [x0 + dx/2, y0 + dy/2, z0],
                    [x0 - dx/2, y0 + dy/2, z0],
                    [x0 - dx/2, y0 - dy/2, z0 + dz],
                    [x0 + dx/2, y0 - dy/2, z0 + dz],
                    [x0 + dx/2, y0 + dy/2, z0 + dz],
                    [x0 - dx/2, y0 + dy/2, z0 + dz]
                ])
                faces = [
                    [box[0], box[1], box[2], box[3]],
                    [box[4], box[5], box[6], box[7]],
                    [box[0], box[1], box[5], box[4]],
                    [box[2], box[3], box[7], box[6]],
                    [box[1], box[2], box[6], box[5]],
                    [box[0], box[3], box[7], box[4]],
                ]
                ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=0.3, linewidths=0.3, edgecolor='k'))
                ax.text(x0 + dx/2, y0 + dy/2, z0 + dz + 0.002, key, fontsize=7, ha='center', va='bottom')

    # 设置视图
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # 自动调整视图范围
    x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    # 保存视图
    for view, elev, azim in [("top", 90, -90), ("front", 0, -90), ("angled", 20, -60)]:
        ax.view_init(elev=elev, azim=azim)
        out_path = f"{exp_name}/{filename}_{view}.png"
        plt.savefig(out_path, dpi=200)
        print(f"[Saved] {out_path}")

    plt.close()
