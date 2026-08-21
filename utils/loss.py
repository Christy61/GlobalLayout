import torch
from PIL import Image
from types import SimpleNamespace
from typing import Optional
from utils.geometry import compute_rotated_corners

try:
    import clip
except ImportError:
    clip = None

def soft_margin_loss(x, a, b, margin):
    lower_bound = a - margin
    upper_bound = b + margin
    return torch.relu(lower_bound - x) + torch.relu(x - upper_bound)

def soft_near_loss(x, min_distance, max_distance, margin=0.01, alpha=1e-2):
    """
    alpha < 1: 缩小在 max_distance 内的惩罚力度
    """
    # base_loss = alpha * torch.relu(x - min_distance - margin)
    penalty = torch.relu(x - (max_distance + margin)) + torch.relu((min_distance - margin) - x)
    # return base_loss + penalty
    return penalty

def diou_loss_2d(pos1, pos2, corners1, corners2):
    """
    Compute the 2D Distance-IoU loss for collision avoidance, focusing only on the x-y plane.
    Assuming that asset.position represents the position of the bottom of the centre of the item, the
    size represents the [width, depth, height] of the item, but only width and depth are used here.
    """
    # x, y plane
    rotated_corners1 = pos1 + corners1
    rotated_corners2 = pos2 + corners2
    box1_min = torch.min(rotated_corners1, dim=0).values
    box1_max = torch.max(rotated_corners1, dim=0).values
    box2_min = torch.min(rotated_corners2, dim=0).values
    box2_max = torch.max(rotated_corners2, dim=0).values

    inter_min = torch.max(box1_min, box2_min)
    inter_max = torch.min(box1_max, box2_max)
    inter_wh = torch.clamp(inter_max - inter_min, min=0)
    inter_area = inter_wh[0] * inter_wh[1]

    area1 = (box1_max[0] - box1_min[0]) * (box1_max[1] - box1_min[1])
    area2 = (box2_max[0] - box2_min[0]) * (box2_max[1] - box2_min[1])
    min_area = torch.min(area1, area2)
    iou = inter_area / min_area
    return 1 - iou


def get_rotated_bbox_2d(center, size):
    w, h = size[0] / 2, size[1] / 2
    corners = torch.tensor([[-w, -h], [-w, h], [w, h], [w, -h]], dtype=center.dtype, device=center.device)
    rotated_corners = corners + center[:2]
    box_min = torch.min(rotated_corners, dim=0).values
    box_max = torch.max(rotated_corners, dim=0).values
    return box_min, box_max


def get_rotated_bbox_3d(center, size):
    w, d, h = size[0] / 2, size[1] / 2, size[2]
    corners = torch.tensor([
        [-w, -d, 0], [-w, -d, h], [-w, d, 0], [-w, d, h],
        [w, -d, 0], [w, -d, h], [w, d, 0], [w, d, h]
    ], dtype=center.dtype, device=center.device)
    rotated_corners = corners + center
    box_min = torch.min(rotated_corners, dim=0).values
    box_max = torch.max(rotated_corners, dim=0).values
    return box_min, box_max


def iou_loss_2d(pos1, pos2, size1, size2):
    """
    Compute the 2D IoU loss for collision avoidance, focusing only on the x-y plane.
    Assuming that asset.position represents the position of the bottom of the centre of the item, the
    size represents the [width, depth, height] of the item, but only width and depth are used here.
    """

    # x, y plane
    box1_min, box1_max = get_rotated_bbox_2d(pos1, size1)
    box2_min, box2_max = get_rotated_bbox_2d(pos2, size2)

    inter_min = torch.max(box1_min, box2_min)
    inter_max = torch.min(box1_max, box2_max)
    inter_wh = torch.clamp(inter_max - inter_min, min=0)
    inter_area = inter_wh[0] * inter_wh[1]

    area1 = (box1_max[0] - box1_min[0]) * (box1_max[1] - box1_min[1])
    area2 = (box2_max[0] - box2_min[0]) * (box2_max[1] - box2_min[1])
    union_area = area1 + area2 - inter_area + 1e-6

    iou = inter_area / union_area

    return iou


def iou_loss_3d(pos1, pos2, size1, size2):
    """
    Calculate the 3D IoU loss for collision avoidance.
    Assuming that asset.position represents the position of the bottom of the centre of the item, the
    size represents the [width, depth, height] of the item.
    """

    box1_min, box1_max = get_rotated_bbox_3d(pos1, size1)
    box2_min, box2_max = get_rotated_bbox_3d(pos2, size2)

    inter_min = torch.max(box1_min, box2_min)
    inter_max = torch.min(box1_max, box2_max)
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0))

    vol1 = torch.prod(size1)
    vol2 = torch.prod(size2)
    union_vol = vol1 + vol2 - inter_vol + 1e-6

    iou = inter_vol / (union_vol)

    return iou

def _is_tuple_solver(solver) -> bool:
    return (
        solver is not None
        and getattr(solver, "constraints", None)
        and solver.constraints
        and isinstance(solver.constraints[0], tuple)
    )


def _is_dict_assets(assets) -> bool:
    if not assets:
        return False
    return isinstance(next(iter(assets.values())), dict)


def _to_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x, dtype=dtype)


def _asset_has_volume_dict(asset: dict) -> bool:
    bbox = asset.get("bbox")
    if bbox is None:
        return True
    bbox_t = _to_tensor(bbox)
    return bool(torch.all(bbox_t[:2] > 0).item())


def _asset_has_volume_node(asset) -> bool:
    bbox = getattr(asset, "bbox", None)
    if bbox is None:
        return True
    bbox_t = _to_tensor(bbox)
    return bool(torch.all(bbox_t[:2] > 0).item())


def _filter_volume_assets(assets: dict) -> dict:
    if not assets:
        return assets
    if _is_dict_assets(assets):
        return {k: v for k, v in assets.items() if _asset_has_volume_dict(v)}
    return {k: v for k, v in assets.items() if _asset_has_volume_node(v)}


def semantic_loss(solver, assets=None):
    if _is_tuple_solver(solver):
        total_loss = torch.tensor(0.0)
        for constraint, args, kwargs in solver.constraints:
            total_loss = total_loss + constraint(*args, **kwargs)
        return total_loss
    return solver.solve(assets)


def semantic_loss_group(solver, groups, assets=None):
    if _is_tuple_solver(solver):
        loss_g = torch.zeros(len(groups))
        total_loss = torch.tensor(0.0)
        for constraint, args, kwargs in solver.constraints:
            l_c = constraint(*args, **kwargs)
            total_loss = total_loss + l_c
            src = args[0] if args else None
            src_id = src.get("id") if isinstance(src, dict) else getattr(src, "id", None)
            for idx, group in enumerate(groups):
                if src_id in group:
                    loss_g[idx] += l_c
                    break
        return loss_g, total_loss
    loss_group = solver.solve_group(groups, assets)
    total_loss = solver.solve(assets)
    return loss_group, total_loss

# ------------------ Physics Loss ------------------
# Paper Eq.(30)-(31): harmonic-mean depth + optional existence term for corner contact.
SOFTSAT_EPS = 1e-6
# τ in σ(δ/τ): at δ=τ=1cm, σ(1)≈0.73 — catches shallow corner overlap missed by L_depth≈δ.
COLLISION_EXIST_TAU = 0.01
COLLISION_EXIST_WEIGHT = 1.0


def _raw_axis_overlaps(min1, max1, min2, max2) -> torch.Tensor:
    """Per-axis 1D overlap lengths on SAT axes (Eq.30, no floor)."""
    return torch.relu(torch.min(max1, max2) - torch.max(min1, min2))


def _harmonic_depth_loss(raw: torch.Tensor) -> torch.Tensor:
    """Eq.(31): L_depth = M / sum_i 1/(o_i + eps)."""
    overlap = raw + SOFTSAT_EPS
    return overlap.shape[-1] / (1.0 / overlap).sum(dim=-1)


def _collision_existence_loss(raw: torch.Tensor) -> torch.Tensor:
    """
    Existence term: prod_i σ(o_i / τ).

    When corner contact makes min(o_i)=δ very small, L_depth≈δ is tiny but
    other axes still have o_i>0; this term stays large if any axis overlaps.
    Zero when SAT-separated (any axis o_i <= 0).
    """
    separated = raw.min(dim=-1).values <= 0
    exist = torch.sigmoid(raw / COLLISION_EXIST_TAU).prod(dim=-1)
    return torch.where(separated, torch.zeros_like(exist), exist)


def _aggregate_axis_overlaps(raw: torch.Tensor) -> torch.Tensor:
    depth = _harmonic_depth_loss(raw)
    exist = _collision_existence_loss(raw)
    return depth + COLLISION_EXIST_WEIGHT * exist


def _legacy_harmonic_axis_loss(raw: torch.Tensor) -> torch.Tensor:
    """Depth-only term (for debug comparison)."""
    return _harmonic_depth_loss(raw)


def get_box_axes_batch(angles):
    """批量获取每个物体的两个轴向"""
    angles = angles.reshape(-1)
    c, s = torch.cos(angles), torch.sin(angles)
    axes_x = torch.stack([c, s], dim=-1)   # (N, 2)
    axes_y = torch.stack([-s, c], dim=-1)  # (N, 2)
    return torch.stack([axes_x, axes_y], dim=1)  # (N, 2, 2)

def project_corners_batch(corners, axis):
    """批量投影到某个轴上
    corners: (N, 4, 2)
    axis: (M, 2)
    返回 (N, M) 的 min 和 max
    """
    # (N,4,2) @ (M,2)T => (N,4,M)
    proj = torch.einsum('nij,mj->nim', corners, axis)
    return proj.min(dim=1).values, proj.max(dim=1).values  # (N, M)


def _pairwise_sat_raw_overlaps(corners_a, corners_b, angles_a, angles_b) -> torch.Tensor:
    """Raw SAT overlaps for every A/B pair on the two boxes' own axes."""
    axes_a = get_box_axes_batch(angles_a)  # (Na, 2, 2)
    axes_b = get_box_axes_batch(angles_b)  # (Nb, 2, 2)

    proj_aa = torch.einsum("npd,nad->npa", corners_a, axes_a)
    min_aa = proj_aa.min(dim=1).values
    max_aa = proj_aa.max(dim=1).values

    proj_b_on_a = torch.einsum("mpd,nad->nmpa", corners_b, axes_a)
    min_b_on_a = proj_b_on_a.min(dim=2).values
    max_b_on_a = proj_b_on_a.max(dim=2).values
    raw_on_a = _raw_axis_overlaps(
        min_aa[:, None, :],
        max_aa[:, None, :],
        min_b_on_a,
        max_b_on_a,
    )

    proj_bb = torch.einsum("mpd,mad->mpa", corners_b, axes_b)
    min_bb = proj_bb.min(dim=1).values
    max_bb = proj_bb.max(dim=1).values

    proj_a_on_b = torch.einsum("npd,mad->nmpa", corners_a, axes_b)
    min_a_on_b = proj_a_on_b.min(dim=2).values
    max_a_on_b = proj_a_on_b.max(dim=2).values
    raw_on_b = _raw_axis_overlaps(
        min_a_on_b,
        max_a_on_b,
        min_bb[None, :, :],
        max_bb[None, :, :],
    )

    return torch.cat([raw_on_a, raw_on_b], dim=-1)


def softsat_overlap_batch(assets):
    """
    并行 Soft-SAT overlap
    assets: dict of assets
    """
    sample_before_filter = next(iter(assets.values()), None) if assets else None
    assets = _filter_volume_assets(assets)
    if len(assets) < 2:
        if _is_dict_assets(assets):
            sample = next(iter(assets.values()), None)
            if sample is not None:
                return torch.zeros((), device=sample["pos"].device, dtype=sample["pos"].dtype)
        sample = next(iter(assets.values()), None)
        if sample is not None:
            return torch.zeros((), device=sample.pos.device, dtype=sample.pos.dtype)
        if isinstance(sample_before_filter, dict):
            return torch.zeros(
                (),
                device=sample_before_filter["pos"].device,
                dtype=sample_before_filter["pos"].dtype,
            )
        if sample_before_filter is not None:
            return torch.zeros(
                (),
                device=sample_before_filter.pos.device,
                dtype=sample_before_filter.pos.dtype,
            )
        return torch.tensor(0.0)
    if _is_dict_assets(assets):
        device = next(iter(assets.values()))["pos"].device
        asset_list = list(assets.values())
        N = len(asset_list)
        angles = torch.stack([a["phy"] for a in asset_list])
        corners = torch.stack([a["corners"] + a["pos"][:2] for a in asset_list])
    else:
        device = next(iter(assets.values())).pos.device
        asset_list = list(assets.values())
        N = len(asset_list)
        angles = torch.stack([a.rot for a in asset_list])
        corners = torch.stack([a.corners + a.pos[:2] for a in asset_list])
    raw = _pairwise_sat_raw_overlaps(corners, corners, angles, angles)  # (N,N,4)
    loss = _aggregate_axis_overlaps(raw)  # (N,N)

    # 只取上三角避免重复
    triu_mask = torch.triu(torch.ones(N,N, device=device), diagonal=1).bool()
    loss_val = loss[triu_mask].sum()

    return loss_val

def _ensure_asset_corners(assets):
    for asset in assets.values():
        if getattr(asset, "corners", None) is not None:
            continue
        rot = asset.rot
        bbox = asset.bbox
        if not torch.is_tensor(rot):
            rot = torch.tensor(rot, dtype=torch.float32)
            asset.rot = rot
        if not torch.is_tensor(bbox):
            bbox = torch.tensor(bbox, dtype=torch.float32)
            asset.bbox = bbox
        asset.corners = compute_rotated_corners(rot, bbox)


def _node_asset_to_dict(asset):
    pos = _to_tensor(asset.pos)
    phy = _to_tensor(asset.rot)
    bbox = _to_tensor(asset.bbox)
    if getattr(asset, "corners", None) is not None:
        corners = _to_tensor(asset.corners)
    else:
        corners = compute_rotated_corners(phy, bbox)
    return {
        "pos": pos,
        "phy": phy,
        "corners": corners,
        "bbox": bbox,
    }


def _node_assets_to_dict_assets(assets):
    """Convert node asset objects to the dict format expected by softsat_overlap_batch."""
    return {key: _node_asset_to_dict(asset) for key, asset in assets.items()}


def softsat_overlap_pair(assets_a, assets_b):
    """
    Soft-SAT overlap between two sets (A vs B only).
    """
    sample_before_filter = next(iter(assets_a.values()), None) if assets_a else None
    assets_a = _filter_volume_assets(assets_a)
    assets_b = _filter_volume_assets(assets_b)
    if not assets_a or not assets_b:
        if isinstance(sample_before_filter, dict):
            return torch.zeros(
                (),
                device=sample_before_filter["pos"].device,
                dtype=sample_before_filter["pos"].dtype,
            )
        if sample_before_filter is not None:
            return torch.zeros(
                (),
                device=sample_before_filter.pos.device,
                dtype=sample_before_filter.pos.dtype,
            )
        return torch.tensor(0.0)
    if _is_dict_assets(assets_a):
        device = next(iter(assets_a.values()))["pos"].device
        assets_a_list = list(assets_a.values())
        angles_a = torch.stack([a["phy"] for a in assets_a_list])
        corners_a = torch.stack([a["corners"] + a["pos"][:2] for a in assets_a_list])
    else:
        device = next(iter(assets_a.values())).pos.device
        assets_a_list = list(assets_a.values())
        angles_a = torch.stack([a.rot for a in assets_a_list])
        corners_a = torch.stack([a.corners + a.pos[:2] for a in assets_a_list])

    if _is_dict_assets(assets_b):
        assets_b_list = list(assets_b.values())
        angles_b = torch.stack([b["phy"].to(device=device, dtype=angles_a.dtype) for b in assets_b_list])
        corners_b = torch.stack([(b["corners"] + b["pos"][:2]).to(device=device, dtype=corners_a.dtype) for b in assets_b_list])
    else:
        assets_b_list = list(assets_b.values())
        angles_b = torch.stack([b.rot.to(device=device, dtype=angles_a.dtype) for b in assets_b_list])
        corners_b = torch.stack([(b.corners + b.pos[:2]).to(device=device, dtype=corners_a.dtype) for b in assets_b_list])

    raw = _pairwise_sat_raw_overlaps(corners_a, corners_b, angles_a, angles_b)
    loss = _aggregate_axis_overlaps(raw)  # (Na,Nb)
    return loss.sum()

def _wall_plane_overlap_loss(
    wall_assets: dict,
    wall_assignments: dict[str, str],
    walls: dict,
) -> torch.Tensor:
    """
    Compute overlap loss for wall-mounted assets per wall by projecting them
    to each wall's local 2D plane (u along wall, v as z).
    """
    if not wall_assets or not wall_assignments or not walls:
        return torch.tensor(0.0)
    device = next(iter(wall_assets.values())).pos.device
    dtype = next(iter(wall_assets.values())).pos.dtype
    total = torch.zeros((), device=device, dtype=dtype)

    grouped: dict[str, list[str]] = {}
    for key in wall_assets.keys():
        wall_key = wall_assignments.get(key)
        if not wall_key or wall_key not in walls:
            continue
        grouped.setdefault(wall_key, []).append(key)

    base_corners = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        device=device,
        dtype=dtype,
    )

    for wall_key, keys in grouped.items():
        if len(keys) < 2:
            continue
        wall = walls[wall_key]
        a = wall.rot.to(device=device, dtype=dtype)
        v = torch.cat([torch.cos(a), torch.sin(a)], dim=0)  # wall normal
        t = torch.stack([-v[1], v[0]])  # wall tangent (u axis)

        pseudo_assets = {}
        for k in keys:
            asset = wall_assets[k]
            corners_world = asset.corners + asset.pos[:2]  # (4, 2)
            u_vals = corners_world @ t  # (4,)
            u_min = u_vals.min()
            u_max = u_vals.max()
            u_center = 0.5 * (u_min + u_max)
            half_u = 0.5 * (u_max - u_min)

            half_v = 0.5 * asset.bbox[2]
            v_center = asset.pos[2] + half_v

            corners_uv = base_corners * torch.stack([half_u, half_v])
            pos_uv = torch.stack([u_center, v_center, u_center.new_zeros(())])
            pseudo_assets[k] = SimpleNamespace(
                pos=pos_uv,
                rot=torch.zeros_like(asset.rot),
                corners=corners_uv,
            )
        total = total + softsat_overlap_batch(pseudo_assets)

    return total


def _infer_wall_assignments_from_layer(assets: dict, walls: dict) -> dict[str, str]:
    """
    Infer wall assignments for assets marked as layer == "wall" by nearest wall.
    """
    if not assets or not walls:
        return {}
    wall_keys = [k for k, a in assets.items() if getattr(a, "layer", None) == "wall"]
    if not wall_keys:
        return {}
    assignments: dict[str, str] = {}
    for key in wall_keys:
        asset = assets[key]
        pos = asset.pos
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, dtype=torch.float32)
        pos_xy = pos[:2]
        device = pos_xy.device
        dtype = pos_xy.dtype
        best_key = None
        best_dist = None
        for wkey, wall in walls.items():
            wpos = wall.pos[:2].to(device=device, dtype=dtype)
            a = wall.rot.to(device=device, dtype=dtype)
            v = torch.cat([torch.cos(a), torch.sin(a)], dim=0)
            dist = torch.abs(torch.dot(pos_xy - wpos, v))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_key = wkey
        if best_key is not None:
            assignments[key] = best_key
    return assignments


def physics_loss(
    assets,
    room_bound=None,
    other_assets=None,
    weight=2.0,
    *,
    wall_assignments: Optional[dict[str, str]] = None,
    walls: Optional[dict] = None,
):
    if _is_dict_assets(assets):
        assets_dict = assets
    else:
        assets_dict = _node_assets_to_dict_assets(assets)

    # 并行 N×N 计算
    if walls and wall_assignments is None and not _is_dict_assets(assets):
        wall_assignments = _infer_wall_assignments_from_layer(assets, walls)

    # primary_assets = assets
    # if wall_assignments and walls:
    #     wall_keys = {
    #         k for k in wall_assignments.keys()
    #         if k in assets and getattr(assets[k], "layer", None) == "wall"
    #     }
    #     xy_assets = {k: v for k, v in assets.items() if k not in wall_keys}
    #     wall_assets = {k: v for k, v in assets.items() if k in wall_keys}
    #     wall_assignments = {k: v for k, v in wall_assignments.items() if k in wall_keys}
    #     loss = torch.zeros((), device=next(iter(assets.values())).pos.device, dtype=next(iter(assets.values())).pos.dtype)
    #     if xy_assets:
    #         loss = loss + softsat_overlap_batch(_node_assets_to_dict_assets(xy_assets) if not _is_dict_assets(xy_assets) else xy_assets)
    #     if wall_assets:
    #         loss = loss + _wall_plane_overlap_loss(wall_assets, wall_assignments, walls)
    #     primary_assets = xy_assets
    # else:
    loss = softsat_overlap_batch(assets_dict)

    if other_assets is not None:
        other_assets_dict = other_assets if _is_dict_assets(other_assets) else _node_assets_to_dict_assets(other_assets)
        # Move other assets tensors to the primary assets device/dtype and recompute corners.
        # Keep their computation graph intact; callers decide whether they are trainable.
        primary_device = next(iter(assets_dict.values()))["pos"].device
        primary_dtype = next(iter(assets_dict.values()))["pos"].dtype
        detached_other = {}
        for k, v in other_assets_dict.items():
            # ensure phy/pos/bbox are tensors on primary device/dtype and detached
            pos = v.get("pos")
            phy = v.get("phy")
            bbox = v.get("bbox")
            if torch.is_tensor(pos):
                pos_t = pos.to(device=primary_device, dtype=primary_dtype)
            else:
                pos_t = torch.as_tensor(pos, device=primary_device, dtype=primary_dtype)
            if torch.is_tensor(phy):
                phy_t = phy.to(device=primary_device, dtype=phy.dtype if hasattr(phy, 'dtype') else primary_dtype)
            else:
                phy_t = torch.as_tensor(phy, device=primary_device, dtype=primary_dtype)
            # GenesisVLM2 sometimes stores `phy` in degrees; convert to radians if magnitude indicates degrees
            try:
                if float(phy_t.abs().max().item()) > 2 * 3.1416:
                    phy_t = phy_t * (3.141592653589793 / 180.0)
            except Exception:
                pass
            if torch.is_tensor(bbox):
                bbox_t = bbox.to(device=primary_device, dtype=bbox.dtype if hasattr(bbox, 'dtype') else primary_dtype)
            else:
                bbox_t = torch.as_tensor(bbox, device=primary_device, dtype=primary_dtype)
            # recompute local corners (no pos added)
            corners_t = compute_rotated_corners(phy_t, bbox_t)
            detached_other[k] = {
                "pos": pos_t,
                "phy": phy_t,
                "bbox": bbox_t,
                "corners": corners_t,
            }
        loss = loss + softsat_overlap_pair(assets_dict, detached_other)

    return loss * weight

def project_loss(assets, region_center, x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0, z_min=0.0, weight=4.0, bounds_by_key=None):
    """
    Vectorized PGD projection loss for all assets.
    Ensures that the position of the items are within predetermined boundaries.
    """
    if not assets:
        return torch.tensor(0.0)
    if _is_dict_assets(assets):
        device = next(iter(assets.values()))["pos"].device
        n = len(assets)
        asset_items = list(assets.items())
        corners = torch.stack([a["corners"] for _, a in asset_items], dim=0).to(device)
        pos = torch.stack([a["pos"] for _, a in asset_items], dim=0).to(device)
    else:
        device = next(iter(assets.values())).pos.device
        n = len(assets)
        asset_items = list(assets.items())
        corners = torch.stack([a.corners for _, a in asset_items], dim=0).to(device)
        pos = torch.stack([a.pos for _, a in asset_items], dim=0).to(device)

    # (N, 2)
    corners_min = corners.amin(dim=1)
    corners_max = corners.amax(dim=1)

    margin = 0.0
    if bounds_by_key:
        x_min_arr = torch.empty(n, device=device)
        x_max_arr = torch.empty(n, device=device)
        y_min_arr = torch.empty(n, device=device)
        y_max_arr = torch.empty(n, device=device)
        for i, (key, _) in enumerate(asset_items):
            if key in bounds_by_key:
                bxmin, bxmax, bymin, bymax = bounds_by_key[key]
                x_min_arr[i] = bxmin
                x_max_arr[i] = bxmax
                y_min_arr[i] = bymin
                y_max_arr[i] = bymax
            else:
                x_min_arr[i] = x_min
                x_max_arr[i] = x_max
                y_min_arr[i] = y_min
                y_max_arr[i] = y_max
        loss_x = soft_margin_loss(pos[:, 0], x_min_arr - corners_min[:, 0], x_max_arr - corners_max[:, 0], margin=margin)
        loss_y = soft_margin_loss(pos[:, 1], y_min_arr - corners_min[:, 1], y_max_arr - corners_max[:, 1], margin=margin)
    else:
        loss_x = soft_margin_loss(pos[:, 0], x_min - corners_min[:, 0], x_max - corners_max[:, 0], margin=margin)
        loss_y = soft_margin_loss(pos[:, 1], y_min - corners_min[:, 1], y_max - corners_max[:, 1], margin=margin)
    loss_z = soft_margin_loss(pos[:, 2], torch.full((n,), z_min, device=device), torch.full((n,), float('inf'), device=device), margin=margin)
    # if region_center == None:
    return (loss_x + loss_y + loss_z).sum() * weight
    # else:
    #     loss_center = (pos[:, 0] - region_center[0])**2 + (pos[:, 1] - region_center[1])**2
    #     return (loss_x + loss_y + loss_z).sum() + loss_center.sum() * 0.005
