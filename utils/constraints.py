from typing import Callable, Dict, Union
import numpy as np
import torch
from types import SimpleNamespace
from utils.loss import soft_near_loss, diou_loss_2d
from typing import Dict, List
from utils.node import Object, ObjectSet
from utils.tool import Point

# -------------------------------------------------
# Constraint registry
# -------------------------------------------------

CONSTRAINT_REGISTRY: Dict[str, Callable] = {}


def register_constraint(name: str):
    """
    Decorator used to expose constraint functions to DSL.
    """

    def wrapper(fn: Callable):
        if name in CONSTRAINT_REGISTRY:
            raise ValueError(f"Constraint '{name}' already registered")
        CONSTRAINT_REGISTRY[name] = fn
        return fn

    return wrapper
           
# ------------------ Constraint Methods ------------------
@register_constraint("distance")
def distance(src: Union[Object, ObjectSet], dst: Union[Object, ObjectSet], min_distance: float, max_distance: float, weight=0.9, margin=0.05, update=True):
    # ablation-aligned near(): margin=0.05, weight=0.9, clamp max=10.0
    margin_val = margin
    if isinstance(dst, Point) or getattr(dst, "type", None) == "point":
        margin_val = 0.03

    if getattr(dst, "type", None) == "asset":
        if update:
            bb1 = src.bbox[:2]
            bb2 = dst.bbox[:2]
            min_dist_1 = torch.min(bb1) / 2
            min_dist_2 = torch.min(bb2) / 2
            max_dist_1 = torch.max(bb1) / 2
            max_dist_2 = torch.max(bb2) / 2
            min_distance = torch.max(torch.tensor(min_distance, device=bb1.device), min_dist_1 + min_dist_2)
            max_distance = torch.max(torch.tensor(max_distance, device=bb1.device), max_dist_1 + max_dist_2)
    pos_i = src.pos[:2]
    pos_j = dst.pos[:2].detach()

    if getattr(dst, "type", None) == "edge":
        a_j = dst.rot
        v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)
        dist_val = torch.abs(torch.dot(pos_i - pos_j, v_j))
    else:
        dist_val = torch.norm(pos_i - pos_j)

    term = soft_near_loss(dist_val, min_distance, max_distance, margin_val)
    loss = torch.clamp(term, max=15.0) 
    return loss * weight

_DIRECTION_TO_ANGLE = {
    "down": -90.0,
    "right": 0.0,
    "up": 90.0,
    "left": 180.0,
}

_DIRECTION_TO_PERP = {
    "up": (1.0, 0.0),
    "down": (1.0, 0.0),
    "right": (0.0, 1.0),
    "left": (0.0, 1.0),
}


@register_constraint("place_align")
def place_align(src: Union[Object, ObjectSet], dst: Union[Object, ObjectSet], direction: str, weight=1.0):
    """
    Place src relative to dst along a global top-view direction.
    direction: one of "up", "down", "left", "right" in the world frame.
    """
    if direction not in _DIRECTION_TO_ANGLE:
        raise ValueError(
            f"place_align: direction must be one of {sorted(_DIRECTION_TO_ANGLE.keys())!r}, "
            f"got {direction!r}."
        )

    center_1 = src.pos[:2]
    center_2 = dst.pos[:2]

    rel_vec = center_1 - center_2.detach()
    norm = torch.sqrt((rel_vec ** 2).sum() + 1e-8)
    unit_vec = rel_vec / norm

    world_angle = _DIRECTION_TO_ANGLE[direction] % 360.0
    angle_tensor = torch.deg2rad(torch.tensor(world_angle, device=rel_vec.device, dtype=rel_vec.dtype))
    target_vec = torch.stack([torch.cos(angle_tensor), torch.sin(angle_tensor)])

    angle_loss = 1 - torch.dot(unit_vec, target_vec)

    perp = _DIRECTION_TO_PERP[direction]
    axis_perp = torch.tensor(perp, device=rel_vec.device, dtype=rel_vec.dtype)
    proj_perp = torch.abs(torch.dot(rel_vec, axis_perp))

    loss = 0.7 * angle_loss + 0.3 * proj_perp
    return loss * weight

@register_constraint("align_with")
def align_with(src: Union[Object, ObjectSet], dst: Union[Object, ObjectSet], angle: float = 0.0, weight=1.0):
    """
    Rotate src so that it is parallel to dst plus an angle offset.
    angle: degrees, counter-clockwise offset
    """
    if isinstance(dst.rot, list):
        dst.rot = torch.tensor(dst.rot, dtype=torch.float32, requires_grad=False)
    if isinstance(src.rot, list):
        src.rot = torch.tensor(src.rot, dtype=torch.float32, requires_grad=True)

    target_angle = dst.rot[0].detach() + torch.deg2rad(torch.tensor(angle, device=src.rot.device))
    loss = 1 - torch.cos(src.rot[0] - target_angle)
    return loss * weight

@register_constraint("point_towards")
def point_towards(src: Union[Object, ObjectSet], dst: Union[Object, ObjectSet], angle: float = 0.0, weight=1.0):
    """
    Rotate src to face dst (ablation-aligned).

    ``desired_direction = dst.pos[:2].detach() - src.pos[:2]``; loss = 1 - dot(dir, facing) / |dir|.
    """
    theta_i = src.rot.view(-1)

    v_i = torch.stack([torch.cos(theta_i), torch.sin(theta_i)]).view(-1)

    if dst.type == "edge":
        a_j = dst.rot.detach()
        rad = torch.deg2rad(torch.tensor(a_j))
        target = torch.tensor([torch.cos(rad), torch.sin(rad)])
        loss = 1 - torch.dot(v_i, target)
    else:
        if isinstance(dst.pos, list):
            dst.pos = torch.tensor(dst.pos, dtype=torch.float32, requires_grad=False)

        desired_direction = dst.pos[:2].detach() - src.pos[:2]
        align_angle = torch.dot(desired_direction, v_i) / (torch.norm(desired_direction) + 1e-6)
        loss = 1 - align_angle
    return loss * weight

@register_constraint("against")
def against(src: Union[Object, ObjectSet], dst: Union[Object, ObjectSet], height: float = 0.0, weight=1.0, dist_tolerance: float = 0.01):
    """
    Place src against wall dst at specified height.
    """
    pos_i = src.pos
    a_i = src.rot
    v_i = torch.cat([torch.cos(a_i), torch.sin(a_i)], dim=0)
    pos_j = dst.pos
    a_j = dst.rot
    v_j = torch.cat([torch.cos(a_j), torch.sin(a_j)], dim=0)
    distance_center = torch.abs(torch.dot(pos_i[:2] - pos_j[:2], v_j))
    gap = src.bbox[0]/2
    dist_error = distance_center - gap
    dist_loss = torch.relu(-dist_error) + torch.relu(dist_error - dist_tolerance)
    if torch.dot(v_i, v_j) < -0.99:
        eps = 8e-2
        v_i = v_i + torch.tensor([eps, -eps], device=v_i.device, dtype=v_i.dtype)
    angle_loss = 1 - torch.dot(v_i, v_j)
    loss_xy = angle_loss * 1.5 + dist_loss

    # z-axis constraint
    loss_z = torch.abs(src.pos[2] - height)
    return (loss_xy + loss_z) * weight

@register_constraint("on")
def on(src: Union[Object, ObjectSet], dst: Union[Object, ObjectSet], surface_id=0, weight=1.0):

    loss_xy = diou_loss_2d(src.pos[:2], dst.pos[:2], src.corners, dst.surface[surface_id].corners)
    top_z = dst.pos[2].detach()
    loss_z = torch.abs(src.pos[2] - top_z)

    return (loss_xy + loss_z) * weight

@register_constraint("above")
def above(src, dst, height: float, weight=1.0):
    loss_xy = torch.norm(src.pos[:2] - dst.pos[:2])
    top_z = dst.pos[2].detach()
    loss_z = torch.abs(src.pos[2] - top_z - height)

    return (loss_xy + loss_z) * weight

def _asset_has_bbox(asset) -> bool:
    bbox = getattr(asset, "bbox", None)
    if bbox is None:
        return False
    if isinstance(bbox, torch.Tensor):
        return bbox.numel() >= 2
    try:
        return len(bbox) >= 2
    except TypeError:
        return False


def _bbox_xy_halves(asset):
    bbox = asset.bbox
    if isinstance(bbox, torch.Tensor):
        bbox_val = bbox.detach().cpu().numpy()
    else:
        bbox_val = np.asarray(bbox, dtype=float)
    return float(bbox_val[0]) / 2.0, float(bbox_val[1]) / 2.0


def _ensure_tensor_pos(asset):
    if isinstance(asset.pos, list):
        asset.pos = torch.tensor(asset.pos, dtype=torch.float32, requires_grad=False)


def _surround_fixed(
    center_asset: Union[Object, ObjectSet],
    surrounding_assets: List[Union[Object, ObjectSet]],
    gap: float = 0.25,
    dist_tol: float = 0.05,
    weight: float = 1.0,
):
    """Place surrounding assets on evenly spaced directions around center bbox (ablation-aligned)."""
    _ensure_tensor_pos(center_asset)
    center_pos = center_asset.pos[:2]
    hx, hy = _bbox_xy_halves(center_asset)

    if float(center_asset.bbox[0]) >= float(center_asset.bbox[1]):
        start_angle = 0.0
    else:
        start_angle = 0.5 * np.pi

    n = len(surrounding_assets)
    if n == 0:
        return torch.tensor(0.0, device=center_pos.device, dtype=center_pos.dtype)

    total_loss = torch.tensor(0.0, device=center_pos.device, dtype=center_pos.dtype)
    eps = 1e-6
    for i, asset in enumerate(surrounding_assets):
        angle = start_angle + i * (2.0 * np.pi / n)
        angle_t = torch.tensor(angle, device=center_pos.device, dtype=center_pos.dtype)
        ca = torch.cos(angle_t)
        sa = torch.sin(angle_t)
        hx_s, hy_s = _bbox_xy_halves(asset)
        abs_c = torch.clamp(torch.abs(ca), min=eps)
        abs_s = torch.clamp(torch.abs(sa), min=eps)
        tx = torch.tensor(hx, device=center_pos.device, dtype=center_pos.dtype) / abs_c
        ty = torch.tensor(hy, device=center_pos.device, dtype=center_pos.dtype) / abs_s
        use_x = tx <= ty
        t = torch.where(use_x, tx, ty)
        gap_x = torch.tensor(gap + hx_s, device=center_pos.device, dtype=center_pos.dtype)
        gap_y = torch.tensor(gap + hy_s, device=center_pos.device, dtype=center_pos.dtype)
        offset = torch.where(use_x, gap_x / abs_c, gap_y / abs_s)
        direction = torch.stack([ca, sa])
        edge_point = center_pos + t * direction
        target_point = center_pos + (t + offset) * direction

        dist = torch.norm(asset.pos[:2] - target_point)
        total_loss += soft_near_loss(dist, 0.0, dist_tol, margin=0.0, alpha=0.005)

        edge_target = SimpleNamespace(pos=edge_point, type="asset")
        total_loss += point_towards(asset, edge_target)

    return total_loss * weight


@register_constraint("surround")
def surround(
    src_list: List[Union[Object, ObjectSet]],
    dst: Union[Object, ObjectSet],
    distance_: float,
    look_mode: str = "axis",
    weight: float = 1.0,
):
    """
    Make src_list surround dst from different directions (ablation ``surround`` / ``surround_fixed``).

    GraphLayout DSL order: ``surround(src_list, dst, distance)``.
    Center with bbox → ``surround_fixed``; Point / no bbox → distance + point_towards path.
    """
    center_asset = dst
    surrounding_assets = src_list
    if distance_ is None:
        distance_ = 1.0
    if not isinstance(look_mode, str):
        look_mode = "axis"
    max_distance = distance_

    if not _asset_has_bbox(center_asset):
        _ensure_tensor_pos(center_asset)
        n = len(surrounding_assets)
        if n == 0:
            return torch.tensor(0.0)
        pos_j = center_asset.pos[:2].detach()
        pos_a = surrounding_assets[0].pos[:2]
        for asset in surrounding_assets[1:]:
            pos_a = pos_a + asset.pos[:2]
        avg_dist = torch.norm(pos_a / n - pos_j)
        loss = soft_near_loss(avg_dist, 0.0, 0.01) * 1.5
        if look_mode == "axis":
            for asset in surrounding_assets:
                loss += distance(asset, center_asset, 0.0, max_distance, update=True)
                loss += point_towards(asset, center_asset)
        else:
            for asset in surrounding_assets:
                loss += distance(asset, center_asset, max_distance, max_distance, update=False)
                loss += point_towards(asset, center_asset)
        return loss * weight

    return _surround_fixed(
        center_asset,
        surrounding_assets,
        gap=0.25,
        dist_tol=0.05,
        weight=weight,
    )
