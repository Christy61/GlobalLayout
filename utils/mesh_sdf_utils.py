"""Helpers for robust mesh SDF collision queries."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import trimesh


def repair_mesh_for_sdf(mesh: trimesh.Trimesh, *, label: str = "") -> Tuple[trimesh.Trimesh, bool]:
    """
    Best-effort repair so signed_distance is more reliable.

    Notes:
    - Hollow assets (table shell, open bowl) may stay non-watertight after fill_holes.
    - fill_holes can seal small gaps; it does NOT turn a hollow shell into a solid block
      unless the mesh already has boundary holes.
    """
    m = mesh.copy()
    if len(m.faces) == 0:
        return m, False
    try:
        m.remove_duplicate_faces()
        m.remove_degenerate_faces()
        m.merge_vertices(merge_tex=True, merge_norm=True)
        trimesh.repair.fix_normals(m)
        if not m.is_watertight:
            trimesh.repair.fill_holes(m)
            trimesh.repair.fix_normals(m)
    except Exception as exc:
        tag = f" ({label})" if label else ""
        print(f"[SDF] mesh repair failed{tag}: {exc}")
    watertight = bool(m.is_watertight)
    if not watertight and label:
        print(f"[SDF] mesh still non-watertight after repair: {label}")
    return m, watertight


def _max_inside_depth(mesh, pts) -> Optional[float]:
    """Max signed distance over sample points that are inside/on mesh (sdf > 0)."""
    if not len(pts):
        return 0.0
    try:
        sdf = trimesh.proximity.signed_distance(mesh, pts)
    except Exception:
        return None
    if not np.all(np.isfinite(sdf)):
        return None
    inside = sdf[sdf > 0.0]
    if inside.size == 0:
        return 0.0
    return float(np.max(inside))


def _signed_penetration(mesh_i, pts_i, mesh_j, pts_j) -> Optional[float]:
    """
    Overlap depth = how far surface samples penetrate into the other solid.

    trimesh signed_distance:
      outside -> negative (ignored)
      on-surface / inside -> positive (kept)

    Adjacent/touching meshes have no inside samples -> depth 0 (not a collision).
    """
    pen_ij = _max_inside_depth(mesh_j, pts_i)
    if pen_ij is None:
        return None
    pen_ji = _max_inside_depth(mesh_i, pts_j)
    if pen_ji is None:
        return None
    return max(pen_ij, pen_ji)


def _unsigned_overlap_depth(mesh_i, pts_i, mesh_j, pts_j) -> float:
    """
    Fallback when signed_distance fails on open meshes.
    Only counts points verified inside the other mesh (contains), not surface gap.
    """
    depths = []
    for target_mesh, pts in ((mesh_j, pts_i), (mesh_i, pts_j)):
        if not len(pts):
            continue
        try:
            inside_mask = target_mesh.contains(pts)
        except Exception:
            continue
        if not np.any(inside_mask):
            continue
        inside_pts = pts[inside_mask]
        depth = _max_inside_depth(target_mesh, inside_pts)
        if depth is not None and depth > 0.0:
            depths.append(depth)
    return max(depths) if depths else 0.0


def pair_penetration(
    mesh_i,
    pts_i,
    mesh_j,
    pts_j,
    *,
    sdf_collision_threshold: float,
    pair_label: str = "",
) -> Tuple[float, str]:
    """
    Returns (overlap_depth_m, method).
    overlap_depth >= threshold means collision.
    """
    signed = _signed_penetration(mesh_i, pts_i, mesh_j, pts_j)
    if signed is not None:
        return signed, "signed"
    unsigned = _unsigned_overlap_depth(mesh_i, pts_i, mesh_j, pts_j)
    if unsigned > 0.0:
        if pair_label:
            print(
                f"[SDF] signed_distance failed for {pair_label}; "
                f"using contains-based fallback (overlap={unsigned:.4f} m)"
            )
        return unsigned, "unsigned"
    if pair_label:
        print(f"[SDF] signed_distance failed for {pair_label}; overlap depth 0")
    return 0.0, "failed"


def is_collision(overlap_depth: float, sdf_collision_threshold: float) -> bool:
    """True when overlap depth reaches the allowed limit."""
    return overlap_depth >= sdf_collision_threshold
