"""Shared geometry helpers (ablation-aligned corner math)."""
import numpy as np
import torch

_LOCAL_CORNER_OFFSETS = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float32)
_LOCAL_CORNER_OFFSETS_NP = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=float)


def compute_rotated_corners(rot, bbox_xy, device=None, dtype=None):
    """Compute oriented box corners (ablation: local @ R.T)."""
    if not torch.is_tensor(rot):
        rot = torch.tensor(rot, dtype=dtype or torch.float32, device=device)
    elif device is not None or dtype is not None:
        rot = rot.to(device=device or rot.device, dtype=dtype or rot.dtype)
    if not torch.is_tensor(bbox_xy):
        bbox_xy = torch.tensor(bbox_xy, dtype=rot.dtype, device=rot.device)
    device = rot.device
    dtype = rot.dtype
    rotation_matrix = torch.stack([
        torch.stack([torch.cos(rot[0]), -torch.sin(rot[0])]),
        torch.stack([torch.sin(rot[0]),  torch.cos(rot[0])]),
    ])
    local_corners = _LOCAL_CORNER_OFFSETS.to(device=device, dtype=dtype) * bbox_xy[:2] / 2
    return torch.matmul(local_corners, rotation_matrix.T)


def compute_rotated_corners_np(rot, bbox_xy):
    rotation_matrix = np.array([
        [np.cos(rot[0]), -np.sin(rot[0])],
        [np.sin(rot[0]),  np.cos(rot[0])],
    ])
    local_corners = _LOCAL_CORNER_OFFSETS_NP * np.asarray(bbox_xy)[:2] / 2
    return np.dot(local_corners, rotation_matrix.T)
