import torch
import numpy as np
from typing import List, Any, Optional


# Wall
class Wall:
    def __init__(self, key: str, id: int, pos: list, rot: list, length: float, group: str, otype: str):
        self.key = key
        self.id = id
        self.group = group
        self.type = otype

        self.pos = torch.tensor(pos).float()
        self.rot = torch.tensor(rot).float()
        self.length = length

# Door
class Door:
    def __init__(
        self,
        key: str,
        id: int,
        pos: list,
        rot: list,
        bbox: list,
        group: str,
        otype: str,
        center: Optional[list] = None,
        wall_id: Optional[int] = None,
        hinge: Optional[str] = None,
    ):
        self.key = key
        self.id = id
        self.group = group
        self.type = otype

        self.pos = torch.tensor(pos).float()
        self.rot = torch.tensor(rot).float()
        self.bbox = torch.tensor(bbox).float()      
        self.center = center
        self.wall_id = wall_id
        self.hinge = hinge or "right"

# Singe scene object
class Object:
    def __init__(
        self, 
        key: str, 
        id: int, 
        bbox: list, 
        scale: int, 
        group: str, 
        otype: str
    ):
        self.key = key
        self.id = id
        self.group = group
        self.type = otype

        # will be defined later
        self.pos = None
        self.rot = None
        self.corners = None
        
        self.bbox = torch.tensor(bbox).float()
        self.scale = scale

        self.surface = None   # will be defined later
        self.depth = None     # depth determines the order of solving, will be defined later
        self.parent = None    # parent determines the z-value of object
        self.layer = None     # optional semantic layer tag (e.g. "wall")


    def init_from_individual(self, individual, i):
        x = individual[4*i]
        y = individual[4*i + 1]
        z = individual[4*i + 2]

        self.pos = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True)
        self.rot = torch.tensor([individual[4*i+3] * torch.pi / 180], dtype=torch.float32, requires_grad=True)


    def get_assets(self, layout_full):
        self.pos = self.pos.detach().cpu().numpy() if isinstance(self.pos, torch.Tensor) else self.pos
        self.rot = self.rot.detach().cpu().numpy() if isinstance(self.rot, torch.Tensor) else self.rot
        self.degree_rot = self.rot/torch.pi*180
        self.corners = self.corners.detach().cpu().numpy() if isinstance(self.corners, torch.Tensor) else self.corners
        self.bbox = self.bbox.detach().cpu().numpy() if isinstance(self.bbox, torch.Tensor) else self.bbox
        layout_full[self.key] = self

# Treat multiple objects as a single rigid set
class ObjectSet:
    def __init__(
        self,
        key: str,
        id: int,
        bbox: list,
        scale: int,
        assets_list: List[Object],
        group: str,
        otype: str,
        axis: str = "x",
        spacing: float = 0.0,
    ):
        self.key = key
        self.id = id
        self.group = group
        self.assets = assets_list
        self.type = otype
        self.axis = axis
        self.spacing = spacing

        # will be defined later
        self.pos = None
        self.rot = None
        self.corners = None
        
        self.bbox = torch.tensor(bbox).float()
        self.scale = scale

        self.surface = None   # will be defined later
        self.depth = None     # depth determines the order of solving, will be defined later
        self.parent = None    # parent determines the z-value of object
        self.layer = None     # optional semantic layer tag (e.g. "wall")


    def init_from_individual(self, individual, i):
        x = individual[4*i]
        y = individual[4*i + 1]
        z = individual[4*i + 2]

        self.pos = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True)
        self.rot = torch.tensor([individual[4*i+3] * torch.pi / 180], dtype=torch.float32, requires_grad=True)


    def get_assets(self, layout_full):
        base_pos = self.pos.clone() if isinstance(self.pos, torch.Tensor) else torch.tensor(self.pos, dtype=torch.float32)
        count = len(self.assets)

        if self.group == "stack":
            # arrange in z axis around base_pos
            current_z = base_pos[2]
            for asset in self.assets:
                asset_height = float(asset.bbox[2])
                asset_pos = torch.stack([base_pos[0], base_pos[1], current_z + asset_height / 2])

                asset.pos = asset_pos.detach().cpu().numpy()
                asset.rot = self.rot.detach().cpu().numpy() if isinstance(self.rot, torch.Tensor) else np.array(self.rot)
                asset.degree_rot = asset.rot/np.pi*180
                asset.bbox = asset.bbox.detach().cpu().numpy() if isinstance(asset.bbox, torch.Tensor) else asset.bbox
                layout_full[asset.key] = asset
                # Update z for next asset
                current_z = asset_pos[2] + asset_height / 2

        elif self.group == "row":
            # arrange in a single row on xy plane along the given axis
            base_pos_row = base_pos.clone()
            axis_idx = 0 if self.axis == "x" else 1

            rot_t = self.rot if isinstance(self.rot, torch.Tensor) else torch.tensor(self.rot, dtype=base_pos_row.dtype, device=base_pos_row.device)
            rot_val = rot_t[0] if rot_t.ndim > 0 else rot_t
            cos_t = torch.cos(rot_val)
            sin_t = torch.sin(rot_val)
            rot_mat = torch.stack([torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])])
            dir_local = torch.tensor(
                [1.0, 0.0] if self.axis == "x" else [0.0, 1.0],
                dtype=base_pos_row.dtype,
                device=base_pos_row.device,
            )
            widths = []
            for asset in self.assets:
                w = float(asset.bbox[axis_idx]) if isinstance(asset.bbox, torch.Tensor) else float(asset.bbox[axis_idx])
                widths.append(w)
            total_len = sum(widths) + self.spacing * max(count - 1, 0)
            cursor = -0.5 * total_len
            for idx, asset in enumerate(self.assets):
                w = widths[idx]
                offset = cursor + 0.5 * w
                cursor += w + self.spacing
                local_vec = dir_local * offset
                world_vec = rot_mat @ local_vec
                pos = base_pos_row.clone()
                pos[0] = pos[0] + world_vec[0]
                pos[1] = pos[1] + world_vec[1]
                asset.pos = pos.detach().cpu().numpy()
                asset.rot = rot_t.detach().cpu().numpy() if isinstance(rot_t, torch.Tensor) else rot_t
                asset.degree_rot = asset.rot/np.pi*180
                asset.bbox = asset.bbox.detach().cpu().numpy() if isinstance(asset.bbox, torch.Tensor) else asset.bbox
                layout_full[asset.key] = asset
        else:
            raise ValueError(f"Unknown group type: {self.group}")
        
        return layout_full
    

# Fixed anchor object (no volume) for constraints
class FixedObject:
    def __init__(self, key: str, pos, rot):
        self.key = key
        self.id = -1
        self.group = "fixed"
        self.type = "asset"
        self.pos = torch.tensor(pos, dtype=torch.float32, requires_grad=False)
        self.rot = torch.tensor(rot, dtype=torch.float32, requires_grad=False)
        self.bbox = torch.zeros(3, dtype=torch.float32)
        self.corners = torch.zeros((4, 2), dtype=torch.float32)
        self.surface = [self]

    def init_from_individual(self, individual, i):
        x = individual[4*i]
        y = individual[4*i + 1]
        z = individual[4*i + 2]

        self.pos = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=True)
        self.rot = torch.tensor([individual[4*i+3] * torch.pi / 180], dtype=torch.float32, requires_grad=True)


# Helper factory functions for asset sets
def row(name: str, axis: str = "x", count: int = 2, space: float = 0.2, vertical: bool = False):
    """
    Describe a row of identical items.
    """
    return {
        "function": f"row('{name}', axis='{axis}', count={count}, space={space}, vertical={str(vertical).lower()})"
    }


def stack(name: str, count: int = 2, space: float = 0.0, vertical: bool = False):
    """
    Describe a vertical stack of identical items.
    """
    return {
        "function": f"stack('{name}', count={count}, space={space}, vertical={str(vertical).lower()})"
    }
