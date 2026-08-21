import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math
import json
from matplotlib.patches import Polygon
from shapely.geometry import LineString, Point
import torch

GRID_STEP = 0.25
DOOR_WIDTH = 0.8
WINDOW_WIDTH = 0.8

def polygon_is_clockwise(pts):
    area2 = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % n]
        area2 += (x2 - x1) * (y2 + y1)
    return area2 > 0  # 该准则下：正为clockwise

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def inward_normal(u, clockwise=True):
    # 对于clockwise多边形，边方向 p1->p2 的“右法线”指向室内；反之用左法线
    if clockwise:
        return np.array([u[1], -u[0]], dtype=float)  # right normal
    else:
        return np.array([-u[1], u[0]], dtype=float)  # left normal

def draw_grid(ax, xmin, xmax, ymin, ymax, step=GRID_STEP):
    xs = np.arange(np.floor((xmin-0.001)/step)*step, np.ceil((xmax+0.001)/step)*step + step/2, step)
    ys = np.arange(np.floor((ymin-0.001)/step)*step, np.ceil((ymax+0.001)/step)*step + step/2, step)
    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], linewidth=0.5, alpha=0.35, color='gray')
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], linewidth=0.5, alpha=0.35, color='gray')

def draw_door(ax, p1, p2, center, hinge='right', swing='inward', clockwise_poly=True, zorder=5):
    """
    绘制门：
    - 铰链点（pivot） = hinge 左/右
    - 门关闭状态线段（贴墙）
    - 门叶线段（沿开门方向）
    - 90° 弧（从门边缘到铰链向内）
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    center = np.array(center, dtype=float)
    wall_dir = unit(p2 - p1)
    n_in = unit(inward_normal(wall_dir, clockwise_poly))
    n_out = -n_in

    # 铰链点
    half = DOOR_WIDTH / 2
    if hinge == 'left':
        hinge_pt = center - wall_dir*half
        edge_pt = center + wall_dir*half
    else:  # right
        hinge_pt = center + wall_dir*half
        edge_pt = center - wall_dir*half

    # 门贴墙线段（关闭状态）
    ax.plot([edge_pt[0], hinge_pt[0]], [edge_pt[1], hinge_pt[1]], color='green', linewidth=3, zorder=zorder)

    # 门叶线段
    n_dir = n_in if swing == 'inward' else n_out
    leaf_end = hinge_pt + n_dir*DOOR_WIDTH
    ax.plot([hinge_pt[0], leaf_end[0]], [hinge_pt[1], leaf_end[1]], color='green', linewidth=3, zorder=zorder)

    # 90° 弧：从门边缘到铰链向内
    vec = edge_pt - hinge_pt
    theta_start = math.degrees(math.atan2(vec[1], vec[0]))
    theta_end = math.degrees(math.atan2(n_dir[1], n_dir[0]))
    
    # 调整方向，保证从门边缘顺时针到门叶
    diff = (theta_end - theta_start) % 360
    if diff < 0:
        diff += 360
    if swing == 'inward':
        if hinge == 'left':
            theta1, theta2 = theta_start, theta_end
        else:  # right
            theta1, theta2 = theta_end, theta_start
    else:  # outward
        if hinge == 'left':
            theta1, theta2 = theta_end, theta_start
        else:
            theta1, theta2 = theta_start, theta_end

    arc = patches.Arc(hinge_pt, 2*DOOR_WIDTH, 2*DOOR_WIDTH,
                      angle=0, theta1=theta1, theta2=theta2,
                      color='green', linewidth=3, zorder=zorder)
    ax.add_patch(arc)
    
def draw_window(ax, floorplan, center, width=WINDOW_WIDTH, zorder=5):
    """
    绘制普通窗户：
    - walls: 墙的端点列表 [(p1, p2), ...]
    - center: 窗户中心坐标
    - width: 窗户沿墙的宽度
    """
    center = np.array(center, dtype=float)
    walls = floorplan.copy()
    walls.append((0.0, 0.0))
    for idx in range(len(walls)-1):
        p1 = np.array(walls[idx], dtype=float)
        p2 = np.array(walls[idx+1], dtype=float)
        # 检查 center 是否在当前墙的范围内
        wall_line = LineString([p1, p2])
        center_point = Point(center)

        if wall_line.contains(center_point):
            # 窗户左右端点
            wall_dir = p2 - p1
            wall_length = np.linalg.norm(wall_dir)
            u = wall_dir / wall_length  # 墙方向单位向量
            half = width / 2
            left_pt = center - u * half
            right_pt = center + u * half
            
            # 绘制窗户线段
            ax.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]],
                color='blue', linewidth=3, zorder=zorder)
            break

def draw_area_centers_and_faders(ax, areas, floorplan,
                                 default_radius=None,
                                 cmap_name='tab10',
                                 base_alpha=0.45,
                                 gradient_res=200,
                                 label_offset=(0.06, 0.06)):
    """
    areas: list of dicts, each at least contains 'area_name' and 'location':[x,y]
           optionally 'radius' or 'boundary' (list of pts)
    floorplan: list of (x,y) pts to derive room scale if default_radius not provided
    """
    # derive default radius from room size
    xs = [p[0] for p in floorplan]
    ys = [p[1] for p in floorplan]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if default_radius is None:
        default_radius = max(0.5, min(width, height) * 0.18)  # 18% of min dim, min 0.5m

    cmap = plt.get_cmap(cmap_name)
    legend_handles = []

    for i, area in enumerate(areas):
        name = area.get('area_name', f"A{i}")
        cx, cy = area.get('location', [0.0, 0.0])
        r = area.get('radius', default_radius)
        color = cmap(i % cmap.N)
        rgb = color[:3]

        # --- radial gradient image centered at (cx,cy) ---
        res = gradient_res
        x = np.linspace(-r, r, res)
        y = np.linspace(-r, r, res)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X**2 + Y**2)
        # radial alpha: smooth falloff; adjust exponent for softer/harder edge
        alpha_map = np.clip(1 - (dist / r), 0.0, 1.0) ** 1.2
        alpha_map = alpha_map * base_alpha

        img = np.zeros((res, res, 4), dtype=float)
        img[..., :3] = rgb
        img[..., 3] = alpha_map

        extent = [cx - r, cx + r, cy - r, cy + r]
        ax.imshow(img, extent=extent, origin='lower', zorder=3, interpolation='bilinear')

        # --- center marker & number badge ---
        ax.scatter([cx], [cy], s=140, marker='o',
                   facecolor=rgb, edgecolors='white', linewidth=1.6, zorder=6)
        # small cross inside for clarity
        cross_size = r * 0.12
        ax.plot([cx - cross_size, cx + cross_size], [cy, cy], linewidth=1.2, color='white', zorder=7)
        ax.plot([cx, cx], [cy - cross_size, cy + cross_size], linewidth=1.2, color='white', zorder=7)

        # label with name and coordinates
        label = f"{i}: {name}\n({cx:.2f}, {cy:.2f})"
        ax.text(cx + label_offset[0], cy + label_offset[1], label,
                fontsize=9, ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='none', alpha=0.95),
                zorder=8)

        # --- if boundary provided, draw filled polygon + multi-layer stroked edges for fade ---
        if 'boundary' in area and area['boundary']:
            pts = np.array(area['boundary'], dtype=float)
            # base filled polygon (low alpha)
            ax.add_patch(Polygon(pts, closed=True, facecolor=(rgb[0], rgb[1], rgb[2], 0.12),
                                 edgecolor=(rgb[0], rgb[1], rgb[2], 0.6), linewidth=1.2, zorder=4))
            # layered edges: shrink slightly towards centroid to create 'fading edge'
            centroid = pts.mean(axis=0)
            n_layers = 6
            for k in range(n_layers):
                t = k / float(n_layers)
                scale = 1.0 - 0.06 * k  # shrink toward centroid
                scaled = centroid + (pts - centroid) * scale
                edge_alpha = 0.28 * (1.0 - t)
                ax.add_patch(Polygon(scaled, closed=True, fill=False,
                                     edgecolor=(rgb[0], rgb[1], rgb[2], edge_alpha),
                                     linewidth=1.6 * (1.0 - t), zorder=4 + k * 0.01))

        # collect legend handle
        legend_handles.append(patches.Patch(facecolor=(rgb[0], rgb[1], rgb[2], 0.6),
                                           label=f"{i}: {name}"))

    # draw a small legend (try to place it in an unobtrusive corner)
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)    

def _coerce_draw_asset(asset):
    """Normalize dict / Object-like records for ``draw_assets``."""
    if isinstance(asset, dict):
        return asset
    pos = getattr(asset, "pos", None)
    if pos is None:
        return None
    corners = getattr(asset, "corners", None)
    return {"pos": pos, "corners": corners}


def draw_assets(ax, results, zorder=6):
    """
    绘制所有物体及其边界框
    results: dict of assets
        每个元素类似：
        {
            'pos': (x, y),          # 中心坐标
            'phy': angle_rad,       # 朝向（弧度）
            'bbox': (w, h),         # 物体的宽/深
            'scale': (sx, sy),      # 缩放因子
            'description': str,     # 物体描述
            'region_idx': int       # 区域编号
        }
    """
    for name, asset in results.items():
        asset = _coerce_draw_asset(asset)
        if asset is None or "pos" not in asset:
            continue
        pos = asset["pos"]
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        else:
            pos = np.asarray(pos, dtype=float)
        if pos.shape[0] < 2:
            continue
        cx, cy = float(pos[0]), float(pos[1])

        corners = asset.get("corners", None)
        if corners is None:
            continue  # 没有corners就跳过
        if isinstance(corners, torch.Tensor):
            corners = corners.detach().cpu().numpy()
        centroid = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        order = np.argsort(angles)
        world_corners = corners[order] + np.array([cx, cy])

        poly = patches.Polygon(
            world_corners,
            closed=True,
            edgecolor='blue',
            facecolor='darkblue',
            alpha=0.75,
            linewidth=1.5,
            zorder=zorder
        )
        ax.add_patch(poly)

        ax.plot(cx, cy, 'o', color='blue', zorder=zorder+1)

        ax.text(cx, cy,
                f"{name}",
                ha='center', va='center',
                fontsize=6, color='blue',
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="none", alpha=0.6),
                zorder=zorder+2)


def plot_floorplan_with_doors_windows(floorplan, door_location, window_locations, areas=None, result=None, out_path="input_cache_areas/floor_plan.png"):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 房间多边形（浅蓝填充 + 红色边界）
    poly = patches.Polygon(floorplan, closed=True, facecolor=(0.7, 0.85, 1.0, 0.5),
                       edgecolor='red', linewidth=2, zorder=2)
    ax.add_patch(poly)

    # 多边形方向（用于内法线）
    cw = polygon_is_clockwise(floorplan)

    # 墙体编号（仅编号，不显示长度），把标注稍微向室内偏移，避免和边重合
    n = len(floorplan)
    for i in range(n):
        x1, z1 = floorplan[i]
        x2, z2 = floorplan[(i + 1) % n]
        p1 = np.array([x1, z1], dtype=float); p2 = np.array([x2, z2], dtype=float)
        u = unit(p2 - p1)
        n_in = unit(inward_normal(u, cw))
        mid = (p1 + p2) / 2.0
        label_pos = mid + n_in * 0.15   # 向室内偏 0.15m
        ax.text(label_pos[0], label_pos[1],
                f"{i}", color="red", ha="center", va="center",
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # 顶点坐标（加粗/放大，并向室内偏移，避免与边/节点重叠）
    for i in range(n):
        p_prev = np.array(floorplan[(i - 1) % n], dtype=float)
        p_curr = np.array(floorplan[i], dtype=float)
        p_next = np.array(floorplan[(i + 1) % n], dtype=float)
        u_prev = unit(p_curr - p_prev)
        u_next = unit(p_next - p_curr)
        n_in_prev = unit(inward_normal(u_prev, cw))
        n_in_next = unit(inward_normal(u_next, cw))
        n_avg = unit(n_in_prev + n_in_next)
        if np.linalg.norm(n_avg) < 1e-6:
            n_avg = np.array([0.15, 0.15])
        label_pos = p_curr + n_avg * 0.18  # 更明显的偏移
        ax.plot(p_curr[0], p_curr[1], 'o', color='red', markersize=4)
        ax.text(label_pos[0], label_pos[1],
                f"({p_curr[0]:.2f}, {p_curr[1]:.2f})",
                color="black", fontsize=12,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        
    # 网格
    max_x = max(x for x,_ in floorplan) + 1
    max_z = max(y for _,y in floorplan) + 1
    ax.set_xticks(np.arange(0,max_x,0.25), minor=True)
    ax.set_yticks(np.arange(0,max_z,0.25), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xticks(np.arange(0,max_x,0.5))
    ax.set_yticks(np.arange(0,max_z,0.5))
    ax.grid(which='major', color='gray', linestyle='-', linewidth=1, alpha=0.8)

    # 门
    if door_location:
        wid = int(door_location["wall_id"])
        p1 = floorplan[wid]
        p2 = floorplan[(wid + 1) % n]
        draw_door(ax, p1, p2, door_location["center"], door_location["hinge"], "inward", clockwise_poly=cw, zorder=5)

    # 窗
    if window_locations:
        for win in window_locations:
            wid = int(win["wall_id"])
            p1 = floorplan[wid]
            p2 = floorplan[(wid + 1) % n]
            draw_window(ax, floorplan, win["center"], width=WINDOW_WIDTH, zorder=5)

    if result:
        draw_assets(ax, result)

    if areas:
        draw_area_centers_and_faders(ax, areas, floorplan,
                                     default_radius=None, cmap_name='tab10',
                                     base_alpha=0.42, gradient_res=200)
        
    # 视图设置：只保留格子与图形，不要坐标轴/标题
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_z)
    ax.set_aspect('equal')
    ax.set_title("Floorplan with Door and Windows")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    # 示例数据
    floorplan = [(0.0, 0.0), (6.0, 0.0), (6.0, 8.0), (0.0, 8.0)]
    json_str = """{
    "door_location": {
        "wall_id": 0,
        "center": [1.0, 0.0],
        "hinge": "right",
        "swing": "inward"
    },
    "window_locations": [
        {
        "wall_id": 1,
        "center": [6.0, 2.0]
        },
        {
        "wall_id": 3,
        "center": [2.0, 8.0]
        }
    ]
    }"""
    json_data = json.loads(json_str)
    door_location = json_data['door_location']
    window_locations = json_data['window_locations']

    area_str = """
    {
    "room_analysis": "The room is a rectangular space measuring 6m x 8m, designed to function as a small buffet restaurant. It's compact but is structured to maintain a functional layout. Windows are located on the north and east walls, and the entrance door is at the southwest corner, facilitating natural light and movement flow.",
    "areas": [
        {
        "area_name": "Buffet Table Area",
        "description": "Central area for buffet tables symmetrically arranged for food service.",
        "location_reasoning": "Placing the buffet tables centrally ensures accessibility from all sides and optimizes flow. Patrons can easily move around the tables while maintaining symmetry.",
        "location": [3.0, 4.0]
        },
        {
        "area_name": "Dessert Station",
        "description": "Separate area for dessert offerings, providing a focused space for dessert selection.",
        "location_reasoning": "Positioned against the wall near window 2 for visibility and efficient use of space while keeping the dessert station distinct from the main buffet.",
        "location": [5.0, 6.0]
        },
        {
        "area_name": "Seating Area",
        "description": "Area for patrons to dine, arranged to maximize capacity without obstructing movement.",
        "location_reasoning": "Located near the north wall to take advantage of natural light and to be close to both buffet and dessert stations.",
        "location": [3.0, 7.0]
        }
    ]
    }"""
    areas = json.loads(area_str)
    plot_floorplan_with_doors_windows(floorplan, door_location, window_locations, areas=areas["areas"])
    print("Saved to input_cache_areas/floor_plan.png")
