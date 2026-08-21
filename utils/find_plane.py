import trimesh
import numpy as np
import os
from collections import deque, defaultdict
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.patheffects as patheffects
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, MultiLineString, Point, Polygon, MultiPolygon, GeometryCollection
from sklearn.cluster import DBSCAN
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Slight Z bump on reported support planes to reduce numerical contact / interpenetration.
SUPPORT_HEIGHT_Z_EPS = 1e-4
# If max(horizontal ``height``) is within this (m) of mesh AABB top Z ``mesh.bounds[1,2]``, snap that layer to bbox top.
_TOP_HEIGHT_SNAP_TO_BBOX_Z_TOL = 0.1

# Defaults tuned to tolerate mildly curved tessellations (weaker normal agreement, slightly
# non-horizontal “tops” still classified as horizontal).
_DEFAULT_PLANAR_T_NORM = 0.85
_DEFAULT_PLANAR_T_ADJ = 0.83
_DEFAULT_CLASSIFY_HORIZONTAL_THRESH = 0.92
_DEFAULT_CLASSIFY_VERTICAL_THRESH = 0.08
_DEFAULT_CLASSIFY_R_PLANE = 0.17
_DEFAULT_MAX_ASPECT = 15
_DEFAULT_MIN_TOP_AREA = 0.03
_DEFAULT_MIN_TRI_OVER_RECT = 0.28


def is_similar_normal(n1, n2, threshold):
    return abs(np.dot(n1, n2)) >= threshold


def build_triangle_adjacency(mesh):
    adjacency = defaultdict(list)
    faces = mesh.faces
    edges = {}

    for idx, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            if edge in edges:
                adjacency[idx].append(edges[edge])
                adjacency[edges[edge]].append(idx)
            else:
                edges[edge] = idx
    return adjacency


def extract_planar_clusters(
    mesh,
    t_norm=_DEFAULT_PLANAR_T_NORM,
    t_adj=_DEFAULT_PLANAR_T_ADJ,
    min_cluster_size=2,
    horizontal_eps_z=1e-3,
    orphan_face_ids=None,
):
    """
    Grow planar-ish connected components by shared edges + normal similarity.

    Faces that never reach ``min_cluster_size`` are dropped from the returned
    ``clusters`` list (historical behavior). If ``orphan_face_ids`` is a list,
    those face indices are appended there so every triangle can be accounted
    for: ``union(cluster faces) ∪ orphan_face_ids`` partitions ``0..n-1``.
    """
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    adjacency = build_triangle_adjacency(mesh)

    unclustered = set(range(len(mesh.faces)))
    clusters = []

    while unclustered:
        queue = deque()
        f0 = unclustered.pop()
        cluster = [f0]
        queue.extend([
            f for f in adjacency[f0]
            if is_similar_normal(face_normals[f], face_normals[f0], t_adj)
        ])

        while queue:
            f = queue.popleft()
            if f not in unclustered:
                continue
            if is_similar_normal(face_normals[f], face_normals[f0], t_norm):
                cluster.append(f)
                unclustered.remove(f)
                queue.extend([
                    n for n in adjacency[f]
                    if is_similar_normal(face_normals[n], face_normals[f], t_adj)
                ])

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
        elif orphan_face_ids is not None:
            orphan_face_ids.extend(int(x) for x in cluster)

    return clusters


def planar_clusters_and_orphans(mesh, **kwargs):
    """
    Same as ``extract_planar_clusters`` but also returns faces discarded for
    being below ``min_cluster_size`` (singletons / undersized BFS patches).
    """
    orphans = []
    clusters = extract_planar_clusters(mesh, orphan_face_ids=orphans, **kwargs)
    return clusters, orphans


def sample_face_indices_by_area(mesh, face_indices, n_samples, *, seed=None):
    """
    Weighted sample (without replacement when possible) by triangle area.
    Useful to subsample ``orphan_face_ids`` for a second-pass / diagnostics.
    """
    if n_samples <= 0 or not face_indices:
        return []
    ids = np.asarray(face_indices, dtype=int).reshape(-1)
    n = len(ids)
    if n_samples >= n:
        return ids.tolist()
    rng = np.random.default_rng(seed)
    w = mesh.area_faces[ids].astype(float)
    s = float(w.sum())
    if s <= 0:
        return rng.choice(ids, size=n_samples, replace=False).tolist()
    p = w / s
    pick = rng.choice(np.arange(n, dtype=int), size=n_samples, replace=False, p=p)
    return ids[pick].tolist()

def classify_cluster(
    mesh,
    face_ids,
    tol=0.015,
    horizontal_thresh=_DEFAULT_CLASSIFY_HORIZONTAL_THRESH,
    vertical_thresh=_DEFAULT_CLASSIFY_VERTICAL_THRESH,
    r_plane=_DEFAULT_CLASSIFY_R_PLANE,
    max_aspect=_DEFAULT_MAX_ASPECT,
    min_top_area=_DEFAULT_MIN_TOP_AREA,
    min_tri_over_rect=_DEFAULT_MIN_TRI_OVER_RECT,
):
    if face_ids is None or len(face_ids) == 0:
        print("[Warn] Empty face_ids cluster, skip.")
        return None
    tris = mesh.faces[face_ids]
    pts = mesh.vertices[tris].reshape(-1, 3)
    try:
        pc = trimesh.points.PointCloud(pts)
        obb_u = pc.bounding_box_oriented
    except Exception as e:
        print(f"[Warn] Failed to compute OBB (faces={len(face_ids)}): {e}")
        return None
    extents = obb_u.extents
    center = obb_u.centroid

    e = np.sort(extents)
    if e[0] > r_plane * e[1]:
        return None
    
    # 形状过滤 太细的片不算支撑面
    e = np.sort(extents)[::-1]  # 降序排列，e[0]: 最大，e[1]: 中间，e[2]: 最小（thickness）
    aspect_ratio = e[0] / e[1]
    area = e[0] * e[1]

    if aspect_ratio > max_aspect or area < min_top_area:  # 太细 or 太小
        return None

    tris_area = trimesh.Trimesh(vertices=pts, faces=np.arange(len(pts)).reshape(-1, 3)).area
    rect_area = area
    if rect_area > 0 and tris_area / rect_area < min_tri_over_rect:
        return None

    tris = mesh.faces[face_ids]
    pts = mesh.vertices[tris].reshape(-1, 3)
    
    # 1. 用面片法向判断类型
    face_normals = mesh.face_normals[face_ids]
    avg_normal = np.mean(face_normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)
    nz = abs(avg_normal[2])

    if nz >= horizontal_thresh:
        typ = "horizontal"
        height = center[2]
    elif nz <= vertical_thresh:
        typ = "vertical"
        height = center[2]  # for XY merge
    else:
        typ = "slanted"
        height = None

    return {
        "faces": np.array(face_ids),
        "center": center,
        "height": height,
        "normal": avg_normal,
        "type": typ,
    }

def center_mesh_on_bottom_surface(mesh):
    # 计算 mesh 所有顶点的最低 Z 坐标
    min_z = np.min(mesh.vertices[:, 2])
    translation = np.array([0, 0, -min_z])
    mesh.apply_translation(translation)
    return mesh

def project_faces_to_plane_polygon(mesh, face_ids, plane="xy", min_area=1e-6, max_aspect=15, eps=1e-6):
    """
    更稳健的投影：先把每个三角形做为 Polygon（不先 buffer），
    再做一次 union，最后对 union 的结果做一次小 buffer(eps) 用于填缝隙。
    如果 union 的结果是 GeometryCollection，尝试抽取其中的多边形部分。
    """
    verts = mesh.vertices[mesh.faces[face_ids]].reshape(-1, 3)
    tris = verts.reshape(-1, 3, 3)
    polygons = []
    for tri in tris:
        if plane == "xy":
            coords = tri[:, :2]
        elif plane == "xz":
            coords = tri[:, [0, 2]]
        elif plane == "yz":
            coords = tri[:, [1, 2]]
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        poly = Polygon(coords)
        if not poly.is_valid or poly.area <= min_area:
            continue
        # 简单的长宽过滤，保守一些
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        aspect = max(w, h) / (min(w, h) + 1e-8)
        if aspect <= max_aspect:
            polygons.append(poly)

    if not polygons:
        return Polygon()

    unioned = unary_union(polygons)
    # 保持 Polygon 或 MultiPolygon
    if isinstance(unioned, (Polygon, MultiPolygon)):
        return unioned
    return Polygon()


def get_vertical_face_intersections(mesh, vertical_info, z_height, tol=1e-6, min_dist=1e-3, snap_tol=1e-3):
    """
    - 在 OBB 的 8 条边上求与 z=z_height 的交点（或端点在平面上的点）
    - 去重后：
        * if len(points) == 2: 返回一条线段
        * if len(points) >= 3: 按角度排序，返回闭合多段边（顺时针/逆时针）
    返回：list of segments (each seg is np.array([[x1,y1],[x2,y2]]))
    """
    # 收集相关顶点
    verts_list = []
    for face_idx in vertical_info['faces']:
        verts = mesh.vertices[mesh.faces[face_idx]]
        verts_list.append(verts)
    if len(verts_list) == 0:
        return []
    pts = np.concatenate(verts_list, axis=0)

    # 使用 OBB
    pc = trimesh.points.PointCloud(pts)
    obb = pc.bounding_box_oriented
    extents = obb.extents
    tf = obb.primitive.transform

    # 构造 8 个顶点（齐次）
    corners_unit = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ])
    corners_scaled = corners_unit * extents
    corners_homo = np.hstack([corners_scaled, np.ones((8, 1))])
    corners_world = (tf @ corners_homo.T).T[:, :3]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 下底面边
        (0, 4), (1, 5), (2, 6), (3, 7)   # 竖直边
    ]

    points = []
    for i1, i2 in edges:
        p1 = corners_world[i1]; p2 = corners_world[i2]
        z1, z2 = p1[2], p2[2]

        # 精确的“跨越”判定（带一些容差）
        if (z1 - z_height) * (z2 - z_height) < - (tol):
            # 严格跨越：插值
            t = (z_height - z1) / (z2 - z1)
            if -tol <= t <= 1 + tol:
                pt = p1 + t * (p2 - p1)
                points.append(pt[:2])
            continue

        # 端点恰好在平面附近（snap）
        if abs(z1 - z_height) <= snap_tol:
            points.append(p1[:2])
        if abs(z2 - z_height) <= snap_tol:
            points.append(p2[:2])

    # 去掉重复点（按距离）
    uniq = []
    for pt in points:
        p = np.array(pt)
        if not any(np.linalg.norm(p - u) < min_dist for u in uniq):
            uniq.append(p)
    points = uniq

    if len(points) < 2:
        return []

    points = np.array(points)

    # len==2 -> 直接一条线段
    if len(points) == 2:
        return [np.stack([points[0], points[1]], axis=0)]

    # len>=3 -> 按角度排序，生成闭合边（多边形边）
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
    order = np.argsort(angles)
    sorted_pts = points[order]

    segs = []
    for i in range(len(sorted_pts)):
        p1 = sorted_pts[i]
        p2 = sorted_pts[(i+1) % len(sorted_pts)]
        if np.linalg.norm(p2 - p1) >= min_dist:
            segs.append(np.stack([p1, p2], axis=0))
    return segs 

def extend_line(line, extension=0.01):
    """
    line: shapely LineString，两点线段
    extension: 延长距离
    返回：延长后的 LineString
    """
    if not isinstance(line, LineString) or len(line.coords) != 2:
        return line  # 只处理两点线段

    (x1, y1), (x2, y2) = line.coords
    dx, dy = x2 - x1, y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length == 0:
        return line

    # 计算单位方向向量
    ux, uy = dx / length, dy / length

    # 延长两端点
    new_p1 = (x1 - ux * extension, y1 - uy * extension)
    new_p2 = (x2 + ux * extension, y2 + uy * extension)

    return LineString([new_p1, new_p2])

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        self.parent[rb] = ra


def merge_clusters_by_polygon(
    mesh,
    clusters_info,
    plane="xy",
    normal_align_thresh=0.95,
    height_thresh=0.05,
    buffer_tol=0.02,
    prefer_upper_merged_horizontal=True,
    upper_height_tol=None,
):
    """
    clusters_info: list of dict, 每个包含 faces, center, height, normal, type
    plane: 用于投影和相交检测的平面
    prefer_upper_merged_horizontal: 若 True，同一合并组内多个水平面且高度差 <= height_thresh 时，
        只保留最高的一层（避免上下近邻面合成一个厚板）。
    upper_height_tol: 判定“与最高面同高”的容差；默认 SUPPORT_HEIGHT_Z_EPS。
    返回: merged_info_list
    """
    if upper_height_tol is None:
        upper_height_tol = SUPPORT_HEIGHT_Z_EPS
    n = len(clusters_info)
    if n == 0:
        return []

    polys = []
    for info in clusters_info:
        poly = project_faces_to_plane_polygon(mesh, info['faces'], plane=plane)
        polys.append(poly)
    
    uf = UnionFind(n)
    for i in range(n):
        if polys[i].is_empty:
            continue
        for j in range(i + 1, n):
            if polys[j].is_empty:
                continue
            # 法向一致性（考虑绝对值）
            if not is_similar_normal(clusters_info[i]['normal'], clusters_info[j]['normal'], normal_align_thresh):
                continue
            # 对水平面考虑高度差
            if clusters_info[i]['type'] == 'horizontal' and clusters_info[j]['type'] == 'horizontal':
                if abs(clusters_info[i]['height'] - clusters_info[j]['height']) > height_thresh:
                    continue
            # 投影边界是否相交或接近
            if polys[i].buffer(buffer_tol).intersects(polys[j].buffer(buffer_tol)):
                uf.union(i, j)

    # collect groups
    groups = defaultdict(list)
    for idx in range(n):
        root = uf.find(idx)
        groups[root].append(idx)

    merged = []
    for gidxs in groups.values():
        if not gidxs:
            continue
        
        # 处理水平面的特殊情况：近邻多层水平投影相交时，只保留最上面一层
        if (
            prefer_upper_merged_horizontal
            and len(gidxs) > 1
            and all(clusters_info[i]['type'] == 'horizontal' for i in gidxs)
        ):
            heights = [clusters_info[i]['height'] for i in gidxs]
            min_h, max_h = min(heights), max(heights)

            if max_h - min_h <= height_thresh:
                top_indices = []
                for i, h in enumerate(heights):
                    if abs(h - max_h) <= upper_height_tol:
                        top_indices.append(gidxs[i])

                for idx in top_indices:
                    merged.append(clusters_info[idx])
                continue

        # 正常合并逻辑
        face_ids = np.concatenate([clusters_info[i]['faces'] for i in gidxs])
        face_ids = np.unique(face_ids.astype(int))
        orig_types = [clusters_info[i].get("type") for i in gidxs]
        info = classify_cluster(mesh, face_ids)
        if info is None:
            info = clusters_info[gidxs[0]]
        elif all(t == "horizontal" for t in orig_types) and (
            info.get("type") != "horizontal" or info.get("height") is None
        ):
            # 合并后三角形集合变大，classify 可能变成 slanted（height=None），
            # 但输入全是水平支撑簇：保留水平语义与可用高度，避免后续排序/比较崩溃。
            heights_sub = [
                float(clusters_info[i]["height"])
                for i in gidxs
                if clusters_info[i].get("height") is not None
            ]
            if heights_sub:
                centers = np.stack(
                    [np.asarray(clusters_info[i]["center"], dtype=float) for i in gidxs],
                    axis=0,
                )
                w = np.array(
                    [len(np.asarray(clusters_info[i]["faces"]).reshape(-1)) for i in gidxs],
                    dtype=float,
                )
                wsum = float(w.sum()) + 1e-12
                merged_center = np.sum(centers * (w[:, None] / wsum), axis=0)
                n0 = np.asarray(clusters_info[gidxs[0]]["normal"], dtype=float)
                n0 = n0 / (np.linalg.norm(n0) + 1e-12)
                info = {
                    "faces": face_ids,
                    "center": merged_center,
                    "height": float(max(heights_sub)),
                    "normal": n0,
                    "type": "horizontal",
                }
            else:
                info = clusters_info[gidxs[0]]
        merged.append(info)
    return merged


def extract_support_surfaces(
    mesh,
    t_norm=_DEFAULT_PLANAR_T_NORM,
    t_adj=_DEFAULT_PLANAR_T_ADJ,
    min_cluster_size=2,
    merge_thresh=0.08,
    r_plane=_DEFAULT_CLASSIFY_R_PLANE,
    prefer_upper_merged_horizontal=True,
    upper_height_tol=None,
):
    raw = extract_planar_clusters(mesh, t_norm, t_adj, min_cluster_size)
    infos = []
    for c in raw:
        info = classify_cluster(
            mesh,
            c,
            horizontal_thresh=_DEFAULT_CLASSIFY_HORIZONTAL_THRESH,
            vertical_thresh=_DEFAULT_CLASSIFY_VERTICAL_THRESH,
            tol=0.01,
            r_plane=r_plane,
        )
        if info is not None:
            infos.append(info)

    horiz = [i for i in infos if i['type'] == 'horizontal']
    vert = [i for i in infos if i['type'] == 'vertical']

    merge_kw = dict(
        plane="xy",
        normal_align_thresh=0.88,
        height_thresh=merge_thresh,
        buffer_tol=merge_thresh * 0.5,
        prefer_upper_merged_horizontal=prefer_upper_merged_horizontal,
        upper_height_tol=upper_height_tol,
    )
    merged_h = merge_clusters_by_polygon(mesh, horiz, **merge_kw)
    merged_v = merge_clusters_by_polygon(
        mesh, vert, **{**merge_kw, "prefer_upper_merged_horizontal": False}
    )

    if merged_h:
        z_bbox_top = float(mesh.bounds[1, 2])
        h_vals = [float(s["height"]) for s in merged_h if s.get("height") is not None]
        if h_vals:
            h_max = max(h_vals)
            if abs(z_bbox_top - h_max) < _TOP_HEIGHT_SNAP_TO_BBOX_Z_TOL:
                h_match_tol = 1e-3
                for s in merged_h:
                    if s.get("height") is None:
                        continue
                    if abs(float(s["height"]) - h_max) <= h_match_tol:
                        s["height"] = z_bbox_top

    return merged_h, merged_v


def segment_horizontal_surface(mesh, horiz_info, vertical_infos, tol=1e-3, min_area=1e-3, min_cut_gap=0.05, aspect_ratio=5):
    """
    用竖直面与当前水平面高度的交线切割水平面，得到分块区域。
    如果两条切割线距离很近，则忽略它们之间的区域（不输出该区域）。
    """
    # 1. 投影水平面到 XY，得到初始范围
    base_poly = project_faces_to_plane_polygon(mesh, horiz_info['faces'], plane="xy")
    if base_poly.is_empty or base_poly.area == 0:
        return []

    # 2. 收集所有竖直面与当前水平面高度的交线
    z_height = horiz_info['height']
    cut_lines = []
    for v in vertical_infos:
        lines = get_vertical_face_intersections(mesh, v, z_height)
        for seg in lines:
            cut_lines.append(seg)
    if not cut_lines:
        # 没有切割线，直接返回整个水平面
        z_sup = float(horiz_info["height"]) + SUPPORT_HEIGHT_Z_EPS
        return [{
            'polygon': base_poly,
            'centroid_xy': np.array([base_poly.centroid.x, base_poly.centroid.y]),
            'support_height': z_sup,
            'normal': horiz_info['normal']
        }]
    # 3. 使用容错 buffer 策略进行切割
    linestrings = []
    for seg in cut_lines:
        if np.linalg.norm(seg[0] - seg[1]) > 1e-8:
            line = LineString(seg)
            # 可延长5cm
            line_ext = extend_line(line, extension=0.05)
            linestrings.append(line_ext)
    multi_line = MultiLineString(linestrings)
    cut_buffer = multi_line.buffer(1e-3)  # 容差宽度可调
    # 4. 用 difference + polygonize 切割 base_poly
    diffed = base_poly.difference(cut_buffer)
    splitted = list(polygonize(diffed))

    # 过滤面积极小且距离切割线极近的区域，其余保留
    regions = []
    for poly in splitted:
        # 计算多边形的长和宽（包围盒的尺寸）
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        aspect = max(width, height) / (min(width, height) + 1e-8)
        min_edge = min(height, width)

        if min_edge < min_cut_gap and aspect > aspect_ratio:
            continue

        if poly.area < min_area:
            continue

        z_sup = float(horiz_info["height"]) + SUPPORT_HEIGHT_Z_EPS
        regions.append({
            'polygon': poly,
            'centroid_xy': np.array([poly.centroid.x, poly.centroid.y]),
            'support_height': z_sup,
            'normal': horiz_info['normal']
        })
    return regions

def segment_all_horizontal_surfaces(
    mesh, horizontal_list, vertical_list,
    clearance_thresh=0.05,       # 最终保留区域的最低 clearance（可调）
    same_slab_tol=0.02,          # 认为是“同一块板子”的最大高度差（m）
    same_slab_overlap=0.7,       # 若两个投影重合度 >= 此值且高度接近，视为同 slab
    eps=1e-3,                    # 投影微膨胀用的小量
    min_overlap=0.05,            # region 与上方面重叠判定阈值（用 region 面积为分母）
    only_top_in_same_slab=True,
    debug=False
):
    """
    改进逻辑：
    - 不在前期剔除任何水平面（保留所有表面用于分割）。
    - 建立 'same-slab' 分组（dz <= same_slab_tol && XY overlap >= same_slab_overlap）。
    - 对每个 'same-slab' 组只标记其 topmost 表面为该组的“有效阻挡者”。
    - 对每个 region 计算 clearance 时，只用这些 topmost 表面作为 candidates。
    """
    # 1. 复制水平面并规范化 normal（仅为了输出一致，不作为筛选依据）
    horiz_copy = []
    for h in horizontal_list:
        h2 = dict(h)
        n = np.array(h2.get('normal', [0,0,1]), dtype=float)
        if np.linalg.norm(n) > 1e-12:
            # 仅把法线“朝上化”方便展示（不作为过滤依据）
            if n[2] < 0:
                n = -n
            h2['normal'] = n / (np.linalg.norm(n) + 1e-12)
        horiz_copy.append(h2)

    # 丢弃无高度的水平面（不应出现；若 merge 后误分类为 slanted 会带 None，避免排序/compare 崩溃）
    horiz_copy = [h for h in horiz_copy if h.get("height") is not None]
    if not horiz_copy:
        return []

    # 2. 排序与投影（保留全部）
    horiz_sorted = sorted(horiz_copy, key=lambda x: float(x["height"]))
    heights = [h['height'] for h in horiz_sorted]
    h_polys = [project_faces_to_plane_polygon(mesh, h['faces'], plane="xy", eps=eps) for h in horiz_sorted]

    if debug:
        print("Horiz count:", len(horiz_sorted))
        for i,h in enumerate(horiz_sorted):
            pa = h_polys[i].area if (h_polys[i] is not None and not h_polys[i].is_empty) else 0.0
            print(f"  idx{i}: h={h['height']:.4f}, poly_area={pa:.6f}")

    # 3. 构建 same-slab 图（pairwise 检查）
    n = len(horiz_sorted)
    uf = UnionFind(n)
    for i in range(n):
        pi = h_polys[i]
        if pi is None or pi.is_empty:
            continue
        for j in range(i+1, n):
            pj = h_polys[j]
            if pj is None or pj.is_empty:
                continue
            dz = abs(heights[j] - heights[i])
            if dz > same_slab_tol:
                continue
            inter = pi.intersection(pj)
            if inter.is_empty:
                continue
            denom = max(min(pi.area, pj.area), 1e-12)
            overlap_ratio = inter.area / denom
            if overlap_ratio >= same_slab_overlap:
                uf.union(i, j)

    # 4. 得到每个组的 topmost 索引（代表该组的“阻挡面”）
    groups = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)
    top_indices = set()
    for root, members in groups.items():
        # 如果组只有一个元素也会返回该元素
        top = max(members, key=lambda k: heights[k])
        top_indices.add(top)

    if only_top_in_same_slab:
        base_indices_for_seg = sorted(list(top_indices))
    else:
        base_indices_for_seg = list(range(n))
    if debug:
        print("same-slab groups:", {r: groups[r] for r in groups})
        print("top_indices (used as blockers):", sorted(list(top_indices)))

    all_regions = []
    for i in base_indices_for_seg:
        h = horiz_sorted[i]
        regs = segment_horizontal_surface(mesh, h, vertical_list)
        if not (regs and h['height']):
            continue
        for reg in regs:
            geom = reg.get('polygon', None)
            if geom is None or geom.is_empty:
                geom = Point(reg['centroid_xy']).buffer(eps)
            z0 = h['height']
            candidates = []
            # 只用 same-slab 的 top 面作为“上方面”阻挡者（保持原来的逻辑）
            for j in range(n):
                if j not in top_indices:
                    continue
                if heights[j] <= z0 + 1e-12:
                    continue
                poly_above = h_polys[j]
                if poly_above is None or poly_above.is_empty:
                    continue
                inter = geom.buffer(eps).intersection(poly_above.buffer(eps))
                overlap = 0.0 if inter.is_empty else inter.area / max(geom.area, 1e-12)
                if overlap < min_overlap:
                    continue
                dz = heights[j] - z0
                candidates.append(dz)
            reg['clearance'] = min(candidates) if candidates else 1.0

        regs_filtered = [reg for reg in regs if reg['clearance'] >= clearance_thresh]
        all_regions.extend(regs_filtered)

    return all_regions

def visualize_support_regions_3d(mesh, support_regions, exp_name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 显示原始 mesh（半透明灰色）
    all_verts = [mesh.vertices[face] for face in mesh.faces]
    ax.add_collection3d(Poly3DCollection(all_verts, color=(0.8, 0.8, 0.8, 0.15), linewidths=0.10))

    for i, region in enumerate(support_regions):
        poly = region['polygon']
        z = region['support_height']
        clearance = region.get('clearance', 0.0)
        color = plt.cm.tab20(i % 20)

        polygons = []
        if isinstance(poly, Polygon):
            polygons = [poly]
        elif isinstance(poly, MultiPolygon):
            polygons = list(poly.geoms)
        else:
            continue  # 不支持的类型

        for sub_poly in polygons:
            coords = np.array(sub_poly.exterior.coords)

            # 平面本身
            verts3d = [np.column_stack((coords, np.full(len(coords), z)))]
            patch = Poly3DCollection(verts3d, color=color, alpha=0.30)
            ax.add_collection3d(patch)

            # clearance 区域（用同色更低透明度填充）
            if clearance > 0:
                verts3d_clearance = [
                    np.column_stack((coords, np.full(len(coords), z + clearance)))
                ]
                # 侧面
                for j in range(len(coords) - 1):
                    v1 = coords[j]
                    v2 = coords[j + 1]
                    side = np.array([
                        [v1[0], v1[1], z],
                        [v2[0], v2[1], z],
                        [v2[0], v2[1], z + clearance],
                        [v1[0], v1[1], z + clearance]
                    ])
                    ax.add_collection3d(Poly3DCollection([side], color=color, alpha=0.05, linewidths=1, edgecolor='none'))
                # 顶面
                patch_clearance = Poly3DCollection(verts3d_clearance, color=color, alpha=0.05, linewidths=1, edgecolor='none')
                ax.add_collection3d(patch_clearance)

        cx, cy = region['centroid_xy']
        ax.text(cx, cy, z, f"{i}", fontsize=12, ha='center', va='center',
                color='black', weight='bold',
                path_effects=[patheffects.withStroke(linewidth=4, foreground='white')], clip_on=False)

    ax.set_axis_off()

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
    for view, elev, azim in [("top", 90, -90), ("front", 0, -90), ("angled", 20, -60)]:
        ax.view_init(elev=elev, azim=azim)
        out_path = f"{exp_name}/support_regions_3d_{view}.png"
        plt.savefig(out_path, dpi=100)
        print(f"[Saved] {out_path}")
    plt.close()


def visualize_support_surfaces(mesh, support_surfaces, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 显示原始 mesh（半透明灰色）
    all_verts = [mesh.vertices[face] for face in mesh.faces]
    ax.add_collection3d(Poly3DCollection(all_verts, color=(0.7, 0.7, 0.7, 0.15), linewidths=0.2))

    # 显示支持面（不同颜色）
    cmap = cm.get_cmap("tab20")
    for i, surf in enumerate(support_surfaces):
        color = cmap(i % 20)
        verts_list = []
        for face_idx in surf['faces']:
            verts = mesh.vertices[mesh.faces[face_idx]]
            verts_list.append(verts)
        ax.add_collection3d(Poly3DCollection(verts_list, color=color, linewidths=0.5))

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
    plt.savefig(filename, dpi=300)

def load_mesh(filename):
    """
    加载 mesh 文件，支持多种格式。
    """
    mesh = trimesh.load(filename)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 1:
            mesh = list(mesh.geometry.values())[0]
        else:
            meshes = [g for g in mesh.geometry.values()]
            mesh = trimesh.util.concatenate(meshes)
    # 绕x轴旋转90度
    angle_rad = np.pi / 2  # 90度
    rot_matrix = R.from_euler('x', angle_rad).as_matrix()
    mesh.apply_transform(np.vstack([np.hstack([rot_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))
    return mesh

if __name__ == "__main__":
    # mesh = load_mesh("3D-FUTURE-model/0e011eab-116b-48ed-bff3-217dd74134ea/raw_model.obj")
    # ./3D-FUTURE-model/7ea1ab97-a7e1-40d2-894b-4ea10e46c26b/raw_model.obj
    mesh = load_mesh("../../GenesisVLM2/3D-FUTURE-model/67b7a6a8-2f40-4db9-9180-e1a5772e21f2/raw_model.obj")
    mesh.process() 
    # 对齐底部支持面到 (0,0,0)
    mesh = center_mesh_on_bottom_surface(mesh)

    support_h, support_v = extract_support_surfaces(mesh)

    for i, surf in enumerate(support_h):
        print(f"Horizontal surface {i}: {len(surf['faces'])} faces, center height = {surf['height']:.2f} m")

    for i, surf in enumerate(support_v):
        print(f"Vertical surface {i}: {len(surf['faces'])} faces, center height = {surf['height']:.2f} m")

    # 可视化支持面
    print("visulizing support surfaces...")
    visualize_support_surfaces(mesh, support_h, filename="support_h.png")
    visualize_support_surfaces(mesh, support_v, filename="support_v.png")
    support_regions = segment_all_horizontal_surfaces(mesh, support_h, support_v)
    
    print(f"Segmented support regions: {len(support_regions)}")
    os.makedirs("test", exist_ok=True)
    visualize_support_regions_3d(mesh, support_regions, "test")
