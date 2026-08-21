import os
import json
import faiss
import torch
import clip
import pickle
from PIL import Image
from tqdm import tqdm
import re
import trimesh
from pygltflib import GLTF2
from utils.tool import get_mesh_bbox_dimensions


device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load(name="/workspace/huangjialu/GraphLayout/clip_model/ViT-B-32.pt", device=device)  # Here should be the path to the local CLIP model
model, preprocess = clip.load(name="ViT-B/32", device=device)

# ========== Step 1: Encode all asset images and save ==========

def encode_assets(root_dir, img_dir, output_dir):
    embeddings = []
    glb_paths = []

    for sec_dir in os.listdir(root_dir):
        fname_list = os.listdir(os.path.join(root_dir, sec_dir))
        for fname_glb in tqdm(fname_list):
            fname = fname_glb.split(".")[0]
            # 可以增加新的视图，但俯视图可能会导致错误的结果，比如冰箱的俯视图可能被看作是电视。
            for view in ["front"]:
                img_path = os.path.join(img_dir, sec_dir, f"{fname}_{view}.png")
                if not os.path.exists(img_path):
                    continue
                try:
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model.encode_image(image).cpu()
                        emb /= emb.norm(dim=-1, keepdim=True)
                    embeddings.append(emb.squeeze(0))
                    glb_path = extract_name_from_path(img_path, "hssd")
                    glb_paths.append(glb_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    embeddings = torch.stack(embeddings, dim=0) # shape: (N_images, feature_dim)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(embeddings, f"{output_dir}/image_embeddings.pt")
    with open(f"{output_dir}/asset_filenames.pkl", "wb") as f:
        pickle.dump(glb_paths, f)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings.numpy())
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    print(f"[FAISS] Index built with {len(glb_paths)} image views.")


def encode_assets_3d_future(root_dir, output_dir):
    embeddings = []
    glb_paths = []

    # for sec_dir in tqdm(os.listdir(root_dir)):
    #     img_path = os.path.join(root_dir, sec_dir, "image.jpg")
    #     if not os.path.exists(img_path):
    #         continue
    #     try:
    #         image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             emb = model.encode_image(image).cpu()
    #             emb /= emb.norm(dim=-1, keepdim=True)
    #         embeddings.append(emb.squeeze(0))
    #         glb_path = extract_name_from_path(img_path, "3d_future")
    #         glb_paths.append(glb_path)
    #     except Exception as e:
    #         print(f"Error processing {img_path}: {e}")

    root_dir = root_dir.replace("3D-FUTURE-model", "test_asset_dir")
    for sec_dir in tqdm(os.listdir(root_dir)):
        img_path = os.path.join(root_dir, sec_dir, "blender_renders/render_90.0.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image).cpu()
                emb /= emb.norm(dim=-1, keepdim=True)
            glb_path = extract_name_from_path(img_path, "objaverse", id=sec_dir)
            glb = GLTF2().load(glb_path)
            if not glb.materials or not any(has_valid_material(m) for m in glb.materials):
                print(f"[Warning] Invalid material: {glb_path}")
                continue
            embeddings.append(emb.squeeze(0))
            glb_paths.append(glb_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    embeddings = torch.stack(embeddings, dim=0) # shape: (N_images, feature_dim)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(embeddings, f"{output_dir}/image_embeddings.pt")
    with open(f"{output_dir}/asset_filenames.pkl", "wb") as f:
        pickle.dump(glb_paths, f)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings.numpy())
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    print(f"[FAISS] Index built with {len(glb_paths)} image views.")


# ========== Step 2: Load everything ==========

def load_index_and_assets(cache_dir="cache"):
    embeddings = torch.load(os.path.join(cache_dir, "image_embeddings.pt"))
    with open(os.path.join(cache_dir, "asset_filenames.pkl"), "rb") as f:
        filenames = pickle.load(f)
    index = faiss.read_index(os.path.join(cache_dir, "faiss.index"))
    return embeddings, filenames, index


# ========== Step 3: Parse region JSON and form queries ==========

def _resolve_bound_key(bound, region_id):
    if bound is None:
        return None
    if region_id in bound:
        return region_id
    if isinstance(region_id, str) and region_id.isdigit():
        alt = int(region_id)
        if alt in bound:
            return alt
    if isinstance(region_id, int):
        alt = str(region_id)
        if alt in bound:
            return alt
    return None


def extract_queries_from_json(data):
    queries = []
    for region in data["regions"]:
        items = region.get("item")
        if not isinstance(items, dict):
            continue
        for label, item_info in items.items():
            if not isinstance(item_info, dict):
                continue
            if label == "standing book" or label == "flat book":
                query_label = "book"
            else:
                query_label = label
            query = f"{query_label}."
            queries.append({
                "id": region["id"],
                "item": label,
                "count": item_info["count"],
                "query": query,
                "z-axis": item_info.get("z-axis", item_info.get("vertical", False)),
                "center": item_info.get("center", False),
                "vertical": item_info.get("z-axis", item_info.get("vertical", False)),
                "dimensions": item_info["dimensions"],
            })
    return queries

def extract_queries_on_floor(region_data, assets_path=None, skip_rugs=False, skip_wall_mounts=False):
    from utils.task_assets import is_rug_label, is_wall_mount_label, resolve_path_for_label

    queries = []
    for region_idx, region in enumerate(region_data["areas"]):
        objs = region.get("objects") or region.get("floor_objects") or []
        for obj in objs:
            label = obj['name']
            if skip_rugs and is_rug_label(label):
                continue
            if skip_wall_mounts and is_wall_mount_label(label):
                continue
            query_label = obj['description']
            query = f"{query_label}"
            entry = {
                "id": str(region_idx),
                "item": label,
                "count": obj["amount"],
                "query": query,
                "dimensions": obj["dimensions"],
                "vertical": obj.get("vertical", False),
                "layer": obj.get("layer"),
                "location": obj.get("location"),
            }
            if assets_path:
                entry["path"] = resolve_path_for_label(label, assets_path)
            queries.append(entry)

    return queries


def _result_entry_from_fixed_path(q, force_unit_scale: bool = False, mesh_x_extent_max=None):
    """Resolve a single query to its pre-assigned task asset path (no CLIP/FAISS)."""
    from utils.task_assets import resolve_mesh_path

    matched_in_bbox = []
    scales = []
    mesh_path = resolve_mesh_path(q.get("path")) if q.get("path") else None
    if mesh_path and os.path.isfile(mesh_path):
        bbox = get_mesh_bbox_dimensions(mesh_path, q.get("vertical", False))
        if bbox:
            if mesh_x_extent_max is not None and bbox[0] > float(mesh_x_extent_max):
                return {
                    "id": q["id"],
                    "item": q["item"],
                    "matches": [],
                    "count": q["count"],
                    "scale": [],
                    "z-axis": q.get("z-axis", q.get("vertical", False)),
                    "center": q.get("center", False),
                    "vertical": q.get("vertical", False),
                    "layer": q.get("layer"),
                    "location": q.get("location"),
                    "dimensions": q.get("dimensions"),
                }
            length, width, h = bbox
            scale = 1.0
            if not force_unit_scale:
                dims = q.get("dimensions") or []
                if isinstance(dims, (list, tuple)) and len(dims) >= 2 and max(length, width) > 1e-8:
                    scale = min(max(dims[1], dims[0]) / max(length, width), 1.0)
            matched_in_bbox.append((mesh_path, 1.0, (length, width, h), scale))
            scales.append(scale)
    return {
        "id": q["id"],
        "item": q["item"],
        "matches": matched_in_bbox,
        "count": q["count"],
        "scale": scales,
        "z-axis": q.get("z-axis", q.get("vertical", False)),
        "center": q.get("center", False),
        "vertical": q.get("vertical", False),
        "layer": q.get("layer"),
        "location": q.get("location"),
        "dimensions": q.get("dimensions"),
    }


def assign_assets_from_task_paths(queries, *, force_unit_scale: bool = True):
    """
    Use task JSON asset paths directly (no FAISS index / CLIP search).
    """
    results = []
    for q in queries:
        entry = _result_entry_from_fixed_path(q, force_unit_scale=force_unit_scale)
        if not entry["matches"]:
            print(
                f"[AssetMatch] missing task path: region={q['id']} "
                f"item={q.get('item')!r} path={q.get('path')!r}"
            )
        results.append(entry)
    return results


# ========== Step 4: Match query to assets using FAISS ==========
def match_text_queries(
    queries,
    bound,
    index,
    filenames,
    task,
    top_k=3,
    force_unit_scale=False,
    mesh_x_extent_max=None,
):
    """
    Match text queries to mesh assets.

    ``mesh_x_extent_max``: if set (for example 0.2 for wall-mounted assets),
    skip candidates whose raw mesh X bbox extent exceeds this value.
    """
    from utils.task_assets import resolve_glb_path, resolve_mesh_path

    results = []
    scale_thr = [0.95, 1.0, 0.95]
    rug_fixed_id = "49b464ed824340ad8004396212171c13"
    rug_candidates = [
        f"./test_asset_dir/{rug_fixed_id}/{rug_fixed_id}.glb",
        f"./test_asset_dir/{rug_fixed_id}",
    ]
    for q in queries:
        if q.get("path"):
            mesh_path = resolve_mesh_path(q.get("path"))
            if mesh_path and os.path.isfile(mesh_path):
                results.append(
                    _result_entry_from_fixed_path(
                        q,
                        force_unit_scale=force_unit_scale,
                        mesh_x_extent_max=mesh_x_extent_max,
                    )
                )
                continue

        layer = str(q.get("layer") or "").lower()
        if layer == "rug":
            matched = [(p, 1.0) for p in rug_candidates if os.path.exists(p)]
            if not matched:
                matched = [(rug_candidates[0], 1.0)]
        else:
            text = clip.tokenize([q["query"]]).to(device)
            with torch.no_grad():
                text_emb = model.encode_text(text).cpu()
                text_emb /= text_emb.norm(dim=-1, keepdim=True)

            search_k = max(top_k, 10) if mesh_x_extent_max is not None else top_k
            D, I = index.search(text_emb.numpy(), search_k)
            matched = [(filenames[i], float(D[0][j])) for j, i in enumerate(I[0])]
            if layer == "ceiling":
                keywords = ("painting", "picture", "frame", "art", "wall_art", "wall-art", "wallart")
                matched = [m for m in matched if any(k in m[0].lower() for k in keywords)]
        matched_in_bbox = []
        scales = []
        if bound is None:
            fixed_path = resolve_glb_path(q.get("path")) if q.get("path") else None
            if fixed_path and os.path.isfile(fixed_path):
                length, width, h = get_mesh_bbox_dimensions(fixed_path, q["vertical"])
                if length is not None:
                    scale = 1.0 if force_unit_scale else 1.0
                    matched_in_bbox.append((fixed_path, 1.0, (length, width, h), scale))
            if not matched_in_bbox:
                for glb_path, score in matched:
                    glb_path = resolve_glb_path(glb_path) or glb_path
                    length, width, h = get_mesh_bbox_dimensions(glb_path, q["vertical"])
                    if mesh_x_extent_max is not None and length > float(mesh_x_extent_max):
                        continue
                    mses = -(
                        abs(length - q['dimensions'][1])+
                        abs(width - q['dimensions'][0])+ 
                        abs(h - q['dimensions'][2])* 0.5
                    )
                    score = score * 0.75 + mses * 0.25  # Penalize based on size mismatch
                    scale = 1.0
                    if not force_unit_scale and max(length, width) > 1e-8:
                        scale = min(max(q['dimensions'][1], q['dimensions'][0]) / max(length, width), 1.0)
                    matched_in_bbox.append((glb_path, score, (length, width, h), scale))
            scales = [m[3] for m in matched_in_bbox]
        else:
            bound_key = _resolve_bound_key(bound, q["id"])
            if bound_key is None:
                print(
                    f"[Warning] No bound for region id={q['id']} item={q.get('item')!r}; skipping."
                )
                results.append({
                    "id": q["id"],
                    "item": q["item"],
                    "matches": [],
                    "count": q["count"],
                    "scale": [],
                    "z-axis": q.get("z-axis", q.get("vertical", False)),
                    "center": q.get("center", False),
                    "vertical": q.get("vertical", False),
                    "layer": q.get("layer"),
                    "location": q.get("location"),
                    "dimensions": q.get("dimensions"),
                })
                continue
            max_allowed = [a * b for a, b in zip(bound[bound_key], scale_thr)]
            max_extended = [1.3 * b for b in max_allowed]
            is_rug = str(q.get("layer") or "").lower() == "rug"
            dims_q = q.get("dimensions") or []
            scales = []
            for glb_path, score in matched:
                glb_path = resolve_glb_path(glb_path) or glb_path
                bbox = get_mesh_bbox_dimensions(glb_path, q["vertical"])
                if not bbox:
                    continue
                if mesh_x_extent_max is not None and bbox[0] > float(mesh_x_extent_max):
                    continue
                x_ok = bbox[0] < max_allowed[0]
                y_ok = bbox[1] < max_allowed[1]
                z_ok = bbox[2] < max_allowed[2]
                if x_ok and y_ok and z_ok:
                    scale = 1.0
                elif bbox[0] <= max_extended[0] and bbox[1] <= max_extended[1] and bbox[2] <= max_extended[2]:
                    lwh = [bbox[0], bbox[1], bbox[2]]
                    scale = min([max_allowed[i] / lwh[i] * scale_thr[i] for i in range(3)])
                else:
                    continue

                if force_unit_scale:
                    scale = 1.0

                # Optional: allow rugs to scale up to meet target xy dimensions (still capped by region bounds).
                if (
                    is_rug
                    and isinstance(dims_q, (list, tuple))
                    and len(dims_q) >= 2
                    and bbox[0] > 1e-8
                    and bbox[1] > 1e-8
                ):
                    l_d, w_d = max(dims_q[0], dims_q[1]), min(dims_q[0], dims_q[1])
                    length_s, width_s = bbox[0] * scale, bbox[1] * scale
                    l_n, w_n = max(length_s, width_s), min(length_s, width_s)
                    if l_n > 1e-8 and w_n > 1e-8:
                        up = max(l_d / l_n, w_d / w_n)
                        if up > 1.0:
                            try:
                                max_scale_region = min(
                                    max_allowed[i] / bbox[i]
                                    for i in range(3)
                                    if bbox[i] > 1e-8
                                )
                            except Exception:
                                max_scale_region = scale * up
                            scale = min(scale * up, max_scale_region)

                # Non-rug: never enlarge beyond the chosen fit scale; optionally shrink toward target size.
                if (not is_rug) and isinstance(dims_q, (list, tuple)) and len(dims_q) >= 2:
                    l_d, w_d = max(dims_q[0], dims_q[1]), min(dims_q[0], dims_q[1])
                    length_s, width_s = bbox[0] * scale, bbox[1] * scale
                    l_n, w_n = max(length_s, width_s), min(length_s, width_s)
                    if l_n > 1e-8:
                        scale = min(l_d / l_n, scale)

                length, width, height = bbox[0] * scale, bbox[1] * scale, bbox[2] * scale
                if isinstance(dims_q, (list, tuple)) and len(dims_q) >= 2:
                    l_d, w_d = max(dims_q[0], dims_q[1]), min(dims_q[0], dims_q[1])
                else:
                    l_d, w_d = 0.0, 0.0
                l_n, w_n = max(length, width), min(length, width)
                mses = -(
                    abs(l_n - l_d)+
                    abs(w_n - w_d)+ 
                    abs(height - (dims_q[2] if len(dims_q) > 2 else height)) * 0.8
                )
                if is_rug:
                    score = score
                else:
                    score = score * 0.75 + mses * 0.25 # Penalize based on size mismatch
                matched_in_bbox.append((glb_path, score, (length, width, height), scale))

            matched_in_bbox.sort(key=lambda x: x[1], reverse=True)
            scales = [m[3] for m in matched_in_bbox]
                
        results.append({
            "id": q["id"],
            "item": q["item"],
            "matches": matched_in_bbox,
            "count": q["count"],
            "scale": scales,
            "z-axis": q.get("z-axis", q.get("vertical", False)),
            "center": q.get("center", False),
            "vertical": q.get("vertical", False),
            "layer": q.get("layer"),
            "location": q.get("location"),
            "dimensions": q.get("dimensions"),
        })

    return results


def print_results(results):
    for r in results:
        print(f"\nRegion {r['id']} [{r['item']}]:")
        for fname, score in r["matches"]:
            print(f"  → {fname} (score={score:.4f})")


def filter_low_score(results, threshold=0.25):
    low_confidence_regions = []
    for r in results:
        top_score = r["matches"][0][1] if r["matches"] else 0.0
        if top_score < threshold:
            print(f"[!] Region {r['id']} (item: {r['item']}) has low confidence match: {top_score:.4f}")
            low_confidence_regions.append({
                "id": r["id"],
                "item": r["item"],
                "reason": f"Top CLIP similarity score {top_score:.4f} < threshold {threshold}"
            })
    return low_confidence_regions

def extract_name_from_path(path, dataset, id=None):
    """
    从路径中提取形如 Book_25_front.png → Book_25
    """
    if not path:
        return path
    path_str = str(path)
    low = path_str.lower()
    if low.endswith((".glb", ".obj", ".gltf")):
        return path_str
    if dataset == "ai2thorhub":
        glb_path = path.replace('_front.png', '').replace('_top.png', '').replace('.png', '') + '.glb'
        glb_path = glb_path.replace("render", "ai2thorhab-uncompressed/assets")
    elif dataset == "hssd":
        glb_path = path.replace('_front.png', '').replace('_top.png', '').replace('.png', '') + '.glb'
        glb_path = glb_path.replace("hssd_render", "hssd-models")
    elif dataset == "3d_future":
        glb_path = path.replace("image.jpg", "raw_model.obj")
    elif dataset == "objaverse":
        glb_path = path.replace("blender_renders/render_90.0.jpg", f"{id}.glb")
    return glb_path

def has_valid_material(mat):
    # 缺贴图或是纯黑都视为“无材质”
    if mat is None:
        return False

    pbr = mat.pbrMetallicRoughness
    has_color_tex = pbr.baseColorTexture is not None
    has_nonblack_color = (
        any(c > 0.05 for c in pbr.baseColorFactor[:3])
        if pbr.baseColorFactor else False
    )
    has_normal = mat.normalTexture is not None

    # 有颜色贴图、或颜色不是纯黑、或有法线贴图 -> 有效
    return has_color_tex or has_nonblack_color or has_normal




if __name__ == "__main__":
    root_dir = "3D-FUTURE-model"
    encode_dir = "assets_feature"
    dataset = "3d_future"
    objects_in_areas = {
"areas": [
{
    "area_name": "Sleeping Area",
    "objects": [
    {
        "name": "fluffy carpet",
        "description": "rug",
        "dimensions": [2.6, 2.7, 0.04],
        "amount": 1
    }]
}]
}
    embeddings, filenames, index = load_index_and_assets(encode_dir)
    queries = extract_queries_on_floor(objects_in_areas)
    # if no region bound, set bound to None
    results = match_text_queries(queries, None, index, filenames, dataset, top_k=10)

    print(results)
