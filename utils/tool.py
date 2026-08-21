from dataclasses import dataclass
import ast
import trimesh
import re
import torch
from typing import Any, Optional

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float
    type: str = "point"

    def __post_init__(self):
        object.__setattr__(self, 'pos', torch.tensor([self.x, self.y, self.z], dtype=torch.float32))

def _normalize_asset_key_token(token: str) -> str:
    """Normalize GPT asset key variants to underscore-index form."""
    token = re.sub(r"-(\d+)\b", r"_\1", token)
    token = re.sub(r"_row(\d+)\b", r"_\1", token)
    return token


def normalize_dsl_fixed_points(code: str) -> str:
    """
    Rewrite legacy GPT fixed-point dict assignments to Point(x, y, z).
    e.g. central_point = {'pos': [4.0, 5.5, 0.0], 'description': 'central row 1'}
      -> central_point = Point(4.0, 5.5, 0.0)
    """
    dict_pattern = re.compile(
        r"(?P<lhs>\b[a-zA-Z_]\w*\s*=\s*)"
        r"\{\s*['\"]pos['\"]\s*:\s*"
        r"\[\s*(?P<x>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*,\s*(?P<y>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*,\s*(?P<z>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*\]\s*"
        r"(?:,\s*['\"]description['\"]\s*:\s*['\"][^'\"]*['\"])?"
        r"\s*\}",
        re.MULTILINE,
    )

    def _repl(match: re.Match) -> str:
        return (
            f"{match.group('lhs')}Point("
            f"{match.group('x')}, {match.group('y')}, {match.group('z')})"
        )

    return dict_pattern.sub(_repl, code)


def normalize_dsl_asset_keys(code: str) -> str:
    """
    Rewrite common GPT key mistakes before AST parsing.
    e.g. wall_display-0 -> wall_display_0, conference_chair_row0 -> conference_chair_0
    """
    def _rewrite(match: re.Match) -> str:
        return _normalize_asset_key_token(match.group(0))

    # Only rewrite identifier-like tokens, not numeric literals.
    return re.sub(r"\b[a-zA-Z_][\w]*(?:-(?:\d+)|_row\d+)\b", _rewrite, code)


def build_dsl_obj_env(assets) -> dict[str, dict]:
    """Build ``obj`` lookup for DSL expressions like ``obj['bed_0']['bbox'][1]``."""
    obj: dict[str, dict] = {}
    if not assets:
        return obj
    for name, asset in assets.items():
        bbox = getattr(asset, "bbox", None)
        if bbox is not None:
            if hasattr(bbox, "detach"):
                bbox_vals = bbox.detach().cpu().tolist()
            elif hasattr(bbox, "tolist"):
                bbox_vals = bbox.tolist()
            else:
                bbox_vals = list(bbox)
        else:
            bbox_vals = [0.0, 0.0, 0.0]
        obj[name] = {"bbox": [float(v) for v in bbox_vals]}
    return obj


def _parse_subscript_slice(slice_node: ast.AST, env: Optional[dict[str, Any]]):
    if isinstance(slice_node, ast.Constant):
        return slice_node.value
    if isinstance(slice_node, ast.Index):
        return parse_ast_value(slice_node.value, env)
    if isinstance(slice_node, ast.Slice):
        raise ValueError("Slice subscripts are not supported in DSL")
    return parse_ast_value(slice_node, env)


def _coerce_subscript_value(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except (TypeError, ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return value


def _resolve_subscript(base: Any, key: Any) -> Any:
    key = _coerce_subscript_value(key)
    if isinstance(base, dict):
        if key not in base:
            raise ValueError(f"DSL subscript key {key!r} not found")
        return _coerce_subscript_value(base[key])
    if isinstance(base, (list, tuple)):
        idx = int(key)
        return _coerce_subscript_value(base[idx])
    raise ValueError(f"Unsupported subscript base type {type(base)!r} for key {key!r}")


def parse_ast_value(node: ast.AST, env: Optional[dict[str, Any]] = None):
    """
    Parse a restricted subset of Python AST nodes used by our DSL.

    `env` holds previously-defined variables (e.g. `central_point = Point(...)`), so a
    later reference like `distance('sofa_0', central_point, ...)` resolves to a `Point`
    instead of the string `"central_point"`.
    """

    if isinstance(node, ast.Constant):
        return node.value
    
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = parse_ast_value(node.operand, env)
        # handle negative number
        if isinstance(val, (int, float)):
            return -val
        raise ValueError("Unary minus only supported for numbers")

    elif isinstance(node, ast.Name):
        # Variable (preferred) or asset reference.
        if env is not None and node.id in env:
            return env[node.id]
        return _normalize_asset_key_token(node.id)

    elif isinstance(node, ast.BinOp):
        left = parse_ast_value(node.left, env)
        right = parse_ast_value(node.right, env)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        # Legacy: ``asset_key - N`` treated as indexed asset name variant.
        if isinstance(node.op, ast.Sub) and isinstance(left, str) and isinstance(right, (int, float)):
            suffix = int(right)
            return _normalize_asset_key_token(f"{left}_{suffix}")
        raise ValueError(f"Unsupported arithmetic in DSL: {ast.dump(node)}")

    elif isinstance(node, ast.Subscript):
        base = parse_ast_value(node.value, env)
        key = _parse_subscript_slice(node.slice, env)
        return _resolve_subscript(base, key)

    elif isinstance(node, ast.List):
        return [parse_ast_value(e, env) for e in node.elts]

    elif isinstance(node, ast.Dict):
        keys = [parse_ast_value(k, env) for k in node.keys]
        values = [parse_ast_value(v, env) for v in node.values]
        mapping = dict(zip(keys, values))
        pos = mapping.get("pos")
        if isinstance(pos, list) and len(pos) == 3:
            x, y, z = pos
            return Point(float(x), float(y), float(z))
        raise ValueError(f"Unsupported dict in DSL: {ast.dump(node)}")

    elif isinstance(node, ast.Call):
        # Handle Point(x, y, z)
        if isinstance(node.func, ast.Name) and node.func.id == "Point":
            vals = [parse_ast_value(a, env) for a in node.args]
            return Point(*vals)

        raise ValueError(f"Unsupported call: {ast.dump(node)}")
    else:
        raise ValueError(f"Unsupported DSL element: {ast.dump(node)}")

def extract_name_from_path(path, dataset):
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
    return glb_path

def get_mesh_bbox_dimensions(glb_path, vertical, scale=1.0):
    """
    Given a GLB file path, return the dx, dy, and dz values of its axis-aligned bounding box.
    """
    def _get_bbox(dims, vertical):
        if vertical:
            return [dims[0], dims[2], dims[1]]
        else:
            return [dims[1], dims[0], dims[2]]
    try:
        scene = trimesh.load(glb_path, force='scene')
        mesh = scene.dump(concatenate=True)
        mesh.apply_scale(scale)
        bbox = mesh.bounds  # shape: (2, 3)
        dx, dy, dz = bbox[1] - bbox[0]
        
        # trimesh: X (right), Y (up), Z (depth)
        # genesis: X (right), Y (depth), Z (up)
        length = dx
        width = dz   # depth -> Genesis.y
        height = dy  # up -> Genesis.z
        dims = [length, width, height]
        return _get_bbox(dims, vertical)
    except Exception as e:
        print(f"[Error] Failed to load {glb_path}: {e}")
        return None
    
def parse_dsl_fixed_points(code: str, env: Optional[dict] = None) -> dict[str, Point]:
    """Collect ``name -> Point(...)`` assignments from constraint DSL."""
    if not code or not code.strip():
        return {}
    env = dict(env or {})
    points: dict[str, Point] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return points
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                val = parse_ast_value(node.value, env)
                if isinstance(val, Point):
                    env[target.id] = val
                    points[target.id] = val
        elif isinstance(node, ast.AnnAssign):
            if node.value is None or not isinstance(node.target, ast.Name):
                continue
            val = parse_ast_value(node.value, env)
            if isinstance(val, Point):
                env[node.target.id] = val
                points[node.target.id] = val
    return points


def clean_pattern(all_constraints: str) -> str:
    code_text_match = re.search(r"```DSL\n(.+?)```", all_constraints, re.DOTALL)
    if not code_text_match:
        code_text = ""
    else:
        code_text = code_text_match.group(1) 
    clean_patterns = [r'^\s*#.*$', r'#.*$', r'^\s*$\n?', r'[ \t]+$']
    for pattern in clean_patterns:
        flags = re.MULTILINE if '^' in pattern else 0
        code_text = re.sub(pattern, '', code_text, flags=flags)
    code_text = normalize_dsl_fixed_points(code_text.strip())
    code_text = normalize_dsl_asset_keys(code_text)
    return code_text

if __name__ == "__main__":
    with open(f"results_areas/dining_room_0/constraints_0.txt", "r") as f:
        all_constraints = f.read()
    code_text = clean_pattern(all_constraints)
    print("Cleaned DSL Code:")
    print(code_text)
