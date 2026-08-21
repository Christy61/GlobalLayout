import numpy as np
import math
import json
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.node import Wall
from utils.tool import Point, parse_ast_value, clean_pattern, build_dsl_obj_env
from utils.gpt import GPT
from utils.graph_conflict import detect_scene_graph_conflicts
import ast
import torch
# from scipy.optimize import linprog
import numpy as np
import math


# --------------------------
# Object State
# --------------------------
@dataclass
class ObjectState:
    name: str
    bbox: np.ndarray = field(default_factory=lambda: np.ones(3))
    region: Optional[str] = None

# --------------------------
# Wall State
# --------------------------
@dataclass
class WallState:
    name: str
    pos: np.ndarray
    rot: np.ndarray
    length: float
    normal: np.ndarray = field(init=False)

    def __post_init__(self):
        self.normal = np.array([math.cos(self.rot.item()), math.sin(self.rot.item())], dtype=np.float32)

# --------------------------
# Relation abstraction
# --------------------------
@dataclass(frozen=True)
class Relation:
    name: str
    src: str
    dst: str
    params: Dict[str, Any]
    region: Optional[str] = None


# --------------------------
# Node & Edge Graph
# --------------------------
class SceneGraph:
    """
    SceneGraph with node-center and edge-center graphs.
    Handles constraints, centrality, and group processing.
    """

    def __init__(self):
        self.relations: List[Relation] = []  # list of Relation objects
        self.nodes: Dict[str, ObjectState] = {}  # node_name -> ObjectState, without walls
        self.walls: Dict[str, WallState] = {}
        self.edges: List[Relation] = []  # list of Relation objects
        self.node_center = nx.MultiDiGraph()  # graph over nodes
        self.edge_center = nx.MultiDiGraph()  # graph over relations/edges
        self.room_bound = [] # room bound (4, 1)
        self.groups = []  # list of node lists
        self.centrality = {}  # normalized centrality per node
        self.center_nodes = {}  # group index -> center node
        self.fixed_points: Dict[str, Point] = {}  # DSL anchor names -> Point

    def __str__(self):
        """
        Returns a string representation of the SceneGraph.
        """
        lines = ["SceneGraph:"]
        
        # Print nodes
        lines.append("\n  Nodes:")
        for name, obj in self.nodes.items():
            lines.append(f"    - {name}")
            lines.append(f"      bbox: {obj.bbox}")
        
        # Print edges
        lines.append("\n  Edges:")
        for rel in self.edges:
            lines.append(f"    - {rel.name}({rel.src} -> {rel.dst})")
            if rel.params:
                lines.append(f"      params: {rel.params}")
        
        # Print groups if available
        if self.groups:
            lines.append("\n  Groups:")
            for i, group in enumerate(self.groups):
                lines.append(f"    - Group {i}: {list(group)}")
        
        # Print centrality if available
        if self.centrality:
            lines.append("\n  Centrality:")
            for node, cent in self.centrality.items():
                lines.append(f"    - {node}: {cent:.4f}")
        
        return "\n".join(lines)

    def compute_depths(self, against_threshold: float = 0.1, height_bucket: float = 0.1) -> Dict[str, int]:
        """
        Compute depth for each object based on relations.
        Rules:
        - default depth = 0
        - on(a, b): depth[a] = depth[b] + 1
        - against height > threshold: all such objects share one layer
        - above: layer by height (bucketed)
        """
        nodes = list(self.nodes.keys())
        depth: Dict[str, int] = {n: 0 for n in nodes}

        def _bucket(h: float) -> float:
            if height_bucket <= 0:
                return h
            return round(float(h) / height_bucket) * height_bucket

        height_layers: Dict[str, float] = {}
        against_nodes = set()
        against_heights = []

        for rel in self.edges:
            if rel.name == "against":
                h = float(rel.params.get("arg0", 0.0))
                if h > against_threshold:
                    against_nodes.add(rel.src)
                    against_heights.append(h)
            elif rel.name == "above":
                h = float(rel.params.get("arg0", 0.0))
                if rel.src in height_layers:
                    height_layers[rel.src] = max(height_layers[rel.src], h)
                else:
                    height_layers[rel.src] = h

        against_layer_height = None
        if against_heights:
            against_layer_height = min(against_heights)

        height_set = set()
        for h in height_layers.values():
            height_set.add(_bucket(h))
        if against_layer_height is not None:
            height_set.add(_bucket(against_layer_height))

        sorted_heights = sorted(height_set)
        height_to_depth = {h: i + 1 for i, h in enumerate(sorted_heights)}

        for n in nodes:
            if n in against_nodes and against_layer_height is not None:
                depth[n] = max(depth[n], height_to_depth[_bucket(against_layer_height)])
            if n in height_layers:
                depth[n] = max(depth[n], height_to_depth[_bucket(height_layers[n])])

        changed = True
        while changed:
            changed = False
            for rel in self.edges:
                if rel.name != "on":
                    continue
                if rel.src not in depth or rel.dst not in depth:
                    continue
                new_depth = depth[rel.dst] + 1
                if new_depth > depth[rel.src]:
                    depth[rel.src] = new_depth
                    changed = True

        return depth


    def __repr__(self):
        return self.__str__()


    def add_node(self, node_name: str, bbox: torch.Tensor, region_idx: Optional[str] = None):
        self.nodes[node_name] = ObjectState(node_name, bbox=np.array(bbox), region=region_idx)
        self.node_center.add_node(node_name)


    def add_walls(self, walls: dict[str, Wall], room_bound: tuple):
        for name, wall in walls.items():
            self.walls[name] = WallState(name=name, pos=np.array(wall.pos), rot=np.array(wall.rot), length=wall.length)
        self.room_bound = room_bound
        if not hasattr(self, "forbidden_polygons"):
            self.forbidden_polygons = []

    def add_forbidden_polygon(self, poly: list[tuple[float, float]]):
        if not hasattr(self, "forbidden_polygons"):
            self.forbidden_polygons = []
        if poly:
            self.forbidden_polygons.append(poly)


    def add_edge(self, relation: Relation):
        if relation in self.edges:
            return
        if relation.name == "surround":
            self.edges.append(relation)
            for src in relation.src:
                self.edge_center.add_edge(src, relation.dst, relation=relation)
        else:
            self.edges.append(relation)
            self.edge_center.add_edge(relation.src, relation.dst, relation=relation)


    # Tools: transform graph to origin dsl
    def _format_dsl_value(self, value: Any, *, as_identifier: bool = False):

        if as_identifier and isinstance(value, str):
            return value
        
        if isinstance(value, str):
            return f"\"{value}\""
        
        if isinstance(value, bool):
            return "True" if value else "False"
        
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        
        if isinstance(value, (float, np.floating)):
            return repr(float(value))
        
        if isinstance(value, Point):
            return f"Point({value.x}, {value.y}, {value.z})"
        
        if isinstance(value, (list, tuple)):
            inner = ", ".join(self._format_dsl_value(v, as_identifier=as_identifier) for v in value)
            return f"[{inner}]"
        
        return repr(value)

    def _relation_to_dsl(self, rel: Relation):

        if rel.name == "surround":
            src_list = self._format_dsl_value(rel.src, as_identifier=True)
            dst = self._format_dsl_value(rel.dst, as_identifier=True)
            distance = rel.params.get("distance")
            if distance is None:
                distance = rel.params.get("arg0")
            args = [src_list, dst, self._format_dsl_value(distance)]
            extra_keys = [k for k in rel.params.keys() if k not in {"angles", "distance", "arg0"}]
            extra = [f"{k}={self._format_dsl_value(rel.params[k])}" for k in sorted(extra_keys)]
            args_str = ", ".join(args + extra)
            return f"{rel.name}({args_str})"

        src = self._format_dsl_value(rel.src, as_identifier=True)
        dst = self._format_dsl_value(rel.dst, as_identifier=True)
        args = [src, dst]
        i = 0

        while f"arg{i}" in rel.params:
            args.append(self._format_dsl_value(rel.params[f"arg{i}"]))
            i += 1

        extra_keys = [k for k in rel.params.keys() if not k.startswith("arg")]
        extra = [f"{k}={self._format_dsl_value(rel.params[k])}" for k in sorted(extra_keys)]
        args_str = ", ".join(args + extra)

        return f"{rel.name}({args_str})"


    def _find_relation_from_conflict(self, conflict_rel: Any):
        if isinstance(conflict_rel, Relation):
            return conflict_rel
        return None


    def _remove_edge_center_relation(self, src: str, dst: str, relation: Relation):
        edge_data = self.edge_center.get_edge_data(src, dst, default={})
        for key, data in list(edge_data.items()):
            if data.get("relation") == relation:
                self.edge_center.remove_edge(src, dst, key=key)

    
    def get_all_relations(
        self,
        dsl_code: str,
        region_idx: Optional[str] = None,
        assets: Optional[dict] = None,
    ) -> List[Relation]:
        """
        Parse LLM-generated DSL into a list of Relation objects.
        """
        tree = ast.parse(dsl_code)
        env: dict[str, Any] = {}
        if assets:
            env["obj"] = build_dsl_obj_env(assets)
        relations = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                value = parse_ast_value(node.value, env)
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        raise ValueError("Only simple assignments are allowed (e.g. name = Point(...))")
                    env[target.id] = value
                    if isinstance(value, Point):
                        self.fixed_points[target.id] = value
                continue

            if isinstance(node, ast.AnnAssign):
                if node.value is None or not isinstance(node.target, ast.Name):
                    raise ValueError("Only simple annotated assignments are allowed (e.g. name: Point = Point(...))")
                value = parse_ast_value(node.value, env)
                env[node.target.id] = value
                if isinstance(value, Point):
                    self.fixed_points[node.target.id] = value
                continue

            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue

            call = node.value
            rel = self._parse_call_to_relation(call, region_idx, env)
            relations.append(rel)
        self.relations.extend(relations)

        return relations

    def add_relations_from_dsl(
        self,
        dsl_code: str,
        region_idx: Optional[str] = None,
        assets: Optional[dict] = None,
    ):
        for rel in self.get_all_relations(dsl_code, region_idx, assets=assets):
            self.add_edge(rel)


    def _parse_call_to_relation(
        self,
        call: ast.Call,
        region_idx: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> Relation:
        # -------- function name --------
        if not isinstance(call.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")

        name = call.func.id
        # -------- positional arguments --------
        args = [parse_ast_value(a, env) for a in call.args]

        # -------- keyword arguments --------
        kwargs = {
            kw.arg: parse_ast_value(kw.value, env)
            for kw in call.keywords
        }

        # -------- convention-based mapping --------
        if name == "surround":
            # surround(src_list, dst, distance)
            src, dst, distance = args
            params = {
                "distance": distance,
                **kwargs
            }
            return Relation(name=name, src=src, dst=dst, params=params, region=region_idx)

        else:
            # only handle binary relation: fn(src, dst, param0, param1, ...)
            if len(args) < 2:
                raise ValueError(f"{name} requires at least src and dst")

            src = args[0]
            dst = args[1]

            # remaining args go into params by position
            params = {}
            for i, v in enumerate(args[2:]):
                params[f"arg{i}"] = v

            params.update(kwargs)

            return Relation(name=name, src=src, dst=dst, params=params, region=region_idx)


    # -----------------------------
    # Constraint propagation
    # Group assets 
    # -----------------------------
    def group_assets(self):
        """Group connected components for EA / conflict detection."""
        G_undirected = self.edge_center.to_undirected().copy()
        groups = list(nx.connected_components(G_undirected))
        self.groups = groups

    def group_assets_for_region(
        self,
        active_srcs: set,
        existing_srcs: Optional[set] = None,
    ) -> List[set]:
        """
        Ablation-aligned grouping for per-region conflict detection:
        within each connected component, split already-placed (existing) vs new assets.
        """
        existing_srcs = set(existing_srcs or ())
        active_srcs = set(active_srcs or ())
        scope = active_srcs | existing_srcs

        G_undirected = self.edge_center.to_undirected().copy()
        nodes_to_remove = [
            n for n in G_undirected.nodes
            if n in self.nodes and n not in scope
        ]
        G_undirected.remove_nodes_from(nodes_to_remove)

        groups: List[set] = []
        for comp in nx.connected_components(G_undirected):
            comp_assets = [n for n in comp if n in self.nodes]
            existing_in_comp = [n for n in comp_assets if n in existing_srcs]
            non_existing_in_comp = [
                n for n in comp_assets if n in active_srcs and n not in existing_srcs
            ]
            if existing_in_comp:
                groups.append(set(existing_in_comp))
                if non_existing_in_comp:
                    groups.append(set(non_existing_in_comp))
            elif comp_assets:
                groups.append(set(comp_assets))

        if not groups:
            for n in active_srcs:
                if n in self.nodes:
                    groups.append({n})
        return groups

    def get_ea_groups(self, G_undirected=None):
        """
        for CCEA: group with wall and no wall
        """
        if G_undirected is None:
            G_undirected = self.edge_center.to_undirected().copy()
        nodes_to_remove = [n for n in G_undirected.nodes if n not in self.nodes]
        if nodes_to_remove:
            G_undirected.remove_nodes_from(nodes_to_remove)
        groups = list(nx.connected_components(G_undirected))
        # split by region to avoid mixing different areas
        separated = []
        for group in groups:
            region_map = {}
            for n in group:
                if n not in self.nodes:
                    continue
                region = self.nodes[n].region
                key = region if region is not None else "__none__"
                region_map.setdefault(key, set()).add(n)
            for g in region_map.values():
                if g:
                    separated.append(g)
        return separated
    
    def clean_group(self):
        """
        remove walls and points in group
        """
        cleaned = []
        for group in self.groups:
            if not isinstance(group, (set, list, tuple)):
                continue
            filtered = {n for n in group if n in self.nodes}
            if filtered:
                cleaned.append(filtered)
        self.groups = cleaned


    def compute_centrality(self, active_srcs: Optional[set] = None):
        """
        1. Compute degree centrality
        2. Normalize within groups
        """
        if active_srcs is None:
            if self.centrality:
                return self.centrality
            G = self.edge_center
            groups = self.groups
        else:
            active_nodes = [n for n in self.edge_center.nodes if n in active_srcs]
            G = self.edge_center.subgraph(active_nodes)
            groups = list(nx.connected_components(G.to_undirected()))

        centrality = nx.degree_centrality(G)
        node_to_norm_centrality = {}

        for group in groups:
            csum = sum(centrality.get(n, 0.0) for n in group)
            size = len(group)
            if csum == 0:
                # distribute equally
                for n in group:
                    node_to_norm_centrality[n] = 1.0
            else:
                for n in group:
                    node_to_norm_centrality[n] = (centrality.get(n, 0.0) / csum) * size

        if active_srcs is None:
            self.centrality = node_to_norm_centrality
            return self.centrality
        return node_to_norm_centrality

    def segment_graph(self, active_srcs: Optional[set] = None, inplace: bool = True):
        """
        Build centered groups based on centrality. Updates self.center_nodes and self.groups.
        """
        if active_srcs is None:
            groups = self.get_ea_groups()
            centrality = self.centrality
            G = self.edge_center
        else:
            active_nodes = [n for n in self.edge_center.nodes if n in active_srcs]
            G = self.edge_center.subgraph(active_nodes)
            groups = self.get_ea_groups(G_undirected=G.to_undirected().copy())
            centrality = self.compute_centrality(active_srcs=active_srcs)

        center_nodes = {}
        new_groups = []
        i = 0

        for group_keys in groups:
            group_keys = list(group_keys)
            if len(group_keys) == 1:
                node = group_keys[0]
                center_nodes[i] = node
                new_groups.append({node})
                i += 1
                continue

            scores = {k: centrality.get(k, 0.0) for k in group_keys}
            high_nodes = [k for k, v in scores.items() if v >= 1.0]

            if len(group_keys) >= 2 and len(high_nodes) >= 2:
                sub_groups = {c: set([c]) for c in high_nodes}
                for node in group_keys:
                    if node in high_nodes:
                        continue
                    link_counts = {}
                    for c in high_nodes:
                        out_w = G.out_degree(node, weight=None) if G.has_edge(node, c) else 0
                        in_w = G.in_degree(node, weight=None) if G.has_edge(c, node) else 0
                        link_counts[c] = out_w + in_w
                    best_center = max(link_counts, key=link_counts.get)
                    if link_counts[best_center] > 0:
                        sub_groups[best_center].add(node)

                for c_node, members in sub_groups.items():
                    center_nodes[i] = c_node
                    new_groups.append(members)
                    i += 1
                continue

            node = max(scores, key=scores.get)
            center_nodes[i] = node
            new_groups.append(set(group_keys))
            i += 1

        if inplace and active_srcs is None:
            self.center_nodes = center_nodes
            self.groups = new_groups
        return center_nodes, new_groups

    def detect_relation_conflicts(
        self,
        active_srcs: Optional[set] = None,
        existing_srcs: Optional[set] = None,
        existing_assets: Optional[dict] = None,
        dsl_code: Optional[str] = None,
        groups: Optional[List[set]] = None,
        verbose: bool = True,
        first_pass: bool = True,
        region_bounds: Optional[Dict[str, tuple]] = None,
    ):
        """
        Rule-based conflict detection on edge_center (Relation graph).
        Per-region when active_srcs is set; wall occupancy uses full scene (all regions).
        Returns removed_edges (DSL strings), low_outdegree_nodes, log_lines.
        """
        if groups is None:
            if active_srcs is not None:
                groups = self.group_assets_for_region(active_srcs, existing_srcs)
            elif getattr(self, "groups", None):
                groups = self.groups
            else:
                groups = None

        removed_rels, low_outdegree_nodes, log_lines = detect_scene_graph_conflicts(
            self.edge_center,
            nodes=self.nodes,
            room_bound=self.room_bound,
            groups=groups,
            active_srcs=active_srcs,
            existing_srcs=existing_srcs,
            existing_assets=existing_assets,
            dsl_code=dsl_code,
            fixed_point_map=dict(self.fixed_points),
            verbose=verbose,
            first_pass=first_pass,
            region_bounds=region_bounds,
        )

        removed_edges = []
        for rel in removed_rels:
            if rel in self.edges:
                self.edges.remove(rel)
            removed_edges.append(self._relation_to_dsl(rel))

        return removed_edges, low_outdegree_nodes, log_lines

    def refine_graph(
        self,
        log_dir: str,
        gpt_api: GPT,
        walls_input: Dict[str, Any],
        task: Dict[str, Any],
        current_code: str,
        output_dir: Optional[str] = "./",
        max_iters: int = 5,
        active_srcs: Optional[set] = None,
        existing_srcs: Optional[set] = None,
        existing_assets: Optional[dict] = None,
        depth_label: Optional[str] = None,
    ):
        """
        Iteratively refine the scene graph:
        1) Run rule-based conflict detector per region (removes conflicting edges in-place).
        2) Wall-length aggregate checks use the full scene across all regions.
        3) Write conflict/feedback logs per iteration (global + per region).
        4) Ask GPT for new constraints based on region-filtered log when active_srcs is set.
        5) Update graph with new constraints.
        Stop when no errors are detected or max_iters reached.
        """
        os.makedirs(log_dir, exist_ok=True)
        if depth_label is None:
            depth_label = os.path.basename(log_dir.rstrip(os.sep))
        region_label = None
        if active_srcs:
            for src in active_srcs:
                region_label = self._node_region(src)
                if region_label is not None:
                    break
        if output_dir:
            refine_root = os.path.join(output_dir, "refine_logs")
            os.makedirs(refine_root, exist_ok=True)
            self._write_text(
                os.path.join(refine_root, "README.txt"),
                (
                    "Refine logs layout:\n"
                    "- refine_logs/summary.jsonl: one record per iteration\n"
                    "- refine_logs/<depth>/global/: full-scene logs per iteration\n"
                    "- refine_logs/<depth>/region_<id>/: per-area logs per iteration\n"
                    "Each iteration saves log_*.txt (conflicts), feedback_*.txt (GPT raw), "
                    "refine_*.txt (new DSL), current_code_*.txt (constraints before refine).\n"
                ),
            )

        for iter_idx in range(max_iters):
            iter_current_code = current_code
            conflict_groups = None
            if active_srcs is not None:
                conflict_groups = self.group_assets_for_region(active_srcs, existing_srcs)
            removed_edges, low_outdegree_nodes, log_lines = self.detect_relation_conflicts(
                active_srcs=active_srcs,
                existing_srcs=existing_srcs,
                existing_assets=existing_assets,
                dsl_code=iter_current_code,
                groups=conflict_groups,
                first_pass=(iter_idx == 0),
            )
            gpt_removed = removed_edges
            gpt_low = low_outdegree_nodes
            gpt_logs = log_lines
            if region_label is not None:
                gpt_removed, gpt_low, gpt_logs = self._filter_conflict_for_region(
                    region_label,
                    removed_edges,
                    low_outdegree_nodes,
                    log_lines,
                )
            log_text = self._format_log_text(
                iter_idx,
                gpt_removed,
                gpt_low,
                gpt_logs,
                region=region_label,
                current_code=iter_current_code,
            )

            if removed_edges:
                removed_set = {line.strip() for line in removed_edges if line.strip()}
                if removed_set:
                    current_lines = [
                        line for line in current_code.splitlines()
                        if line.strip() not in removed_set
                    ]
                    current_code = "\n".join(current_lines).strip()

            if not log_lines:
                self.write_refine_logs(
                    log_dir,
                    iter_idx,
                    removed_edges,
                    low_outdegree_nodes,
                    log_lines,
                    current_code=iter_current_code,
                    active_srcs=active_srcs,
                    output_dir=output_dir,
                    depth_label=depth_label,
                )
                print(f"No conflicts detected at iteration {iter_idx}. Stopping refinement.")
                break

            content_system, content_user = gpt_api.get_reflection(
                task,
                walls_input,
                self._reflection_nodes(active_srcs, iter_current_code, gpt_logs),
                current_code,
                log_text,
                output_dir=output_dir,
                low_outdegree_nodes=gpt_low,
            )
            refine_constraints = gpt_api(content_system, content_user)
            dsl_code = clean_pattern(refine_constraints)
            print(f"Refine constraints: {refine_constraints}")
            print(f"Cleaned DSL code: {dsl_code}")

            self.write_refine_logs(
                log_dir,
                iter_idx,
                removed_edges,
                low_outdegree_nodes,
                log_lines,
                current_code=iter_current_code,
                feedback_raw=refine_constraints,
                feedback_dsl=dsl_code,
                active_srcs=active_srcs,
                output_dir=output_dir,
                depth_label=depth_label,
            )

            if not dsl_code.strip():
                print(f"No dsl code generated at iteration {iter_idx}. Skipping refinement for this iter.")
                continue

            region_idx = self.get_dsl_region(current_code, dsl_code)
            self.add_relations_from_dsl(dsl_code, region_idx)
            if active_srcs is not None:
                self.groups = self.group_assets_for_region(active_srcs, existing_srcs)
            else:
                self.group_assets()
            current_code = (current_code + "\n" + dsl_code).strip() if current_code else dsl_code.strip()

    def _reflection_nodes(self, active_srcs, current_code: str, log_lines) -> Dict[str, Any]:
        """Limit regional reflection context to objects relevant to this refinement."""
        if active_srcs is None:
            return dict(self.node_center.nodes)
        context = "\n".join([current_code or "", *(log_lines or [])])
        relevant = set(active_srcs)
        relevant.update(node for node in self.nodes if node in context)
        return {
            node: self.node_center.nodes[node]
            for node in relevant
            if node in self.node_center
        }

    def get_dsl_region(self, current_code, dsl_code):
        current_code = (current_code + "\n" + dsl_code).strip()
        region_idx = None
        try:
            tree = ast.parse(dsl_code)
            first_node = None
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    args = node.value.args
                    if args:
                        arg0 = args[0]
                        if isinstance(arg0, ast.Name):
                            first_node = arg0.id
                        elif isinstance(arg0, ast.List) and arg0.elts:
                            for elt in arg0.elts:
                                if isinstance(elt, ast.Name):
                                    first_node = elt.id
                                    break
                        if first_node is None and len(args) > 1:
                            arg1 = args[1]
                            if isinstance(arg1, ast.Name):
                                first_node = arg1.id
                    break
            if first_node is not None:
                obj = self.nodes.get(first_node)
                if obj is not None and obj.region is not None:
                    region_idx = str(obj.region)
        except Exception:
            region_idx = None
        return region_idx

    def _node_region(self, node_name: str) -> Optional[str]:
        obj = self.nodes.get(node_name)
        if obj is None or obj.region is None:
            return None
        return str(obj.region)

    def _all_scene_regions(self) -> List[str]:
        regions: Set[str] = set()
        for obj in self.nodes.values():
            if obj.region is not None:
                regions.add(str(obj.region))
        return sorted(regions)

    def _affected_regions_in_iter(
        self,
        removed_edges,
        low_outdegree_nodes,
        log_lines,
        active_srcs=None,
        dsl_code: Optional[str] = None,
    ) -> List[str]:
        regions: Set[str] = set()
        for node in low_outdegree_nodes:
            region = self._node_region(node)
            if region is not None:
                regions.add(region)
        if active_srcs:
            for node in active_srcs:
                region = self._node_region(node)
                if region is not None:
                    regions.add(region)
        for line in list(log_lines or []) + list(removed_edges or []):
            for node in self.nodes:
                if node in line:
                    region = self._node_region(node)
                    if region is not None:
                        regions.add(region)
        if dsl_code:
            for node in self.nodes:
                if node in dsl_code:
                    region = self._node_region(node)
                    if region is not None:
                        regions.add(region)
        return sorted(regions)

    def _filter_conflict_for_region(
        self,
        region: str,
        removed_edges,
        low_outdegree_nodes,
        log_lines,
    ):
        def in_region(node_name: str) -> bool:
            return self._node_region(node_name) == region

        filtered_low = [n for n in low_outdegree_nodes if in_region(n)]
        filtered_removed = []
        for edge in removed_edges:
            if any(
                str(self.nodes[node].region) == region and node in edge
                for node in self.nodes
                if node in edge
            ):
                filtered_removed.append(edge)
        filtered_logs = []
        for line in log_lines or []:
            if any(in_region(node) for node in self.nodes if node in line):
                filtered_logs.append(line)
            elif "[PhysicalConflict]" in line or "aggregate_wall" in line:
                filtered_logs.append(line)
        return filtered_removed, filtered_low, filtered_logs

    def _filter_dsl_for_region(self, dsl_code: str, region: str) -> str:
        if not dsl_code:
            return ""
        kept = []
        for line in dsl_code.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for node, obj in self.nodes.items():
                if str(obj.region) == region and node in stripped:
                    kept.append(stripped)
                    break
        return "\n".join(kept)

    def _format_log_text(
        self,
        iter_idx: int,
        removed_edges,
        low_outdegree_nodes,
        log_lines,
        *,
        region: Optional[str] = None,
        current_code: Optional[str] = None,
    ) -> str:
        semantic_lines = [l for l in log_lines if "[LogicError]" in l]
        semantic_common = [l for l in log_lines if "[SemanticCommon]" in l]
        physical_lines = [
            l for l in log_lines
            if "[PhysicalConflict]" in l or "[ProjectorError]" in l
        ]
        low_outdegree_lines = [l for l in log_lines if "[ConstraintCompleteness]" in l]
        return "\n".join(
            [
                "Edges conflicts due to semantics logic:",
                *(semantic_lines if semantic_lines else ["(none)"]),
                "Low outdegree nodes:",
                *(low_outdegree_lines if low_outdegree_lines else ["(none)"]),
                "Edges conflicts due to physical:",
                *(physical_lines if physical_lines else ["(none)"]),
                "semantic common error:",
                *(semantic_common if semantic_common else ["(none)"]),
            ]
        )

    def _write_text(self, path: str, content: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def write_refine_logs(
        self,
        log_dir: str,
        iter_idx: int,
        removed_edges,
        low_outdegree_nodes,
        log_lines,
        *,
        current_code: Optional[str] = None,
        feedback_raw: Optional[str] = None,
        feedback_dsl: Optional[str] = None,
        active_srcs: Optional[set] = None,
        output_dir: Optional[str] = None,
        depth_label: Optional[str] = None,
    ) -> str:
        """Save conflict + feedback logs globally and per region for one refine iteration."""
        log_text = self._format_log_text(
            iter_idx,
            removed_edges,
            low_outdegree_nodes,
            log_lines,
            current_code=current_code,
        )

        # Backward-compatible root files
        self._write_text(os.path.join(log_dir, f"log_{iter_idx}.txt"), log_text)
        if current_code is not None:
            self._write_text(os.path.join(log_dir, f"current_code_{iter_idx}.txt"), current_code)
        if feedback_raw is not None:
            self._write_text(os.path.join(log_dir, f"feedback_{iter_idx}.txt"), feedback_raw)
        if feedback_dsl is not None:
            self._write_text(os.path.join(log_dir, f"refine_{iter_idx}.txt"), feedback_dsl)

        global_dir = os.path.join(log_dir, "global")
        self._write_text(os.path.join(global_dir, f"log_{iter_idx}.txt"), log_text)
        if current_code is not None:
            self._write_text(os.path.join(global_dir, f"current_code_{iter_idx}.txt"), current_code)
        if feedback_raw is not None:
            self._write_text(os.path.join(global_dir, f"feedback_{iter_idx}.txt"), feedback_raw)
        if feedback_dsl is not None:
            self._write_text(os.path.join(global_dir, f"refine_{iter_idx}.txt"), feedback_dsl)

        regions = sorted(
            set(self._all_scene_regions())
            | set(
                self._affected_regions_in_iter(
                    removed_edges,
                    low_outdegree_nodes,
                    log_lines,
                    active_srcs=active_srcs,
                    dsl_code=feedback_dsl,
                )
            )
        )
        for region in regions:
            region_dir = os.path.join(log_dir, f"region_{region}")
            r_removed, r_low, r_logs = self._filter_conflict_for_region(
                region, removed_edges, low_outdegree_nodes, log_lines
            )
            region_log = self._format_log_text(
                iter_idx,
                r_removed,
                r_low,
                r_logs,
                region=region,
                current_code=current_code,
            )
            self._write_text(os.path.join(region_dir, f"log_{iter_idx}.txt"), region_log)
            if feedback_raw is not None:
                self._write_text(os.path.join(region_dir, f"feedback_{iter_idx}.txt"), feedback_raw)
            if feedback_dsl:
                region_dsl = self._filter_dsl_for_region(feedback_dsl, region)
                if region_dsl.strip():
                    self._write_text(os.path.join(region_dir, f"refine_{iter_idx}.txt"), region_dsl)

        if output_dir:
            summary_path = os.path.join(output_dir, "refine_logs", "summary.jsonl")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            record = {
                "iteration": iter_idx,
                "depth": depth_label or os.path.basename(log_dir.rstrip("/")),
                "regions": regions,
                "removed_count": len(removed_edges or []),
                "low_outdegree_count": len(low_outdegree_nodes or []),
                "conflict_line_count": len(log_lines or []),
                "has_feedback": bool((feedback_raw or "").strip()),
                "has_refine_dsl": bool((feedback_dsl or "").strip()),
                "log_dir": log_dir,
            }
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return log_text

    def write_log(self, log_dir, iter_idx, removed_edges, low_outdegree_nodes, log_lines):
        return self.write_refine_logs(
            log_dir,
            iter_idx,
            removed_edges,
            low_outdegree_nodes,
            log_lines,
        )

def update_graph(scene_graph, assets, dsl_code, region_idx: Optional[str] = None):
    # Initialize SceneGraph
    if scene_graph is None:
        scene_graph = SceneGraph()

    # Add node (asset or assets_set)
    for name, value in assets.items():
        scene_graph.add_node(name, bbox=value.bbox, region_idx=region_idx)

    # Add edge (constraints)
    for rel in scene_graph.get_all_relations(dsl_code, region_idx, assets=assets):
        scene_graph.add_edge(rel)
    
    scene_graph.group_assets()
    
    return scene_graph



if __name__ == "__main__":
    wall: dict[str, Wall] = {}
    wall["left_wall"] = Wall(
        key="left_wall",
        id=0,
        pos=[0.0, 3.0, 0.0],
        rot=[0.],
        length=6.0,
        group="single",
        otype="edge")
    wall["right_wall"] = Wall(
        key="right_wall",
        id=1,
        pos=[6.0, 3.0, 0.0],
        rot=[180*torch.pi/180],
        length=6.0,
        group="single",
        otype="edge")
    wall["front_wall"] = Wall(
        key="front_wall",
        id=2,
        pos=[3.0, 0.0, 0.0],
        rot=[90*torch.pi/180],
        length=6.0,
        group="single",
        otype="edge")
    wall["back_wall"] = Wall(
        key="back_wall",
        id=2,
        pos=[3.0, 6.0, 0.0],
        rot=[270*torch.pi/180],
        length=6.0,
        group="single",
        otype="edge")
    room_bound = (0.0, 6.0, 0.0, 6.0)

    sg = SceneGraph()

    sg.add_node("sofa", bbox=np.array([2.0, 3.0, 1.0]))
    sg.add_node("nightstand", bbox=np.array([0.5, 0.5, 0.5]))
    sg.add_node("bed", bbox=np.array([2.0, 3.0, 1.0]))
    sg.add_edge(Relation(name="against", src="sofa", dst="left_wall", params={"arg0": 0.}))
    sg.add_edge(Relation(name="against", src="nightstand", dst="left_wall", params={"arg0": 0.}))
    sg.add_edge(Relation(name="against", src="bed", dst="left_wall", params={"arg0": 0.}))
    sg.add_edge(Relation(name="align_with", src="sofa", dst="nightstand", params={"arg0": 0.}))
    sg.add_edge(Relation(name="against", src="nightstand", dst="right_wall", params={"arg0": 0.}))
    sg.add_edge(Relation(name="place_align", src="bed", dst="sofa", params={"arg0": "down"}))
    sg.add_edge(Relation(name="place_align", src="bed", dst="nightstand", params={"arg0": "down"}))
    sg.add_edge(Relation(name="against", src="bed", dst="back_wall", params={"arg0": 0.}))
    sg.add_walls(wall, room_bound)
    removed_edges, low_outdegree_nodes, log_lines = sg.detect_relation_conflicts()

    print("Removed edges due to semantics conflicts:", removed_edges)
    print("Weak nodes:", low_outdegree_nodes)
    print("Conflict logs:", log_lines)
