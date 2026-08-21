"""Batch scene evaluation helpers (GenesisVLM2 run_sceneeval_33x3 aligned)."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_GPT_SEEDS = (1111, 1234, 2026)
METRIC_KEYS = ("NAV", "COL", "OOB", "POP", "POS", "ROT", "OVR")


def scene_stem_from_json(scene_json_file: str | Path) -> str:
    return Path(scene_json_file).stem


def build_scene_run_dir_name(
    scene_name: str,
    *,
    gpt_seed: Optional[int] = None,
    run_tag: str = "",
) -> str:
    """Genesis-style: plain scene name unless batch run_tag is set."""
    if not run_tag:
        return scene_name
    parts = [scene_name]
    if gpt_seed is not None:
        parts.append(f"seed{gpt_seed}")
    parts.append(f"run{run_tag}")
    return "_".join(parts)


def resolve_scene_output_dir(
    output_root: str | Path,
    scene_json_file: str | Path,
    *,
    gpt_seed: Optional[int] = None,
    run_tag: str = "",
) -> Path:
    scene_name = scene_stem_from_json(scene_json_file)
    dir_name = build_scene_run_dir_name(scene_name, gpt_seed=gpt_seed, run_tag=run_tag)
    return Path(output_root) / dir_name


def resolve_metrics_file(run_folder: Path) -> Optional[Path]:
    candidates = [
        run_folder / "metrics_results.json",
        run_folder / "ccea" / "metrics_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_metrics_payload(metrics_path: Path) -> Optional[dict[str, Any]]:
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
        return data["metrics"]
    if isinstance(data, dict):
        return data
    return None


def _mean_std(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def summarize_metric_rows(rows: Iterable[dict[str, Any]], metric_keys: tuple[str, ...] = METRIC_KEYS) -> dict[str, Any]:
    rows = list(rows)
    summary: dict[str, Any] = {"n_runs": len(rows), "metrics": {}}
    for key in metric_keys:
        vals: list[float] = []
        for row in rows:
            metrics = row.get("metrics") or {}
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        summary["metrics"][key] = _mean_std(vals)
    return summary


def aggregate_batch_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run JSONL records into per-seed and per-scene summaries."""
    by_scene_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    by_scene: dict[str, list[dict[str, Any]]] = {}

    for rec in records:
        scene = rec.get("scene_json_file") or rec.get("scene_stem")
        seed = rec.get("gpt_seed")
        if scene is None:
            continue
        scene_key = str(scene)
        by_scene.setdefault(scene_key, []).append(rec)
        if seed is not None:
            by_scene_seed.setdefault((scene_key, int(seed)), []).append(rec)

    per_seed_summary: dict[str, dict[str, Any]] = {}
    for (scene, seed), rows in sorted(by_scene_seed.items(), key=lambda x: (x[0][0], x[0][1])):
        key = f"{Path(scene).stem}_seed{seed}"
        per_seed_summary[key] = {
            "scene_json_file": scene,
            "gpt_seed": seed,
            **summarize_metric_rows(rows),
        }

    per_scene_summary: dict[str, Any] = {}
    for scene, rows in sorted(by_scene.items(), key=lambda x: x[0]):
        per_scene_summary[Path(scene).stem] = {
            "scene_json_file": scene,
            **summarize_metric_rows(rows),
        }

    return {
        "overall": summarize_metric_rows(records),
        "per_scene": per_scene_summary,
        "per_scene_seed": per_seed_summary,
    }
