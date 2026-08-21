"""Resolve OpenAI API key from CLI, environment, or local .env file."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_dotenv_key(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"OPENAI_API_KEY", "GPT_API_KEY", "OPENAI_ADMIN_KEY"}:
            value = _strip_quotes(value.strip())
            if value:
                return value
    return None


def resolve_openai_api_key(explicit: Optional[str] = None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()

    for env_name in ("OPENAI_API_KEY", "GPT_API_KEY", "OPENAI_ADMIN_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    search_roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[1],
    ]
    seen: set[Path] = set()
    for root in search_roots:
        for name in (".env", "secrets/openai_api_key", "openai_api_key.txt"):
            path = (root / name).resolve()
            if path in seen:
                continue
            seen.add(path)
            if name == ".env":
                value = _read_dotenv_key(path)
            elif path.is_file():
                value = path.read_text(encoding="utf-8").strip()
            else:
                value = None
            if value:
                return value

    raise RuntimeError(
        "Missing OpenAI API key. Set OPENAI_API_KEY (or GPT_API_KEY), "
        "create a .env file in the project root, or pass --gpt_api_key."
    )


def ensure_gpt_api_key(args) -> str:
    key = resolve_openai_api_key(getattr(args, "gpt_api_key", None))
    args.gpt_api_key = key
    return key
