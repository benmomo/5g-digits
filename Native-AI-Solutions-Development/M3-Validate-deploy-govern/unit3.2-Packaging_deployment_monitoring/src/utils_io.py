# src/utils_io.py
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

def timestamp_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
