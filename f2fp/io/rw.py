from __future__ import annotations
import json
from pathlib import Path

def write_jsonl(fps: list[dict], path: str | Path) -> None:
    p = Path(path)
    with p.open('w', encoding='utf-8') as f:
        for d in fps:
            f.write(json.dumps(d) + '\n')

def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    out = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
