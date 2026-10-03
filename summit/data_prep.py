"""DeepSWE data prep: turn deep-swe/tasks/<id> dirs into NexRL parquet rows.

Each row's `prompt` is a JSON string consumed by DeepSWERolloutWorker:
  {"task_id", "instruction", "image"}

Runs on the VM (pandas/pyarrow come from NexRL's deps).
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def build_rows(tasks_dir: Path, task_ids: list[str]) -> list[dict]:
    rows = []
    for task_id in task_ids:
        tdir = tasks_dir / task_id
        toml_path = tdir / "task.toml"
        instr_path = tdir / "instruction.md"
        if not toml_path.exists() or not instr_path.exists():
            raise FileNotFoundError(f"task {task_id}: expected task.toml + instruction.md in {tdir}")
        meta = tomllib.loads(toml_path.read_text())
        image = (meta.get("environment") or {}).get("docker_image")
        payload = {
            "task_id": task_id,
            "instruction": instr_path.read_text(),
            "image": image,
        }
        rows.append({"prompt": json.dumps(payload), "task_id": task_id})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated task ids")
    ap.add_argument("--out", required=True, help="output dir for train.parquet")
    args = ap.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rows = build_rows(Path(args.tasks_dir), task_ids)

    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "train.parquet", index=False)
    print(f"wrote {len(df)} rows to {out}/train.parquet")


if __name__ == "__main__":
    main()
