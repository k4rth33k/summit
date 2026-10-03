#!/usr/bin/env python3
"""Create a case-isolated Summit train/calibration split from typed-decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from evaluate_typed_decisions import DATASET_REVISION, convert, load_rows, sha256
from summit.recipes.decision.data import TeacherTarget, canonical_json, check_split_isolation


TRAIN_SHA256 = "46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--validation-cases-per-workflow", type=int, default=30)
    args = parser.parse_args()

    # Do not label an arbitrary (or held-out test) parquet as this pinned train
    # split in the manifest. Changing sources requires a reviewed converter.
    if file_sha256(args.dataset) != TRAIN_SHA256:
        parser.error("dataset is not the pinned Typed Decisions train parquet (SHA-256 mismatch)")
    if not 0 < args.validation_cases_per_workflow < 300:
        parser.error("validation-cases-per-workflow must be between 1 and 299")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_rows = load_rows(args.dataset)
    records, metadata = convert(source_rows)
    workflow_groups = {}
    for source in source_rows:
        workflow_groups.setdefault(source["workflow"], []).append(source["id"])
    validation_groups = set()
    rng = random.Random(args.seed)
    for workflow, group_ids in sorted(workflow_groups.items()):
        selected = list(group_ids)
        rng.shuffle(selected)
        validation_groups.update(selected[:args.validation_cases_per_workflow])

    splits = {"train": [], "validation": []}
    for record in records:
        detail = metadata[record.id]
        probabilities = detail["gold_probabilities"]
        teacher = TeacherTarget(
            input_sha256=record.input_hash(),
            model="typed-decisions-three-sample-gold",
            method="mean of three benchmark teacher samples",
            probabilities=probabilities,
            metadata={
                "dataset": "LocalLLaMA/typed-decisions",
                "dataset_revision": DATASET_REVISION,
                "soft_target": True,
            },
        )
        split = "validation" if record.group_id in validation_groups else "train"
        splits[split].append(record.model_copy(update={"teacher": teacher}))

    check_split_isolation(splits)
    for name, split_records in splits.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in split_records))

    manifest = {
        "schema_version": 1,
        "dataset": "LocalLLaMA/typed-decisions",
        "dataset_revision": DATASET_REVISION,
        "source_file": str(args.dataset.resolve()),
        "source_sha256": sha256(args.dataset),
        "split_method": f"stratified case-level holdout; {args.validation_cases_per_workflow} of 300 cases per workflow",
        "seed": args.seed,
        "training": {
            "cases": len({row.group_id for row in splits["train"]}),
            "decisions": len(splits["train"]),
            "sha256": file_sha256(output / "train.jsonl"),
        },
        "validation": {
            "cases": len({row.group_id for row in splits["validation"]}),
            "decisions": len(splits["validation"]),
            "sha256": file_sha256(output / "validation.jsonl"),
        },
        "test_access": "not read or converted by this preparation script",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
