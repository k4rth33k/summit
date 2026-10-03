"""Build a grouped v3 pilot and deterministic option-order training views."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .data import DecisionRecord, canonical_json, check_split_isolation, digest, read_records
from .dataset import write_dataset


SAMPLER_VERSION = "decision-pilot-v3"


def _stable_key(seed, scope, row):
    return hashlib.sha256(f"{seed}:{scope}:{row.id}".encode()).hexdigest()


def _choose(rows, source, stratum, count, seed):
    if source == "clinc150":
        candidates = [row for row in rows if row.provenance.get("source") == source
                      and row.provenance.get("negative_type") == stratum]
    else:
        candidates = [row for row in rows if row.provenance.get("source") == source
                      and row.provenance.get("reference_label") == stratum]
    if len(candidates) < count:
        raise ValueError(f"insufficient {source}/{stratum}: need {count}, found {len(candidates)}")
    chosen, group_counts = [], Counter()
    for _ in range(count):
        row = min(candidates, key=lambda value: (
            group_counts[value.group_id], _stable_key(seed, f"{source}:{stratum}", value)))
        candidates.remove(row)
        chosen.append(row)
        group_counts[row.group_id] += 1
    return chosen


def select_pilot(train, validation, seed=20260927):
    """Select 512 train and 256 development rows without reading test/calibration."""
    train_spec = [
        # The length-filtered v3 training pool contains 26 true-OOS rows. Keep
        # every one and retain 64 total explicit negative cases via shortlist misses.
        ("clinc150", "true_oos", 26), ("clinc150", "shortlist_miss", 38),
        ("clinc150", "in_scope", 192),
        ("contractnli", "e", 86), ("contractnli", "c", 85), ("contractnli", "n", 85),
    ]
    validation_spec = [
        ("clinc150", "true_oos", 16), ("clinc150", "shortlist_miss", 16),
        ("clinc150", "in_scope", 96),
        # Only 39 length-qualified contradiction rows exist in development.
        ("contractnli", "e", 44), ("contractnli", "c", 39), ("contractnli", "n", 45),
    ]
    result = {
        "train": [row for source, stratum, count in train_spec
                  for row in _choose(train, source, stratum, count, seed)],
        "validation": [row for source, stratum, count in validation_spec
                       for row in _choose(validation, source, stratum, count, seed)],
    }
    for name, rows in result.items():
        rows.sort(key=lambda row: _stable_key(seed, name, row))
    check_split_isolation(result)
    return result


def augment_order(records, seed=20260927, copies=1):
    """Keep originals and add semantic-ID-preserving candidate permutations."""
    if copies < 1:
        raise ValueError("copies must be positive")
    result = []
    for row in records:
        result.append(row)
        original = list(row.candidate_options)
        for copy in range(1, copies + 1):
            options = list(original)
            rng = random.Random(int(digest([seed, row.id, copy])[:16], 16))
            rng.shuffle(options)
            if [item.id for item in options] == [item.id for item in original]:
                options = options[1:] + options[:1]
            provenance = {**row.provenance, "order_augmentation": {
                "version": SAMPLER_VERSION, "copy": copy, "seed": seed,
                "semantic_option_ids_preserved": True,
            }}
            value = row.model_dump(mode="json", exclude={"teacher"})
            value.update({"id": f"{row.id}::order-{copy}", "candidate_options": [
                option.model_dump(mode="json") for option in options], "provenance": provenance})
            augmented = DecisionRecord.model_validate(value)
            if row.teacher is not None:
                teacher = row.teacher.model_copy(deep=True)
                teacher.input_sha256 = augmented.input_hash()
                teacher.metadata = {**teacher.metadata, "order_augmentation": {
                    "method": "semantic_id_transport", "copy": copy, "seed": seed,
                    "source_input_sha256": row.teacher.input_sha256,
                    "new_teacher_request": False,
                }}
                value["teacher"] = teacher.model_dump(mode="json")
                augmented = DecisionRecord.model_validate(value)
            result.append(augmented)
    if len({row.id for row in result}) != len(result):
        raise ValueError("augmentation produced duplicate IDs")
    return result


def _summary(rows):
    return {
        "rows": len(rows), "groups": len({row.group_id for row in rows}),
        "by_source": dict(Counter(row.provenance["source"] for row in rows)),
        "clinc_negative_type": dict(Counter(row.provenance.get("negative_type") for row in rows
                                            if row.provenance["source"] == "clinc150")),
        "reference_classes": dict(Counter(f"{row.provenance['source']}:{row.provenance['reference_label']}"
                                          for row in rows)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--data-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=20260927)
    augment = sub.add_parser("augment-order")
    augment.add_argument("--data", type=Path, required=True)
    augment.add_argument("--output", type=Path, required=True)
    augment.add_argument("--seed", type=int, default=20260927)
    augment.add_argument("--copies", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "select":
        splits = select_pilot(read_records(args.data_dir / "train.jsonl"),
                              read_records(args.data_dir / "validation.jsonl"), args.seed)
        metadata = {
            "sampler": SAMPLER_VERSION, "seed": args.seed, "parent": str(args.data_dir),
            "parent_hashes": {name: hashlib.sha256((args.data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                              for name in splits},
            "scope": "group-spread 512-train/256-development pilot; test and calibration unread",
            "statistics": {name: _summary(rows) for name, rows in splits.items()},
        }
        print(json.dumps(write_dataset(args.output, splits, metadata), indent=2))
    else:
        if args.output.exists():
            parser.error("output already exists")
        source = read_records(args.data, require_teacher=False)
        rows = augment_order(source, args.seed, args.copies)
        with args.output.open("x") as stream:
            stream.writelines(canonical_json(row.model_dump(mode="json", exclude_none=True)) + "\n" for row in rows)
        manifest = {
            "version": SAMPLER_VERSION, "seed": args.seed, "copies": args.copies,
            "source": str(args.data), "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "teacher_transport": "semantic IDs only; derived rows are not new teacher requests",
            "statistics": _summary(rows),
        }
        manifest_path = args.output.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
