"""Token-length audit/filter using the actual student prompts; never truncate evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .data import canonical_json, digest, read_records, check_split_isolation
from .dataset import write_dataset
from .prompts import prompt
from .train import load_tokenizer


def audit(splits, tokenizer, max_length):
    retained, report, exclusions = {}, {}, []
    for split, rows in splits.items():
        retained[split] = []
        counts = Counter()
        longest = 0
        for row in rows:
            texts = [prompt(row, tokenizer)] + [prompt(row, tokenizer, i) for i in range(len(row.candidate_options))]
            lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
            length = max(lengths)
            longest = max(longest, length)
            source = row.provenance["source"]
            counts[f"{source}:total"] += 1
            if length <= max_length:
                retained[split].append(row)
                counts[f"{source}:retained"] += 1
            else:
                counts[f"{source}:overlength"] += 1
                exclusions.append({"id": row.id, "split": split, "max_prompt_tokens": length, "reason": "overlength_no_truncation"})
        report[split] = {"by_source": dict(counts), "max_prompt_tokens": longest, "retained": len(retained[split])}
    check_split_isolation(retained)
    return retained, report, exclusions


def qualification_subset(splits, seed=42):
    selected = {}
    for split, limit in (("train", 6), ("validation", 2)):
        by_source = Counter()
        selected[split] = []
        for row in sorted(splits[split], key=lambda r: digest([seed, r.id])):
            source = row.provenance["source"]
            if by_source[source] < limit:
                selected[split].append(row)
                by_source[source] += 1
    check_split_isolation(selected)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--qualification-output", type=Path, help="Optional tiny train/validation subset, no test inputs")
    args = parser.parse_args(argv)
    if args.max_length < 16 or args.output.exists():
        parser.error("max-length must be >=16 and output must not exist")
    splits = {name: read_records(args.data_dir / f"{name}.jsonl") for name in ("train", "validation", "calibration", "test")
              if (args.data_dir / f"{name}.jsonl").exists()}
    tokenizer = load_tokenizer(args.tokenizer, args.revision)
    retained, report, exclusions = audit(splits, tokenizer, args.max_length)
    write_dataset(args.output, {name: rows for name, rows in retained.items() if rows},
                  {"parent_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
                   "tokenizer": args.tokenizer, "revision": args.revision, "max_length": args.max_length,
                   "scope": "length-filtered conversion, not full original benchmark", "length_audit": report})
    (args.output / "exclusions.jsonl").write_text("".join(canonical_json(value) + "\n" for value in exclusions))
    if args.qualification_output:
        selected = qualification_subset(retained)
        write_dataset(args.qualification_output, selected, {"scope": "pipeline qualification only, not a quality benchmark",
                                                            "parent": str(args.output), "selection_seed": 42})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
