"""Evaluate a saved decision artifact on a held-out canonical JSONL split."""

import argparse
import json
from pathlib import Path

from .data import canonical_json, dataset_summary, read_records
from .train import evaluate, load_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if not (args.artifact / "complete.json").exists():
        parser.error("artifact does not have a completion marker")
    rows = read_records(args.data)
    model, tokenizer, metadata = load_artifact(args.artifact, args.device)
    metrics, predictions = evaluate(model, tokenizer, rows, metadata["max_length"], args.batch_size)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps({"metrics": metrics, "dataset": dataset_summary(args.data, rows)}, indent=2) + "\n")
    (args.output / "predictions.jsonl").write_text("".join(canonical_json(row) + "\n" for row in predictions))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
