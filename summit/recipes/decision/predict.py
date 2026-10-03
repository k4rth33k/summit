"""Target-free inference: choose a valid candidate and build JSON in application code."""

import argparse
from pathlib import Path

from .data import DecisionInput, canonical_json
from .train import load_artifact


def predict(model, tokenizer, records, max_length):
    import torch
    model.eval()
    results = []
    with torch.no_grad():
        for row in records:
            vectors = model.score([row], tokenizer, max_length)
            if len(vectors) != 1:
                raise ValueError("adapter must return exactly one vector per input")
            values = vectors[0]
            if values.shape != (len(row.candidate_options),) or not torch.isfinite(values).all():
                raise ValueError("invalid prediction logits")
            probs = values.float().softmax(-1).cpu()
            choice = row.candidate_options[int(probs.argmax())]
            results.append({"id": row.id, "choice": choice.id, "description": choice.description,
                            "probabilities": {o.id: p for o, p in zip(row.candidate_options, probs.tolist())}})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL: id, state, question, candidate_options only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if not (args.artifact / "complete.json").exists():
        parser.error("artifact does not have a completion marker")
    records = [DecisionInput.model_validate_json(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not records or len({row.id for row in records}) != len(records):
        parser.error("inputs must be nonempty with unique IDs")
    if args.output.exists():
        parser.error("output already exists")
    model, tokenizer, metadata = load_artifact(args.artifact, args.device)
    predictions = predict(model, tokenizer, records, metadata["max_length"])
    with args.output.open("x") as stream:
        stream.writelines(canonical_json(row) + "\n" for row in predictions)


if __name__ == "__main__":
    main()
