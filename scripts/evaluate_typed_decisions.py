#!/usr/bin/env python3
"""Evaluate a Summit causal decision checkpoint on LocalLLaMA/typed-decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import string
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from summit.recipes.decision.data import Candidate, DecisionRecord, Target, canonical_json
from summit.recipes.decision.prompts import prompt


DATASET_REVISION = "561333a8576d22875380b14d25a13065b046538c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path):
    return pq.read_table(path).to_pylist()


def candidates(question: dict) -> list[Candidate]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [Candidate(id=str(key), description=str(description))
                for key, description in criteria.items()]
    if kind == "noul":
        criteria = criteria or {
            "false": "The statement is false.",
            "true": "The statement is true.",
        }
        return [Candidate(id=key, description=str(criteria[key])) for key in ("false", "true")]
    if kind == "score":
        return [Candidate(id=str(index), description=str(description))
                for index, description in enumerate(criteria)]
    raise ValueError(f"unsupported question type: {kind}")


def gold_distribution(question: dict, gold: dict, option_ids: list[str]) -> list[float]:
    probabilities = gold.get("probabilities", {})
    if question["type"] == "noul" and not probabilities:
        true_probability = float(gold["noul"])
        probabilities = {"false": 1 - true_probability, "true": true_probability}
    values = np.asarray([float(probabilities.get(option_id, 0.0)) for option_id in option_ids], dtype=float)
    if values.sum() <= 0:
        values[option_ids.index(str(gold["label"]).lower())] = 1.0
    return (values / values.sum()).tolist()


def convert(rows):
    records = []
    metadata = {}
    for source in rows:
        state = json.loads(source["state"])
        questions = json.loads(source["questions"])
        gold = json.loads(source["gold"])
        for question_id, question in questions.items():
            options = candidates(question)
            option_ids = [option.id for option in options]
            label = str(gold[question_id]["label"]).lower()
            record_id = f"{source['id']}::{question_id}"
            record = DecisionRecord(
                id=record_id,
                group_id=source["id"],
                state=state,
                question=question["instructions"],
                candidate_options=options,
                target=Target(correct_option_id=label),
                provenance={
                    "source": "LocalLLaMA/typed-decisions",
                    "workflow": source["workflow"],
                    "question_id": question_id,
                    "question_type": question["type"],
                },
            )
            records.append(record)
            metadata[record_id] = {
                "workflow": source["workflow"],
                "question_id": question_id,
                "question_type": question["type"],
                "gold_probabilities": dict(zip(option_ids, gold_distribution(
                    question, gold[question_id], option_ids))),
                "gold_score": gold[question_id].get("score"),
            }
    return records, metadata


def ece(confidence, correct, bins=15):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = ((confidence >= low) if index == 0 else (confidence > low)) & (confidence <= high)
        if selected.any():
            value += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return float(value)


def aggregate(predictions):
    accuracy = np.asarray([row["choice"] == row["target"] for row in predictions], dtype=float)
    confidence = np.asarray([max(row["probabilities"].values()) for row in predictions], dtype=float)
    nll = []
    hard_brier = []
    soft_accuracy = []
    soft_brier = []
    kl = []
    score_mae = []
    for row in predictions:
        option_ids = list(row["probabilities"])
        predicted = np.asarray([row["probabilities"][key] for key in option_ids], dtype=float)
        hard = np.zeros_like(predicted)
        hard[option_ids.index(row["target"])] = 1.0
        gold = np.asarray([row["gold_probabilities"][key] for key in option_ids], dtype=float)
        nll.append(-math.log(max(predicted[option_ids.index(row["target"])], 1e-12)))
        hard_brier.append(float(((predicted - hard) ** 2).sum()))
        soft_accuracy.append(float((predicted * gold).sum()))
        soft_brier.append(float(((predicted - gold) ** 2).sum()))
        positive = gold > 0
        kl.append(float((gold[positive] * np.log(gold[positive] / np.clip(predicted[positive], 1e-12, None))).sum()))
        if row["question_type"] == "score":
            expected = float((np.arange(len(predicted)) * predicted).sum())
            score_mae.append(abs(expected - float(row["gold_score"])))
    return {
        "n": len(predictions),
        "accuracy": float(accuracy.mean()),
        "correct": int(accuracy.sum()),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(hard_brier)),
        "ece": ece(confidence, accuracy),
        "mean_confidence": float(confidence.mean()),
        "soft_accuracy": float(np.mean(soft_accuracy)),
        "brier_vs_soft": float(np.mean(soft_brier)),
        "kl_from_gold": float(np.mean(kl)),
        "score_mae": float(np.mean(score_mae)) if score_mae else None,
    }


def grouped(predictions, key):
    groups = {}
    for row in predictions:
        groups.setdefault(row[key], []).append(row)
    return {name: aggregate(rows) for name, rows in sorted(groups.items())}


def grouped_questions(predictions):
    groups = {}
    for row in predictions:
        name = f"{row['workflow']}/{row['question_id']}"
        groups.setdefault(name, []).append(row)
    return {name: aggregate(rows) for name, rows in sorted(groups.items())}


def case_bootstrap_accuracy(predictions, samples=10_000, seed=42):
    """Bootstrap whole five-question cases, preserving within-case dependence."""
    groups = {}
    for row in predictions:
        groups.setdefault(row["group_id"], []).append(float(row["choice"] == row["target"]))
    case_accuracy = np.asarray([np.mean(values) for values in groups.values()], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        resample = rng.integers(0, len(case_accuracy), size=len(case_accuracy))
        estimates[index] = case_accuracy[resample].mean()
    return {
        "method": "percentile bootstrap over 400 independent cases (five decisions retained per case)",
        "samples": samples,
        "seed": seed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def run(args):
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = load_rows(dataset)
    records, metadata = convert(rows)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompts = [prompt(record, tokenizer) for record in records]
    token_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in prompts]
    if max(token_lengths) > args.max_length:
        raise ValueError(f"prompt exceeds {args.max_length}: {max(token_lengths)}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint / "model",
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="eager",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to(args.device)
    if args.peft_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft_adapter.resolve(), is_trainable=False)
    model = model.eval()
    decoder = model.get_decoder()
    head = model.get_output_embeddings()
    predictions = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start:start + args.batch_size]
            batch_prompts = prompts[start:start + args.batch_size]
            encoded = tokenizer(batch_prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            hidden = decoder(**encoded, use_cache=False, return_dict=True).last_hidden_state
            ends = encoded["attention_mask"].sum(-1) - 1
            final = hidden[torch.arange(len(batch_records), device=args.device), ends]
            for offset, (record, vector, text) in enumerate(zip(batch_records, final, batch_prompts)):
                prefix = tokenizer.encode(text, add_special_tokens=False)
                answer_ids = []
                for letter in string.ascii_uppercase[:len(record.candidate_options)]:
                    full = tokenizer.encode(text + letter, add_special_tokens=False)
                    if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
                        raise ValueError(f"answer {letter} is not one token for {record.id}")
                    answer_ids.append(full[-1])
                logits = torch.nn.functional.linear(
                    vector, head.weight[answer_ids], head.bias[answer_ids] if head.bias is not None else None
                ).float()
                probabilities = logits.softmax(-1).cpu().tolist()
                option_ids = [option.id for option in record.candidate_options]
                detail = metadata[record.id]
                predictions.append({
                    "id": record.id,
                    "group_id": record.group_id,
                    "workflow": detail["workflow"],
                    "question_id": detail["question_id"],
                    "question_type": detail["question_type"],
                    "choice": option_ids[int(np.argmax(probabilities))],
                    "target": record.target.correct_option_id,
                    "probabilities": dict(zip(option_ids, probabilities)),
                    "gold_probabilities": detail["gold_probabilities"],
                    "gold_score": detail["gold_score"],
                })
            if (start + len(batch_records)) % 100 == 0 or start + len(batch_records) == len(records):
                print(canonical_json({"completed": start + len(batch_records), "total": len(records)}), flush=True)
    elapsed = time.monotonic() - started
    metrics = aggregate(predictions)
    metrics["accuracy_95_ci"] = case_bootstrap_accuracy(predictions)
    metrics["by_workflow"] = grouped(predictions, "workflow")
    metrics["by_question_type"] = grouped(predictions, "question_type")
    metrics["by_question"] = grouped_questions(predictions)
    metrics["elapsed_seconds"] = elapsed
    metrics["decisions_per_second"] = len(predictions) / elapsed
    metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None

    (output / "predictions.jsonl").write_text("".join(canonical_json(row) + "\n" for row in predictions))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "dataset": "LocalLLaMA/typed-decisions",
        "dataset_revision": DATASET_REVISION,
        "dataset_file": str(dataset),
        "dataset_sha256": sha256(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_revision": checkpoint.name,
        "checkpoint_weight_sha256": sha256(checkpoint / "model" / "model.safetensors"),
        "peft_adapter": str(args.peft_adapter.resolve()) if args.peft_adapter else None,
        "peft_adapter_sha256": sha256(args.peft_adapter.resolve() / "adapter_model.safetensors")
        if args.peft_adapter else None,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "mode": (
            "benchmark specialist; PEFT adapter trained on typed-decisions train; one Summit decision prompt per question"
            if args.peft_adapter else
            "cross-domain zero-shot; one Summit decision prompt per question"
        ),
        "protocol_deviation": "The published benchmark sends all five questions in one System One request; Summit's current causal adapter scores one question per prompt.",
        "cases": len(rows),
        "decisions": len(records),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "prompt_tokens": {"minimum": min(token_lengths), "maximum": max(token_lengths),
                          "mean": float(np.mean(token_lengths))},
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peft-adapter", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
