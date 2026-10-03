"""Single-device offline training, evaluation, and complete reloadable artifacts."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time

from .data import read_records, check_split_isolation, dataset_summary, canonical_json


def add_code_paths(cfg):
    for source in cfg.code:
        parent = str(cfg.resolve_path(source).parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)


def load_tokenizer(name, revision=None):
    from transformers import AutoTokenizer
    source = Path(str(name))
    if source.is_dir() and (source / "tokenizer").is_dir():
        source = source / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(source), revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def optimizer_parameters(model, training):
    """Return AdamW parameter groups, separating a fresh scorer head when requested."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if training.head_learning_rate is None:
        return trainable
    head = getattr(model, "head", None)
    if head is None:
        raise ValueError("head_learning_rate requires an adapter with a head module")
    head_ids = {id(p) for p in head.parameters() if p.requires_grad}
    backbone = [p for p in trainable if id(p) not in head_ids]
    head_parameters = [p for p in trainable if id(p) in head_ids]
    if not backbone or not head_parameters:
        raise ValueError("head_learning_rate requires trainable backbone and head parameters")
    return [
        {"params": backbone, "lr": training.learning_rate},
        {"params": head_parameters, "lr": training.head_learning_rate},
    ]


def evaluate(model, tokenizer, records, max_length, batch_size=1):
    import torch
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            scores = model.score(rows, tokenizer, max_length)
            if len(scores) != len(rows):
                raise ValueError("adapter returned the wrong number of decisions")
            for row, logits in zip(rows, scores):
                if logits.shape != (len(row.candidate_options),) or not torch.isfinite(logits).all():
                    raise ValueError("invalid evaluation logits")
                probs = logits.float().softmax(-1).cpu()
                ids = [option.id for option in row.candidate_options]
                gold = ids.index(row.target.correct_option_id)
                y = torch.zeros_like(probs)
                y[gold] = 1
                predictions.append({
                    "id": row.id, "group_id": row.group_id, "input_sha256": row.input_hash(),
                    "source": row.provenance.get("source", "unknown"),
                    "choice": ids[int(probs.argmax())], "target": row.target.correct_option_id,
                    "probabilities": dict(zip(ids, probs.tolist())),
                    "nll": float(-probs[gold].clamp_min(1e-30).log()),
                    "brier": float(((probs - y) ** 2).sum()),
                })
    n = len(predictions)
    metrics = {"examples": n, "accuracy": sum(p["choice"] == p["target"] for p in predictions) / n,
               "nll": sum(p["nll"] for p in predictions) / n, "brier": sum(p["brier"] for p in predictions) / n}
    return metrics, predictions


def reverse_options(records):
    # Order changes invalidate cached targets; evaluation uses independent labels.
    return [row.model_copy(update={"candidate_options": list(reversed(row.candidate_options)),
                                   "teacher": None}) for row in records]


def load_artifact(path, device="cpu"):
    import torch
    from .config import ModelConfig
    from .models import resolve_adapter
    path = Path(path)
    metadata = json.loads((path / "adapter.json").read_text())
    for relative in metadata.get("code", []):
        source = (path / relative).resolve()
        if not source.is_relative_to(path.resolve()):
            raise ValueError("artifact code path escapes artifact directory")
        sys.path.insert(0, str(source.parent))
    config = ModelConfig.model_validate(metadata["model"])
    model = resolve_adapter(config.adapter)(config, checkpoint=path)
    if not getattr(model, "manages_device_placement", False):
        model = model.to(torch.device(device))
    model.eval()
    return model, load_tokenizer(path / "tokenizer"), metadata


def train(cfg):
    if cfg.training.comparison == "ce_kd":
        from .matched import train_matched
        return train_matched(cfg)
    import torch
    from .models import resolve_adapter
    from .objective import decision_loss
    from .recipe import preflight

    report = preflight(cfg)
    if not report.passed:
        raise ValueError(report.format_text())
    if cfg.training.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use training.device: cpu for small verification models")
    data = {name: read_records(cfg.resolve_path(getattr(cfg.data, name)),
                              require_teacher=(name == "train" and cfg.objective.type == "candidate_distillation"),
                              max_candidates=cfg.data.max_candidates) for name in ("train", "validation")}
    check_split_isolation(data)
    output = cfg.resolve_path(cfg.summitConfig.output.directory)
    # Never overwrite an old run, including failed runs.
    output.mkdir(parents=True, exist_ok=False)
    add_code_paths(cfg)
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    started = time.monotonic()
    try:
        resolved_model = cfg.resolved_model()
        tokenizer = load_tokenizer(resolved_model.name, resolved_model.revision)
        model = resolve_adapter(cfg.model.adapter)(resolved_model, checkpoint=None)
        if not getattr(model, "manages_device_placement", False):
            model = model.to(cfg.training.device)
        if any(p.requires_grad and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("torch_single AdamW requires FP32 trainable parameters; use BF16 autocast for compute")
        model_config = getattr(getattr(model, "lm", getattr(model, "backbone", None)), "config", None)
        training_metadata = getattr(model, "training_metadata", None)
        logical_base = training_metadata.get("logical_base_parameters") if training_metadata else None
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        version_names = ["torch", "transformers", "safetensors"]
        if cfg.model.adaptation == "qlora":
            version_names.extend(["peft", "bitsandbytes"])
        manifest = {
            "schema_version": 1, "config": cfg.model_dump(mode="json"),
            "resolved_model_revision": getattr(model_config, "_commit_hash", None),
            "datasets": {name: dataset_summary(cfg.resolve_path(getattr(cfg.data, name)), rows) for name, rows in data.items()},
            "versions": {name: importlib.metadata.version(name) for name in version_names},
            "parameters": logical_base + trainable_parameters if logical_base else sum(p.numel() for p in model.parameters()),
            "trainable_parameters": trainable_parameters,
            "precision": training_metadata or {
                "parameters": "float32", "compute": cfg.model.dtype, "optimizer": "AdamW-float32"
            },
        }
        package = Path(__file__).resolve().parents[2]
        manifest["code_sha256"] = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(package.rglob("*.py"))}
        artifact_code = []
        for source in cfg.code:
            source = cfg.resolve_path(source)
            destination = output / "code" / source.name
            destination.parent.mkdir(exist_ok=True)
            sources = sorted(source.rglob("*.py")) if source.is_dir() else [source]
            if not sources or any(p.suffix != ".py" for p in sources):
                raise ValueError("custom code bundles require Python source files")
            for item in sources:
                target = destination / item.relative_to(source) if source.is_dir() else destination
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"duplicate custom code destination: {target}")
                shutil.copy2(item, target)
            artifact_code.append(str(destination.relative_to(output)))
        manifest["custom_code_sha256"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted((output / "code").rglob("*.py"))}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        before, before_predictions = evaluate(model, tokenizer, data["validation"], cfg.data.max_length, cfg.training.batch_size)
        print(canonical_json({"event": "evaluation", "stage": "before", "metrics": before}), flush=True)
        (output / "predictions-before.jsonl").write_text("".join(canonical_json(p) + "\n" for p in before_predictions))
        reversed_before = None
        if cfg.training.evaluate_reversed_options:
            reversed_before, reversed_predictions = evaluate(model, tokenizer, reverse_options(data["validation"]),
                                                             cfg.data.max_length, cfg.training.batch_size)
            print(canonical_json({"event": "evaluation", "stage": "reversed_before", "metrics": reversed_before}), flush=True)
            (output / "predictions-reversed-before.jsonl").write_text("".join(canonical_json(p) + "\n" for p in reversed_predictions))
        optimizer = torch.optim.AdamW(optimizer_parameters(model, cfg.training), lr=cfg.training.learning_rate,
                                      weight_decay=cfg.training.weight_decay, foreach=False)
        effective_batch = cfg.training.batch_size * cfg.training.gradient_accumulation
        steps = 0
        rng = random.Random(cfg.training.seed)
        if cfg.training.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with (output / "metrics.jsonl").open("x") as metrics_stream:
            for epoch in range(cfg.training.epochs):
                rows = list(data["train"])
                rng.shuffle(rows)
                for offset in range(0, len(rows), effective_batch):
                    group = rows[offset:offset + effective_batch]
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    for micro in range(0, len(group), cfg.training.batch_size):
                        batch = group[micro:micro + cfg.training.batch_size]
                        loss = decision_loss(model.score(batch, tokenizer, cfg.data.max_length), batch, cfg.objective)
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite training loss")
                        scaled = loss * (len(batch) / len(group))
                        scaled.backward()
                        total_loss += float(scaled.detach())
                        del loss, scaled
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    metrics = {"step": steps, "epoch": epoch, "examples": len(group), "loss": total_loss, "grad_norm": float(norm)}
                    metrics_stream.write(canonical_json(metrics) + "\n")
                    metrics_stream.flush()
                    print(canonical_json(metrics), flush=True)
                    if cfg.training.max_steps and steps >= cfg.training.max_steps:
                        break
                if cfg.training.max_steps and steps >= cfg.training.max_steps:
                    break
        after, predictions = evaluate(model, tokenizer, data["validation"], cfg.data.max_length, cfg.training.batch_size)
        print(canonical_json({"event": "evaluation", "stage": "after", "metrics": after}), flush=True)
        (output / "evaluation.json").write_text(json.dumps({"before": before, "after": after}, indent=2) + "\n")
        (output / "predictions.jsonl").write_text("".join(canonical_json(p) + "\n" for p in predictions))
        if cfg.training.evaluate_reversed_options:
            reversed_after, reversed_predictions = evaluate(model, tokenizer, reverse_options(data["validation"]),
                                                            cfg.data.max_length, cfg.training.batch_size)
            print(canonical_json({"event": "evaluation", "stage": "reversed_after", "metrics": reversed_after}), flush=True)
            (output / "evaluation-reversed.json").write_text(json.dumps({"before": reversed_before, "after": reversed_after}, indent=2) + "\n")
            (output / "predictions-reversed.jsonl").write_text("".join(canonical_json(p) + "\n" for p in reversed_predictions))
        model.save(output)
        tokenizer.save_pretrained(output / "tokenizer")
        (output / "adapter.json").write_text(json.dumps({"schema_version": 1, "model": resolved_model.model_dump(mode="json"),
                                                         "max_length": cfg.data.max_length, "code": artifact_code}, indent=2) + "\n")
        peak = torch.cuda.max_memory_allocated() if cfg.training.device == "cuda" else None
        del optimizer, model
        gc.collect()
        if cfg.training.device == "cuda":
            torch.cuda.empty_cache()
        reloaded, saved_tokenizer, _ = load_artifact(output, cfg.training.device)
        _, probe = evaluate(reloaded, saved_tokenizer, data["validation"][:2], cfg.data.max_length, cfg.training.batch_size)
        for expected, actual in zip(predictions[:2], probe):
            if expected["choice"] != actual["choice"] or any(abs(expected["probabilities"][k] - v) > 1e-4 for k, v in actual["probabilities"].items()):
                raise ValueError("checkpoint reload changed predictions")
        print(canonical_json({"event": "checkpoint_reload_verified", "steps": steps}), flush=True)
        destination = cfg.summitConfig.output.hf_push
        complete = {"status": "completed", "steps": steps, "reload_verified": True,
                    "elapsed_seconds": time.monotonic() - started, "peak_allocated_bytes": peak,
                    "elapsed_scope": "training, evaluation, save and reload; excludes HF publication",
                    "hf_repo": destination.repo if destination else None}
        if destination:
            from .publication import publish_artifact
            revision = publish_artifact(output, destination, complete)
            print(canonical_json({"event": "checkpoint_published", "repo": destination.repo, "revision": revision}), flush=True)
        (output / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
        print(f"decision training completed: {output}", flush=True)
        return complete
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
        # A publication failure must not leave a success marker for automation.
        (output / "complete.json").unlink(missing_ok=True)
        raise


def main():
    from summit.config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    from .config import DecisionJobConfig
    if not isinstance(cfg, DecisionJobConfig):
        parser.error("this trainer requires recipe: decision")
    train(cfg)


if __name__ == "__main__":
    main()
