"""Budgeted, resumable all-candidate echo scoring. Dry-run unless --execute."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import string
from typing import Literal

import httpx
from pydantic import Field
import yaml

from .config import StrictModel
from .data import canonical_json, digest, read_records, TeacherTarget
from .prompts import prompt, VERSION


class TeacherConfig(StrictModel):
    model: str = "accounts/fireworks/models/qwen3p8-max"
    tokenizer: str = "Qwen/Qwen3.8-2.4T-A95B"
    tokenizer_revision: str = Field(min_length=1)
    prompt_mode: Literal["nonthinking", "forced_answer", "reasoning_cached"] = "nonthinking"
    max_input_tokens: int = Field(default=4096, gt=0)
    max_records: int = Field(default=8, gt=0)
    max_requests: int = Field(default=64, gt=0)
    max_usd: float = Field(gt=0)
    # Explicit rates must be rechecked at run time. No cache discounts assumed.
    input_usd_per_million: float = Field(gt=0)
    output_usd_per_million: float = Field(gt=0)


def prepare(row, tokenizer, config, rationale=None):
    text = prompt(row, tokenizer, thinking=config.prompt_mode != "nonthinking")
    if config.prompt_mode == "forced_answer":
        if not text.rstrip().endswith("<think>"):
            raise ValueError("forced_answer requires a verified open <think> template suffix")
        text += "</think>\n\n"
    elif config.prompt_mode == "reasoning_cached":
        if not text.rstrip().endswith("<think>") or not rationale or not rationale.rstrip().endswith("</think>"):
            raise ValueError("reasoning_cached requires a verified rationale ending at </think>")
        text += rationale + "\n\n"
    prefix = tokenizer.encode(text, add_special_tokens=False)
    sequences = []
    for letter in string.ascii_uppercase[:len(row.candidate_options)]:
        tokens = tokenizer.encode(text + letter, add_special_tokens=False)
        if tokens[:-1] != prefix or len(tokens) != len(prefix) + 1:
            raise ValueError(f"teacher answer {letter} is not a single token at the prompt boundary")
        if len(tokens) > config.max_input_tokens:
            raise ValueError(f"teacher input exceeds {config.max_input_tokens} tokens: {row.id}")
        sequences.append(tokens)
    if len({tokens[-1] for tokens in sequences}) != len(sequences):
        raise ValueError("teacher answer tokens are not distinct")
    return sequences


def parse_echo(response, tokens):
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    echoed = logprobs.get("token_ids") or choice.get("prompt_token_ids")
    values = logprobs.get("token_logprobs")
    # Stronger than the legacy OPD client's optional token-ID verification.
    if not echoed or echoed[:len(tokens)] != tokens:
        raise ValueError("teacher did not echo the exact requested token IDs")
    if not values or len(values) < len(tokens):
        raise ValueError("teacher returned incomplete prompt scores")
    value = values[len(tokens) - 1]
    if value is None or not math.isfinite(value) or value > 1e-6:
        raise ValueError("invalid teacher answer logprob")
    usage = response.get("usage") or {}
    if usage.get("prompt_tokens") != len(tokens) or not 0 <= usage.get("completion_tokens", -1) <= 1:
        raise ValueError("teacher usage does not match the bounded request")
    return float(value)


def normalize(scores):
    top = max(scores)
    weights = [math.exp(value - top) for value in scores]
    return [value / sum(weights) for value in weights]


def collect(records, tokenizer, config, output, *, execute=False, api_key=None, transport=None, rationales=None):
    records = records[:config.max_records]
    rationales = rationales or {}
    if (config.prompt_mode == "reasoning_cached") != bool(rationales):
        raise ValueError("rationales are required only for reasoning_cached mode")
    prepared = [(row, prepare(row, tokenizer, config, rationales.get(row.id))) for row in records]
    model_settings = {"model": config.model, "tokenizer": config.tokenizer,
                      "tokenizer_revision": config.tokenizer_revision, "prompt_version": VERSION,
                      "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest(),
                      "method": "all_candidate_single_token_echo", "reasoning": config.prompt_mode,
                      "input_rate": config.input_usd_per_million, "output_rate": config.output_usd_per_million}
    if config.prompt_mode == "reasoning_cached":
        model_settings["rationale_set_sha256"] = digest({row.id: digest(rationales[row.id]) for row in records})
    identity = digest({"settings": model_settings, "inputs": [(r.id, r.input_hash()) for r, _ in prepared]})
    planned_requests = sum(len(sequences) for _, sequences in prepared)
    planned_tokens = sum(len(tokens) for _, sequences in prepared for tokens in sequences)
    planned_cost = (planned_tokens * config.input_usd_per_million + planned_requests * config.output_usd_per_million) / 1e6
    plan = {"identity": identity, "settings": model_settings, "records": len(records), "requests": planned_requests,
            "input_tokens": planned_tokens, "estimated_max_usd": planned_cost, "max_usd": config.max_usd,
            "scope": "direct candidate preferences, not reasoning-enabled teacher accuracy"}
    if planned_requests > config.max_requests or planned_cost > config.max_usd:
        raise ValueError("planned collection exceeds request or estimated dollar cap")
    if not execute:
        return {"status": "dry_run", **plan}
    if not api_key and transport is None:
        raise ValueError("FIREWORKS_API_KEY is required for execution")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive advisory lock prevents concurrent collectors double-spending.
    import fcntl
    lock = (output / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("another collector holds this cache lock") from None
    try:
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["identity"] != identity:
                raise ValueError("cache directory belongs to different inputs/settings")
        else:
            if any(path.name != ".lock" for path in output.iterdir()):
                raise ValueError("nonempty cache directory has no manifest")
            manifest_path.write_text(json.dumps(plan, indent=2) + "\n")
        ledger_path = output / "requests.jsonl"
        previous = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
        reserved = sum(item["reserved_usd"] for item in previous)
        requests = len(previous)
        (output / "responses").mkdir(exist_ok=True)
    except BaseException:
        lock.close()
        raise
    try:
        with httpx.Client(timeout=60, headers={"Authorization": "Bearer " + (api_key or "test")}, transport=transport) as client:
            for row, sequences in prepared:
                scores, usage, response_models = [], [], []
                for option, tokens in zip(row.candidate_options, sequences):
                    key = digest([identity, row.id, option.id, tokens])
                    path = output / "responses" / f"{key}.json"
                    if path.exists():
                        response = json.loads(path.read_text())
                    else:
                        estimate = (len(tokens) * config.input_usd_per_million + config.output_usd_per_million) / 1e6
                        if requests >= config.max_requests or reserved + estimate > config.max_usd:
                            raise ValueError("remaining request/dollar budget exhausted (failed attempts remain reserved)")
                        # Reserve before networking; an interrupted/ambiguous attempt may have been billed.
                        entry = {"key": key, "record_id": row.id, "candidate_id": option.id,
                                 "reserved_usd": estimate, "input_tokens": len(tokens),
                                 "at": datetime.now(timezone.utc).isoformat()}
                        with ledger_path.open("a") as stream:
                            stream.write(canonical_json(entry) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        requests += 1
                        reserved += estimate
                        response_http = client.post("https://api.fireworks.ai/inference/v1/completions", json={
                            "model": config.model, "prompt": tokens, "echo": True, "return_token_ids": True,
                            "logprobs": 0, "max_tokens": 1, "temperature": 0,
                            "context_length_exceeded_behavior": "error"})
                        if response_http.status_code != 200:
                            raise ValueError(f"teacher HTTP {response_http.status_code}; no automatic retries")
                        response = response_http.json()
                        # Save even malformed responses for diagnostics; resume does not silently re-spend.
                        path.write_text(json.dumps(response) + "\n")
                    scores.append(parse_echo(response, tokens))
                    usage.append(response["usage"])
                    response_models.append(response.get("model"))
                teacher = TeacherTarget(input_sha256=row.input_hash(), model=config.model,
                    method="candidate_logprobs", probabilities={o.id: p for o, p in zip(row.candidate_options, normalize(scores))},
                    metadata={**model_settings, "cache_key": digest([identity, row.id, row.input_hash()]),
                              "rationale_sha256": digest(rationales[row.id]) if row.id in rationales else None,
                              "logprobs": dict(zip([o.id for o in row.candidate_options], scores)),
                              "usage": usage, "response_models": response_models})
                (output / f"target-{digest(row.id)}.json").write_text(canonical_json({"id": row.id, "teacher": teacher.model_dump(mode="json")}) + "\n")
        values = [json.loads((output / f"target-{digest(row.id)}.json").read_text()) for row in records]
        (output / "cache.jsonl").write_text("".join(canonical_json(value) + "\n" for value in values))
        report = {"status": "completed", "records": len(values), "requests_reserved": requests,
                  "estimated_usd_reserved": reserved, "billing": "conservative token-rate estimate, not provider invoice"}
        (output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reasoning-reference", type=Path, help="completed native-thinking reference directory")
    parser.add_argument("--reasoning-repair", type=Path, action="append", help="explicit completed repair for invalid traces only; repeat in acquisition order")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    config = TeacherConfig.model_validate(yaml.safe_load(args.config.read_text()))
    from .train import load_tokenizer
    tokenizer = load_tokenizer(config.tokenizer, config.tokenizer_revision)
    api_key = None
    if args.execute:
        from summit.env import load_env
        api_key = load_env(args.env_file).get("FIREWORKS_API_KEY")
    records = read_records(args.data)
    if args.reasoning_repair and not args.reasoning_reference:
        parser.error('--reasoning-repair requires --reasoning-reference')
    rationales = load_rationales(args.reasoning_reference, records[:config.max_records], config,
                                repair=args.reasoning_repair) if args.reasoning_reference else None
    if (config.prompt_mode == "reasoning_cached") != bool(args.reasoning_reference):
        parser.error("reasoning_cached requires --reasoning-reference, and only that mode accepts it")
    result = collect(records, tokenizer, config, args.output, execute=args.execute, api_key=api_key, rationales=rationales)
    print(json.dumps(result, indent=2))


def load_rationales(directory, records, config, repair=None):
    repairs = [] if repair is None else list(repair) if isinstance(repair, (list, tuple)) else [repair]
    if not (directory / "complete.json").exists():
        raise ValueError("reasoning reference is incomplete")
    manifest = json.loads((directory / "manifest.json").read_text())
    for key in ("model", "tokenizer", "tokenizer_revision"):
        if manifest["config"][key] != getattr(config, key):
            raise ValueError("reasoning reference model/tokenizer mismatch")
    rationales = {}
    for row in records:
        if manifest["inputs"].get(row.id) != row.input_hash():
            raise ValueError("reasoning reference input hash mismatch")
        response = json.loads((directory / f"response-{digest(row.id)}.json").read_text())
        choice = response["choices"][0]
        from .reasoning_reference import parse_answer
        if parse_answer(choice, row.candidate_options) is None or "</think>" not in choice["text"]:
            if repairs:
                repair_manifest = json.loads((repairs[0] / 'manifest.json').read_text())
                if repair_manifest.get('repair_manifest_sha256') != digest(manifest):
                    raise ValueError('repair is not bound to this original reference')
                rationales.update(load_rationales(repairs[0], [row], config, repair=repairs[1:]))
                continue
            raise ValueError("reasoning trace has no complete parseable answer")
        # Drop the generated final answer completely. Score every candidate after
        # the SAME self-generated trace; the reference label was never provided.
        rationales[row.id] = choice["text"].rsplit("</think>", 1)[0] + "</think>"
    return rationales


if __name__ == "__main__":
    main()
