"""Bounded native-thinking teacher reference; never used as a soft-target cache."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import string

import httpx
import yaml

from .data import canonical_json, digest, read_records
from .prompts import prompt
from .teacher import TeacherConfig
from .train import load_tokenizer


def parse_answer(choice, options):
    text = choice.get("text", "").split("</think>")[-1].strip()
    if choice.get("finish_reason") == "length" or text not in string.ascii_uppercase[:len(options)] or len(text) != 1:
        return None
    return options[string.ascii_uppercase.index(text)].id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--retry-incomplete-from", type=Path,
                        help="explicit new paid collection of only truncated/unparseable rows from a completed reference")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = TeacherConfig.model_validate(yaml.safe_load(args.config.read_text()))
    if args.max_output_tokens < 1:
        parser.error("max-output-tokens must be positive")
    tokenizer = load_tokenizer(config.tokenizer, config.tokenizer_revision)
    rows = read_records(args.data)[:config.max_records]
    if args.retry_incomplete_from:
        previous = args.retry_incomplete_from
        if not (previous / 'complete.json').exists():
            parser.error('previous reference must have completed before an explicit repair')
        manifest = json.loads((previous / 'manifest.json').read_text())
        if any(manifest['config'][k] != getattr(config, k) for k in ('model', 'tokenizer', 'tokenizer_revision')):
            parser.error('repair model/tokenizer mismatch')
        if not set(manifest['inputs']).issubset({row.id for row in rows}):
            parser.error('repair dataset must include every original reference input')
        rows = [row for row in rows if row.id in manifest['inputs']]
        incomplete = []
        for row in rows:
            if manifest['inputs'].get(row.id) != row.input_hash():
                parser.error('repair input hash mismatch')
            payload = json.loads((previous / f'response-{digest(row.id)}.json').read_text())
            if parse_answer(payload['choices'][0], row.candidate_options) is None or '</think>' not in payload['choices'][0]['text']:
                incomplete.append(row)
        rows = incomplete
        if not rows:
            parser.error('no incomplete rows to repair; no request sent')
    sequences = [tokenizer.encode(prompt(row, tokenizer, thinking=True), add_special_tokens=False) for row in rows]
    if any(len(seq) > config.max_input_tokens for seq in sequences):
        parser.error("overlength reference input; do not truncate evidence")
    estimate = sum((len(seq) * config.input_usd_per_million + args.max_output_tokens * config.output_usd_per_million) / 1e6
                   for seq in sequences)
    if len(rows) > config.max_requests or estimate > config.max_usd:
        parser.error("reference probe exceeds request or estimated dollar budget")
    plan = {"scope": "native-thinking reference, not a candidate distribution", "records": len(rows),
            "config": config.model_dump(), "max_output_tokens": args.max_output_tokens, "estimated_max_usd": estimate,
            "inputs": {r.id: r.input_hash() for r in rows}}
    if args.retry_incomplete_from:
        plan['repair_of'] = str(args.retry_incomplete_from)
        plan['repair_manifest_sha256'] = digest(manifest)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    from summit.env import load_env
    key = load_env(".env").get("FIREWORKS_API_KEY")
    if not key:
        parser.error("FIREWORKS_API_KEY is required")
    # Single-shot directory: rerunning never silently duplicates a previous spend.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    results, estimated_used = [], 0.0
    with httpx.Client(timeout=180, headers={"Authorization": "Bearer " + key}) as client:
        for row, tokens in zip(rows, sequences):
            reserved = (len(tokens) * config.input_usd_per_million + args.max_output_tokens * config.output_usd_per_million) / 1e6
            with (args.output / "requests.jsonl").open("a") as stream:
                stream.write(canonical_json({"id": row.id, "reserved_usd": reserved, "at": datetime.now(timezone.utc).isoformat()}) + "\n")
                stream.flush()
                import os
                os.fsync(stream.fileno())
            response = client.post("https://api.fireworks.ai/inference/v1/completions", json={
                "model": config.model, "prompt": tokens, "max_tokens": args.max_output_tokens,
                "temperature": 0, "context_length_exceeded_behavior": "error"})
            if response.status_code != 200:
                raise ValueError(f"reference teacher HTTP {response.status_code}; no retry")
            payload = response.json()
            (args.output / f"response-{digest(row.id)}.json").write_text(json.dumps(payload, indent=2) + "\n")
            usage = payload.get("usage", {})
            if usage.get("prompt_tokens") != len(tokens) or not 0 <= usage.get("completion_tokens", -1) <= args.max_output_tokens:
                raise ValueError("reference response usage violates bounded request")
            estimated_used += (usage["prompt_tokens"] * config.input_usd_per_million + usage["completion_tokens"] * config.output_usd_per_million) / 1e6
            choice = parse_answer(payload["choices"][0], row.candidate_options)
            results.append({"id": row.id, "input_sha256": row.input_hash(), "choice": choice,
                            "target": row.target.correct_option_id, "usage": usage})
            print(canonical_json({"id": row.id, "parsed": choice is not None, "correct": choice == row.target.correct_option_id}), flush=True)
    (args.output / "predictions.jsonl").write_text("".join(canonical_json(r) + "\n" for r in results))
    report = {"records": len(results), "correct": sum(r["choice"] == r["target"] for r in results),
              "unparsed_or_truncated": sum(r["choice"] is None for r in results), "estimated_usd": estimated_used,
              "billing": "undiscounted token-rate estimate, not an invoice"}
    (args.output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
