"""Deterministic policy fixtures and offline teacher-cache assembly (no API calls)."""

import argparse
import json
from pathlib import Path
import random

from .data import DecisionRecord, TeacherTarget, canonical_json, digest, read_records, check_split_isolation, dataset_summary


GENERATOR_VERSION = "policy-v1"


def policy_decision(policy, facts):
    """Executable reference: known disqualifiers win; missing evidence -> clarify."""
    required = ("amount", "age_days", "verified")
    if ((facts.get("amount") is not None and facts["amount"] > policy["max_amount"])
            or (facts.get("age_days") is not None and facts["age_days"] > policy["max_age_days"])
            or facts.get("verified") is False):
        return "deny"
    if any(facts.get(key) is None for key in required):
        return "clarify"
    return "approve"


def generate_records(*, policies=30, seed=42):
    if policies < 6:
        raise ValueError("at least six policy groups are required")
    rng = random.Random(seed)
    groups = list(range(policies))
    rng.shuffle(groups)
    n_test = max(1, policies // 10)
    n_validation = max(1, policies // 5)
    membership = {group: ("test" if i < n_test else "validation" if i < n_test + n_validation else "train")
                  for i, group in enumerate(groups)}
    splits = {name: [] for name in ("train", "validation", "test")}
    for group in range(policies):
        # Unique threshold pairs define policy groups. Every scenario derived from
        # a policy stays in one split; the generator template itself is shared.
        policy = {"max_amount": 50 + group * 7, "max_age_days": 7 + group * 3}
        base = {"amount": policy["max_amount"], "age_days": policy["max_age_days"], "verified": True}
        cases = [base, {**base, "amount": policy["max_amount"] - 1},
                 {**base, "age_days": policy["max_age_days"] - 1},
                 {**base, "amount": policy["max_amount"] + 1},
                 {**base, "age_days": policy["max_age_days"] + 1},
                 {**base, "verified": False}, {**base, "amount": None},
                 {**base, "age_days": None}, {**base, "verified": None},
                 {**base, "amount": None, "verified": False}]
        for case, facts in enumerate(cases):
            answers = ["approve", "deny", "clarify"]
            rng.shuffle(answers)
            # Opaque IDs and independently shuffled order; reference is separate.
            candidates = [{"id": f"option_{i}", "description": canonical_json({"action": answer})}
                          for i, answer in enumerate(answers)]
            state = {"policy": policy, "request": facts,
                     "rules": "Approve iff amount <= max_amount, age_days <= max_age_days, and verified is true. "
                              "Deny if any known condition fails, even if other evidence is missing. "
                              "Otherwise ask for clarification if any required value is null."}
            row = DecisionRecord.model_validate({
                "id": f"{GENERATOR_VERSION}-{seed}-{group}-{case}", "group_id": f"{GENERATOR_VERSION}-policy-{group}",
                "state": state, "question": "Which action should the refund service take?",
                "candidate_options": candidates,
                "target": {"correct_option_id": candidates[answers.index(policy_decision(policy, facts))]["id"]},
                "provenance": {"source": "summit-original-policy", "generator": GENERATOR_VERSION,
                               "seed": seed, "reference": "executable_rules", "case": case},
            })
            splits[membership[group]].append(row)
    check_split_isolation(splits)
    return splits


def write_dataset(output, splits, metadata):
    # No overwrite: dataset hashes must remain stable once used by an experiment.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical_json(row.model_dump(mode="json", exclude_none=True)) + "\n" for row in rows))
        summaries[name] = dataset_summary(path, rows)
    (output / "manifest.json").write_text(json.dumps({"schema_version": 1, **metadata, "splits": summaries}, indent=2) + "\n")
    return summaries


def attach_teacher(records, cache):
    """Strict one-to-one join by ID plus content hash; never change references."""
    by_id = {}
    for item in cache:
        if set(item) != {"id", "teacher"}:
            raise ValueError("cache rows require exactly id and teacher")
        if item["id"] in by_id:
            raise ValueError(f"duplicate teacher cache ID: {item['id']}")
        by_id[item["id"]] = TeacherTarget.model_validate(item["teacher"])
    if set(by_id) != {row.id for row in records}:
        raise ValueError("teacher cache must cover exactly the requested dataset IDs")
    combined = []
    for row in records:
        if row.teacher is not None:
            raise ValueError(f"record already contains a teacher: {row.id}")
        value = row.model_dump(mode="json")
        value["teacher"] = by_id[row.id].model_dump(mode="json")
        combined.append(DecisionRecord.model_validate(value))
    return combined


def teacher_requests(records):
    # This export never includes reference labels, provenance, or source groups.
    return [{"id": row.id, "input_sha256": row.input_hash(), "input": row.payload()} for row in records]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate original executable-policy fixtures, not a benchmark")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--policies", type=int, default=30)
    generate.add_argument("--seed", type=int, default=42)
    validate = sub.add_parser("validate", help="Validate canonical splits and exact/group overlap")
    validate.add_argument("--train", type=Path, required=True)
    validate.add_argument("--validation", type=Path, required=True)
    validate.add_argument("--test", type=Path)
    requests = sub.add_parser("teacher-inputs", help="Export target-free inputs; does not call a teacher")
    requests.add_argument("--data", type=Path, required=True)
    requests.add_argument("--output", type=Path, required=True)
    attach = sub.add_parser("attach-teacher", help="Validate and attach an externally collected probability cache")
    attach.add_argument("--data", type=Path, required=True)
    attach.add_argument("--cache", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = write_dataset(args.output, generate_records(policies=args.policies, seed=args.seed),
                               {"generator": GENERATOR_VERSION, "seed": args.seed,
                                "scope": "synthetic workflow fixture; shared rule template, held-out threshold pairs"})
        print(json.dumps(result, indent=2))
    elif args.command == "validate":
        splits = {name: read_records(getattr(args, name)) for name in ("train", "validation", "test") if getattr(args, name)}
        check_split_isolation(splits)
        print(json.dumps({name: dataset_summary(getattr(args, name), rows) for name, rows in splits.items()}, indent=2))
    else:
        records = read_records(args.data)
        if args.command == "teacher-inputs":
            values = teacher_requests(records)
        else:
            cache = [json.loads(line) for line in args.cache.read_text().splitlines() if line.strip()]
            values = [row.model_dump(mode="json") for row in attach_teacher(records, cache)]
        with args.output.open("x") as stream:
            stream.writelines(canonical_json(value) + "\n" for value in values)


if __name__ == "__main__":
    main()
