"""Auditable native-format dataset adapters; no network or teacher calls."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
import zipfile

import yaml
from pydantic import Field

from .config import StrictModel
from .data import DecisionRecord, canonical_json, check_split_isolation, digest
from .dataset import write_dataset


VERSION = "native-decision-v3"
CATALOG = {
    "contractnli": {"license": "CC-BY-4.0", "url": "https://stanfordnlp.github.io/contract-nli/",
                    "attribution": "Yuta Koreeda and Christopher D. Manning, ContractNLI (2021)", "restricted": False},
    "clinc150": {"license": "CC-BY-3.0", "url": "https://github.com/clinc/oos-eval",
                 "attribution": "Stefan Larson et al., An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction (2019)", "restricted": False},
    "anli": {"license": "CC-BY-NC-4.0", "url": "https://github.com/facebookresearch/anli",
             "attribution": "Yixin Nie et al., Adversarial NLI (2020)", "restricted": True},
    "sharc": {"license": "UNRESOLVED", "url": "https://sharc-data.github.io/",
              "attribution": "Marzieh Saeidi et al., Interpretation of Natural Language Rules in Conversational Machine Reading (2018)", "restricted": True},
}
NLI_CHOICES = {"e": "The evidence supports the statement.", "c": "The evidence contradicts the statement.",
               "n": "The evidence neither supports nor contradicts the statement."}
SHARC_CHOICES = {"yes": "Yes.", "no": "No.", "irrelevant": "The rule does not apply to this question.",
                 "clarify": "Ask for more information before answering yes or no."}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_id(source, text):
    return f"{source}:{digest(normalized(text))}"


def destination(group, official_split, seed=42, calibration_percent=5):
    split = {"dev": "validation", "val": "validation"}.get(official_split, official_split)
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown original split: {official_split}")
    if split == "train" and int(digest([seed, group])[:8], 16) % 100 < calibration_percent:
        return "calibration"
    return split


def record(source, original_id, group, state, question, choices, label, provenance, seed):
    if label not in choices:
        raise ValueError(f"unknown {source} label: {label}")
    ordered = list(choices)
    random.Random(digest([seed, source, str(original_id)])).shuffle(ordered)
    options = [{"id": f"option_{i}", "description": choices[key]} for i, key in enumerate(ordered)]
    return DecisionRecord.model_validate({
        "id": f"{source}:{original_id}", "group_id": group, "state": state, "question": question,
        "candidate_options": options, "target": {"correct_option_id": options[ordered.index(label)]["id"]},
        "provenance": {"source": source, "converter": VERSION, "original_id": str(original_id),
                       "reference_label": label, **provenance},
    })


def convert_anli(rows, provenance, seed=42):
    for item in rows:
        # Annotator explanations and verifier labels never enter state/question.
        yield record("anli", item["uid"], group_id("anli", item["premise"]),
                     {"premise": item["premise"]}, f"Assess this statement: {item['hypothesis']}",
                     NLI_CHOICES, item["label"], provenance, seed)


def convert_sharc(rows, provenance, seed=42):
    for item in rows:
        answer = item["answer"].strip()
        if not answer:
            raise ValueError("ShARC record has no reference answer; do not invent labels")
        label = answer.lower() if answer.lower() in {"yes", "no", "irrelevant"} else "clarify"
        history = [{"question": h["follow_up_question"], "answer": h["follow_up_answer"]} for h in item["history"]]
        yield record("sharc", item["utterance_id"], group_id("sharc", item["snippet"]),
                     {"rule": item["snippet"], "scenario": item["scenario"], "observed_history": history},
                     item["question"], SHARC_CHOICES, label,
                     {**provenance, "tree_id": item["tree_id"], "conversion": "follow_up_generation_to_clarification_class"}, seed)


def convert_contractnli(document, provenance, seed=42):
    mapping = {"Entailment": "e", "Contradiction": "c", "NotMentioned": "n"}
    for item in document["documents"]:
        if len(item["annotation_sets"]) != 1:
            raise ValueError("ContractNLI requires exactly one resolved annotation set")
        for key, annotation in item["annotation_sets"][0]["annotations"].items():
            # Retain the FULL contract; never select evidence using gold spans.
            yield record("contractnli", f"{item['id']}:{key}", group_id("contractnli", item["text"]),
                         {"contract": item["text"]}, f"Assess this statement: {document['labels'][key]['hypothesis']}",
                         NLI_CHOICES, mapping[annotation["choice"]],
                         {**provenance, "document_id": item["id"], "hypothesis_id": key}, seed)


def features(text):
    return Counter(re.findall(r"[a-z0-9]+", normalized(text)))


class IntentRetriever:
    """Small train-only TF-IDF centroid retriever, with leave-query-group-out scoring."""

    def __init__(self, rows, labels):
        self.labels = labels
        self.counts = {label: Counter() for label in labels}
        self.examples = defaultdict(list)
        self.queries = defaultdict(lambda: defaultdict(Counter))
        df = Counter()
        for text, label in rows:
            if label == "oos":
                continue
            self.examples[label].append(text)
            values = features(text)
            self.counts[label].update(values)
            self.queries[normalized(text)][label].update(values)
            df.update(values.keys())
        self.idf = {token: math.log((1 + len(rows)) / (1 + count)) + 1 for token, count in df.items()}
        self.vectors = {label: {key: value * self.idf.get(key, 1) for key, value in values.items()}
                        for label, values in self.counts.items()}
        self.norms = {label: sum(value * value for value in values.values()) for label, values in self.vectors.items()}

    def description(self, label, query, count=2):
        """Describe an intent using only capped training rows, excluding this query."""
        if label not in self.labels:
            raise ValueError(f"unknown intent label: {label}")
        values, seen = [], set()
        for text in sorted(self.examples[label], key=lambda value: digest([label, normalized(value)])):
            key = normalized(text)
            if key == normalized(query) or key in seen:
                continue
            values.append(text)
            seen.add(key)
            if len(values) == count:
                break
        name = label.replace("_", " ")
        if not values:
            return f"Intent: {name}."
        return f"Intent: {name}. Training examples: " + canonical_json(values) + "."

    def shortlist(self, text, count):
        query = features(text)
        q = {key: value * self.idf.get(key, 1) for key, value in query.items()}
        ranked = []
        for label in self.labels:
            weights = self.vectors[label]
            removed = {key: value * self.idf.get(key, 1) for key, value in
                       self.queries.get(normalized(text), {}).get(label, {}).items()}
            norm_squared = self.norms[label] + sum(value * value - 2 * value * weights[key] for key, value in removed.items())
            norm = math.sqrt(max(0, norm_squared))
            similarity = sum(value * (weights.get(key, 0) - removed.get(key, 0)) for key, value in q.items()) / (norm or 1)
            ranked.append((-similarity, digest([normalized(text), label]), label))
        return [label for _, _, label in sorted(ranked)[:count]]


def convert_clinc150(data, provenance, seed=42, candidate_count=8, calibration_percent=5, max_train_rows=None):
    if not 2 <= candidate_count <= 26:
        raise ValueError("candidate_count must be 2..26 including none")
    labels = sorted({label for _, label in data["train"]})
    if "oos" in labels:
        raise ValueError("oos rows belong in oos_train, not train")
    if candidate_count - 1 > len(labels):
        raise ValueError("too few intent labels for requested candidate count")
    heldout = {normalized(text) for key in ("val", "test", "oos_val", "oos_test") for text, _ in data[key]}
    eligible = defaultdict(list)
    for text, label in data["train"] + data["oos_train"]:
        group = group_id("clinc150", text)
        if normalized(text) not in heldout and destination(group, "train", seed, calibration_percent) == "train":
            eligible[group].append((text, label))
    fit_rows = []
    for group in sorted(eligible, key=lambda key: digest([seed, key])):
        if max_train_rows is None or len(fit_rows) + len(eligible[group]) <= max_train_rows:
            fit_rows.extend(eligible[group])
    retriever = IntentRetriever(fit_rows, labels)
    fit_hash = digest(fit_rows)
    for key in ("train", "val", "test", "oos_train", "oos_val", "oos_test"):
        original_split = key.removeprefix("oos_")
        for index, (text, label) in enumerate(data[key]):
            if (key.startswith("oos_") and label != "oos") or (not key.startswith("oos_") and label not in labels):
                raise ValueError("unexpected CLINC intent label")
            shortlist = retriever.shortlist(text, candidate_count - 1)
            choices = {intent: retriever.description(intent, text) for intent in shortlist}
            choices["__none__"] = (
                "None of the listed intents applies. Use this when the request is truly outside the supported "
                "taxonomy or when its correct intent is absent from the listed options."
            )
            target = label if label in shortlist else "__none__"
            negative_type = "true_oos" if label == "oos" else "in_scope" if label in shortlist else "shortlist_miss"
            group = group_id("clinc150", text)
            yield destination(group, original_split, seed, calibration_percent), record(
                "clinc150", f"{key}:{index}", group, {"request": text}, "Which listed intent best matches the request?",
                choices, target, {**provenance, "original_split": original_split, "original_intent": label,
                                  "gold_in_candidates": label in shortlist, "negative_type": negative_type,
                                  "candidate_builder": "train-tfidf-centroids-with-exemplars-v2",
                                  "retrieval_fit_sha256": fit_hash, "retrieval_training_rows": len(fit_rows)}, seed)


class SourceSpec(StrictModel):
    source: str
    path: Path
    revision: str = Field(min_length=1)
    split: str | None = None
    member: str | None = None
    research_only: bool = False
    max_train_rows: int | None = Field(default=None, gt=0)


class BuildConfig(StrictModel):
    sources: list[SourceSpec] = Field(min_length=1)
    seed: int = 42
    calibration_percent: int = Field(default=5, ge=0, le=30)
    candidate_count: int = Field(default=8, ge=2, le=26)
    overlap_policy: str = "error"


def read_native(path, member=None):
    content = path.read_bytes()
    if member:
        with zipfile.ZipFile(path) as archive:
            content = archive.read(member)  # No archive extraction/path traversal.
    try:
        return json.loads(content), hashlib.sha256(content).hexdigest()
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.decode().splitlines() if line.strip()], hashlib.sha256(content).hexdigest()


def isolate(splits, policy):
    if policy not in {"error", "quarantine"}:
        raise ValueError("overlap_policy must be error or quarantine")
    if policy == "error":
        check_split_isolation(splits)
        return splits, []
    # Never move held-out rows into training. Keep test, quarantine any entire
    # lower-priority group sharing a group, input state, or ID with a held-out row.
    result, exclusions, owned = {}, [], set()
    for split in ("test", "validation", "calibration", "train"):
        split_keys = set()
        groups = defaultdict(list)
        for row in splits.get(split, []):
            groups[row.group_id].append(row)
        result[split] = []
        for group, rows in sorted(groups.items()):
            keys = {key for row in rows for key in (("group", group), ("state", digest(row.state)), ("id", row.id))}
            if keys & owned:
                exclusions.extend({"id": row.id, "split": split, "reason": "cross_split_group_overlap"} for row in rows)
            else:
                result[split].extend(rows)
                split_keys.update(keys)
        owned.update(split_keys)
    check_split_isolation(result)
    return result, exclusions


def build(config, base):
    splits = {name: [] for name in ("train", "validation", "calibration", "test")}
    source_manifest, limits = [], {}
    for spec in config.sources:
        if spec.source not in CATALOG:
            raise ValueError(f"unsupported source: {spec.source}")
        info = CATALOG[spec.source]
        if info["restricted"] and not spec.research_only:
            raise ValueError(f"{spec.source} is excluded from publication recipes ({info['license']}); explicit research_only acknowledgement required")
        path = (base / spec.path).resolve()
        native, content_hash = read_native(path, spec.member)
        provenance = {"revision": spec.revision, "original_split": spec.split,
                      "native_sha256": content_hash, "license": info["license"], "research_only": spec.research_only}
        source_manifest.append({**spec.model_dump(mode="json"), **info, "native_sha256": content_hash,
                                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if spec.max_train_rows is not None:
            if spec.source in limits and limits[spec.source] != spec.max_train_rows:
                raise ValueError("all shards of a source must use the same training limit")
            limits[spec.source] = spec.max_train_rows
        if spec.source == "clinc150":
            for split, row in convert_clinc150(native, provenance, config.seed, config.candidate_count, config.calibration_percent, spec.max_train_rows):
                splits[split].append(row)
        else:
            convert = {"anli": convert_anli, "sharc": convert_sharc, "contractnli": convert_contractnli}[spec.source]
            for row in convert(native, provenance, config.seed):
                splits[destination(row.group_id, spec.split, config.seed, config.calibration_percent)].append(row)
    # Detect repeated native shards/IDs even within a split.
    for name, rows in splits.items():
        if len({row.id for row in rows}) != len(rows):
            raise ValueError(f"duplicate record IDs in {name}")
    splits, exclusions = isolate(splits, config.overlap_policy)
    selected, counts = [], Counter()
    groups = defaultdict(list)
    for row in splits["train"]:
        groups[row.group_id].append(row)
    for group in sorted(groups, key=lambda key: digest([config.seed, key])):
        rows = groups[group]
        source = rows[0].provenance["source"]
        if source in limits and counts[source] + len(rows) > limits[source]:
            exclusions.extend({"id": row.id, "split": "train", "reason": "group_preserving_train_cap"} for row in rows)
        else:
            selected.extend(rows)
            counts[source] += len(rows)
    splits["train"] = selected
    for rows in splits.values():
        rows.sort(key=lambda row: row.id)
    splits = {name: rows for name, rows in splits.items() if rows}
    if "train" not in splits or "validation" not in splits:
        raise ValueError("build must retain nonempty train and validation splits")
    check_split_isolation(splits)
    summary = {name: {"by_source": dict(Counter(r.provenance["source"] for r in rows)),
                      "reference_classes": dict(Counter(f"{r.provenance['source']}:{r.provenance['reference_label']}" for r in rows)),
                      "characters_max": max(len(canonical_json(r.payload())) for r in rows)} for name, rows in splits.items()}
    manifest = {"converter": VERSION, "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "config": config.model_dump(mode="json"), "sources": source_manifest,
                "publication_status": "research-only" if any(s.research_only for s in config.sources) else "attribution-and-release-review-required",
                "statistics": summary, "excluded_rows": len(exclusions),
                "limitations": ["public benchmarks may occur in pretraining", "CLINC task is retrieved-candidate routing, not original 150-way accuracy",
                                "intent exemplars are automatic train-only descriptions, not human-authored boundaries",
                                "no semantic near-duplicate audit", "no token-length filtering or human semantic audit yet"]}
    return splits, manifest, exclusions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = BuildConfig.model_validate(yaml.safe_load(args.config.read_text()))
    if args.output.exists():
        parser.error("output already exists")
    splits, manifest, exclusions = build(config, args.config.resolve().parent)
    summaries = write_dataset(args.output, splits, manifest)
    (args.output / "exclusions.jsonl").write_text("".join(canonical_json(row) + "\n" for row in exclusions))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
