"""Dependency-light canonical records, provenance, and split validation."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .config import StrictModel


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Candidate(StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Target(StrictModel):
    correct_option_id: str


class TeacherTarget(StrictModel):
    input_sha256: str
    model: str = Field(min_length=1)
    method: str = Field(min_length=1)
    probabilities: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionInput(StrictModel):
    id: str = Field(min_length=1)
    state: Any
    question: str = Field(min_length=1)
    candidate_options: list[Candidate] = Field(min_length=2, max_length=26)

    def payload(self):
        return {
            "state": self.state,
            "question": self.question,
            "candidate_options": [option.model_dump() for option in self.candidate_options],
        }

    def input_hash(self):
        return digest(self.payload())

    @model_validator(mode="after")
    def validate_candidates(self):
        canonical_json(self.state)  # Reject non-JSON values and NaN nested in state.
        ids = [option.id for option in self.candidate_options]
        descriptions = [" ".join(option.description.lower().split()) for option in self.candidate_options]
        if len(set(ids)) != len(ids) or len(set(descriptions)) != len(descriptions):
            raise ValueError("candidate IDs and normalized descriptions must be unique")
        return self


class DecisionRecord(DecisionInput):
    group_id: str = Field(min_length=1)
    target: Target
    provenance: dict[str, Any]
    teacher: TeacherTarget | None = None

    @model_validator(mode="after")
    def validate_targets(self):
        ids = [option.id for option in self.candidate_options]
        if self.target.correct_option_id not in ids:
            raise ValueError("reference target is not a candidate")
        if self.teacher:
            teacher = self.teacher
            if teacher.input_sha256 != self.input_hash():
                raise ValueError("stale teacher target: input hash does not match")
            p = teacher.probabilities
            if set(p) != set(ids):
                raise ValueError("teacher probabilities must cover exactly the candidate IDs")
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in p.values()) or not math.isclose(
                sum(p.values()), 1.0, abs_tol=1e-6
            ):
                raise ValueError("teacher probabilities must be finite, nonnegative, and sum to one")
        return self


def read_records(path: Path, *, require_teacher=False, max_candidates=26):
    records = []
    ids = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = DecisionRecord.model_validate_json(line)
                if row.id in ids:
                    raise ValueError("duplicate record ID")
                if len(row.candidate_options) > max_candidates:
                    raise ValueError("candidate count exceeds configured limit")
                if require_teacher and row.teacher is None:
                    raise ValueError("distillation requires cached teacher targets on every training row")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            ids.add(row.id)
            records.append(row)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def check_split_isolation(splits: dict[str, list[DecisionRecord]]):
    owners = {}
    for split, records in splits.items():
        for row in records:
            # The same state belongs together even when the question/options differ.
            for key in (("id", row.id), ("group", row.group_id), ("state", digest(row.state))):
                if key in owners and owners[key] != split:
                    raise ValueError(f"cross-split {key[0]} overlap between {owners[key]} and {split}: {row.id}")
                owners[key] = split


def dataset_summary(path, records):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(records),
        "groups": len({r.group_id for r in records}),
        "max_candidates": max(len(r.candidate_options) for r in records),
        "teacher_rows": sum(r.teacher is not None for r in records),
    }
