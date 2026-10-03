import json
from pathlib import Path

import pytest

from summit.recipes.decision.sources import (
    BuildConfig, build, convert_anli, convert_sharc, convert_contractnli, convert_clinc150,
    destination, group_id, isolate, IntentRetriever,
)


def test_anli_explanations_never_enter_inputs():
    native = [{"uid": "1", "premise": "Cats are animals.", "hypothesis": "Cats are rocks.", "label": "c", "reason": "SECRET EXPLANATION"}]
    row = next(convert_anli(native, {"original_split": "train"}))
    assert "SECRET" not in json.dumps(row.payload())
    assert row.provenance["reference_label"] == "c"
    assert group_id("anli", "Cats  ARE animals.") == row.group_id


def test_sharc_hidden_evidence_is_not_observed_history():
    item = {"utterance_id": "1", "tree_id": "t", "snippet": "A rule", "question": "Can I?", "scenario": "A scenario",
            "history": [{"follow_up_question": "Paid?", "follow_up_answer": "Yes"}],
            "evidence": [{"follow_up_answer": "SECRET"}], "answer": "Are you employed?"}
    row = next(convert_sharc([item], {}))
    assert row.provenance["reference_label"] == "clarify"
    assert "SECRET" not in json.dumps(row.payload())
    assert "Are you employed?" not in json.dumps(row.payload())
    assert row.state["observed_history"][0]["answer"] == "Yes"


def contract_document():
    return {"documents": [{"id": 1, "text": "FULL CONTRACT INCLUDING EXCEPTIONS", "spans": [[0, 4]],
            "annotation_sets": [{"annotations": {"nda-1": {"choice": "NotMentioned", "spans": []}}}]}],
            "labels": {"nda-1": {"hypothesis": "Term lasts forever."}}}


def test_contract_uses_full_document_not_gold_spans():
    row = next(convert_contractnli(contract_document(), {}))
    assert row.state == {"contract": "FULL CONTRACT INCLUDING EXCEPTIONS"}
    assert row.provenance["reference_label"] == "n"
    assert "spans" not in json.dumps(row.payload())


def test_retrieval_does_not_inject_reference_label():
    data = {"train": [["flight plane ticket", "flight"], ["bank money balance", "bank"]],
            "val": [["plane ticket", "bank"]], "test": [["money", "bank"]],
            "oos_train": [], "oos_val": [], "oos_test": [["unrelated", "oos"]]}
    rows = list(convert_clinc150(data, {}, candidate_count=2, calibration_percent=0))
    val = next(r for split, r in rows if split == "validation")
    assert val.provenance["reference_label"] == "__none__"
    assert not val.provenance["gold_in_candidates"]
    assert val.provenance["negative_type"] == "shortlist_miss"
    assert any("flight" in o.description for o in val.candidate_options)
    data["val"][0][1] = "flight"
    changed = next(r for split, r in convert_clinc150(data, {}, candidate_count=2, calibration_percent=0) if split == "validation")
    assert changed.payload() == val.payload()
    assert changed.target != val.target


def test_retriever_leaves_own_query_out():
    retriever = IntentRetriever([("unique", "a"), ("other", "b")], ["a", "b"])
    # Both similarities become zero when the only member of a is the query.
    expected = IntentRetriever([], ["a", "b"]).shortlist("unique", 2)
    assert retriever.shortlist("unique", 2) == expected
    assert "unique" not in retriever.description("a", "unique")


def test_clinc_distinguishes_true_oos_from_shortlist_miss():
    data = {"train": [["flight plane ticket", "flight"], ["bank money balance", "bank"]],
            "val": [["plane ticket", "bank"]], "test": [], "oos_train": [],
            "oos_val": [["unrelated request", "oos"]], "oos_test": []}
    rows = [row for split, row in convert_clinc150(data, {}, candidate_count=2, calibration_percent=0)
            if split == "validation"]
    assert {row.provenance["negative_type"] for row in rows} == {"shortlist_miss", "true_oos"}
    assert all("outside the supported taxonomy" in next(
        option.description for option in row.candidate_options if option.id == row.target.correct_option_id)
        for row in rows)


def test_retriever_obeys_training_cap_and_never_fits_heldout_queries():
    data = {"train": [["flight", "flight"], ["bank", "bank"], ["shared", "bank"]],
            "val": [["shared", "bank"]], "test": [["test query", "flight"]],
            "oos_train": [], "oos_val": [], "oos_test": []}
    converted = list(convert_clinc150(data, {}, candidate_count=2, calibration_percent=0, max_train_rows=1))
    assert all(row.provenance["retrieval_training_rows"] == 1 for _, row in converted)


def test_quarantine_keeps_test_not_training():
    row = next(convert_anli([{"uid": "1", "premise": "Shared", "hypothesis": "X", "label": "n"}], {}))
    other = row.model_copy(update={"id": "new"})
    splits, removed = isolate({"train": [row], "test": [other]}, "quarantine")
    assert splits["train"] == [] and splits["test"] == [other]
    assert removed[0]["id"] == row.id
    with pytest.raises(ValueError, match="cross-split"):
        isolate({"train": [row], "test": [other]}, "error")
    sibling = row.model_copy(update={"id": "sibling", "group_id": "another-group"})
    retained, removed = isolate({"train": [row, sibling]}, "quarantine")
    assert len(retained["train"]) == 2 and not removed


def test_restricted_sources_require_explicit_acknowledgement(tmp_path):
    cfg = BuildConfig.model_validate({"sources": [{"source": "anli", "path": "missing", "revision": "v1"}]})
    with pytest.raises(ValueError, match="excluded from publication"):
        build(cfg, tmp_path)


def test_calibration_only_comes_from_training():
    assert destination("g", "test", calibration_percent=100) == "test"
    assert destination("g", "dev", calibration_percent=100) == "validation"
    assert destination("g", "train", calibration_percent=100) == "calibration"
