from summit.recipes.decision.data import DecisionRecord, TeacherTarget
from summit.recipes.decision.pilot import augment_order


def row(with_teacher=False):
    value = {
        "id": "x", "group_id": "g", "state": {"x": 1}, "question": "Choose",
        "candidate_options": [
            {"id": "a", "description": "A"}, {"id": "b", "description": "B"},
            {"id": "c", "description": "C"}],
        "target": {"correct_option_id": "b"},
        "provenance": {"source": "fixture", "reference_label": "b"},
    }
    base = DecisionRecord.model_validate(value)
    if with_teacher:
        value["teacher"] = TeacherTarget(input_sha256=base.input_hash(), model="teacher", method="hard",
            probabilities={"a": 0.0, "b": 1.0, "c": 0.0}).model_dump(mode="json")
    return DecisionRecord.model_validate(value)


def test_order_augmentation_preserves_semantics_and_rebinds_hard_teacher():
    original = row(with_teacher=True)
    first = augment_order([original], copies=1)
    second = augment_order([original], copies=1)
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert len(first) == 2 and first[1].id.endswith("::order-1")
    assert [option.id for option in first[1].candidate_options] != [option.id for option in original.candidate_options]
    assert first[1].target == original.target
    assert first[1].teacher.probabilities == original.teacher.probabilities
    assert first[1].teacher.input_sha256 == first[1].input_hash()
    assert first[1].teacher.metadata["order_augmentation"]["new_teacher_request"] is False
