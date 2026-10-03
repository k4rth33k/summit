"""Release entry points must remain usable without private workspace artifacts."""

import importlib.util
from pathlib import Path

from summit.config import load_config


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_release", ROOT / "scripts/check_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_public_recipe_settings():
    opd = load_config(ROOT / "examples/deepswe_opd.yaml")
    assert opd.model_name == "Qwen/Qwen3.5-4B"
    assert opd.total_train_steps == 4
    assert opd.summitConfig.resources.gpu_spec().count == 2
    parent = load_config(ROOT / "examples/decision_9b.yaml")
    assert parent.model.adapter == "causal_lm"
    assert parent.training.comparison == "ce_kd"
    assert parent.summitConfig.output.hf_push.replace is False
    specialist = load_config(ROOT / "examples/specialize_9b.yaml")
    assert specialist.model.adapter == "qlora_causal"
    assert specialist.model.adaptation == "qlora"
    assert specialist.summitConfig.output.hf_push is None


def test_release_inventory_and_documentation():
    names = release.inventory(ROOT)
    assert not release.audit(ROOT, names)
    assert "README.md" in names and "AGENTS.md" in names
    assert "tests/test_release.py" in names
    assert ".env" not in names


def test_inventory_rejects_private_paths_before_opening(tmp_path):
    errors = release.audit(tmp_path, sorted(release.EXAMPLES | {".env", "data/private.jsonl"}))
    assert len(errors) == 2


def test_public_decision_recipes_preflight_with_explicit_fixture_assets(tmp_path):
    from summit.recipes.decision.dataset import generate_records, attach_teacher, write_dataset
    from summit.recipes.decision.recipe import preflight
    splits = generate_records(policies=6)
    splits["train"] = attach_teacher(splits["train"], [
        {"id": row.id, "teacher": {"input_sha256": row.input_hash(), "model": "fixture",
         "method": "synthetic plumbing check, not research data",
         "probabilities": {option.id: 1 / len(row.candidate_options) for option in row.candidate_options}}}
        for row in splits["train"]])
    write_dataset(tmp_path / "data", splits, {})
    local_base = tmp_path / "base"
    local_base.mkdir()  # Static-only validation does not load weights.
    for name in ("decision_9b", "specialize_9b"):
        cfg = load_config(ROOT / f"examples/{name}.yaml")
        cfg.data.train = tmp_path / "data/train.jsonl"
        cfg.data.validation = tmp_path / "data/validation.jsonl"
        if name == "specialize_9b":
            cfg.model.name = str(local_base)
        report = preflight(cfg)
        assert report.passed, report.format_text()
        if name == "decision_9b":
            assert report.rendered.execution_plan["services"] == []
            assert report.rendered.execution_plan["comparison"] == "ce_kd"
        else:
            assert report.rendered is None  # Local base is never auto-uploaded.


def test_qlora_runtime_check_requires_optional_dependencies(monkeypatch):
    import importlib
    from summit.recipes.decision.recipe import runtime_check
    cfg = load_config(ROOT / "examples/specialize_9b.yaml")
    imported = []
    def fake_import(name):
        imported.append(name)
        if name == "peft":
            raise ImportError("missing optional PEFT dependency")
    monkeypatch.setattr(importlib, "import_module", fake_import)
    report = runtime_check(cfg, "runtime")
    assert imported == ["torch", "transformers", "safetensors", "peft"]
    assert report["status"] == "failed"
    assert report["findings"][0]["code"] == "RUNTIME.IMPORT"
