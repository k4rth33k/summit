"""Offline recipe contracts; core tests need no ML runtime or credentials."""

import json
from pathlib import Path

import pytest
import yaml

from summit.config import load_config
from summit.preflight import run_preflight
from summit.render import render
from summit.phantora import build_phantora_manifest
from summit.recipes.decision.config import DecisionJobConfig
from summit.recipes.decision.data import DecisionRecord, check_split_isolation, read_records
from summit.recipes.decision.dataset import generate_records, write_dataset, attach_teacher, teacher_requests, policy_decision
from summit.recipes.decision.recipe import forwarded_env, runtime_check


@pytest.fixture
def decision_config(tmp_path):
    write_dataset(tmp_path / "data", generate_records(policies=6), {"fixture": True})
    raw = {"schema_version": 2, "recipe": "decision", "experiment_name": "test-decision",
           "data": {"train": "data/train.jsonl", "validation": "data/validation.jsonl"}}
    path = tmp_path / "decision.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_default_opd_schema_is_unchanged():
    assert not isinstance(load_config("tests/fixtures/deepswe_opd.yaml"), DecisionJobConfig)


def test_offline_schema_is_strict(decision_config):
    cfg = decision_config
    assert cfg.summitConfig.resources.rollout_gpus == 0
    assert not hasattr(cfg, "teacher") and not hasattr(cfg, "rollout")
    for update in ({"teacher": {}}, {"rollout": {}}, {"schema_version": 1}):
        with pytest.raises(ValueError):
            DecisionJobConfig.model_validate({**cfg.model_dump(), **update})
    for resources in ({"gpu": "H100:2"}, {"rollout_gpus": 1}):
        with pytest.raises(ValueError):
            DecisionJobConfig.model_validate({**cfg.model_dump(), "summitConfig": {"resources": resources}})


def test_render_is_service_free_and_bundles_inputs(decision_config):
    cfg = decision_config
    report = run_preflight(cfg)
    assert report.passed, report.format_text()
    r = report.rendered
    assert r.recipe_filename == "decision.yaml"
    assert r.execution_plan["services"] == []
    assert r.execution_plan["resources"] == {"training_gpus": 1, "rollout_gpus": 0}
    assert "run/inputs/train.jsonl" in r.files
    assert r.execution_plan["datasets"]["train"]["rows"] == 40
    assert "nexrl" not in r.bootstrap_sh.lower()
    assert "sglang" not in r.bootstrap_sh.lower()
    assert "fireworks" not in r.recipe_yaml.lower()
    assert "modal" not in r.recipe_yaml.lower()


def test_cloud_requires_retention_but_local_needs_no_secrets(decision_config):
    assert run_preflight(decision_config).passed
    report = run_preflight(decision_config, env={"RUNPOD_API_KEY": "fake"})
    assert not report.passed
    assert "OUTPUT.CLOUD_RETENTION" in report.format_text()
    assert forwarded_env(decision_config, {"FIREWORKS_API_KEY": "secret", "MODAL_TOKEN_SECRET": "secret", "HF_TOKEN": "secret"}) == {}


def test_matched_mode_requires_teacher_objective_and_order_check(decision_config):
    data = decision_config.model_dump()
    data['training']['comparison'] = 'ce_kd'
    with pytest.raises(ValueError, match='ce_kd comparison requires'):
        DecisionJobConfig.model_validate(data)
    data['objective']['type'] = 'candidate_distillation'
    with pytest.raises(ValueError, match='ce_kd comparison requires'):
        DecisionJobConfig.model_validate(data)
    data['training']['evaluate_reversed_options'] = True
    assert DecisionJobConfig.model_validate(data).training.comparison == 'ce_kd'


def test_head_learning_rate_is_scorer_only(decision_config):
    data = decision_config.model_dump()
    data['training']['head_learning_rate'] = 1e-4
    with pytest.raises(ValueError, match='builtin candidate_scorer'):
        DecisionJobConfig.model_validate(data)
    data['model']['adapter'] = 'candidate_scorer'
    assert DecisionJobConfig.model_validate(data).training.head_learning_rate == 1e-4


def test_qlora_adapter_is_explicit(decision_config):
    data = decision_config.model_dump()
    data["model"]["adaptation"] = "qlora"
    with pytest.raises(ValueError, match="requires the qlora_causal adapter"):
        DecisionJobConfig.model_validate(data)
    data["model"]["adapter"] = "qlora_causal"
    assert DecisionJobConfig.model_validate(data).model.adaptation == "qlora"
    data["model"]["adaptation"] = "full"
    with pytest.raises(ValueError, match="requires model.adaptation: qlora"):
        DecisionJobConfig.model_validate(data)


def test_simulation_never_claims_coverage(decision_config):
    assert build_phantora_manifest(decision_config)["status"] == "unsupported"
    assert runtime_check(decision_config, "simulation")["status"] == "inconclusive"


def test_generator_deterministic_and_group_disjoint():
    first = generate_records(policies=10)
    second = generate_records(policies=10)
    assert first == second
    check_split_isolation(first)
    assert sum(map(len, first.values())) == 100
    policy = {"max_amount": 10, "max_age_days": 5}
    assert policy_decision(policy, {"amount": 10, "age_days": 5, "verified": True}) == "approve"
    assert policy_decision(policy, {"amount": None, "age_days": 5, "verified": True}) == "clarify"
    assert policy_decision(policy, {"amount": None, "age_days": 5, "verified": False}) == "deny"


def test_split_leakage_and_duplicate_ids(tmp_path):
    row = generate_records(policies=6)["train"][0]
    for modified in (row, row.model_copy(update={"id": "new"}), row.model_copy(update={"id": "new", "group_id": "new"})):
        with pytest.raises(ValueError, match="cross-split"):
            check_split_isolation({"train": [row], "test": [modified]})
    path = tmp_path / "duplicate.jsonl"
    path.write_text(row.model_dump_json() + "\n" + row.model_dump_json())
    with pytest.raises(ValueError, match="duplicate record ID"):
        read_records(path)


def cache_row(row):
    return {"id": row.id, "teacher": {"input_sha256": row.input_hash(), "model": "test-teacher-pinned",
            "method": "candidate_logprobs", "probabilities": {o.id: 1 / len(row.candidate_options) for o in row.candidate_options}}}


def test_teacher_hash_coverage_and_reference_preservation():
    row = generate_records(policies=6)["train"][0]
    joined = attach_teacher([row], [cache_row(row)])
    assert joined[0].target == row.target
    request = teacher_requests([row])[0]
    assert set(request) == {"id", "input_sha256", "input"}
    assert set(request["input"]) == {"state", "question", "candidate_options"}
    assert "target" not in request["input"]
    for mutate in (lambda c: c["teacher"].update(input_sha256="stale"),
                   lambda c: c["teacher"].update(probabilities={"fake": 1}),
                   lambda c: c["teacher"].update(probabilities={o.id: 0.1 for o in row.candidate_options})):
        cached = cache_row(row)
        mutate(cached)
        with pytest.raises(ValueError):
            attach_teacher([row], [cached])
    with pytest.raises(ValueError, match="exactly"):
        attach_teacher([row], [])
    with pytest.raises(ValueError, match="duplicate"):
        attach_teacher([row], [cache_row(row), cache_row(row)])


def test_custom_code_bundle_and_local_path_resolution(decision_config, tmp_path):
    source = tmp_path / "custom.py"
    source.write_text("def create(config, checkpoint=None):\n    pass\n")
    cfg = decision_config.model_copy(update={"code": [Path("custom.py")]})
    cfg.model.adapter = "custom:create"
    r = render(cfg)
    assert r.files["run/code/custom.py"] == source.read_bytes()
    assert yaml.safe_load(r.recipe_yaml)["code"] == ["code/custom.py"]
    local = tmp_path / "local-model"
    local.mkdir()
    cfg.model.name = "./local-model"
    assert cfg.resolved_model().name == str(local)
    assert run_preflight(cfg).passed
    assert not run_preflight(cfg, env={}).passed


def test_decision_dstack_task_and_virtual_repo(decision_config):
    from summit.dstack_ops import _build_repo, _build_task
    task = _build_task(decision_config, "decision-test", {})
    assert task.resources.gpu.count.min == 1
    repo = _build_repo(render(decision_config))
    assert "run/decision.yaml" in repo.files
    assert "run/inputs/train.jsonl" in repo.files
    assert "run/execution-plan.json" in repo.files
    assert "run/rl_train.yaml" not in repo.files
    decision_config.summitConfig.max_price = 3
    assert _build_task(decision_config, "capped", {}).max_price == 3


def test_dry_run_needs_no_env_and_writes_complete_bundle(decision_config, tmp_path, monkeypatch):
    import argparse
    from summit import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_env", lambda *_: pytest.fail("dry-run read credentials"))
    monkeypatch.setattr("summit.dstack_ops.ensure_server", lambda *_: pytest.fail("cloud server started"))
    args = argparse.Namespace(config=tmp_path / "decision.yaml", dry_run=True, simulate=False, runtime_check=False)
    cli.cmd_launch(args)
    bundle = tmp_path / "summit-dry-run/bundle"
    runtime = yaml.safe_load((bundle / "run/decision.yaml").read_text())
    assert (bundle / "run" / runtime["data"]["train"]).is_file()
    assert (bundle / "summit/recipes/decision/train.py").is_file()
    assert (bundle / "summit/recipes/decision/requirements.txt").is_file()


def test_requested_unsupported_simulation_blocks_launch(decision_config, tmp_path, monkeypatch):
    import argparse
    from summit import cli
    monkeypatch.setattr("summit.dstack_ops.ensure_server", lambda *_: pytest.fail("cloud server started"))
    args = argparse.Namespace(config=tmp_path / "decision.yaml", dry_run=True, simulate=True, runtime_check=False)
    with pytest.raises(SystemExit) as error:
        cli.cmd_launch(args)
    assert error.value.code == 2
