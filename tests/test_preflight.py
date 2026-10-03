import json

import yaml

from summit.config import load_config
from summit.phantora import build_phantora_manifest
from summit.preflight import run_preflight
from summit.render import render


def _env():
    return {
        "RUNPOD_API_KEY": "test",
        "VAST_API_KEY": "test",
        "HF_TOKEN": "test",
        "FIREWORKS_API_KEY": "test",
        "WANDB_API_KEY": "test",
        "MODAL_TOKEN_ID": "test",
        "MODAL_TOKEN_SECRET": "test",
    }


def test_vast_only_environment_does_not_require_runpod():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    cfg.summitConfig.backends = ["vastai"]
    env = _env()
    del env["RUNPOD_API_KEY"]
    report = run_preflight(cfg, env=env)

    assert "ENV.MISSING" not in {finding.code for finding in report.errors}


def test_example_passes_preflight():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    report = run_preflight(cfg, env=_env())

    assert report.passed, report.format_text()
    assert not report.errors
    assert report.rendered is not None
    assert report.checks_run >= 15
    assert json.loads(report.to_json())["status"] == "pass"


def test_preflight_rejects_sequence_cap_smaller_than_prompt_plus_response():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    cfg.data.max_sequence_length = 4095

    report = run_preflight(cfg, env=_env())

    assert not report.passed
    assert "CFG.SEQUENCE_LENGTH" in {finding.code for finding in report.errors}


def test_preflight_catches_nexrl_per_rank_batch_flooring(monkeypatch):
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    original_render = render(cfg)
    recipe = yaml.safe_load(original_render.recipe_yaml)
    recipe["service"]["train_service"]["student"]["actor"][
        "ppo_mini_batch_size"
    ] = 1

    class FakeRendered:
        recipe_yaml = yaml.safe_dump(recipe)
        bootstrap_sh = original_render.bootstrap_sh

    monkeypatch.setattr("summit.preflight.render", lambda _cfg: FakeRendered())
    report = run_preflight(cfg, env=_env())

    assert "NEXRL.MINI_BATCH_ZERO" in {finding.code for finding in report.errors}


def test_phantora_manifest_uses_training_gpu_partition():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    manifest = build_phantora_manifest(cfg)

    assert manifest["adapter"] == "summit-nexrl-fsdp"
    assert manifest["virtual_cluster"]["gpus_per_host"] == 4
    assert manifest["virtual_cluster"]["vram_mib"] == 141 * 1024
    assert manifest["workload"]["max_sequence_length"] == 4096


def test_stub_sandbox_is_rejected_before_provisioning():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    cfg.rollout.sandbox = "daytona"
    report = run_preflight(cfg)
    assert "RUNTIME.SANDBOX_UNSUPPORTED" in {finding.code for finding in report.errors}
