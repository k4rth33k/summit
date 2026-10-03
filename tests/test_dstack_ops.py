"""Validate that our Task construction passes dstack's pydantic validation
(no server needed — coercion happens client-side)."""

from summit.config import load_config
from summit.dstack_ops import _build_repo, _build_task
from summit.render import render


def test_run_name_respects_provider_length_and_characters():
    from datetime import datetime, timezone
    import re
    from summit.dstack_ops import _run_name
    value = _run_name("decision-scorer-qual-20260925_Extra_LONG", datetime.now(timezone.utc))
    assert len(value) <= 41
    assert re.fullmatch(r"[a-z][a-z0-9-]{1,40}", value)


def test_task_builds_and_validates():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    task = _build_task(cfg, "summit-test", {"HF_TOKEN": "x"})
    assert task.type == "task"
    assert task.commands == ["bash run/bootstrap.sh"]
    assert task.working_dir == "/workflow"
    gpu = task.resources.gpu
    assert gpu.count.min == 5
    assert "H200" in gpu.name
    assert str(task.spot_policy) in ("on-demand", "SpotPolicy.ONDEMAND")
    assert task.retry is not None and "no-capacity" in [e.value for e in task.retry.on_events]


def test_task_can_limit_retries_to_capacity_waiting():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    cfg.summitConfig.retry_on_error = False
    cfg.summitConfig.retry_on_interruption = False
    task = _build_task(cfg, "summit-budgeted-test", {"HF_TOKEN": "x"})
    assert task.retry is not None
    assert [event.value for event in task.retry.on_events] == ["no-capacity"]


def test_task_accepts_multiple_compatible_gpu_models():
    cfg = load_config("tests/fixtures/deepswe_opd_learning.yaml")
    task = _build_task(cfg, "summit-flexible-gpu-test", {"HF_TOKEN": "x"})
    assert task.resources.gpu.name == ["H200", "H200NVL"]
    assert task.resources.gpu.memory.min == 141
    assert task.resources.gpu.count.min == 5


def test_virtual_repo_contains_package_and_run_files():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    repo = _build_repo(render(cfg))
    names = set(repo.files)
    assert "summit/cli.py" in names
    assert "summit/nexrl_ext/opd_trainer.py" in names
    assert "summit/rollout/deepswe_worker.py" in names
    assert "run/rl_train.yaml" in names and "run/bootstrap.sh" in names
    assert "pyproject.toml" in names
    assert "summit/runtime-requirements.txt" in names
    assert "summit/sglang-overrides.txt" in names
