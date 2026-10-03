import json
from pathlib import Path
import subprocess

import pytest

from summit import cli
from summit.config import load_config
from summit.phantora import build_phantora_manifest
from summit.preflight import run_preflight
from summit.render import render
from summit.simulation.runner import run_simulation


@pytest.fixture
def job():
    return load_config("tests/fixtures/deepswe_opd.yaml")


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "model metadata"
    path.mkdir()
    (path / "config.json").write_text('{"model_type": "qwen3_5"}')
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_text("do not copy weights")
    (path / ".env").write_text("PRIVATE_KEY=not-forwarded")
    (path / "credentials.json").write_text('{"token": "not-forwarded"}')
    return path


def docker_stub(monkeypatch, *, status="completed", missing_rank=False, invalid_hash=False,
                returncode=0, timeout=False, no_report=False):
    commands = []
    monkeypatch.setattr("summit.simulation.runner.shutil.which", lambda _: "/usr/bin/docker")

    def run(command, **kwargs):
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "sha256:test-image\n", "")
        if command[1] == "run":
            if timeout:
                raise subprocess.TimeoutExpired(command, 1)
            mounts = {}
            for arg in command:
                if arg.startswith("type=bind,"):
                    fields = dict(v.split("=", 1) for v in arg.split(",") if "=" in v)
                    mounts[fields["dst"]] = Path(fields["src"])
            manifest = json.loads((mounts["/input"] / "manifest.json").read_text())
            simulation = manifest["mode"] == "simulation"
            ranks = list(range(manifest["virtual_cluster"]["gpus_per_host"])) if simulation else []
            payload = {
                "schema_version": 1, "status": status,
                "input_sha256": "wrong" if invalid_hash else (mounts["/input"] / "input.sha256").read_text(),
                "completed_ranks": ranks[:-1] if missing_rank else ranks,
                "findings": [],
                "coverage": {"runtime": "completed", "imports": "completed", "nexrl_config": "completed",
                             "training": "completed" if simulation else "not_run"},
            }
            if not no_report:
                (mounts["/results"] / "result.json").write_text(json.dumps(payload))
            return subprocess.CompletedProcess(command, returncode)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("summit.simulation.runner.subprocess.run", run)
    return commands


def test_manifest_does_not_assume_a100_has_80gb(job):
    job.summitConfig.resources.gpu = "A100:8"
    assert build_phantora_manifest(job)["virtual_cluster"]["vram_mib"] is None


def test_manifest_binds_optimizer_to_simulated_update(job):
    job.optimizer = "adafactor"
    job.memory_efficient_fsdp = True
    assert build_phantora_manifest(job)["workload"]["optimizer"] == "adafactor"
    assert build_phantora_manifest(job)["workload"]["memory_efficient_fsdp"] is True


def test_manifest_records_blackwell_runtime_profile():
    job = load_config("tests/fixtures/deepswe_opd_learning_96gb.yaml")
    job.summitConfig.resources.gpu = "RTXPRO6000-96GB:5"
    workload = build_phantora_manifest(job)["workload"]
    assert workload["pytorch_cuda_build"] == "cu128"
    assert workload["gpu_architecture_profile"] == "blackwell"


def test_completed_simulation_is_bound_to_inputs_and_isolated(job, metadata, tmp_path, monkeypatch):
    commands = docker_stub(monkeypatch)
    result = run_simulation(job, render(job), model_dir=metadata, output_dir=tmp_path)
    assert result["status"] == "completed"
    expected_ranks = build_phantora_manifest(job)["virtual_cluster"]["gpus_per_host"]
    assert result["completed_ranks"] == list(range(expected_ranks))
    assert result["image_id"] == "sha256:test-image"
    run = next(command for command in commands if command[1] == "run")
    assert "--network=none" in run and "--pull=never" in run
    assert "--gpus" not in run and "HF_TOKEN" not in " ".join(run)
    assert "NVIDIA_VISIBLE_DEVICES=void" in run
    assert "--privileged" not in run
    assert {path.name for path in (Path(result["artifacts"]) / "model").iterdir()} == {"config.json", "tokenizer.json"}
    assert commands[-1][1:3] == ["rm", "-f"]
    assert json.loads((Path(result["artifacts"]) / "report.json").read_text()) == result


@pytest.mark.parametrize("options", [
    {"missing_rank": True}, {"invalid_hash": True}, {"returncode": 1}, {"no_report": True},
])
def test_partial_or_stale_result_never_passes(job, metadata, tmp_path, monkeypatch, options):
    docker_stub(monkeypatch, **options)
    result = run_simulation(job, render(job), model_dir=metadata, output_dir=tmp_path)
    assert result["status"] == "inconclusive"
    assert result["findings"][-1]["code"] == "SIM.RESULT_INVALID"


def test_timeout_removes_only_owned_container(job, metadata, tmp_path, monkeypatch):
    commands = docker_stub(monkeypatch, timeout=True)
    result = run_simulation(job, render(job), model_dir=metadata, output_dir=tmp_path, timeout=1)
    assert result["status"] == "inconclusive"
    assert result["findings"][-1]["code"] == "SIM.TIMEOUT"
    run = next(command for command in commands if command[1] == "run")
    assert commands[-1] == ["docker", "rm", "-f", run[run.index("--name") + 1]]


def test_runtime_stage_needs_no_model_or_gpu_memory(job, tmp_path, monkeypatch):
    job.summitConfig.resources.gpu = "A100:8"
    docker_stub(monkeypatch)
    result = run_simulation(job, render(job), output_dir=tmp_path, mode="runtime")
    assert result["status"] == "completed"
    assert result["completed_ranks"] == []
    assert result["coverage"]["training"] == "not_run"


@pytest.mark.parametrize("status,exitcode", [("failed", 1), ("inconclusive", 2)])
def test_launch_stops_before_cloud_on_failed_requested_stage(job, monkeypatch, status, exitcode):
    import argparse
    monkeypatch.setattr(cli, "_preflight", lambda _, **kwargs: (job, {}, run_preflight(job)))
    monkeypatch.setattr(cli, "_simulate", lambda *args: {
        "status": status, "findings": [], "artifacts": "/tmp/test", "mode": "simulation"})
    monkeypatch.setattr("summit.dstack_ops.ensure_server", lambda *_: pytest.fail("cloud server started"))
    monkeypatch.setattr("summit.dstack_ops.launch", lambda *_: pytest.fail("cloud launch requested"))
    with pytest.raises(SystemExit) as exc:
        cli.cmd_launch(argparse.Namespace(dry_run=False))
    assert exc.value.code == exitcode


def test_static_failure_skips_container(job, monkeypatch):
    job.data.max_sequence_length = 1
    report = run_preflight(job)
    monkeypatch.setattr("summit.simulation.runner.run_simulation", lambda *a, **kw: pytest.fail("container started"))
    import argparse
    assert cli._simulate(argparse.Namespace(simulate=True, runtime_check=False), job, report) is None


def test_bad_config_json_is_structured(tmp_path, monkeypatch, capsys):
    path = tmp_path / "invalid.yaml"
    path.write_text("data:\n  batch_size: -1\n")
    monkeypatch.setattr("sys.argv", ["summit", "check", "-f", str(path), "--no-env", "--json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["findings"][0]["code"] == "CFG.INVALID"


def test_ignored_offload_is_visible(job):
    job.summitConfig.resources.gpu = "H100:2"
    report = run_preflight(job)
    assert "NEXRL.OFFLOAD_IGNORED" in {finding.code for finding in report.warnings}


def test_runtime_json_does_not_claim_simulation_ran(job, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_preflight", lambda *a, **kw: (job, {}, run_preflight(job)))
    monkeypatch.setattr(cli, "_simulate", lambda *a: {
        "status": "completed", "mode": "runtime", "findings": [], "artifacts": "/tmp/test"})
    monkeypatch.setattr("sys.argv", ["summit", "check", "-f", "unused.yaml", "--runtime-check", "--json"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage"]["simulation"] == "not_run"
    assert payload["coverage"]["runtime"] == "completed"
    assert payload["simulation"] is None
    assert payload["runtime_check"]["status"] == "completed"


def test_simulation_failure_is_in_combined_json_findings(job, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_preflight", lambda *a, **kw: (job, {}, run_preflight(job)))
    monkeypatch.setattr(cli, "_simulate", lambda *a: {
        "status": "failed", "mode": "simulation", "artifacts": "/tmp/test",
        "findings": [{"code": "SIM.CUDA_OOM", "message": "Virtual VRAM exceeded"}]})
    monkeypatch.setattr("sys.argv", ["summit", "check", "-f", "unused.yaml", "--simulate", "--json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["error_count"] == 1
    assert payload["findings"][0]["stage"] == "simulation"
