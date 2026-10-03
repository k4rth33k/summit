"""Offline recipe integration with Summit's existing lifecycle."""

import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from summit.env import BACKEND_KEY
from summit.render import RenderedRun
from .data import read_records, check_split_isolation, dataset_summary


def required_env(cfg):
    names = [BACKEND_KEY[b] for b in cfg.summitConfig.backends] + list(cfg.summitConfig.env)
    if cfg.summitConfig.output.hf_push:
        names.append("HF_TOKEN")
    return sorted(set(names))


def forwarded_env(cfg, env):
    names = set(cfg.summitConfig.env)
    if cfg.summitConfig.output.hf_push:
        names.add("HF_TOKEN")
    return {name: env[name] for name in names if env.get(name)}


def code_files(cfg):
    files = {}
    for root in cfg.code:
        source = cfg.resolve_path(root)
        if not source.exists():
            raise ValueError(f"missing custom code: {source}")
        sources = sorted(source.rglob("*.py")) if source.is_dir() else [source]
        if not sources:
            raise ValueError(f"no Python source files in {source}")
        for path in sources:
            if path.suffix != ".py":
                raise ValueError("custom code bundles currently accept Python source only")
            name = f"run/code/{path.relative_to(source.parent).as_posix()}"
            if name in files:
                raise ValueError(f"duplicate custom code destination: {name}")
            files[name] = path.read_bytes()
    return files


def render_decision(cfg):
    files = code_files(cfg)
    runtime = cfg.model_dump(mode="json")
    for split in ("train", "validation"):
        path = cfg.resolve_path(getattr(cfg.data, split))
        target = f"inputs/{split}.jsonl"
        files[f"run/{target}"] = path.read_bytes()
        runtime["data"][split] = target
    runtime["code"] = [f"code/{Path(p).name}" for p in cfg.code]
    runtime["summitConfig"]["output"]["directory"] = "/workflow/output/decision"
    if Path(cfg.resolved_model().name).exists():
        raise ValueError("cloud rendering requires a Hub model ID; use summit train for a local model")
    requirements = Path(__file__).with_name("requirements.txt").read_bytes()
    files["summit/recipes/decision/requirements.txt"] = requirements
    plan = {
        "schema_version": 1,
        "recipe": "decision",
        "backend": cfg.training.backend,
        "model_adapter": cfg.model.adapter,
        "objective": cfg.objective.type,
        "comparison": cfg.training.comparison,
        "services": [],
        "resources": {"training_gpus": 1, "rollout_gpus": 0},
        "required_environment": required_env(cfg),
        "forwarded_environment": sorted(set(cfg.summitConfig.env) | (
            {"HF_TOKEN"} if cfg.summitConfig.output.hf_push else set()
        )),
        "artifacts": {"directory": "/workflow/output/decision", "completion_marker": "complete.json",
                      "hf_push": cfg.summitConfig.output.hf_push.model_dump() if cfg.summitConfig.output.hf_push else None},
        "validation": {"static": "supported", "runtime": "local_imports", "phantora": "unsupported",
                       "gpu": "unqualified"},
        "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    script = """#!/usr/bin/env bash
set -euo pipefail
cd /workflow
python -m venv /opt/summit-decision
/opt/summit-decision/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
/opt/summit-decision/bin/python -m pip install -r summit/recipes/decision/requirements.txt
/opt/summit-decision/bin/python -m summit.recipes.decision.train -f run/decision.yaml
test -f /workflow/output/decision/complete.json
"""
    return RenderedRun(yaml.safe_dump(runtime, sort_keys=False), script, "decision.yaml", files, plan)


def preflight(cfg, env=None):
    from summit.preflight import PreflightReport

    report = PreflightReport()
    try:
        code_files(cfg)
    except (ValueError, OSError) as exc:
        report.check(False, code="CODE.INVALID", severity="error", message=str(exc))
    splits = {}
    for name in ("train", "validation"):
        try:
            path = cfg.resolve_path(getattr(cfg.data, name))
            rows = read_records(path, require_teacher=(name == "train" and cfg.objective.type == "candidate_distillation"),
                                max_candidates=cfg.data.max_candidates)
            splits[name] = rows
            report.check(True, code="DATA.VALID", severity="error", message=name)
        except (ValueError, OSError) as exc:
            report.check(False, code="DATA.INVALID", severity="error", message=str(exc))
    if len(splits) == 2:
        try:
            check_split_isolation(splits)
            report.check(True, code="DATA.SPLITS", severity="error", message="split isolation")
        except ValueError as exc:
            report.check(False, code="DATA.LEAKAGE", severity="error", message=str(exc))
    if env is not None:
        missing = [key for key in required_env(cfg) if not env.get(key)]
        report.check(not missing, code="ENV.MISSING", severity="error", message="missing keys: " + ", ".join(missing))
        report.check(cfg.summitConfig.output.hf_push is not None, code="OUTPUT.CLOUD_RETENTION", severity="error",
                     message="Cloud decision runs currently require hf_push to retain artifacts before VM teardown; use summit train for local artifacts.")
        report.check(cfg.training.device == "cuda", code="CFG.CLOUD_DEVICE", severity="error",
                     message="Cloud GPU runs must select training.device: cuda.")
    report.check(cfg.model.revision is not None, code="MODEL.UNPINNED", severity="warning",
                 message="Pin model.revision to an immutable Hub commit before publishing or comparing runs.")
    if report.passed:
        try:
            if Path(cfg.resolved_model().name).exists():
                # Local checkpoints need no cloud rendering; the local trainer supports them.
                report.check(env is None, code="MODEL.LOCAL", severity="error",
                             message="Local checkpoints require summit train; cloud launch requires a Hub ID.")
            else:
                report.rendered = render_decision(cfg)
                report.recipe = yaml.safe_load(report.rendered.recipe_yaml)
                syntax = subprocess.run(["bash", "-n"], input=report.rendered.bootstrap_sh, text=True, capture_output=True)
                report.check(syntax.returncode == 0, code="RENDER.BASH_SYNTAX", severity="error", message=syntax.stderr)
                report.rendered.execution_plan["datasets"] = {
                    name: dataset_summary(cfg.resolve_path(getattr(cfg.data, name)), rows)
                    for name, rows in splits.items()
                }
        except (ValueError, OSError) as exc:
            report.check(False, code="RENDER.FAILED", severity="error", message=str(exc))
    return report


def runtime_check(cfg, mode):
    if mode == "simulation":
        return {"schema_version": 1, "mode": mode, "status": "inconclusive", "artifacts": None,
                "coverage": {"training": "not_run"}, "findings": [{"code": "SIM.UNSUPPORTED",
                "message": "No qualified Phantora adapter for decision training. Requested simulation blocks launch."}]}
    import importlib.metadata
    import importlib
    try:
        names = ["torch", "transformers", "safetensors"]
        if cfg.model.adaptation == "qlora":
            names.extend(["peft", "bitsandbytes"])
        for name in names:
            importlib.import_module(name)
        from .models import resolve_adapter
        from .train import add_code_paths
        add_code_paths(cfg)
        resolve_adapter(cfg.model.adapter)
        versions = {name: importlib.metadata.version(name) for name in names}
        return {"schema_version": 1, "mode": "runtime", "status": "completed", "artifacts": None,
                "coverage": {"training": "not_run", "imports": "completed", "location": "local_interpreter"},
                "versions": versions, "findings": [{"code": "RUNTIME.SCOPE",
                "message": "Local imports only; model loading, GPU operations, and cloud runtime remain unqualified."}]}
    except Exception as exc:
        return {"schema_version": 1, "mode": "runtime", "status": "failed", "artifacts": None,
                "coverage": {"training": "not_run"}, "findings": [{"code": "RUNTIME.IMPORT", "message": str(exc)}]}
