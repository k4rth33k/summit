"""summit CLI — launch and manage self-hosted RL training runs.

Commands:
  summit init                        verify .env, configure + start local dstack server
  summit probe [--config FILE]       verify the teacher endpoint (echo logprobs)
  summit check -f FILE               validate a run without provisioning a machine
  summit launch -f FILE [--dry-run]  compile + submit a training run
  summit ps                          list runs
  summit logs RUN                    stream run logs
  summit stop RUN                    stop a run (tears down its instances)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .env import BACKEND_KEY, backend_credentials, forwarded_env, load_env


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def cmd_init(args: argparse.Namespace) -> None:
    from .dstack_ops import ensure_fleet, ensure_server, get_client

    env = load_env(args.env_file)
    cfg = load_config(args.config) if args.config else None
    backends = cfg.summitConfig.backends if cfg else None
    credentials = backend_credentials(env, backends)
    if not credentials:
        expected = [BACKEND_KEY[b] for b in (backends or BACKEND_KEY) if b in BACKEND_KEY]
        _die("no cloud backend credentials found; expected one of: " + ", ".join(expected))
    ensure_server(credentials)
    client = get_client()
    fleet = cfg.summitConfig.fleet if cfg else "summit-fleet"
    ensure_fleet(fleet)
    print(f"summit: dstack server ready (project={client.project}), fleet {fleet} ready")


def cmd_probe(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    from .recipes import is_decision
    if is_decision(cfg):
        _die("the offline decision recipe has no live teacher; use cached decision targets")
    env = load_env(args.env_file)
    if cfg.teacher.backend != "fireworks":
        _die(f"probe supports the fireworks teacher backend in v0.1 (got {cfg.teacher.backend})")
    from .teacher.fireworks import FireworksEchoClient

    client = FireworksEchoClient(
        model=cfg.teacher.base_model,
        api_key=env.get(cfg.teacher.api_key_env or ""),
        base_url=cfg.teacher.base_url or "",
    )
    # A tiny student-style sequence: fake "prompt + response" tokens.
    probe_tokens = [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198, 9707, 11, 825]
    lps = client.score(probe_tokens)
    assert len(lps) == len(probe_tokens), "alignment failure"
    assert lps[0] is None and all(v is not None for v in lps[1:]), "unexpected logprob layout"
    print(f"probe OK: {cfg.teacher.base_model} returned {len(lps)} echo logprobs")
    print(f"  sample logprobs: {lps[1:5]}")


def _preflight(args: argparse.Namespace, *, include_env: bool = True):
    from .preflight import PreflightReport, run_preflight
    from pydantic import ValidationError
    import yaml

    try:
        cfg = load_config(args.config)
    except (OSError, ValidationError, ValueError, yaml.YAMLError) as exc:
        report = PreflightReport()
        report.check(False, code="CFG.INVALID", severity="error", message=str(exc),
                     hint="Correct the configuration file before launching.")
        return None, None, report
    env = load_env(args.env_file) if include_env else None
    report = run_preflight(cfg, env=env)
    return cfg, env, report


def cmd_check(args: argparse.Namespace) -> None:
    cfg, _env, report = _preflight(args, include_env=not args.no_env)
    simulation = _simulate(args, cfg, report)
    if args.json:
        payload = report.as_dict()
        from .phantora import build_phantora_manifest

        payload["phantora_manifest"] = build_phantora_manifest(cfg) if cfg else None
        if report.rendered and report.rendered.execution_plan:
            payload["execution_plan"] = report.rendered.execution_plan
        payload["simulation"] = simulation if args.simulate else None
        payload["runtime_check"] = simulation if args.runtime_check else None
        if simulation:
            stage = "simulation" if args.simulate else "runtime"
            payload["coverage"][stage] = simulation["status"]
            # Keep summary counts useful to CI: stage diagnostics belong in
            # the combined report as well as the detailed stage result.
            informational = {"SIM.EXPERIMENTAL", "RUNTIME.SCOPE"}
            for finding in simulation["findings"]:
                severity = "warning" if (
                    finding["code"] in informational or simulation["status"] != "failed"
                ) else "error"
                payload["findings"].append({**finding, "severity": severity, "stage": stage})
                payload[f"{severity}_count"] += 1
            payload["status"] = "fail" if simulation["status"] == "failed" else (
                "inconclusive" if simulation["status"] != "completed" else payload["status"])
        print(json.dumps(payload, indent=2))
    else:
        print(report.format_text())
        if simulation:
            from .simulation.runner import format_simulation
            print(format_simulation(simulation))
    if not report.passed or (simulation and simulation["status"] == "failed"):
        raise SystemExit(1)
    if simulation and simulation["status"] != "completed":
        raise SystemExit(2)


def _simulate(args, cfg, report):
    if not (args.simulate or args.runtime_check) or not report.passed:
        return None
    from .recipes import is_decision
    if is_decision(cfg):
        from .recipes.decision.recipe import runtime_check
        return runtime_check(cfg, "simulation" if args.simulate else "runtime")
    from .simulation.runner import run_simulation

    mode = "simulation" if args.simulate else "runtime"
    try:
        return run_simulation(
            cfg, report.rendered, image=args.simulation_image, model_dir=args.model_dir,
            output_dir=args.simulation_output, timeout=args.simulation_timeout, mode=mode,
        )
    except (OSError, ValueError) as exc:
        return {
            "schema_version": 1, "mode": mode, "status": "inconclusive", "artifacts": None,
            "coverage": {"training": "not_run"},
            "findings": [{"code": "SIM.SETUP_FAILED", "message": str(exc)}],
        }


def cmd_launch(args: argparse.Namespace) -> None:
    cfg, env, report = _preflight(args, include_env=not args.dry_run)
    print(report.format_text())
    if not report.passed:
        _die("preflight failed; no GPU machine was provisioned")
    simulation = _simulate(args, cfg, report)
    if simulation:
        from .simulation.runner import format_simulation
        print(format_simulation(simulation))
        if simulation["status"] != "completed":
            _die("requested check did not complete; no GPU machine was provisioned",
                 code=1 if simulation["status"] == "failed" else 2)
    if args.dry_run:
        from .phantora import build_phantora_manifest

        if report.rendered is None:
            _die("this configuration uses a local checkpoint; use summit train for local execution")
        r = report.rendered
        out = Path("summit-dry-run")
        out.mkdir(exist_ok=True)
        (out / r.recipe_filename).write_text(r.recipe_yaml)
        (out / "bootstrap.sh").write_text(r.bootstrap_sh)
        (out / "preflight.json").write_text(report.to_json() + "\n")
        (out / "phantora.json").write_text(
            json.dumps(build_phantora_manifest(cfg), indent=2) + "\n"
        )
        if r.execution_plan:
            (out / "execution-plan.json").write_text(json.dumps(r.execution_plan, indent=2) + "\n")
        from .dstack_ops import _build_repo
        for name, content in _build_repo(r).files.items():
            target = out / "bundle" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if simulation:
            stage_file = "simulation.json" if args.simulate else "runtime-check.json"
            (out / stage_file).write_text(json.dumps(simulation, indent=2) + "\n")
        print(
            f"dry-run: wrote {out}/{r.recipe_filename}, {out}/bootstrap.sh, "
            f"{out}/preflight.json, and {out}/phantora.json"
        )
        return

    from .dstack_ops import ensure_server, launch

    assert env is not None
    ensure_server(backend_credentials(env, cfg.summitConfig.backends))
    from .recipes import is_decision
    if is_decision(cfg):
        from .recipes.decision.recipe import forwarded_env as decision_env
        run_name = launch(cfg, decision_env(cfg, env))
        print(f"summit: launched run {run_name}\n  monitor: summit logs {run_name}")
        return
    env_names = list(cfg.summitConfig.env)
    fwd = forwarded_env(env, env_names)
    # Keys required by the run that must always be forwarded.
    for k in ("HF_TOKEN", cfg.teacher.api_key_env, "WANDB_KEY", "WANDB_HOST"):
        if k and env.get(k) and k not in fwd:
            fwd[k] = env[k]
    for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        if cfg.rollout.sandbox == "modal" and env.get(k):
            fwd[k] = env[k]

    run_name = launch(cfg, fwd)
    print(f"summit: launched run {run_name}")
    print("  monitor: summit logs " + run_name)
    if cfg.use_wandb:
        print("  wandb:   metrics stream to your W&B project " + cfg.project_name)


def cmd_ps(args: argparse.Namespace) -> None:
    from .dstack_ops import get_client

    client = get_client()
    for run in client.runs.list():
        print(f"{run.name}\t{run.status}")


def cmd_logs(args: argparse.Namespace) -> None:
    from .dstack_ops import get_client

    client = get_client()
    run = client.runs.get(args.run)
    if run.status in ("terminated", "failed", "completed"):
        _die(
            f"run {args.run} is {run.status} — no live logs. "
            "Use `summit ps` for status; ask dstack for history if the server kept it."
        )
    run.attach()
    try:
        for chunk in run.logs():
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    except KeyboardInterrupt:
        pass
    finally:
        run.detach()


def cmd_stop(args: argparse.Namespace) -> None:
    from .dstack_ops import get_client

    client = get_client()
    run = client.runs.get(args.run)
    run.stop(abort=True)
    print(f"stopped {args.run}")


def cmd_train(args):
    """Run an offline recipe locally, with no cloud client or implicit publication."""
    cfg = load_config(args.config)
    from .recipes import is_decision
    if not is_decision(cfg):
        _die("local training is supported by the decision recipe; use launch for DeepSWE OPD")
    from .recipes.decision.train import train
    train(cfg)


def main() -> None:
    p = argparse.ArgumentParser(prog="summit")
    p.add_argument("--env-file", default=".env")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init")
    init.add_argument("-f", "--config")

    probe = sub.add_parser("probe")
    probe.add_argument("--config", default="examples/deepswe_opd.yaml")

    check = sub.add_parser("check")
    check.add_argument("-f", "--config", required=True)
    check.add_argument("--json", action="store_true")
    check.add_argument(
        "--no-env", action="store_true", help="skip local secret-presence checks"
    )

    launch = sub.add_parser("launch")
    launch.add_argument("-f", "--config", required=True)
    launch.add_argument("--dry-run", action="store_true")

    train = sub.add_parser("train", help="run the offline decision recipe locally (no cloud provisioning)")
    train.add_argument("-f", "--config", required=True)

    for parser in (check, launch):
        stages = parser.add_mutually_exclusive_group()
        stages.add_argument("--simulate", action="store_true",
                            help="run experimental GPU-free NexRL/Phantora diagnostics before provisioning")
        stages.add_argument("--runtime-check", action="store_true",
                            help="recipe runtime check: OPD container or decision imports in the local interpreter")
        parser.add_argument("--simulation-image",
                            help="prebuilt local Docker image; check never builds or pulls")
        parser.add_argument("--model-dir", type=Path,
                            help="local model config/tokenizer metadata (weights unnecessary)")
        parser.add_argument("--simulation-output", default="summit-checks", type=Path)
        parser.add_argument("--simulation-timeout", default=600, type=_positive_int)

    sub.add_parser("ps")

    logs = sub.add_parser("logs")
    logs.add_argument("run")

    stop = sub.add_parser("stop")
    stop.add_argument("run")

    args = p.parse_args()
    {
        "init": cmd_init,
        "probe": cmd_probe,
        "check": cmd_check,
        "launch": cmd_launch,
        "ps": cmd_ps,
        "logs": cmd_logs,
        "stop": cmd_stop,
        "train": cmd_train,
    }[args.cmd](args)


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    main()
