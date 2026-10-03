"""dstack integration: local server bootstrap + run lifecycle via dstack.api."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _dstack_exe() -> str:
    exe = shutil.which("dstack")
    if exe:
        return exe
    candidate = Path(sys.executable).parent / "dstack"  # same venv as summit
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("dstack CLI not found (pip install dstack[all])")

from .config import SummitJobConfig
from .render import RenderedRun

DSTACK_PORT = 3131  # 3000 collides with common dev servers (Next.js etc.)
SERVER_URL = f"http://127.0.0.1:{DSTACK_PORT}"
SERVER_CODE_UPLOAD_LIMIT = 64 * 2**20
SERVER_CONFIG = Path.home() / ".dstack" / "server" / "config.yml"
SUMMIT_STATE = Path.home() / ".summit"
SERVER_LOG = SUMMIT_STATE / "dstack-server.log"
TOKEN_FILE = SUMMIT_STATE / "dstack-token"


def _write_server_config(backend_credentials: dict[str, str]) -> None:
    """Add the selected API-key backends to dstack's local `main` project."""
    cfg: dict = {"projects": []}
    if SERVER_CONFIG.exists():
        cfg = yaml.safe_load(SERVER_CONFIG.read_text()) or cfg
    projects = cfg.setdefault("projects", [])
    main = next((p for p in projects if p.get("name") == "main"), None)
    if main is None:
        main = {"name": "main", "backends": []}
        projects.append(main)
    backends = main.setdefault("backends", [])
    for btype, key in backend_credentials.items():
        if not any(b.get("type") == btype for b in backends):
            backends.append({"type": btype, "creds": {"type": "api_key", "api_key": key}})
    SERVER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SERVER_CONFIG.write_text(yaml.safe_dump(cfg))


def _server_healthy() -> bool:
    """True only if a dstack server (not some other app on the port) answers."""
    import httpx

    try:
        # dstack serves /healthcheck and redirects / -> /api/docs
        r = httpx.get(f"{SERVER_URL}/healthcheck", timeout=2.0, follow_redirects=False)
        if r.status_code == 200:
            return True
        r = httpx.get(f"{SERVER_URL}/", timeout=2.0, follow_redirects=False)
        return r.status_code in (301, 302, 307) and "api/docs" in (r.headers.get("location") or "")
    except Exception:  # noqa: BLE001
        return False


def ensure_server(backend_credentials: dict[str, str]) -> None:
    """Write backend config and (re)start a summit-managed local dstack server."""
    SUMMIT_STATE.mkdir(parents=True, exist_ok=True)
    _write_server_config(backend_credentials)

    if _server_healthy():
        return  # a server is already running (summit-managed or user's)

    # Pin the admin token (survives restarts, keeps CLI config stable) and
    # auto-confirm the server's project-config update prompt (-y).
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
    if not token:
        import secrets

        token = secrets.token_urlsafe(32)
        TOKEN_FILE.write_text(token)

    with open(SERVER_LOG, "ab") as log:
        server_env = os.environ.copy()
        # Decision recipes bundle their local JSONL inputs in the virtual repo.
        # dstack's 2 MiB default is too small for even modest training probes.
        server_env.setdefault(
            "DSTACK_SERVER_CODE_UPLOAD_LIMIT", str(SERVER_CODE_UPLOAD_LIMIT)
        )
        proc = subprocess.Popen(  # noqa: S603
            [
                _dstack_exe(), "server", "--port", str(DSTACK_PORT),
                "--token", token, "-y",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=server_env,
        )
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"dstack server exited early; see {SERVER_LOG}")
        if _server_healthy():
            break
        time.sleep(2)
    if not _server_healthy():
        raise RuntimeError(f"dstack server did not become healthy; see {SERVER_LOG}")


def _read_token_from_log() -> str | None:
    if not SERVER_LOG.exists():
        return None
    m = re.search(r"The admin token is (\S+)", SERVER_LOG.read_text(errors="replace"))
    return m.group(1) if m else None


def ensure_fleet(name: str) -> None:
    """Create the summit fleet template if missing (dstack requires a fleet
    before any run). `nodes: 0..1` = template only; the instance provisions
    when a run is submitted, matching the run's resource requirements."""
    fleet_yaml = SUMMIT_STATE / f"{name}.dstack.yml"
    fleet_yaml.write_text(
        yaml.safe_dump(
            {
                "type": "fleet",
                "name": name,
                "nodes": "0..1",
                "idle_duration": "15m",
                "spot_policy": "on-demand",
                "resources": {"gpu": "0..8"},
            }
        )
    )
    r = subprocess.run(
        [_dstack_exe(), "apply", "-f", str(fleet_yaml), "-y", "-d", "--project", "main"],
        capture_output=True,
        text=True,
        cwd=SUMMIT_STATE,  # dstack apply requires the config file under cwd
    )
    if r.returncode != 0:
        raise RuntimeError(f"failed to create fleet {name}: {r.stdout}\n{r.stderr}")


def get_client():
    """Client for the summit-managed server (token from state) or default config."""
    from dstack.api import Client

    if TOKEN_FILE.exists():
        return Client.from_config(
            project_name="main", server_url=SERVER_URL, user_token=TOKEN_FILE.read_text().strip()
        )
    return Client.from_config()


def _build_task(cfg: SummitJobConfig, run_name: str, env_forward: dict[str, str]):
    from dstack.api import GPU, Resources, Task

    gpu = cfg.summitConfig.resources.gpu_spec()
    # dstack accepts multiple compatible accelerator names and will schedule
    # the first available homogeneous offer. Keep the compact config syntax
    # (for example ``H200,H200NVL-141GB:5``) in GPUResource while expanding it
    # at the provider boundary.
    gpu_names = [name.strip() for name in gpu.name.split(",") if name.strip()]
    gpu_kwargs: dict = {"name": gpu_names, "count": gpu.count}
    if gpu.memory_gb is not None:
        gpu_kwargs["memory"] = f"{gpu.memory_gb}GB"
    res_kwargs: dict = {
        "gpu": GPU(**gpu_kwargs),
        "shm_size": cfg.summitConfig.resources.shm_size,
    }
    if cfg.summitConfig.resources.disk:
        res_kwargs["disk"] = cfg.summitConfig.resources.disk
    resources = Resources(**res_kwargs)
    kwargs: dict = {
        "name": run_name,
        "env": dict(env_forward),
        # dstack mounts the (virtual) repo at /workflow; run from there so
        # run/bootstrap.sh and the summit package resolve.
        "working_dir": "/workflow",
        "commands": ["bash run/bootstrap.sh"],
        "resources": resources,
        # profile-level scheduling knobs
        "backends": cfg.summitConfig.backends,
        "spot_policy": cfg.summitConfig.spot_policy,
        "max_duration": cfg.summitConfig.max_duration,
    }
    if cfg.summitConfig.regions:
        kwargs["regions"] = cfg.summitConfig.regions
    if getattr(cfg.summitConfig, "max_price", None) is not None:
        kwargs["max_price"] = cfg.summitConfig.max_price
    retry_events: list[str] = []
    if cfg.summitConfig.retry_on_no_capacity:
        retry_events.append("no-capacity")
    if cfg.summitConfig.retry_on_error:
        retry_events.append("error")
    if cfg.summitConfig.retry_on_interruption:
        retry_events.append("interruption")
    if retry_events:
        kwargs["retry"] = {
            "on_events": retry_events,
            "duration": cfg.summitConfig.retry_duration,
        }
    if cfg.summitConfig.image:
        kwargs["image"] = cfg.summitConfig.image
    else:
        kwargs["python"] = "3.12"
        # SGLang's flashinfer backend JIT-compiles CUDA kernels (head_dim=256
        # configs aren't in prebuilt cubins) — needs the CUDA toolkit's nvcc.
        from .recipes import is_decision
        if not is_decision(cfg):
            kwargs["nvcc"] = True
    return Task(**kwargs)


def _build_repo(rendered: RenderedRun):
    from dstack.api import VirtualRepo

    repo = VirtualRepo(repo_id="summit-cli")
    pkg_root = Path(__file__).parent
    for path in pkg_root.rglob("*.py"):
        repo.add_file(f"summit/{path.relative_to(pkg_root)}", path.read_bytes())
    for path in (pkg_root / "templates").glob("*.jinja"):
        repo.add_file(f"summit/templates/{path.name}", path.read_bytes())
    repo.add_file("summit/runtime-requirements.txt", (pkg_root / "runtime-requirements.txt").read_bytes())
    repo.add_file("summit/sglang-overrides.txt", (pkg_root / "sglang-overrides.txt").read_bytes())
    repo.add_file(f"run/{rendered.recipe_filename}", rendered.recipe_yaml.encode())
    repo.add_file("run/bootstrap.sh", rendered.bootstrap_sh.encode())
    for name, content in rendered.files.items():
        repo.add_file(name, content)
    if rendered.execution_plan:
        import json
        repo.add_file("run/execution-plan.json", json.dumps(rendered.execution_plan, indent=2).encode())
    repo.add_file("pyproject.toml", (pkg_root.parent / "pyproject.toml").read_bytes())
    return repo


def _run_name(experiment_name, now):
    stamp = now.strftime("%m%d-%H%M%S")
    experiment = re.sub(r"[^a-z0-9-]", "-", experiment_name.lower()).strip("-") or "run"
    available = 41 - len(f"summit--{stamp}")
    return f"summit-{experiment[:available].rstrip('-')}-{stamp}"


def launch(cfg: SummitJobConfig, env_forward: dict[str, str]) -> str:
    """Submit the training run. Returns the run name."""
    from datetime import datetime, timezone

    from .render import render

    rendered = render(cfg)
    run_name = _run_name(cfg.experiment_name, datetime.now(timezone.utc))

    client = get_client()
    ensure_fleet(cfg.summitConfig.fleet)
    task = _build_task(cfg, run_name, env_forward)
    repo = _build_repo(rendered)
    client.repos.init(repo)  # repo must be registered before code upload
    run = client.runs.apply_configuration(configuration=task, repo=repo)
    return run.name
