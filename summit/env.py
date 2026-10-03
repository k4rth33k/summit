"""Environment variable loading and validation for summit.

Secrets live in a local `.env` file (or real environment). Nothing here is
ever written to logs or configs — values are only *forwarded* into the dstack
run's environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

# Keys summit itself needs for every launch.
ALWAYS_REQUIRED = ("HF_TOKEN",)

# Cloud backend -> key used by Summit's managed dstack server.
BACKEND_KEY = {
    "runpod": "RUNPOD_API_KEY",
    "vastai": "VAST_API_KEY",
}

# Teacher backend -> required key.
TEACHER_KEY = {
    "fireworks": "FIREWORKS_API_KEY",
    "friendli": "FRIENDLI_API_KEY",
}

# Sandbox provider -> required keys.
SANDBOX_KEYS = {
    "modal": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
    "daytona": (),  # interface stub in v0.1
}

# Optional keys forwarded when present.
OPTIONAL_KEYS = ("WANDB_API_KEY", "WANDB_KEY", "WANDB_HOST", "FRIENDLI_API_KEY")


def load_env(env_file: str | Path = ".env") -> dict[str, str]:
    """Merge `.env` file values with the real environment (env wins)."""
    values: dict[str, str] = {}
    path = Path(env_file)
    if path.exists():
        for k, v in dotenv_values(path).items():
            if v is not None:
                values[k] = v
    for k, v in os.environ.items():
        values[k] = v
    return values


def check_required(
    env: dict[str, str],
    teacher_backend: str,
    sandbox: str,
    use_wandb: bool,
    backends: list[str] | None = None,
) -> list[str]:
    """Return a list of missing required keys (empty = all good)."""
    missing = [k for k in ALWAYS_REQUIRED if not env.get(k)]
    for backend in backends or ["runpod"]:
        backend_key = BACKEND_KEY.get(backend)
        if backend_key and not env.get(backend_key):
            missing.append(backend_key)
    teacher_key = TEACHER_KEY.get(teacher_backend)
    if teacher_key and not env.get(teacher_key):
        missing.append(teacher_key)
    for k in SANDBOX_KEYS.get(sandbox, ()):
        if not env.get(k):
            missing.append(k)
    if use_wandb and not (env.get("WANDB_KEY") or env.get("WANDB_API_KEY")):
        missing.append("WANDB_KEY (or WANDB_API_KEY)")
    return missing


def backend_credentials(
    env: dict[str, str], backends: list[str] | None = None
) -> dict[str, str]:
    """Return configured dstack backend credentials without exposing them."""

    selected = backends or list(BACKEND_KEY)
    return {
        backend: env[key]
        for backend in selected
        if (key := BACKEND_KEY.get(backend)) and env.get(key)
    }


def forwarded_env(env: dict[str, str], extra_names: list[str]) -> dict[str, str]:
    """Build the environment to inject into the dstack run.

    Forwards only known/requested keys. Maps WANDB_API_KEY -> WANDB_KEY
    (NexRL's tracker reads WANDB_KEY).
    """
    out: dict[str, str] = {}
    for name in set(extra_names) | set(OPTIONAL_KEYS):
        if env.get(name):
            out[name] = env[name]
    if not out.get("WANDB_KEY") and env.get("WANDB_API_KEY"):
        out["WANDB_KEY"] = env["WANDB_API_KEY"]
    out.setdefault("WANDB_HOST", env.get("WANDB_HOST", "https://api.wandb.ai"))
    return out
