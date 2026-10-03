"""Host-side runner. Only an explicitly requested stage starts Docker.

Containers get no credentials, network, or GPU devices. Each invocation owns a
fresh results directory and a unique container name; failures never reuse a
previous report. Simulator failures are not labelled as workload failures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from summit.phantora import build_phantora_manifest

DEFAULT_IMAGE = "summit-phantora:torch2.7.1"
DEFAULT_RUNTIME_IMAGE = "summit-runtime:torch2.7.1"
MODEL_METADATA_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "tokenizer.tiktoken",
    "tekken.json", "preprocessor_config.json", "processor_config.json", "chat_template.json",
}


def fingerprint(manifest: dict, recipe: str) -> str:
    return hashlib.sha256(
        (json.dumps(manifest, sort_keys=True) + "\n" + recipe).encode()
    ).hexdigest()


def run_simulation(cfg, rendered, *, image=None, model_dir=None, mode="simulation",
                   output_dir="summit-checks", timeout=600, docker="docker") -> dict:
    """Run a local image; never build or pull as a side effect of a check."""
    if timeout <= 0:
        raise ValueError("simulation timeout must be positive")
    if mode not in {"runtime", "simulation"}:
        raise ValueError("unknown check mode")
    image = image or (DEFAULT_IMAGE if mode == "simulation" else DEFAULT_RUNTIME_IMAGE)
    package_root = Path(__file__).resolve().parents[1]
    output = Path(output_dir).resolve()
    if output == package_root or package_root in output.parents:
        raise ValueError("Choose a check output directory outside the summit Python package")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="phantora-", dir=output))
    manifest = build_phantora_manifest(cfg)
    manifest["mode"] = mode
    manifest["bootstrap_sha256"] = hashlib.sha256(rendered.bootstrap_sh.encode()).hexdigest()
    # Run an immutable copy, so editing the checkout during a long check cannot
    # change the worker halfway through execution. Store hashes for replay.
    package = run_dir / "code" / "summit"
    code_hashes = {}
    for source in sorted(package_root.rglob("*.py")) + [package_root / "runtime-requirements.txt"]:
        relative = source.relative_to(package_root)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        code_hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    manifest["summit_code_sha256"] = code_hashes
    manifest["runtime_lock_sha256"] = code_hashes["runtime-requirements.txt"]
    manifest["adapter_code_sha256"] = {key: value for key, value in code_hashes.items() if key.startswith("simulation/")}
    digest = fingerprint(manifest, rendered.recipe_yaml)
    result = {
        "schema_version": 1, "mode": mode, "status": "inconclusive", "input_sha256": digest,
        "findings": [], "coverage": {"training": "not_run", "timing": "not_validated"},
        "excluded": manifest["scope"]["exclude"], "artifacts": str(run_dir),
    }
    bundle = run_dir / "input"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (bundle / "recipe.yaml").write_text(rendered.recipe_yaml)
    (bundle / "bootstrap.sh").write_text(rendered.bootstrap_sh)
    (bundle / "input.sha256").write_text(digest)

    def finish(code=None, message=None):
        if code:
            result["findings"].append({"code": code, "message": message})
        (run_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    if not shutil.which(docker):
        return finish("SIM.DOCKER_MISSING", "Install Docker and build the simulation image (see docs/validation.md).")
    if mode == "simulation" and not manifest["virtual_cluster"]["vram_mib"]:
        return finish("SIM.VRAM_UNKNOWN", "Specify GPU memory explicitly, for example A100-80GB:8.")
    if model_dir is None and mode == "simulation":
        return finish("SIM.MODEL_METADATA_MISSING", "Pass --model-dir with config.json and tokenizer files from the exact model revision; weights are unnecessary.")
    model = Path(model_dir).resolve() if model_dir is not None else None
    if model is not None and not (model / "config.json").is_file():
        return finish("SIM.MODEL_METADATA_MISSING", f"No config.json in {model}.")
    # Copy only metadata, never weights or arbitrary Python, into the container.
    # Following HF-cache symlinks here works without mounting the entire cache.
    metadata = run_dir / "model"
    metadata.mkdir()
    hashes = {}
    for source in sorted(model.iterdir()) if model else []:
        if source.is_file() and source.name in MODEL_METADATA_FILES:
            if source.stat().st_size > 128 * 1024 * 1024:
                return finish("SIM.MODEL_METADATA_TOO_LARGE", f"Metadata file exceeds 128 MiB: {source.name}")
            target = metadata / source.name
            shutil.copyfile(source, target)
            hashes[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["model_metadata_sha256"] = hashes
    digest = fingerprint(manifest, rendered.recipe_yaml)
    result["input_sha256"] = digest
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (bundle / "input.sha256").write_text(digest)
    results = run_dir / "results"
    results.mkdir()
    result["model_metadata_sha256"] = hashes
    try:
        inspection = subprocess.run([docker, "image", "inspect", image, "--format", "{{.Id}}"],
                                    capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return finish("SIM.DOCKER_UNAVAILABLE", str(exc))
    if inspection.returncode:
        return finish("SIM.IMAGE_UNAVAILABLE", f"Local image {image!r} is unavailable. Build it using simulation/Dockerfile; images are never pulled by check.")
    image_id = inspection.stdout.strip()
    result["image_id"] = image_id
    name = "summit-check-" + uuid.uuid4().hex
    command = [
        docker, "run", "--rm", "--pull=never", "--name", name,
        "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size=1g",
        "--env", "PYTHONPATH=/opt/summit", "--env", "HOME=/tmp",
        # CUDA base images default NVIDIA_VISIBLE_DEVICES=all. Override it so
        # even a daemon whose default runtime is NVIDIA injects no GPU devices.
        "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "WANDB_MODE=disabled", "--env", "PYTHONDONTWRITEBYTECODE=1",
    ]
    for source, dest, readonly in [
        (package, "/opt/summit/summit", True), (bundle, "/input", True),
        (metadata, "/model", True), (results, "/results", False),
    ]:
        # Docker --mount has comma-separated syntax, so reject ambiguous paths.
        if "," in str(source):
            return finish("SIM.PATH_UNSUPPORTED", "Simulation paths must not contain commas.")
        command += ["--mount", f"type=bind,src={source},dst={dest}" + (",readonly" if readonly else "")]
    command += ["--entrypoint", "python", image_id, "-m", "summit.simulation.container",
                "--timeout", str(timeout)]
    failure = None
    cleanup_problem = None
    try:
        with (run_dir / "container.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        failure = ("SIM.TIMEOUT", f"Simulation exceeded {timeout} seconds; inspect container.log and results/*.log.")
    except KeyboardInterrupt:
        failure = ("SIM.INTERRUPTED", "Check interrupted; cleanup was requested for this invocation's container.")
    except OSError as exc:
        failure = ("SIM.START_FAILED", str(exc))
    finally:
        # Killing the docker client alone leaves a running container behind.
        # Remove only the unique container owned by this invocation.
        try:
            cleanup = subprocess.run([docker, "rm", "-f", name], capture_output=True, text=True, timeout=15)
            if cleanup.returncode and "No such container" not in (cleanup.stderr or ""):
                cleanup_problem = f"Could not confirm cleanup of container {name}: {cleanup.stderr}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            cleanup_problem = f"Could not confirm cleanup of container {name}: {exc}"
    if cleanup_problem:
        result["findings"].append({"code": "SIM.CLEANUP_UNCONFIRMED", "message": cleanup_problem})
    if failure:
        return finish(*failure)
    if cleanup_problem:
        return finish()
    try:
        payload = json.loads((results / "result.json").read_text())
        expected_ranks = manifest["virtual_cluster"]["gpus_per_host"] if mode == "simulation" else 0
        valid = (
            isinstance(payload, dict) and payload.get("schema_version") == 1
            and payload.get("input_sha256") == digest
            and payload.get("status") in {"completed", "failed", "inconclusive"}
            and isinstance(payload.get("findings"), list)
            and all(isinstance(item, dict) and isinstance(item.get("code"), str)
                    and isinstance(item.get("message"), str) for item in payload["findings"])
            and isinstance(payload.get("coverage"), dict)
        )
        if not valid:
            raise ValueError("invalid result schema or input fingerprint")
        if payload["status"] == "completed" and (
            completed.returncode != 0 or payload.get("completed_ranks") != list(range(expected_ranks))
            or (mode == "simulation" and payload["coverage"].get("training") != "completed")
            or any(payload["coverage"].get(key) != "completed" for key in ("runtime", "imports", "nexrl_config"))
        ):
            raise ValueError("incomplete ranks or nonzero exit despite completion report")
    except (OSError, ValueError, TypeError) as exc:
        return finish("SIM.RESULT_INVALID", f"No trustworthy completion report ({exc}); container exit={completed.returncode}. Inspect container.log.")
    for key in ("status", "findings", "coverage", "completed_ranks", "versions", "rank_results"):
        if key in payload:
            result[key] = payload[key]
    return finish()


def format_simulation(result: dict) -> str:
    label = "runtime check" if result.get("mode") == "runtime" else "simulation"
    lines = [f"{label}: {result['status'].upper()}"]
    if label == "simulation":
        lines.append("  experimental; timing and numerical correctness unvalidated")
    for finding in result["findings"]:
        lines.append(f"  [{finding['code']}] {finding['message']}")
    if result.get("artifacts"):
        lines.append(f"  artifacts: {result['artifacts']}")
    return "\n".join(lines)
