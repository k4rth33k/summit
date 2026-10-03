"""Single-container supervisor, deliberately outside LD_PRELOAD.

The server runs in replay mode with an empty performance DB: no real GPU is
needed and no timing claims are made. Only torchrun children load CUDA stubs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

from summit.phantora import NEXRL_COMMIT, PHANTORA_REVIEWED_COMMIT, PHANTORA_TORCH_COMMIT


def load_recipe(path="/input/recipe.yaml"):
    from omegaconf import OmegaConf

    os.environ.update(NEXRL_DATA_PATH="/tmp/data", SUMMIT_REPO_DIR="/opt/summit",
                      EXPERIMENT_PATH="/tmp/experiment", INFERENCE_BASE_URL="localhost:8001",
                      API_SERVER_URL="localhost")
    recipe = OmegaConf.load(path)
    OmegaConf.resolve(recipe)
    return recipe


def stop_process(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def verify_runtime(manifest):
    import torch
    from packaging.requirements import Requirement

    provenance = json.loads(Path("/opt/phantora-provenance.json").read_text())
    expected = {"phantora": PHANTORA_REVIEWED_COMMIT if manifest["mode"] == "simulation" else None,
                "torch": PHANTORA_TORCH_COMMIT if manifest["mode"] == "simulation" else None,
                "nexrl": NEXRL_COMMIT}
    if provenance != expected or torch.__version__.split("+")[0] != "2.7.1":
        raise RuntimeError(f"Unqualified simulator image: {provenance}, torch={torch.__version__}")
    if manifest["mode"] == "simulation" and not callable(getattr(torch.profiler, "enable_function_tracer", None)):
        raise RuntimeError("PyTorch lacks Phantora's function tracer; stock torch cannot simulate CUDA")
    if hashlib.sha256(Path("/tmp/runtime-requirements.txt").read_bytes()).hexdigest() != manifest["runtime_lock_sha256"]:
        raise RuntimeError("The simulator image has a stale dependency lock; rebuild simulation/Dockerfile")
    for line in Path("/tmp/runtime-requirements.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = importlib.metadata.version(requirement.name)
        if installed not in requirement.specifier:
            raise RuntimeError(f"Image dependency drift: {requirement.name}={installed}, expected {requirement.specifier}")
    direct_url = json.loads(importlib.metadata.distribution("NexRL").read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != NEXRL_COMMIT:
        raise RuntimeError("Image NexRL installation does not match the pinned source commit")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tensordict", "numpy", "NexRL")}
    versions["cuda_build"] = torch.version.cuda
    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    output = Path("/results")
    manifest = json.loads(Path("/input/manifest.json").read_text())
    result = {
        "schema_version": 1, "input_sha256": Path("/input/input.sha256").read_text(),
        "status": "inconclusive", "findings": [], "completed_ranks": [],
        "coverage": {"runtime": "not_run", "imports": "not_run", "nexrl_config": "not_run",
                     "training": "not_run", "timing": "not_validated"},
    }
    server = workers = None
    stage = "runtime"
    try:
        result["versions"] = verify_runtime(manifest)
        result["coverage"]["runtime"] = "completed"
        stage = "imports"
        from summit.nexrl_ext.compat import apply
        apply()
        from nexrl.train_service_backend.fsdp_worker.fsdp_workers import ModelWorker  # noqa: F401
        from summit.nexrl_ext.opd_trainer import SummitOpdTrainer  # noqa: F401
        from summit.rollout.deepswe_worker import DeepSWERolloutWorker  # noqa: F401
        from nexrl.utils.validate_config import validate_config
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
        if Path("/model/config.json").exists():
            model_config = AutoConfig.from_pretrained("/model", local_files_only=True, trust_remote_code=False)
            AutoTokenizer.from_pretrained("/model", local_files_only=True, trust_remote_code=False)
            factory = AutoModelForVision2Seq if type(model_config) in AutoModelForVision2Seq._model_mapping else AutoModelForCausalLM
            # Import the architecture without allocating its weights.
            factory._model_mapping[type(model_config)]
            result["coverage"]["model_metadata"] = "completed"
        else:
            result["coverage"]["model_metadata"] = "not_run"
        result["coverage"]["imports"] = "completed"
        stage = "patch_targets"
        from nexrl.utils import tracking
        worker_source = inspect.getsource(ModelWorker)
        tracking_source = inspect.getsource(tracking)
        for anchor, source in [
            ('attn_implementation="flash_attention_2"', worker_source),
            ("def convert_ckpt_to_huggingface(self, local_path):", worker_source),
            ("def barrier(self):", worker_source),
            ("wandb_run.get_url()", tracking_source),
        ]:
            if source.count(anchor) != 1:
                raise RuntimeError(f"Production bootstrap patch anchor changed: {anchor}")
        result["coverage"]["patch_targets"] = "completed"
        stage = "nexrl_config"
        validate_config(load_recipe())
        result["coverage"]["nexrl_config"] = "completed"
        if manifest["mode"] == "runtime":
            result["status"] = "completed"
            result["findings"].append({"code": "RUNTIME.SCOPE", "message": "Imports and NexRL configuration validated; CUDA execution, model weights and external services were not checked."})
            return 0
        stage = "training"
        ranks = manifest["virtual_cluster"]["gpus_per_host"]
        prefix = "/tmp/summit-phantora"
        env = os.environ | {
            "PHANTORA_SOCKET_PREFIX": prefix, "PHANTORA_LOG": "warn",
            "PHANTORA_VRAM_MIB": str(manifest["virtual_cluster"]["vram_mib"]),
            "PHANTORA_NGPU": str(ranks),
            "PHANTORA_GPU_NAME": manifest["virtual_cluster"]["gpu_name"],
            "OMP_NUM_THREADS": "1",
            "GLOO_SOCKET_IFNAME": "lo",
        }
        # The virtual CUDA implementation exposes devices using this variable.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(ranks)))
        perfdb = output / "perfdb"
        perfdb.mkdir()
        netconfig = output / "netconfig.toml"
        netconfig.write_text(f'''host_mapping = ["{socket.gethostname()}"]
[simulator]
loopback_speed = 2880
fairness = "PerFlowMaxMin"
[topology]
type = "TwoLayerMultiPath"
[topology.args]
nspines = 2
nracks = 1
rack_size = 2
host_bw = 800
rack_uplink_port_bw = 800
load_balancer_type = "EcmpEverything"
''')
        deadline = time.monotonic() + args.timeout
        with (output / "server.log").open("w") as server_log, (output / "workers.log").open("w") as worker_log:
            server = subprocess.Popen(
                ["/phantora/dist/phantora_server", "--netconfig", str(netconfig), "--perf-db", str(perfdb)],
                env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            while not Path(prefix + ".simulator.sock").exists():
                if server.poll() is not None:
                    raise RuntimeError("Phantora server exited before creating its socket; see server.log")
                if time.monotonic() > min(deadline, deadline - args.timeout + 30):
                    raise TimeoutError("Phantora server did not become ready")
                time.sleep(0.05)
            workers = subprocess.Popen(
                ["/phantora/dist/phantora_run", sys.executable, "-m", "torch.distributed.run",
                 "--nnodes=1", "--node-rank=0", "--master-addr=127.0.0.1",
                 "--master-port=29500", f"--nproc-per-node={ranks}",
                 "--max-restarts=0", "-m", "summit.simulation.worker"],
                env=env, stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
            while workers.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Phantora server exited during training; see server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Synthetic update timed out; possible unsupported operation or collective deadlock")
                time.sleep(0.1)
        unsupported = sorted(set(re.findall(
            r'NOT IMPLEMENTED: "([^"]+)"', (output / "workers.log").read_text(errors="replace"))))
        if unsupported:
            result["findings"].append({"code": "SIM.UNSUPPORTED", "message": "Phantora has no implementation for: " + ", ".join(unsupported)})
        rank_results = []
        for rank in range(ranks):
            path = output / f"rank-{rank}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("rank") != rank or row.get("input_sha256") != result["input_sha256"]:
                    raise RuntimeError(f"Invalid rank {rank} report")
                rank_results.append(row)
                if row["status"] == "completed":
                    result["completed_ranks"].append(rank)
        result["rank_results"] = rank_results
        seen = set()
        for row in rank_results:
            if row["status"] != "completed" and row.get("message"):
                key = (row.get("code"), row.get("phase"), row["message"])
                if key not in seen:
                    seen.add(key)
                    result["findings"].append({
                        "code": row.get("code", "SIM.WORKER_ERROR"),
                        "message": f"Rank {row['rank']} during {row.get('phase')}: {row['message']}",
                    })
        if any(row.get("code") == "SIM.CUDA_OOM" for row in rank_results):
            result["status"] = "failed"
            result["coverage"]["training"] = "failed"
            result["findings"].append({"code": "SIM.CUDA_OOM", "message": "The synthetic training scenario exceeded virtual VRAM. See rank reports for the failing phase."})
        elif workers.returncode != 0 or result["completed_ranks"] != list(range(ranks)):
            raise RuntimeError("Not every rank completed the synthetic update; see workers.log and rank reports")
        else:
            result["status"] = "completed"
            result["coverage"]["training"] = "completed"
        # Unknown operations can fall through upstream's timing model. Do not
        # advertise complete CUDA/kernel coverage even when all ranks return.
        result["findings"].append({"code": "SIM.EXPERIMENTAL", "message": "Synthetic shapes only; value-dependent metrics, checkpoint I/O, allocator segmentation and external services are excluded. No throughput estimate or real-GPU fit guarantee."})
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "failed" if stage in {"imports", "patch_targets", "nexrl_config"} else "inconclusive"
        result["coverage"][stage] = result["status"]
        result["findings"].append({"code": f"SIM.{stage.upper()}_ERROR", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        stop_process(workers)
        stop_process(server)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "completed" else (1 if result["status"] == "failed" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
