"""Translate a Summit run into the contract for the Phantora adapter.

The simulator consumes the rendered recipe alongside this versioned manifest.
No torch, Docker, or cloud dependency is imported by this module.
"""

from __future__ import annotations

from typing import Any
import hashlib
from pathlib import Path

from .config import SummitJobConfig

PHANTORA_REPOSITORY = "https://github.com/QDelta/Phantora"
PHANTORA_REVIEWED_COMMIT = "301d9d976a13067cc0f5438912395a11efba5f8e"
PHANTORA_TORCH_COMMIT = "43d1bb713f140464e7d6ae7014b1de0f140dc94b"
NEXRL_COMMIT = "6860191994413876a9d7f1431d79c29d114a12e6"

_DEFAULT_VRAM_MIB = {
    "H100": 81559,
    "H200": 143771,
}


def build_phantora_manifest(cfg: SummitJobConfig) -> dict[str, Any]:
    """Build the stable input document for Summit's NexRL Phantora driver."""

    from .recipes import is_decision
    if is_decision(cfg):
        return {
            "schema_version": 2, "adapter": None, "recipe": "decision",
            "status": "unsupported", "scope": {"simulate": [], "exclude": [
                "Decision model training has no qualified Phantora adapter yet."
            ]},
        }

    gpu = cfg.summitConfig.resources.gpu_spec()
    gpu_names = {name.strip().upper() for name in gpu.name.split(",")}
    blackwell = bool(gpu_names & {"B200", "RTXPRO6000", "RTXPRO6000WK"})
    train_gpus = gpu.count - cfg.summitConfig.resources.rollout_gpus
    memory_mib = (
        gpu.memory_gb * 1024
        if gpu.memory_gb is not None
        else _DEFAULT_VRAM_MIB.get(gpu.name.upper())
    )
    return {
        "schema_version": 2,
        "adapter": "summit-nexrl-fsdp",
        "runtime_lock_sha256": hashlib.sha256(
            Path(__file__).with_name("runtime-requirements.txt").read_bytes()
        ).hexdigest(),
        "adapter_code_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).with_name("simulation").glob("*.py"))
        },
        "upstream": {
            "repository": PHANTORA_REPOSITORY,
            "reviewed_commit": PHANTORA_REVIEWED_COMMIT,
            "pytorch_commit": PHANTORA_TORCH_COMMIT,
        },
        "virtual_cluster": {
            "hosts": 1,
            "gpus_per_host": train_gpus,
            "gpu_name": gpu.name,
            "vram_mib": memory_mib,
        },
        "workload": {
            "framework": "nexrl",
            "framework_ref": "v1.4.0",
            "framework_commit": NEXRL_COMMIT,
            "pytorch_version": "2.7.1",
            "pytorch_cuda_build": "cu128" if blackwell else "cu126",
            "gpu_architecture_profile": "blackwell" if blackwell else "default",
            "parallelism": "fsdp1",
            "model": cfg.model_name,
            "model_dtype": "bfloat16",
            "optimizer": cfg.optimizer,
            "memory_efficient_fsdp": cfg.memory_efficient_fsdp,
            "batch_size": cfg.data.batch_size,
            "rollout_repeat_n": cfg.data.rollout_repeat_n,
            "max_prompt_length": cfg.data.max_prompt_length,
            "max_response_length": cfg.data.max_response_length,
            "max_sequence_length": cfg.data.max_sequence_length,
            "steps": 1,
            # Each agent step can produce a trajectory. Include a terminal
            # response and pad up to a whole per-rank mini batch. This is a
            # conservative synthetic stress scenario, not a predicted rollout.
            "trajectory_rows_upper_bound": (
                min(cfg.data.batch_size, len(cfg.data.tasks))
                * cfg.data.rollout_repeat_n * (cfg.rollout.step_limit + 1)
            ),
        },
        "scope": {
            "simulate": [
                "FSDP model initialization",
                "one synthetic trainer batch including forward/backward/AdamW",
                "CUDA memory allocations",
                "NCCL collectives",
            ],
            "exclude": [
                "SGLang generation",
                "mini-swe-agent and sandbox control flow",
                "remote teacher, W&B, and Hugging Face APIs",
                "checkpoint export timing",
                "checkpoint loading and real parameter values",
                "value-dependent metrics and nonfinite-gradient decisions",
                "tied embedding identity (simulated as separate parameters, conservatively)",
                "CUDA expandable-segment allocator behavior",
                "cuDNN/cuBLAS/kernel workspace parity (simulator builds without cuDNN)",
                "GPU kernel numerical correctness and throughput estimates",
            ],
        },
    }
