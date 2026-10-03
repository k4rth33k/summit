"""Legacy OPD schema and recipe-aware YAML loading."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from .recipes.decision.config import DecisionJobConfig

DEFAULT_FIREWORKS_URL = "https://api.fireworks.ai/inference"
DEFAULT_FRIENDLI_URL = "https://api.friendli.ai/serverless"


class AlgorithmConfig(BaseModel):
    type: Literal["distillation"] = "distillation"
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    temperature: float = Field(default=0.8, gt=0)
    distillation_coeff: float = 1.0
    entropy_coeff: float = 0.0


class TeacherConfig(BaseModel):
    backend: Literal["fireworks", "friendli", "weaver", "tinker"] = "fireworks"
    base_model: str = "accounts/fireworks/models/qwen3p8-2p4t-a95b"  # Qwen3.8 Max (2.4T-A95B)
    base_url: str | None = None  # defaults per-backend below
    api_key_env: str | None = None  # defaults per-backend below

    @model_validator(mode="after")
    def fill_defaults(self) -> "TeacherConfig":
        if self.base_url is None:
            self.base_url = (
                DEFAULT_FIREWORKS_URL if self.backend == "fireworks" else DEFAULT_FRIENDLI_URL
            )
        if self.api_key_env is None:
            self.api_key_env = (
                "FIREWORKS_API_KEY" if self.backend == "fireworks" else "FRIENDLI_API_KEY"
            )
        return self


class DataConfig(BaseModel):
    source: Literal["deepswe"] = "deepswe"
    tasks: list[str] = Field(min_length=1)  # task ids under deep-swe/tasks/
    batch_size: int = Field(default=4, gt=0)
    shuffle: bool = True
    rollout_repeat_n: int = Field(default=1, gt=0)
    max_prompt_length: int = Field(default=8192, gt=0)
    max_response_length: int = Field(default=4096, gt=0)
    max_sequence_length: int = Field(default=16384, gt=0)


class RolloutConfig(BaseModel):
    agent: Literal["mini-swe-agent"] = "mini-swe-agent"
    sandbox: Literal["modal", "daytona"] = "modal"
    num_workers: int = Field(default=2, gt=0)
    step_limit: int = Field(default=30, gt=0)  # max agent steps per task
    temperature: float | None = Field(default=None, gt=0)  # defaults to algorithm.temperature


class GPUResource(BaseModel):
    """e.g. `A100-80GB:8` or `H100:8`."""

    name: str = Field(default="A100", min_length=1)
    memory_gb: int | None = Field(default=80, gt=0)
    count: int = Field(default=8, gt=0)

    @classmethod
    def parse(cls, spec: str) -> "GPUResource":
        # forms: NAME:COUNT | NAME-MEMGB:COUNT | NAME
        count = 8
        memory = None
        main = spec
        if ":" in main:
            main, count_s = main.rsplit(":", 1)
            count = int(count_s)
        name = main
        if "-" in main and main.rsplit("-", 1)[1].rstrip("GBgb").isdigit():
            name, mem = main.rsplit("-", 1)
            memory = int(mem.rstrip("GBgb"))
        return cls(name=name, memory_gb=memory, count=count)


class ResourcesConfig(BaseModel):
    gpu: str = "A100-80GB:8"
    shm_size: str = "64GB"
    # disk must stay None for vast.ai (its offers don't report allocatable
    # disk, so any disk requirement filters them out); RunPod allocates fine.
    disk: str | None = None
    rollout_gpus: int = Field(default=2, gt=0)  # GPUs reserved for the SGLang student server

    @model_validator(mode="after")
    def validate_gpu(self):
        self.gpu_spec()
        return self

    def gpu_spec(self) -> GPUResource:
        return GPUResource.parse(self.gpu)


class HFPushConfig(BaseModel):
    repo: str
    private: bool = True


class OutputConfig(BaseModel):
    hf_push: HFPushConfig


class SummitOrchestrationConfig(BaseModel):
    """The `summitConfig` block: everything the orchestrator needs."""

    backends: list[str] = Field(default=["runpod"], min_length=1)
    regions: list[str] | None = None  # optional pin, e.g. [CA-MTL-1]
    spot_policy: Literal["on-demand"] = "on-demand"  # spot lands on the roadmap
    fleet: str = "summit-fleet"  # dstack fleet template; auto-created by init/launch
    resources: ResourcesConfig = ResourcesConfig()
    image: str | None = None  # custom docker image; default = dstack base image
    max_duration: str = "6h"
    # Wait for capacity instead of failing instantly when no offers match.
    retry_on_no_capacity: bool = True
    retry_on_error: bool = True
    retry_on_interruption: bool = True
    retry_duration: str = "2h"
    env: list[str] = []  # extra env var names to forward into the run
    output: OutputConfig


class SummitJobConfig(BaseModel):
    project_name: str = "summit"
    experiment_name: str

    # ---- training (tinker-like surface) ----
    model_name: str = "Qwen/Qwen3.5-9B"
    lora_rank: int = 32
    learning_rate: float = Field(default=1.0e-4, gt=0)
    optimizer: Literal["adamw", "adafactor"] = "adamw"
    memory_efficient_fsdp: bool = False
    loss_fn: str = "importance_sampling"

    algorithm: AlgorithmConfig = AlgorithmConfig()
    teacher: TeacherConfig = TeacherConfig()
    data: DataConfig
    rollout: RolloutConfig = RolloutConfig()

    total_train_steps: int = Field(default=2, gt=0)
    save_freq: int = 2
    logger: list[str] = ["console", "wandb"]

    summitConfig: SummitOrchestrationConfig

    @model_validator(mode="after")
    def check_v01_constraints(self) -> "SummitJobConfig":
        if self.summitConfig.spot_policy != "on-demand":
            raise ValueError("v0.1 supports spot_policy: on-demand only")
        if self.algorithm.type != "distillation":
            raise ValueError("v0.1 supports algorithm.type: distillation (OPD) only")
        # NexRL's self-hosted FSDP backend is full fine-tuning only; LoRA exists
        # on the weaver/tinker service backends. Warn rather than silently ignore.
        warnings.warn(
            "note: the self-hosted FSDP backend full-fine-tunes the student; "
            f"lora_rank={self.lora_rank} is recorded but not applied in v0.1",
            stacklevel=2,
        )
        return self

    @property
    def use_wandb(self) -> bool:
        return "wandb" in self.logger


def load_config(path: str | Path) -> SummitJobConfig | DecisionJobConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    if raw.get("recipe") == "decision":
        from .recipes.decision.config import DecisionJobConfig

        cfg = DecisionJobConfig.model_validate(raw)
        cfg._config_dir = Path(path).resolve().parent
        return cfg
    if raw.get("recipe", "deepswe_opd") != "deepswe_opd":
        raise ValueError(f"unknown recipe: {raw['recipe']}")
    return SummitJobConfig.model_validate(raw)
