"""Versioned offline recipe; deliberately independent of the legacy OPD schema."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from summit.config import GPUResource, HFPushConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelConfig(StrictModel):
    name: str = "Qwen/Qwen3.5-4B"
    revision: str | None = None
    # A custom module:factory must implement the documented decision model contract.
    adapter: str = "causal_lm"
    adapter_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Autocast compute precision; builtin trainable weights/AdamW remain FP32.
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    gradient_checkpointing: bool = True
    adaptation: Literal["full", "qlora"] = "full"

    @model_validator(mode="after")
    def validate_adapter(self):
        if self.adapter not in {"causal_lm", "candidate_scorer", "qlora_causal"}:
            module, separator, symbol = self.adapter.partition(":")
            if not separator or not symbol.isidentifier() or not all(
                part.isidentifier() for part in module.split(".")
            ):
                raise ValueError("model.adapter must be a builtin or module:factory")
        if self.adapter == "qlora_causal" and self.adaptation != "qlora":
            raise ValueError("qlora_causal requires model.adaptation: qlora")
        if self.adapter != "qlora_causal" and self.adaptation != "full":
            raise ValueError("model.adaptation: qlora requires the qlora_causal adapter")
        return self


class DataConfig(StrictModel):
    train: Path
    validation: Path
    max_length: int = Field(default=2048, ge=16)
    max_candidates: int = Field(default=24, ge=2, le=26)


class ObjectiveConfig(StrictModel):
    type: Literal["candidate_ce", "candidate_distillation"] = "candidate_ce"
    alpha: float = Field(default=0.5, ge=0, le=1)
    temperature: float = Field(default=1.0, gt=0)


class TrainingConfig(StrictModel):
    backend: Literal["torch_single"] = "torch_single"
    comparison: Literal["single", "ce_kd"] = "single"
    device: Literal["cpu", "cuda"] = "cuda"
    epochs: int = Field(default=1, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0)
    # A freshly initialized decision head generally needs a larger update than
    # its pretrained backbone. When omitted, every parameter uses learning_rate.
    head_learning_rate: float | None = Field(default=None, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    seed: int = 42
    evaluate_reversed_options: bool = False


class Resources(StrictModel):
    gpu: str = "RTXPRO6000-96GB:1"
    rollout_gpus: Literal[0] = 0
    shm_size: str = "16GB"
    disk: str | None = "100GB"

    def gpu_spec(self):
        return GPUResource.parse(self.gpu)

    @model_validator(mode="after")
    def single_gpu(self):
        if self.gpu_spec().count != 1:
            raise ValueError("torch_single currently supports exactly one training GPU")
        return self


class DecisionHFPushConfig(HFPushConfig):
    model_config = ConfigDict(extra="forbid")
    replace: bool = False


class Output(StrictModel):
    directory: Path = Path("outputs/decision")
    hf_push: DecisionHFPushConfig | None = None


class Orchestration(StrictModel):
    backends: list[Literal["runpod", "vastai"]] = Field(default_factory=lambda: ["runpod"], min_length=1)
    regions: list[str] | None = None
    fleet: str = "summit-fleet"
    resources: Resources = Field(default_factory=Resources)
    image: str | None = None
    spot_policy: Literal["on-demand"] = "on-demand"
    max_duration: str = "1h"
    max_price: float | None = Field(default=None, gt=0)
    retry_on_no_capacity: bool = True
    retry_on_error: bool = False
    retry_on_interruption: bool = False
    retry_duration: str = "20m"
    env: list[str] = Field(default_factory=list)
    output: Output = Field(default_factory=Output)


class DecisionJobConfig(StrictModel):
    schema_version: Literal[2] = 2
    recipe: Literal["decision"] = "decision"
    project_name: str = "summit-decision"
    experiment_name: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    # Local Python source files/directories, relative to the YAML. Bundled for cloud runs.
    code: list[Path] = Field(default_factory=list)
    summitConfig: Orchestration = Field(default_factory=Orchestration)
    _config_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def validate_comparison(self):
        if self.training.comparison == "ce_kd" and (
                self.objective.type != "candidate_distillation" or not self.training.evaluate_reversed_options):
            raise ValueError("ce_kd comparison requires cached distillation targets and reversed-option evaluation")
        if self.training.head_learning_rate is not None and self.model.adapter != "candidate_scorer":
            raise ValueError("head_learning_rate is supported only by the builtin candidate_scorer")
        return self

    @property
    def use_wandb(self):
        return False

    def resolve_path(self, path: Path) -> Path:
        return (self._config_dir / path).resolve()

    def resolved_model(self) -> ModelConfig:
        local = self.resolve_path(Path(self.model.name))
        if local.exists():
            return self.model.model_copy(update={"name": str(local)})
        if self.model.name.startswith((".", "/")):
            raise ValueError(f"local model path does not exist: {local}")
        return self.model
