"""Render summit.yaml into the two artifacts the VM needs:
1. `rl_train.yaml` — the compiled NexRL recipe
2. `bootstrap.sh` — the on-VM bring-up/run/finalize script
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from .config import SummitJobConfig
from .phantora import NEXRL_COMMIT

TEMPLATE_DIR = Path(__file__).parent / "templates"
NEXRL_REF = NEXRL_COMMIT  # immutable v1.4.0; shared with the simulation image


@dataclass
class RenderedRun:
    recipe_yaml: str
    bootstrap_sh: str
    recipe_filename: str = "rl_train.yaml"
    files: dict[str, bytes] = field(default_factory=dict)
    execution_plan: dict = field(default_factory=dict)


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )


def render(cfg: SummitJobConfig) -> RenderedRun:
    from .recipes import is_decision
    if is_decision(cfg):
        from .recipes.decision.recipe import render_decision
        return render_decision(cfg)
    env = _env()
    gpu = cfg.summitConfig.resources.gpu_spec()
    gpu_names = {name.strip().upper() for name in gpu.name.split(",")}
    blackwell = bool(gpu_names & {"B200", "RTXPRO6000", "RTXPRO6000WK"})
    rollout_gpus = cfg.summitConfig.resources.rollout_gpus
    train_gpus = gpu.count - rollout_gpus
    if train_gpus < 1:
        raise ValueError(
            f"need at least 1 training GPU: gpu count {gpu.count} - rollout_gpus {rollout_gpus}"
        )

    ctx = {
        "project_name": cfg.project_name,
        "experiment_name": cfg.experiment_name,
        "logger_backends": cfg.logger,
        "student_model": cfg.model_name,
        "served_model_name": f"summit-{cfg.experiment_name}-student",
        "teacher_model": cfg.teacher.base_model,
        "teacher_base_url": cfg.teacher.base_url,
        "learning_rate": cfg.learning_rate,
        "optimizer": cfg.optimizer,
        "memory_efficient_fsdp": cfg.memory_efficient_fsdp,
        "blackwell": blackwell,
        "batch_size": cfg.data.batch_size,
        "shuffle": cfg.data.shuffle,
        "rollout_repeat_n": cfg.data.rollout_repeat_n,
        "max_prompt_length": cfg.data.max_prompt_length,
        "max_response_length": cfg.data.max_response_length,
        "max_sequence_length": cfg.data.max_sequence_length,
        "sandbox": cfg.rollout.sandbox,
        "step_limit": cfg.rollout.step_limit,
        "num_workers": cfg.rollout.num_workers,
        "temperature": cfg.rollout.temperature or cfg.algorithm.temperature,
        "distillation_coeff": cfg.algorithm.distillation_coeff,
        "entropy_coeff": cfg.algorithm.entropy_coeff,
        "total_train_steps": cfg.total_train_steps,
        "save_freq": cfg.save_freq,
        "train_gpus": train_gpus,
        # A DeepSWE task normally emits several turn-level trajectories.  Keep
        # those turns in one optimizer minibatch where possible instead of
        # taking a separate optimizer step for every turn on a 1-GPU trainer.
        # NexRL floor-divides this value by the data-parallel world size, so it
        # must also be at least train_gpus.
        "ppo_mini_batch_size": max(cfg.data.batch_size * 4, train_gpus),
        "rollout_gpus": rollout_gpus,
        # full-FT on 1 GPU needs CPU offload for params/optimizer to fit
        "offload": train_gpus <= 1,
        "tasks_csv": ",".join(cfg.data.tasks),
        "hf_repo": cfg.summitConfig.output.hf_push.repo,
        "nexrl_ref": NEXRL_REF,
    }
    return RenderedRun(
        recipe_yaml=env.get_template("nexrl_recipe.yaml.jinja").render(**ctx),
        bootstrap_sh=env.get_template("vm_bootstrap.sh.jinja").render(**ctx),
    )
