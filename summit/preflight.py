"""Local validation for a Summit run before cloud resources are requested.

The checks in this module deliberately execute no remote calls.  They validate
the user config, both rendered artifacts, NexRL's per-rank batch arithmetic,
and the runtime assumptions Summit currently patches into its VM bootstrap.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import SummitJobConfig
from .env import BACKEND_KEY, check_required
from .render import RenderedRun, render
from .phantora import NEXRL_COMMIT

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    hint: str | None = None


@dataclass
class PreflightReport:
    findings: list[Finding] = field(default_factory=list)
    checks_run: int = 0
    rendered: RenderedRun | None = field(default=None, repr=False)
    recipe: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def errors(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def check(
        self,
        condition: bool,
        *,
        code: str,
        severity: Severity,
        message: str,
        hint: str | None = None,
    ) -> None:
        self.checks_run += 1
        if not condition:
            self.findings.append(Finding(code, severity, message, hint))

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.passed else "fail",
            "checks_run": self.checks_run,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [asdict(finding) for finding in self.findings],
            "coverage": {"static": "completed", "simulation": "not_run",
                         "external_services": "not_checked"},
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    def format_text(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"preflight (static): {status} ({self.checks_run} checks, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings)"
        ]
        for finding in self.findings:
            lines.append(f"  {finding.severity.upper()} [{finding.code}] {finding.message}")
            if finding.hint:
                lines.append(f"    fix: {finding.hint}")
        return "\n".join(lines)


def _check_config(report: PreflightReport, cfg: SummitJobConfig) -> None:
    gpu = cfg.summitConfig.resources.gpu_spec()
    rollout_gpus = cfg.summitConfig.resources.rollout_gpus
    train_gpus = gpu.count - rollout_gpus

    unsupported_backends = [
        backend for backend in cfg.summitConfig.backends if backend not in BACKEND_KEY
    ]
    report.check(
        not unsupported_backends,
        code="CFG.BACKEND_UNSUPPORTED",
        severity="error",
        message=(
            "Summit's managed dstack server has no credential adapter for: "
            + ", ".join(unsupported_backends)
        ),
        hint="Use runpod or vastai, or add a credential adapter before selecting this backend.",
    )

    report.check(
        gpu.count > 0 and rollout_gpus > 0 and train_gpus > 0,
        code="CFG.GPU_SPLIT",
        severity="error",
        message=(
            f"invalid GPU split: {gpu.count} total - {rollout_gpus} rollout = "
            f"{train_gpus} training GPUs"
        ),
        hint="Reserve at least one GPU each for training and rollout serving.",
    )
    report.check(
        cfg.data.max_sequence_length
        >= cfg.data.max_prompt_length + cfg.data.max_response_length,
        code="CFG.SEQUENCE_LENGTH",
        severity="error",
        message=(
            f"max_sequence_length={cfg.data.max_sequence_length} is smaller than "
            f"max_prompt_length + max_response_length="
            f"{cfg.data.max_prompt_length + cfg.data.max_response_length}"
        ),
        hint="Increase max_sequence_length or lower the prompt/response caps.",
    )
    report.check(
        not (
            cfg.model_name.lower().startswith("qwen/qwen3.5")
            and rollout_gpus != 1
        ),
        code="CFG.QWEN35_ROLLOUT_TP",
        severity="error",
        message=(
            "Qwen3.5 rollout serving is configured with tensor parallelism greater "
            "than one; this workflow produced divergent Gated DeltaNet outputs and "
            "unparseable mini-swe-agent commands in live runs"
        ),
        hint="Set summitConfig.resources.rollout_gpus: 1.",
    )
    report.check(
        not (
            cfg.model_name == "Qwen/Qwen3.5-9B"
            and cfg.data.max_sequence_length > 8192
        ),
        code="MEMORY.SEQUENCE_CAP",
        severity="warning",
        message=(
            "Qwen3.5-9B full fine-tuning above an 8192-token cap previously OOMed "
            "during the first Summit distillation update"
        ),
        hint="Use an 8192-token or smaller cap; simulator results still need real-GPU qualification.",
    )
    report.check(
        cfg.teacher.backend == "fireworks",
        code="RUNTIME.TEACHER_BACKEND",
        severity="error",
        message=(
            f"teacher.backend={cfg.teacher.backend!r} is accepted by the schema, but "
            "SummitOpdTrainer currently implements only Fireworks echo scoring"
        ),
        hint="Use teacher.backend: fireworks until another remote teacher client is implemented.",
    )
    report.check(
        cfg.rollout.sandbox == "modal",
        code="RUNTIME.SANDBOX_UNSUPPORTED", severity="error",
        message="The Daytona sandbox provider is an interface stub and cannot execute Summit rollouts.",
        hint="Use rollout.sandbox: modal until the Daytona provider is implemented.",
    )


def _check_recipe(report: PreflightReport, recipe: dict[str, Any]) -> None:
    data = recipe["data"]
    pool = recipe["trajectory_pool"]
    student = recipe["service"]["train_service"]["student"]
    actor = student["actor"]
    resource = student["resource"]
    fsdp = actor.get("fsdp_config", {})
    report.check(
        not any(fsdp.get(key) for key in ("param_offload", "grad_offload", "optimizer_offload")),
        code="NEXRL.OFFLOAD_IGNORED", severity="warning",
        message="NexRL v1.4.0 ignores FSDP CPU-offload flags; parameters and optimizer remain on GPU.",
        hint="Size training GPUs for full fine-tuning without CPU offload.",
    )

    report.check(
        not data.get("keep_batch_order")
        or "loaded_batch_finished" in pool.get("check_batch_ready_function", ""),
        code="NEXRL.TRAJECTORY_GATE",
        severity="error",
        message=(
            "NexRL requires a loaded_batch_finished trajectory gate when "
            "data.keep_batch_order is true"
        ),
        hint=(
            "Use check_batch_ready_function: "
            "batch_size_reached_and_loaded_batch_finished."
        ),
    )

    world_size = int(resource["world_size"]) * int(resource["gpus_per_pod"])
    ulysses = int(actor.get("ulysses_sequence_parallel_size", 1))
    dp_size = world_size // ulysses if ulysses else 0
    # The rendered actor points at Hydra's ${data.rollout_repeat_n}; use the
    # concrete source value because plain yaml.safe_load does not resolve it.
    rollout_n = int(data.get("rollout_repeat_n", 1))
    mini = int(actor["ppo_mini_batch_size"]) * rollout_n
    micro = int(actor["ppo_micro_batch_size"])
    ref_log = int(actor["ref"]["log_prob_micro_batch_size"])
    rollout_log = int(actor["rollout"]["log_prob_micro_batch_size"])

    report.check(
        ulysses > 0 and world_size % ulysses == 0,
        code="NEXRL.PARALLELISM",
        severity="error",
        message=f"world_size={world_size} is not divisible by Ulysses size={ulysses}",
        hint="Use a Ulysses size that divides the number of training workers.",
    )
    if dp_size > 0:
        mini_per_rank = mini // dp_size
        micro_per_rank = micro // dp_size
        report.check(
            mini_per_rank > 0,
            code="NEXRL.MINI_BATCH_ZERO",
            severity="error",
            message=(
                f"ppo_mini_batch_size={mini} floor-divides to zero across "
                f"DP size {dp_size}"
            ),
            hint="Set ppo_mini_batch_size to at least the DP size.",
        )
        report.check(
            micro_per_rank > 0,
            code="NEXRL.MICRO_BATCH_ZERO",
            severity="error",
            message=(
                f"ppo_micro_batch_size={micro} floor-divides to zero across "
                f"DP size {dp_size}"
            ),
            hint="Set ppo_micro_batch_size to at least the DP size.",
        )
        report.check(
            micro_per_rank > 0 and mini_per_rank % micro_per_rank == 0,
            code="NEXRL.BATCH_DIVISIBILITY",
            severity="error",
            message=(
                f"per-rank mini batch {mini_per_rank} is not divisible by "
                f"per-rank micro batch {micro_per_rank}"
            ),
            hint="Choose global mini/micro batches whose per-rank values divide evenly.",
        )
        report.check(
            ref_log // dp_size > 0 and rollout_log // dp_size > 0,
            code="NEXRL.LOGPROB_BATCH_ZERO",
            severity="error",
            message=(
                "a ref or rollout log_prob_micro_batch_size floor-divides to zero "
                f"across DP size {dp_size}"
            ),
            hint="Set both log_prob_micro_batch_size values to at least the DP size.",
        )

    report.check(
        actor["fsdp_config"].get("model_dtype") == "bfloat16",
        code="MEMORY.MODEL_DTYPE",
        severity="error",
        message="the FSDP actor is not configured for bfloat16 model weights",
        hint="Set service.train_service.student.actor.fsdp_config.model_dtype: bfloat16.",
    )
    report.check(
        actor.get("ppo_max_token_len_per_gpu")
        == recipe["rollout_worker"].get("max_sequence_length"),
        code="MEMORY.TOKEN_BUDGET",
        severity="error",
        message="the actor token budget does not match the rollout sequence cap",
        hint="Render ppo_max_token_len_per_gpu from max_sequence_length.",
    )
    report.check(
        recipe["weight"].get("sync_method") == "disk"
        and actor["rollout"].get("use_weight_provider") is False,
        code="NEXRL.WEIGHT_SYNC",
        severity="error",
        message="the open-source single-VM path requires disk sync without weight_provider",
        hint="Use weight.sync_method: disk and use_weight_provider: false.",
    )


def _check_bootstrap(report: PreflightReport, script: str) -> None:
    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    report.check(
        syntax.returncode == 0,
        code="RENDER.BASH_SYNTAX",
        severity="error",
        message=f"rendered bootstrap is not valid Bash: {syntax.stderr.strip()}",
        hint="Fix summit/templates/vm_bootstrap.sh.jinja before launching.",
    )

    required_runtime_guards = {
        "NexRL v1.4.0 pin": "NexRL.git@" + NEXRL_COMMIT,
        "Shared runtime dependency lock": '-r "$SUMMIT_REPO_DIR/summit/runtime-requirements.txt"',
        "CuDNN compatibility pin": '"nvidia-cudnn-cu12==9.16.0.29"',
        "CuDNN resolver override": 'summit/sglang-overrides.txt',
        "SGLang pin": '"sglang[all]==0.5.9"',
        "SGLang router pin": '"sglang-router==0.3.2"',
        "Parallel trainer setup": "run_setup_branch trainer setup_nexrl &",
        "Parallel model prefetch": "run_setup_branch model prefetch_model &",
        "W&B API compatibility patch": "patched tracking.py: get_url guarded",
        "FSDP export recovery patch": "patched fsdp_workers.py: convert clears unshard ctx",
        "SGLang router": "sglang_router.launch_router",
    }
    missing = [
        name for name, needle in required_runtime_guards.items() if needle not in script
    ]
    report.check(
        not missing,
        code="RUNTIME.GUARDS",
        severity="error",
        message="rendered bootstrap is missing runtime guards: " + ", ".join(missing),
        hint="Restore the known-good dependency pins and guarded compatibility patches.",
    )
    report.check(
        'if [ "${N:-0}" -ge "$TRAIN_GPUS" ]' in script,
        code="EXPORT.WORKER_COUNT",
        severity="error",
        message="checkpoint export does not require confirmation from every training worker",
        hint="Compare the converted worker count with $TRAIN_GPUS.",
    )


def _check_tasks(
    report: PreflightReport, cfg: SummitJobConfig, project_root: Path
) -> None:
    tasks_root = project_root / "research" / "deep-swe" / "tasks"
    if not tasks_root.is_dir():
        return
    missing: list[str] = []
    for task_id in cfg.data.tasks:
        task_dir = tasks_root / task_id
        if not (task_dir / "task.toml").is_file() or not (
            task_dir / "instruction.md"
        ).is_file():
            missing.append(task_id)
    report.check(
        not missing,
        code="DATA.TASK_NOT_FOUND",
        severity="error",
        message="DeepSWE tasks are missing required files: " + ", ".join(missing),
        hint="Use task IDs present in deep-swe/tasks with task.toml and instruction.md.",
    )


def run_preflight(
    cfg: SummitJobConfig,
    *,
    env: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> PreflightReport:
    """Run local checks and return a structured report.

    Passing ``env=None`` skips secret-presence checks, which is useful when a
    config is being inspected outside the machine that will submit it.
    """

    from .recipes import is_decision
    if is_decision(cfg):
        from .recipes.decision.recipe import preflight
        return preflight(cfg, env=env)

    report = PreflightReport()
    _check_config(report, cfg)

    if env is not None:
        missing = check_required(
            env,
            cfg.teacher.backend,
            cfg.rollout.sandbox,
            cfg.use_wandb,
            cfg.summitConfig.backends,
        )
        report.check(
            not missing,
            code="ENV.MISSING",
            severity="error",
            message="missing required environment keys: " + ", ".join(missing),
            hint="Add the keys to the selected --env-file or process environment.",
        )

    try:
        rendered = render(cfg)
        report.rendered = rendered
    except Exception as exc:  # render errors should be reported, not traceback the CLI
        report.check(
            False,
            code="RENDER.FAILED",
            severity="error",
            message=str(exc),
            hint="Correct the config values used by the Summit templates.",
        )
        return report

    try:
        recipe = yaml.safe_load(rendered.recipe_yaml)
        if not isinstance(recipe, dict):
            raise TypeError("rendered recipe is not a YAML mapping")
        report.recipe = recipe
        report.check(
            True,
            code="RENDER.YAML",
            severity="error",
            message="rendered recipe is not valid YAML",
        )
    except Exception as exc:
        report.check(
            False,
            code="RENDER.YAML",
            severity="error",
            message=f"rendered recipe is not valid YAML: {exc}",
            hint="Fix summit/templates/nexrl_recipe.yaml.jinja.",
        )
        return report

    _check_recipe(report, recipe)
    _check_bootstrap(report, rendered.bootstrap_sh)
    root = project_root or Path(__file__).resolve().parents[1]
    _check_tasks(report, cfg, root)
    return report
