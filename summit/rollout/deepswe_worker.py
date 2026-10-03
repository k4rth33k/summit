"""DeepSWE rollout worker for NexRL (mini-swe-agent + sandboxed execution).

Loaded by NexRL via `rollout_worker.custom_rollout_worker_module_path`.

Each task row from the dataloader carries `prompt` = JSON string:
  {"task_id": ..., "instruction": ..., "image": ...}
(produced by summit's data-prep step from datacurve-ai/deep-swe).

For every agent turn we emit one NexRL Trajectory (prompt = full conversation
rendered by the student server, response = that turn's completion) — the same
shape AgentRolloutWorker produces, so trainer-side datum assembly is stock.
OPD uses reward=0.0; the only supervision is teacher KL.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from nexrl.nexrl_types import Trajectory
from nexrl.rollout_worker.base_rollout_worker import BaseRolloutWorker

logger = logging.getLogger(__name__)


class DeepSWERolloutWorker(BaseRolloutWorker):
    def __init__(self, config):
        super().__init__(config)
        self._sandbox_name = config.get("sandbox", "modal")
        self._step_limit = int(config.get("step_limit", 30))
        self._traj_dir = os.environ.get("EXPERIMENT_PATH")
        self._result_file = os.environ.get("SUMMIT_ROLLOUT_RESULT_FILE")

    def _record_result(self, produced_trajectory: bool) -> None:
        """Atomically record one result for the paid-VM batch watchdog."""
        if not self._result_file:
            return
        fd = os.open(self._result_file, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, b"1\n" if produced_trajectory else b"0\n")
        finally:
            os.close(fd)

    def _generate(self, messages, **kwargs):
        """Call the student policy; repair prompt_tokens for processor-style
        tokenizers (Qwen3.5's apply_chat_template returns a BatchEncoding,
        which NexRL's isinstance(list) guard silently drops)."""
        from summit.rollout.mini_swe import repair_prompt_tokens

        resp = self._inference_client.generate(messages, **kwargs)
        return repair_prompt_tokens(
            resp,
            lambda: self._inference_client.apply_chat_template(messages, tokenize=True),
        )

    def rollout(self, task: dict[str, Any]) -> str | None:
        from summit.rollout.mini_swe import run_agent_task
        from summit.rollout.sandbox.base import get_provider

        # Unpack the task payload.
        payload = task.get("prompt", "{}")
        try:
            task_info = json.loads(payload) if isinstance(payload, str) else dict(payload)
        except json.JSONDecodeError:
            logger.warning("DeepSWE task prompt is not valid JSON, skipping: %r", payload[:200])
            self._record_result(False)
            return None

        logger.info("Running DeepSWE task %s", task_info.get("task_id"))

        try:
            result = run_agent_task(
                task=task_info,
                generate_fn=lambda messages, **kw: self._generate(messages, **kw),
                sandbox_provider=get_provider(self._sandbox_name),
                step_limit=self._step_limit,
                output_path=self._trajectory_path(task_info.get("task_id", "unknown")),
            )
        except Exception:  # noqa: BLE001
            logger.exception("agent run failed for task %s", task_info.get("task_id"))
            self._record_result(False)
            return None

        identifier = self._config.inference_service.get("identifier", "default")
        ground_truth = (task.get("reward_model") or {}).get("ground_truth", "")

        last_status = None
        n_turns = 0
        n_dropped = 0
        for msg in result["messages"]:
            if msg.get("role") != "assistant":
                continue
            train = (msg.get("extra") or {}).get("nexrl_train") or {}
            prompt_tokens = train.get("prompt_tokens", [])
            response_tokens = train.get("response_tokens", [])
            response_logprobs = train.get("response_logprobs", [])
            if not prompt_tokens or not response_tokens:
                n_dropped += 1
                if n_dropped <= 2:
                    logger.warning(
                        "dropping turn: prompt_tokens=%d response_tokens=%d "
                        "logprobs=%d content=%r",
                        len(prompt_tokens),
                        len(response_tokens),
                        len(response_logprobs),
                        (msg.get("content") or "")[:200],
                    )
                continue

            tokens = list(prompt_tokens) + list(response_tokens)
            loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
            logprobs = [0.0] * len(prompt_tokens) + [float(x) for x in response_logprobs]

            tokens, loss_mask, logprobs, is_truncated = self._check_and_truncate(
                tokens, loss_mask, logprobs
            )

            trajectory = Trajectory(
                tokens=tokens,
                loss_mask=loss_mask,
                reward=0.0,  # OPD: no environment reward; supervision is teacher KL
                is_val=task.get("is_val", False),
                extra_fields={
                    "ground_truth": ground_truth,
                    "group_id": task.get("group_id", ""),
                    "run_id": task.get("run_id", 0),
                    "task_id": task.get("task_id", 0),
                    "temperature": self._config.temperature,
                    "finish_reason": (msg.get("extra") or {}).get("finish_reason", "stop"),
                    "model_tag": identifier,
                    "logprobs": logprobs,
                    "is_truncated": is_truncated,
                },
            )
            last_status = self._put_trajectory(trajectory)
            n_turns += 1

        logger.info(
            "DeepSWE task %s produced %d trajectories, dropped %d turns (exit=%s)",
            task_info.get("task_id"),
            n_turns,
            n_dropped,
            (result.get("exit") or {}).get("exit_status", "unknown"),
        )
        self._record_result(n_turns > 0)
        return last_status

    def _trajectory_path(self, task_id: str):
        if not self._traj_dir:
            return None
        from pathlib import Path

        d = Path(self._traj_dir) / "agent_trajectories"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{task_id}-{os.getpid()}.json"
