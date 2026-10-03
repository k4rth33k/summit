"""OPD trainer with a remote Fireworks teacher.

NexRL's stock `SelfHostedOpdTrainer` computes teacher logprobs via co-located
FSDP teacher workers (backends "http"/"direct-zmq"). A 397B teacher doesn't
fit next to the student on a single VM, so this subclass swaps the teacher
side for Fireworks' echo-logprobs API. The student side (local FSDP training,
distillation update math, weight sync) is 100% stock NexRL.

Loaded via the recipe's `trainer.custom_trainer_module_path`.

Teacher config env vars (exported by the VM bootstrap from the recipe):
  SUMMIT_TEACHER_MODEL   e.g. accounts/fireworks/models/qwen3p5-397b-a17b
  SUMMIT_TEACHER_BASE_URL  default https://api.fireworks.ai/inference
  FIREWORKS_API_KEY
"""

from __future__ import annotations

import logging
import os
from typing import Any

from nexrl.trainer.self_hosted_opd_trainer import SelfHostedOpdTrainer

logger = logging.getLogger(__name__)

# Keys the teacher scoring path needs from the training batch.
_TEACHER_BATCH_KEYS = [
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "scoring_attention_mask",
]


def _to_list(x: Any) -> list:
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


class SummitOpdTrainer(SelfHostedOpdTrainer):
    """SelfHostedOpdTrainer + remote Fireworks teacher (echo logprobs)."""

    def __init__(self, config):
        super().__init__(config)
        from summit.teacher.fireworks import FireworksEchoClient

        model = os.environ.get("SUMMIT_TEACHER_MODEL")
        if not model:
            raise RuntimeError("SUMMIT_TEACHER_MODEL env var must be set for the OPD teacher")
        base_url = os.environ.get(
            "SUMMIT_TEACHER_BASE_URL", "https://api.fireworks.ai/inference"
        )
        self._fireworks = FireworksEchoClient(model=model, base_url=base_url)
        logger.info(
            "[SummitOpdTrainer] remote teacher=%s via %s", model, base_url
        )

    # ---- teacher plumbing overrides ----

    def _initialize_teacher_workers(self) -> None:
        """No teacher workers to launch: the teacher is a remote API."""
        logger.info("[SummitOpdTrainer] remote teacher — skipping worker initialization")
        self._teacher_initialized = True

    def _get_teacher_log_probs(self, batch):
        """Teacher logprobs of student trajectories via Fireworks echo scoring.

        Returns torch.Tensor [batch_size, response_length], matching the stock
        implementation's contract.
        """
        import torch

        from summit.teacher.fireworks import FireworksEchoError, align_echo_logprobs

        teacher_batch = batch.trim_for_backend(_TEACHER_BATCH_KEYS).to_nextrainer_batch()
        # to_nextrainer_batch() nests tensors under "batch"
        inner = teacher_batch.get("batch", teacher_batch)

        input_ids = _to_list(inner["input_ids"])
        attention_mask = _to_list(inner["attention_mask"])
        responses = inner["responses"]
        scoring_mask = _to_list(inner["scoring_attention_mask"])
        resp_width = len(_to_list(responses[0])) if len(responses) else 0

        # Unpad each row and score the full (prompt + response) sequence.
        sequences: list[list[int]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            full = [int(t) for t, m in zip(ids_row, mask_row) if m]
            sequences.append(full)

        all_logprobs = self._fireworks.score_batch(sequences)

        out = torch.zeros((len(sequences), resp_width), dtype=torch.float32)
        for i, (mask_row, lps) in enumerate(zip(attention_mask, all_logprobs)):
            out[i] = torch.tensor(
                align_echo_logprobs(mask_row, lps, resp_width), dtype=torch.float32
            )

        # A misplaced score tensor is numerically valid but makes the teacher
        # signal exactly zero under NexRL's response mask. Refuse to optimize
        # such a batch instead of producing a plausible-looking negative loss.
        score_tensor = torch.as_tensor(scoring_mask, dtype=torch.bool)
        response_mask = score_tensor[:, -resp_width:]
        valid_scores = out[response_mask]
        if valid_scores.numel() and torch.all(valid_scores == 0):
            raise FireworksEchoError(
                "teacher scores are all zero under scoring_attention_mask; "
                "refusing an invalid distillation update"
            )
        if valid_scores.numel():
            logger.info(
                "[SummitOpdTrainer] aligned %d teacher tokens: mean=%.4f min=%.4f max=%.4f",
                valid_scores.numel(),
                valid_scores.mean().item(),
                valid_scores.min().item(),
                valid_scores.max().item(),
            )
        return out

    def train(self, trajectories: list) -> dict:
        """Record a durable success marker only after the full update returns.

        NexRL runs training in a background thread. In v1.4.0 an exception in
        that thread can still let the controller exit successfully once the
        dataloader is drained. The bootstrap checks these markers before it is
        allowed to export a checkpoint.
        """
        metrics = super().train(trajectories)
        success_file = os.environ.get("SUMMIT_TRAIN_SUCCESS_FILE")
        if success_file:
            with open(success_file, "a", encoding="utf-8") as marker:
                marker.write("ok\n")
                marker.flush()
                os.fsync(marker.fileno())
        return metrics
