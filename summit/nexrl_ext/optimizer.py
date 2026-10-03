"""Optimizer adapters for NexRL's self-hosted FSDP worker."""

from __future__ import annotations


def adafactor(params, *, lr, betas=None, weight_decay=0.0):
    """Match NexRL's AdamW call shape while constructing Adafactor.

    Adafactor factors the second-moment state for matrix parameters, which
    substantially reduces full-fine-tuning memory. ``betas`` is accepted only
    to keep the guarded upstream call-site replacement narrow.
    """
    del betas
    from transformers.optimization import Adafactor

    return Adafactor(
        params,
        lr=lr,
        weight_decay=weight_decay,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
