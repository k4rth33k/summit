"""Candidate CE and forward-KL distillation, independent of model architecture."""

import torch
import torch.nn.functional as F


def decision_loss(logits, records, config):
    if len(logits) != len(records):
        raise ValueError("adapter must return one logit vector per record")
    losses = []
    for values, row in zip(logits, records):
        if values.ndim != 1 or len(values) != len(row.candidate_options) or not torch.isfinite(values).all():
            raise ValueError("adapter logits must be a finite vector over exactly the valid candidates")
        ids = [option.id for option in row.candidate_options]
        target = torch.tensor([ids.index(row.target.correct_option_id)], device=values.device)
        ce = F.cross_entropy(values.float().unsqueeze(0), target)
        if config.type == "candidate_ce":
            losses.append(ce)
            continue
        if row.teacher is None:
            raise ValueError("distillation requires a teacher target")
        p = torch.tensor([row.teacher.probabilities[i] for i in ids], dtype=torch.float32, device=values.device)
        # Stored probabilities are at temperature 1. Zeros remain zero after tempering.
        softened = (p.log() / config.temperature).softmax(-1)
        kl = F.kl_div(F.log_softmax(values.float() / config.temperature, dim=-1), softened, reduction="sum")
        losses.append((1 - config.alpha) * ce + config.alpha * config.temperature ** 2 * kl)
    return torch.stack(losses).mean()
