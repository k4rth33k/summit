"""Recipe boundaries: orchestration does not choose a training algorithm."""


def is_decision(cfg) -> bool:
    return getattr(cfg, "recipe", "deepswe_opd") == "decision"
