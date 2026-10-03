"""Driver entrypoint for the training VM.

Runs instead of `python -m nexrl.main` so summit can normalize the
environment first. Usage (hydra args pass through):

    TRAIN_CONFIG=$EXPERIMENT_PATH/rl_train.yaml \
        python -m summit.nexrl_ext.boot --config-path $EXPERIMENT_PATH --config-name rl_train
"""

from __future__ import annotations

import os


def main() -> None:
    # NexRL's tracker reads WANDB_KEY/WANDB_HOST; accept WANDB_API_KEY too.
    if not os.environ.get("WANDB_KEY") and os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ.setdefault("WANDB_HOST", "https://api.wandb.ai")
    os.environ.setdefault("API_SERVER_URL", "127.0.0.1")

    from nexrl.main import main as nexrl_main

    nexrl_main()


if __name__ == "__main__":
    main()
