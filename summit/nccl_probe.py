"""Fast on-node NCCL gate used before loading the training model."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(value)
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"NCCL all-reduce returned {value.item()}, expected {expected}")
        dist.broadcast(value, src=0)
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize(device)
        if rank == 0:
            print(f"summit: NCCL probe passed on {world_size} training GPUs")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
