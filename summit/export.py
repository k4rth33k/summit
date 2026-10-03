"""Finalize step (runs on the VM): push the HF-format checkpoint directory
(produced by NexRL's /convert_checkpoint_to_huggingface) to the Hub.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true", default=False)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    url = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=args.hf_dir,
    )
    print(f"pushed {args.hf_dir} -> {url}")


if __name__ == "__main__":
    main()
