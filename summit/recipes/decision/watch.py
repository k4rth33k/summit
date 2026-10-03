"""Monitor one explicitly named qualification run with a conservative stop guard."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from summit.dstack_ops import get_client
from .data import canonical_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--max-seconds", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.max_usd <= 0 or args.max_seconds <= 0:
        parser.error("positive stop limits required")
    args.output.mkdir(parents=True, exist_ok=False)
    client = get_client()
    stopped = False
    while True:
        run = client.runs.get(args.run)
        if run is None:
            raise ValueError(f"run not found: {args.run}")
        submitted = run._run.submitted_at.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - submitted).total_seconds()
        provisioning = run._run.jobs[0].job_submissions[-1].job_provisioning_data
        rate = provisioning.price if provisioning else 0
        conservative = max(run._run.cost, elapsed * rate / 3600)
        report = {"run": args.run, "status": run.status.value, "elapsed_seconds": elapsed,
                  "reported_run_cost_usd": run._run.cost, "conservative_elapsed_cost_usd": conservative,
                  "hourly_rate": rate, "instance_id": provisioning.instance_id if provisioning else None,
                  "budget_stop_requested": stopped}
        with (args.output / "status.jsonl").open("a") as stream:
            stream.write(canonical_json(report) + "\n")
        print(canonical_json(report), flush=True)
        logs = b"".join(run.logs()).decode(errors="replace")
        (args.output / "run.log").write_text(logs)
        if run.status.is_finished():
            (args.output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
            return
        if not stopped and (conservative >= args.max_usd or elapsed >= args.max_seconds):
            run.stop(abort=True)
            stopped = True
        time.sleep(30)


if __name__ == "__main__":
    main()
