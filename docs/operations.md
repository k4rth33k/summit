# Operations and artifact safety

## Commands and side effects

| Command | Side effects / scope |
| --- | --- |
| `check --no-env` | Static local validation; no credential loading, provisioning or teacher call |
| `check` | Also reads `.env` to check required names; no provisioning |
| `check --runtime-check` | DeepSWE Docker checks or decision imports in current interpreter |
| `check --simulate` | DeepSWE virtual training only; decision unsupported |
| `launch --dry-run` | Writes rendered/upload files locally; no provisioning |
| `probe` | Small **paid** Fireworks capability request; DeepSWE only |
| `init` | Configures/starts local dstack server and fleet configuration |
| `launch` | Configures dstack, submits a **paid** GPU job; returns run ID |
| `train` | Runs decision training on local CPU/CUDA; optional remote HF push |
| `ps`, `logs RUN` | Read cloud status/logs |
| `stop RUN` | Stops the selected job; verify resource termination afterward |

Use the global option before the subcommand for an alternate environment file:
`summit --env-file PATH check -f FILE`. Never put secret values in command-line
arguments. `--runtime-check` and `--simulate` are mutually exclusive gates.

## Credentials

The CLI merges `.env` with process environment; process values win. Keys are
forwarded only as needed, but the selected cloud/teacher/sandbox services receive
the relevant credentials. Protect the local dstack configuration and cloud logs.
Do not dump environments for debugging.

| Recipe | Required credentials for cloud launch |
| --- | --- |
| DeepSWE | Selected provider key, HF_TOKEN, FIREWORKS_API_KEY, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET |
| Decision | Selected provider key and HF_TOKEN for mandatory remote retention |
| Local specialization as shipped | No cloud credentials; local base and data already present |

`RUNPOD_API_KEY` selects RunPod credentials; `VAST_API_KEY` is for `vastai`.
Every configured backend currently needs credentials; avoid listing unused
providers. `WANDB_API_KEY` is optional when DeepSWE logging excludes `wandb`.

Local `summit train` does **not** load `.env` automatically. If enabling local
HF publication, configure SDK credentials securely in the process environment
or your local HF credential store. Do not paste a token into the YAML.

## Budgets and shutdown

`max_price` limits the hourly GPU offer, not accumulated spending. `max_duration`
limits the cloud task, not local `train`, and is not an all-services dollar cap.
Budget for GPU setup, the complete job, export, provider idle time, storage,
teacher inference and Modal separately. Matched CE/KD runs train twice.

The supplied decision recipe disables automatic workload/interruption retries.
DeepSWE permits capacity waiting, but no automatic retry after a failed workload.
Changing these policies can multiply costs. Teacher acquisition has separate
per-output-directory request/dollar ledgers; aggregate limits remain the operator's
responsibility and depend on correct current rates.

After launch, record the run ID, watch progress and check provider billing.
The managed fleet uses a 15-minute idle timeout. Completion/stop is not proof
that every associated resource was deleted instantly. Confirm dstack status,
provider instances and any external sandbox/storage resources. Do not stop
unrelated runs or fleets.

## Checkpoint retention and portability

Decision training performs before/after validation, saves model/adapter and
tokenizer, then reloads and compares the first two validation predictions. This
is a serialization check, not full-model equivalence on every input. Training
writes `complete.json` only after the requested publication succeeds. Failures
produce `failure.json` and must not be reported as complete.

Cloud decision preflight requires `hf_push`. Publication checks visibility,
rejects nonempty repositories unless `replace: true`, and uses a parent-commit
guard. Large artifacts stage small diagnostics before uploading weights. Failed
upload may preserve diagnostics and the previous HF checkpoint, **not the new
weights**. Ephemeral VM teardown can lose those weights. Leave sufficient time
and HF storage quota for export; there is no durable fallback/resume guarantee.

Use a fresh authorized HF destination or explicitly authorize replacement after
archiving its immutable prior revision. Verify the new remote files, successful
marker and hashes. Download full artifacts under ignored `checkpoints/` or
`artifacts/` in this workspace. Do not create repositories merely to work around
retention problems without permission.

A causal Summit checkpoint stores its Transformers model under `model/` and
tokenizer under `tokenizer/`. Use the Summit artifact loader for inference, or
pass those subdirectories explicitly to Transformers. A root Hub ID is not
automatically flattened into a standard model repository.

A QLoRA artifact contains adapter weights, **not the base**. The current saved
`adapter.json` records the resolved base location; a local-base artifact can
therefore contain an absolute path. Moving/publishing it does not make that path
portable. Preserve the exact parent independently. For evaluation on another
machine, provide its new location with `--checkpoint` and the adapter with
`--peft-adapter`. For the generic artifact loader, relocate a copy and update
its base reference deliberately, retaining the original manifest and verified
parent hash. Do not distribute the artifact as self-contained. A fully portable
merged/public model release requires a separate, tested export step and rights
review, not just `hf_push` on this local recipe.

## Common failures

- Missing dataset/base: follow docs/data.md and the checkpoint handoff. No private
  assets are included; do not substitute synthetic rows and claim reproduction.
- Overlength decision: audit prompts with the pinned tokenizer; exclude/report
  whole rows or increase the reviewed limit, not silent evidence truncation.
- Missing Phantora image/metadata: build/download prerequisites explicitly. The
  check itself does not pull images or access the network.
- CUDA OOM: stop the job, inspect precision, sequence length, accumulation and
  adapter. NF4 adapter training and full FP32 AdamW have very different memory.
- Existing output: use a new run directory; do not delete historical evidence.
- HF visibility/quota/permission failure: fix destination and storage before
  rerunning expensive training. A populated `.env` alone proves neither access
  nor available quota.
- Provider capacity unavailable: check selected region/SKU; never silently
  increase the hourly ceiling or switch provider without authorization.
