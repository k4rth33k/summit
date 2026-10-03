# Check training before provisioning

These checks never provision a GPU. Static checks run automatically on launch;
runtime and Phantora checks are opt-in while the simulator is being qualified.

## Coverage by recipe

| Stage | DeepSWE | Decision / QLoRA |
| --- | --- | --- |
| Static check | Config, environment names, rendering, shell syntax | Data/schema, split overlap, cached targets, rendering |
| `--runtime-check` | Isolated local Docker imports/config | Imports in the current Python interpreter |
| `--simulate` | Experimental NexRL/Phantora synthetic update | Unsupported; inconclusive blocks launch |
| Real execution | Remote rollouts and training | Local or remote training, before/after evaluation, two-row reload check |

Static checks do not load the tokenizer or establish actual context fit. Decision
encoding rejects overlength inputs instead of silently truncating evidence.
Neither import checks nor Phantora replace a separately approved small GPU run.
`--no-env` prevents reading the environment file during a check. A local decision
checkpoint must use this mode because cloud preflight requires a Hub base.

## Commands

Run the fast local checks (no Docker required):

```bash
summit check -f examples/deepswe_opd.yaml --no-env
```

Build the dependency/import image once, from the repository root, then run
NexRL's own configuration validator and the Summit training/rollout imports:

```bash
docker build -f simulation/Dockerfile --target runtime-check -t summit-runtime:torch2.7.1 .
summit check -f examples/deepswe_opd.yaml --no-env --runtime-check
```

The VM and checker install the exact versions in
`summit/runtime-requirements.txt`. The NexRL source is pinned to the immutable
v1.4.0 commit. To update dependencies deliberately:

```bash
uv pip compile summit/runtime-requirements.in --python-version 3.12 --output-file summit/runtime-requirements.txt
```

Rebuild both images when that lock changes. Reports capture image ID, dependency
versions and the lock digest. A stale image is an inconclusive check.

## Experimental Phantora stage

For the released 4B recipe, obtain only metadata (not model weights) locally.
Install `huggingface_hub` if its `hf` CLI is unavailable:

```bash
uv pip install huggingface_hub
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --include '*.json' '*.model' 'merges.txt' '*.tiktoken' \
  --local-dir checkpoints/deepswe-metadata
```

This is an explicit network download, unlike the check itself. The legacy OPD
recipe downloads its model by Hub name rather than pinning the same revision;
compare the live model metadata and report any drift before relying on a
simulation. Decision recipes have an explicit `model.revision` field.

Build the source image (requires substantial disk space and CPU build time):

```bash
docker build -f simulation/Dockerfile -t summit-phantora:torch2.7.1 --build-arg MAX_JOBS=4 .
```

Supply `--model-dir` with **config and tokenizer metadata from the intended
model revision**. Existing Hugging Face snapshots work, including symlinks to
cache blobs. Only JSON/tokenizer vocabulary files are copied; weights, Python
files and credentials are not mounted. Model identity is supplied by the user;
file hashes make the chosen metadata auditable. The check does not download
metadata, weights, images, or dependencies.

```bash
summit check -f examples/deepswe_opd.yaml --no-env --simulate \
  --model-dir /path/to/model/snapshot --simulation-timeout 600 --json

# Require this stage to finish before requesting the VM:
summit launch -f examples/deepswe_opd.yaml --simulate --model-dir /path/to/model/snapshot
```

`--simulation-image` overrides the local image for either stage.
`--simulation-output` defaults to `summit-checks/`; each invocation writes a
fresh subdirectory. It contains input recipe, manifest, metadata hashes,
an immutable copy of the Summit Python code, `report.json`, `container.log`, and, when training starts, per-rank reports and
worker/server logs. Containers have no GPU devices, network or forwarded
secrets. Timeout/interrupt cleanup targets only the invocation's own container.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Requested checks completed; read coverage and limitations |
| 1 | Static, dependency/configuration, or virtual allocation failure |
| 2 | Inconclusive: missing image/metadata, unsupported runtime, timeout, or incomplete ranks |

A requested stage that fails **or is inconclusive** blocks launch. Static-only
success says nothing about simulated execution. A successful import check says
nothing about CUDA execution.

## Workload and adaptations

The driver uses the rendered recipe and runs NexRL's `ModelWorker` through FSDP
initialization and `update_actor_with_distillation`, including the actual loss,
backward, gradient clipping, AdamW state creation and scheduler. Each training
rank receives a conservative synthetic batch at the configured sequence and
response caps. Rows are bounded by the smaller of task count and batch size,
times rollout repetitions, times `(step_limit + 1)`, then rounded up to whole
per-rank mini batches. This stresses configured shapes; it does not predict
which trajectories a real agent will produce.

Phantora is pinned to `301d9d976a13067cc0f5438912395a11efba5f8e`, with its
PyTorch 2.7.1 branch pinned to `43d1bb713f140464e7d6ae7014b1de0f140dc94b`.
The build restores the matching tch 0.20 revision and routes the missing
`ncclBroadcast` entry point through Phantora's implemented `ncclBcast` event.
It also exports the legacy CUDA stream-write symbol that PyTorch 2.7.1 loads
when reporting OOM; actual calls return unsupported instead of silently
pretending to perform a stream write.
Qwen3.5's reference Gated DeltaNet additionally needs batched float32 triangular
solves. The adapter supplies a cuBLAS stub that validates dimensions and enums
and lets PyTorch allocate tensors and run autograd, matching upstream's GEMM
approach. Solve values, cuBLAS workspace and solve timing remain unvalidated.
No third-party source is vendored here; guarded edits are in `simulation/patch_upstream.py`.

Simulation-only adapters are guarded by `PHANTORA=1`:

- Build the actual model architecture on meta and let FSDP materialize virtual
  CUDA parameters module by module on every rank. This avoids real CPU weight
  allocations and an artificial whole-model CUDA peak during construction.
- Use eager attention, matching Summit's production FSDP bootstrap patch.
- Disable expandable allocator segments, which the virtual allocator cannot
  reproduce.
- Keep reverse-KL loss math but omit its value-dependent diagnostic metrics.
- Force the finite-gradient branch so fake gradient values cannot skip AdamW.

These differences are explicit exclusions. The simulator uses Summit's
flash-attention import stub; it does not validate the production flash-attention
wheel. Remove-padding, dynamic batch sizing, clipping/filtering and sequence
parallelism greater than one are rejected by the current driver. Tied embeddings
are materialized as separate input and output parameters in simulation. This
conservatively overstates memory and does not validate shared-parameter identity.
The initial architecture scope is dense Llama, Qwen3 and Qwen3.5; other model
types, including MoE routing, return inconclusive rather than executing an
unqualified value-dependent path.

The server runs with an empty replay database. Missing timings are zero-cost
upstream, so **no throughput or execution-time estimate is valid**. Upstream
also ignores some torch operations in its timing model. Even completion of all
ranks is experimental evidence about this synthetic path, not proof that a
real GPU run fits or that numerical results are correct.
The source simulator builds against CUDA 12.8 with cuDNN disabled; the stock
2.7.1 wheel uses CUDA 12.6. Kernel selection and workspace allocation can differ.
Reports record the CUDA build version, and workspace parity is an exclusion.

SGLang serving, agent/sandbox behavior, remote teacher/W&B/HF calls, pretrained
weights, weight synchronization and checkpoint export remain outside the
simulation. Runtime imports and config checks do not start those services.
Real-GPU comparison is required before enabling simulation by default.

## Verification

Earlier built-image qualification included a seven-rank Qwen3.5-9B synthetic
update at 512 tokens and a virtual OOM at 8192 tokens. Those were different
settings from the released four-update 4B recipe; re-run checks for your own
model metadata and configuration. Historical reports are not shipped.

The regular pytest suite exercises failure gates, malformed/stale results,
rank completeness, metadata isolation and timeout cleanup without Docker.
For real container integration tests after building the corresponding images:

```bash
SUMMIT_TEST_RUNTIME=1 pytest tests/test_simulation_container.py -q
SUMMIT_TEST_PHANTORA=1 pytest tests/test_simulation_container.py -q
```

The latter constructs a tiny local Llama architecture and tokenizer, runs a
two-rank FSDP update, and verifies rejection of a model that exceeds virtual
VRAM. It requires neither downloaded model weights nor a GPU.
