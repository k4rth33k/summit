# Summit

**Tinker-style training workflows, on infrastructure you control.**

Summit is an open-source toolkit for validating and running complex LLM training
jobs—from on-policy distillation (OPD) with agent rollouts to offline distillation
and QLoRA specialization. Define your model, training recipe and GPU requirements
in one YAML file; Summit handles validation, cloud provisioning, runtime setup,
training and Hugging Face export.

Use Phantora to exercise supported training workloads on virtual GPUs before
renting hardware, then launch through dstack on your own cloud account. Work
directly from the CLI or let a coding agent prepare and operate your recipes.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Linux is the supported
training environment. You do not need a local GPU for cloud orchestration.

### 1. Install

```bash
git clone https://github.com/k4rth33k/summit.git
cd summit
uv venv --python 3.12
uv pip install -e .
source .venv/bin/activate
```

### 2. Validate an example

Start with the included [DeepSWE OPD recipe](examples/deepswe_opd.yaml), which
distills a teacher into Qwen3.5-4B through sandboxed coding-agent rollouts.

```bash
summit check -f examples/deepswe_opd.yaml --no-env
summit launch -f examples/deepswe_opd.yaml --dry-run
```

These commands require no credentials and provision no GPUs. The dry run writes
the rendered configuration, bootstrap script and upload bundle to
`summit-dry-run/` for inspection.

### 3. Configure and launch

Set your Hugging Face destination and GPU region in the example YAML. Create
a local `.env` using [.env.example](.env.example) as a template; preserve any
existing environment file. For this recipe, supply:

- `RUNPOD_API_KEY` — GPU provisioning.
- `HF_TOKEN` — write access to your model repository.
- `FIREWORKS_API_KEY` — teacher scoring.
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` — coding-agent sandboxes.

Review the GPU duration, retry settings and external-service costs, then run:

```bash
# Small paid request to verify the teacher's token-scoring endpoint.
summit probe --config examples/deepswe_opd.yaml

# Launch the training workflow on your cloud account.
summit launch -f examples/deepswe_opd.yaml
```

One launch command validates the configuration, prepares dstack, provisions the
requested GPUs and starts the recipe. The remote job sets up its dependencies,
runs training and pushes the resulting checkpoint to the configured HF
repository. Launch returns a run name; use it to follow progress or stop the job:

```bash
summit ps
summit logs RUN_NAME
summit stop RUN_NAME
```

To require Phantora validation before provisioning, first follow the
[one-time simulator setup](docs/validation.md), then launch with:

```bash
summit launch -f examples/deepswe_opd.yaml --simulate \
  --model-dir checkpoints/deepswe-metadata --simulation-timeout 600
```

A failed or inconclusive simulation blocks provisioning.

## Features

- **Declarative training jobs.** Configure the model, objective, data, GPU
  resources and artifact destination in YAML. Inspect the exact runtime bundle
  before launching.
- **Validation before provisioning.** Automatic static checks, optional runtime
  checks and Phantora-based virtual-GPU diagnostics for the DeepSWE/NexRL path.
- **Self-hosted orchestration.** Provision and manage GPU jobs through dstack,
  using your own provider accounts. RunPod and Vast.ai integrations are available.
- **On-policy distillation.** Combine NexRL training, SGLang student serving,
  remote teacher scores and mini-swe-agent rollouts in Modal sandboxes.
- **Offline training without OPD services.** Train from reference labels or
  cached teacher targets without a live teacher, SGLang or rollout infrastructure.
- **Composable model adapters.** Use causal decision models, candidate-scoring
  heads or your own Python model factory. Keep architecture, objective and
  orchestration separate.
- **Full fine-tuning and QLoRA.** Choose full-parameter decision training or
  memory-efficient adapter specialization with the appropriate recipe.
- **Evaluation and artifacts.** Decision jobs record data hashes, evaluate before
  and after training, verify checkpoint reloads and optionally publish to HF.
  Matched CE/KD runs retain both arms' diagnostics and one selected checkpoint.

### Included recipes

| Recipe | Use case | Execution |
| --- | --- | --- |
| [DeepSWE OPD](examples/deepswe_opd.yaml) | Distill coding-agent behavior into Qwen3.5-4B | Cloud, two 96 GB GPUs |
| [Train causal 9B](examples/decision_9b.yaml) | Compare reference-label training with cached-target distillation | Cloud, one B300 288 GB |
| [Specialize causal 9B](examples/specialize_9b.yaml) | Adapt the retained parent to Typed Decisions with QLoRA | Local CUDA; qualified on RTX 5090 32 GB |

See the [recipe guide](docs/recipes.md) for data preparation, checkpoint handoff,
evaluation commands and recorded results. Decision recipes require prepared
datasets; specialization also requires the parent checkpoint.

## Agent-first by design

Summit's configuration files, validation reports and documentation give coding
agents a concrete workflow: prepare a recipe, check it, inspect the execution
plan, launch an approved job and verify its artifacts.

Open the repository in Claude Code or Codex. [AGENTS.md](AGENTS.md) provides
shared operating instructions, and [CLAUDE.md](CLAUDE.md) points to the same
guide. Start with a request such as:

> Prepare an OPD training run using the DeepSWE example. Read AGENTS.md, explain
> the GPU and credential requirements, and run the static check and dry run.
> Show me the configuration and expected costs before launching anything.

For a custom model:

> Adapt the decision recipe to my dataset and model factory. Validate the data
> and split isolation, check the adapter contract, and prepare a small
> qualification run. Ask before using GPUs or publishing checkpoints.

Agents can work through the same CLI and YAMLs as a human operator. Credentials
stay outside recipes, paid execution requires explicit approval, and completion
is verified from training and publication evidence—not just job submission.

The [documentation](docs/README.md) covers [data preparation](docs/data.md),
[custom architectures](docs/extending.md), [validation](docs/validation.md)
and [operations](docs/operations.md).

## Caveats

- **Tinker-style, not a drop-in SDK implementation.** Summit currently exposes
  its own YAML/CLI interface. Existing Tinker SDK programs cannot run unchanged.
- **Phantora coverage is recipe-specific.** It currently exercises an experimental
  DeepSWE/NexRL synthetic training path, not numerical quality or end-to-end
  serving, teacher and export behavior. Decision/QLoRA simulation is unsupported;
  requesting it blocks launch.
- **Hardware support has bounds.** Jobs are single-node; decision training uses
  one GPU. The included cloud recipes were qualified on RunPod. DeepSWE performs
  full fine-tuning, not LoRA. Its four-update example is a starting workflow,
  not a finished quality benchmark.
- **Budget settings are not a total-spend guarantee.** GPU limits do not include
  teacher, sandbox, storage or idle-instance charges. Local training has no
  automatic wall-clock cap. Confirm resource termination after each cloud run.
- **Checkpoints need deliberate retention.** Cloud decision runs require HF
  export, which can fail. QLoRA outputs contain adapters, not their parent
  weights; preserve the exact base. There is no general durable-resume guarantee.

Summit is distributed under the [Apache 2.0 license](LICENSE). Installation is
currently supported from a source checkout; datasets and trained checkpoints
are not bundled.
