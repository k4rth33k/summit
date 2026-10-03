# The three release recipes

Run commands from the repository root after installation. Paths inside decision
YAMLs are relative to the YAML, not the shell's working directory. Personal
variants belong in ignored `runs/`; adjust relative paths if copying there.
None of these recipes grants permission to spend money or publish artifacts.

## 1. DeepSWE OPD

Entry point: [examples/deepswe_opd.yaml](../examples/deepswe_opd.yaml).

This is the successful small learning qualification: Qwen3.5-4B, four updates,
12 unique coding tasks with four anchor tasks repeated, Adafactor, and two
96 GB GPUs (one trainer, one rollout server). It uses Fireworks token scoring,
NexRL's self-hosted FSDP trainer, SGLang, mini-swe-agent and Modal. It is full
fine-tuning; a legacy `lora_rank` field would not enable LoRA on this path.

1. Set the HF destination and provisionable GPU region in the YAML.
2. Supply credentials via `.env` as described in [operations](operations.md).
3. Perform static validation and inspect the dry-run bundle.
4. With an approved teacher allowance, run the endpoint capability probe.
5. With an approved cloud/sandbox budget, launch and monitor.

```bash
summit check -f examples/deepswe_opd.yaml --no-env
summit launch -f examples/deepswe_opd.yaml --dry-run
summit probe --config examples/deepswe_opd.yaml
summit launch -f examples/deepswe_opd.yaml
```

For GPU-free validation before provisioning, add `--simulate --model-dir PATH`
after completing [Phantora setup](validation.md). Teacher availability and
tokenizer compatibility are external prerequisites, not guaranteed by a YAML.
Do not infer quality improvement from loss movement on four updates alone.

## 2. Train the causal 9B parent

Entry point: [examples/decision_9b.yaml](../examples/decision_9b.yaml).

This preserves the successful parent training settings: pinned Qwen3.5-9B,
full FP32 trainable weights/AdamW state, BF16 compute, gradient checkpointing,
batch 1, accumulation 8, LR `2e-7`, 128 updates per arm. It runs **two arms**
from the untouched base: reference-label CE and mixed CE/KD. Both use identical
data and seed. The retained checkpoint is selected by development accuracy,
then NLL, then CE on an exact tie. Both arms' diagnostics are retained. There
are 256 total optimizer updates, not 128 total.

The original run used 512 CLINC150/ContractNLI decisions with one shuffled-order
view each (1,024 training views), plus 256 development decisions. Qwen Max
native reasoning answers were encoded as hard one-hot targets, not logits.
The selected KD arm reached 226/256 development decisions in both option orders.
That development set selected the checkpoint; it is not an untouched test result.

Follow [data preparation](data.md) to provide:

```text
data/decision-9b/train.jsonl       # canonical rows with cached targets
data/decision-9b/validation.jsonl  # independent reference labels
```

Set your HF destination, then:

```bash
summit check -f examples/decision_9b.yaml --no-env --json
summit launch -f examples/decision_9b.yaml --dry-run
# Paid; only after reviewing the prepared data, bundle and budget.
summit launch -f examples/decision_9b.yaml
```

The example requests one B300 288 GB; full training does not fit into the same
memory budget as QLoRA. No live teacher, SGLang, NexRL or Modal is used during
this job. `--runtime-check` requires local ML dependencies; `--simulate` is
unsupported and blocks launch. The recipe requires data preparation beforehand.

For local execution, install `.[decision]`, remove `hf_push` unless publication
is intended, and run `summit train -f examples/decision_9b.yaml`. This blocks
until training finishes but has no local wall-clock budget guard. It requires
adequate local GPU RAM. Do not start it as a quick installation test.

### Handoff to specialization

Keep the complete selected artifact, including `adapter.json`, `manifest.json`,
`model/`, `tokenizer/` and diagnostics. A cloud artifact is a Summit folder, not
a flat Transformers repository; do not feed its Hub ID directly to a loader
expecting `config.json` at repository root.

Download an immutable successful HF revision into the workspace:

```bash
hf download YOUR_HF_USERNAME/summit-decision-9b \
  --revision IMMUTABLE_COMMIT --local-dir checkpoints/decision-9b
```

Verify `complete.json` reports completion and reload verification, and retain
the revision/hash record. For a local parent, set specialization `model.name`
to `../outputs/decision-9b` instead of copying the large weights unnecessarily.
The published historical checkpoint is not bundled or publicly distributed here.

## 3. Specialize the retained 9B with QLoRA

Entry point: [examples/specialize_9b.yaml](../examples/specialize_9b.yaml).

Install local CUDA-capable dependencies. For the qualified CUDA 12.8 stack:

```bash
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[qlora,benchmark]'
```

The NVIDIA driver must support the selected wheel/GPU. This is not a CPU
training recipe. Prepare only the public **train** split as described in
[data.md](data.md#typed-decisions-specialization). Confirm that
`checkpoints/decision-9b/` contains the complete parent artifact.

```bash
summit check -f examples/specialize_9b.yaml --no-env --runtime-check
# Local GPU work; no provisioning and no HF push in the supplied YAML.
summit train -f examples/specialize_9b.yaml
```

The recipe freezes an NF4 base and trains rank-32 LoRA with alpha 64, dropout
0.05, BF16 compute and FP32 trainable adapter weights. It uses forward KL to
benchmark-provided distributions (`alpha: 1`), LR `1e-4`, one epoch, batch 1,
accumulation 8 and maximum length 768. Our qualification completed 675 updates
on a local RTX 5090 32 GB, with approximately 12.64 GiB peak allocated GPU memory.
Observed allocation is not a guarantee of total VRAM use on another system.

The YAML's cloud resource fields describe the intended hardware, but its local
base path makes this a **local training** entry point. `summit launch` does not
upload that base checkpoint. To move to cloud, first prepare a loadable, pinned
Hub base and explicitly configure HF retention; that variant is not the retained
local qualification. An adapter remains dependent on its exact parent weights.

The output is `outputs/specialize-9b/`, including `peft_adapter/`, tokenizer,
metadata, metrics, predictions and `complete.json`. It is not a standalone full
9B model. Do not publish only the adapter and imply the private parent is included.
See [artifact portability](operations.md#checkpoint-retention-and-portability).

## Evaluation and evidence

Keep the public test file out of training and model selection. After freezing
the checkpoint, evaluate the parent and specialist separately:

```bash
python scripts/evaluate_typed_decisions.py \
  --checkpoint checkpoints/decision-9b \
  --dataset data/typed-decisions-raw/all/test-00000-of-00001.parquet \
  --output outputs/typed-decisions-parent-test --device cuda --max-length 768

python scripts/evaluate_typed_decisions.py \
  --checkpoint checkpoints/decision-9b \
  --peft-adapter outputs/specialize-9b/peft_adapter \
  --dataset data/typed-decisions-raw/all/test-00000-of-00001.parquet \
  --output outputs/typed-decisions-specialist-test --device cuda --max-length 768
```

These are real inference jobs and require approval/resources. The evaluator
loads the full parent in BF16, not the training-time NF4 base; allow GPU memory
for weights and activations. It scores answer-letter logits rather than
generating a multi-question JSON response. The current evaluator expects a
single parent `model/model.safetensors` file for its weight-hash record.

Recorded qualification at dataset revision
`561333a8576d22875380b14d25a13065b046538c`:

| Checkpoint | Training relevant to this test | Accuracy |
| --- | --- | --- |
| Retained causal KD parent | CLINC150/ContractNLI, not Typed Decisions | 1,222/2,000 (61.1%) |
| Parent + QLoRA specialist | 5,400 Typed Decisions training decisions | 1,516/2,000 (75.8%) |

The 600 validation decisions came from separate cases in the public train split.
The test contains 400 cases and 2,000 decisions; questions from one case are
correlated. Do not mistake a decision-level confidence interval for a
case-clustered statistical test. No paired Jev evaluation was run through this
harness. Our benchmark-specific specialist should not be marketed as a general
zero-shot superiority or Qwen Max parity result. The original run artifacts are
local research evidence, not included in this source release.

For application inference, `python -m summit.recipes.decision.predict --help`
describes the canonical input format. Prediction selects among provided options;
applications may map those IDs to validated JSON structures deterministically.
