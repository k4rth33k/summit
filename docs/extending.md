# Architecture and development

Summit separates job description, validation/rendering, orchestration, and
training. The legacy DeepSWE schema dispatches to NexRL OPD; `schema_version: 2`
with `recipe: decision` dispatches to the offline single-device loop. Adding a
decision model should not introduce a requirement for NexRL, SGLang or a teacher
endpoint. Those are selected implementations of the DeepSWE workflow.

## Decision components

- **Data:** `DecisionRecord` in `summit/recipes/decision/data.py` validates input,
  candidates, independent labels and optional hash-bound teacher targets.
- **Model:** `causal_lm`, `candidate_scorer`, `qlora_causal`, or `module:factory`.
- **Objective:** candidate CE or mixed reference CE / forward-KL distillation.
- **Trainer:** `torch_single` owns optimization, evaluation, save/reload and
  publication. It currently supports one training GPU, not distributed training.
- **Lifecycle:** preflight and rendering create an execution plan with required
  services, environment names and data hashes; dstack submits the bundle.

The causal adapter uses the existing vocabulary head restricted to valid
answer-letter columns. It avoids allocating full sequence-by-vocabulary logits.
The candidate scorer attaches a scalar head to a backbone and scores each
state/question/candidate input. It is a non-standard architecture extension,
not the model behind the retained Typed Decisions specialist.

## Custom adapter contract

Set `model.adapter: my_models.decision:build_model` and bundle local Python via
`code: [../my_models]`, with paths relative to the YAML. Custom sources are
trusted executable code; review them before loading an artifact. Only Python
files are bundled automatically, not arbitrary assets or dependencies.

The factory accepts `(ModelConfig, checkpoint: Path | None)` and returns a
PyTorch `nn.Module` implementing:

```python
def build_model(config, checkpoint=None):
    # Initialize from config.name/config.revision, or reload from checkpoint.
    return MyDecisionModel(config, checkpoint)

class MyDecisionModel(torch.nn.Module):
    def score(self, records, tokenizer, max_length):
        # Return one differentiable, finite Tensor[K] per input, preserving
        # exactly the record's candidate order. K can differ between inputs.
        ...

    def save(self, path):
        # Save every architecture-specific weight needed by the reload factory.
        ...
```

Honor the actual prompt token limit and never silently truncate evidence. Keep
reference labels and teacher metadata out of model inputs. Full-training
parameters must be FP32 for the current AdamW implementation; use BF16 autocast
for compute. Quantized adapters manage placement explicitly; merely changing
`dtype` or `adaptation` does not create a quantization implementation. The current
schema permits QLoRA adaptation only with the builtin `qlora_causal` adapter.

The trainer saves tokenizer, adapter configuration and bundled custom code,
then calls the factory again and verifies two predictions. Your factory must
actually restore the saved weights. Prediction depends on the same contract.
The existing two builtin architecture tests are useful examples for implementing
new adapters with tiny local random models.

## Dependencies

The base editable install supports schemas, rendering and orchestration without
local PyTorch. `.[decision]` adds torch/transformers and core ML dependencies;
`.[qlora]` adds PEFT/bitsandbytes; `.[benchmark]` adds the parquet reader and ML
stack for preparation/evaluation scripts. `.[dev]` adds pytest. CUDA runtime
wheels and drivers must match the qualified environment.

The cloud decision bootstrap has its own pinned requirements. DeepSWE maintains
a separate runtime lock and SGLang overrides. Changing the local `uv.lock` alone
does not change those environments. Rebuild checker images after runtime-lock
changes. Adding an adapter does not automatically add Phantora support.

## Tests

```bash
# Lightweight core tests; ML tests skip if their dependencies are absent.
uv pip install -e '.[dev]'
python -m pytest -q

# ML tests use tiny local random weights, not downloaded 9B weights.
uv pip install -e '.[dev,decision]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

Test configuration validation, rendering, mandatory gate failures, no-secret
bundles, split isolation, target hashing, objective math, forward/backward,
save/reload, and publication failure behavior. HF publication tests use fakes.
Container tests are explicitly gated as described in [validation](validation.md).
QLoRA CUDA qualification and any cloud/teacher probes require separate approval;
a CPU test cannot establish their numerical or memory behavior.

Keep maintained regression tests in the release. Do not confuse test fixtures
with experimental job configs. Archive obsolete one-off experiment/administrative
scripts under ignored `notes/` rather than making them public entry points.
