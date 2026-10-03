# Working with Summit

Read this file before operating this repository. Start with README.md, then the
relevant guide in docs/. These instructions apply to all files in this checkout.

## Project map

- `summit/cli.py`: check, launch, train, probe, logs, ps and stop.
- `summit/config.py`, `preflight.py`, `render.py`: DeepSWE schema/lifecycle.
- `summit/dstack_ops.py`: provider setup, upload bundle and job submission.
- `summit/recipes/decision/`: schema, data, adapters, objectives, trainer,
  prediction and publication. This path must remain independent of OPD.
- `summit/nexrl_ext/`, `rollout/`, `templates/`: DeepSWE integration.
- `summit/simulation/`, `simulation/`: optional runtime/Phantora checks.
- `examples/`: three release jobs. Create personal variants in ignored `runs/`.
- `tests/`: maintained regression suite; tests/fixtures are offline test inputs,
  not additional supported recipes.
- `notes/`, `data/`, `artifacts/`, `outputs/`, `checkpoints/`: ignored local
  material. Do not publish, move or delete these without authorization.

## Default workflow

1. Clarify recipe, dataset, base model, provider, resources, output repository
   and budget. Do not assume that a model fits a GPU based on parameter count.
2. Read the YAML and schema. Decision paths resolve relative to the YAML; adjust
   them when copying a configuration to a different directory.
3. Check provenance, dataset rights, split isolation, candidate coverage, teacher
   hashes and sequence limits. Do not select runs using held-out test results.
4. Run `summit check -f FILE --no-env --json`. For a cloud-compatible model,
   also run `summit launch -f FILE --dry-run`. Neither provisions compute.
5. Report warnings, required credential NAMES, resources, limits, unsupported
   checks and publication scope. Static success does not prove GPU fit.
6. Obtain explicit authorization before teacher requests, provisioning, local
   GPU training, remote publication or destructive replacement/deletion. Confirm
   total budget, teacher/sandbox/storage extras and retry policy.
7. Execute the authorized command. Record the run ID and monitor it. Successful
   submission is not successful training. Do not retry paid work blindly.
8. Verify completion, finite metrics, save/reload evidence and immutable HF
   revision when applicable. Confirm termination and report failures honestly.

## Credentials and data safety

- Never read/print `.env`, environment dumps, credential stores or tokens.
  Let Summit/SDKs load credentials. Use `.env.example` for key names.
- Keep secrets out of Markdown, YAML, command arguments, reports and commits.
- Review the inventory before staging. `.gitignore` does not untrack files.
- Decision `hf_push.replace: false` is the safe default. Replacement requires
  approval and a verified local copy of the previous immutable revision.
- Keep downloads inside ignored workspace directories. Create no additional
  HF repositories beyond authorized destinations.
- Never overwrite output directories or edit historical results to improve
  reported performance. Preserve failed-run evidence.

## Recipe selection and claims

- DeepSWE: online token-scoring OPD, NexRL + SGLang + Modal; Phantora opt-in.
- Causal 9B: cached candidate targets, full training, sequential CE/KD arms.
- Specialized 9B: QLoRA over the retained causal parent; local CUDA qualified.
- The candidate scorer/custom factories are extension points, not the retained
  checkpoint behind the 75.8% Typed Decisions result.
- Native-answer hard labels are not soft logits. Benchmark gold distributions
  are not new Qwen Max supervision. Never conflate these.
- Missing data/checkpoints are prerequisites. Synthetic fixtures may verify
  plumbing, but must not be presented as benchmark reproduction.
- Requested Phantora failure/inconclusive must block launch, never be bypassed.

## Development

Use Python 3.12 and an editable source install. Run `python -m pytest -q` for
the suite; install the `decision` extra for ML tests. QLoRA needs `qlora`, CUDA
and a separate approved hardware qualification. See docs/extending.md.
Keep tests offline by default. Add focused regression coverage for changed
schemas, renderers and lifecycle gates. Do not launch paid work implicitly.

Handoff must summarize changed files, verification results, unverified hardware
paths, remaining prerequisites, and resources left running. Follow
docs/releasing.md before publishing.
