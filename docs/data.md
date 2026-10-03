# Data and targets

Decision training consumes canonical JSONL, one question per line. DeepSWE uses
its own task preparation path and does not consume this schema. Training does
not automatically call a teacher or manufacture missing targets.

## What are x and y?

`x` consists only of `state`, `question` and ordered `candidate_options`.
`y` is the independently recorded correct option; distillation additionally
uses a cached distribution over the same option IDs. IDs and provenance support
auditing and splitting, not hidden answer hints in the prompt.

```json
{
  "id": "refund-001",
  "group_id": "refund-case-001",
  "state": {"policy": "Refund verified orders up to 30 days old.", "age_days": 12, "verified": true},
  "question": "What should the service do?",
  "candidate_options": [
    {"id": "opt-a", "description": "Approve the refund"},
    {"id": "opt-b", "description": "Reject the refund"},
    {"id": "opt-c", "description": "Request missing information"}
  ],
  "target": {"correct_option_id": "opt-a"},
  "provenance": {"source": "my-reviewed-policy-data", "version": "1"}
}
```

The example is pretty-printed for readability; serialize each record on a single
line in a JSONL file. Candidate IDs/descriptions must be unique, the correct
option must exist, and each row must contain at least two options. The recipe's
`max_candidates` may be stricter than the schema's maximum of 26.

For distillation, attach `teacher` using the Python schema rather than manually
inventing a digest:

```python
from summit.recipes.decision.data import DecisionRecord, TeacherTarget

row = DecisionRecord.model_validate(record_dict)
teacher = TeacherTarget(
    input_sha256=row.input_hash(),
    model="my-teacher-at-an-immutable-revision",
    method="document-the-actual-acquisition-method",
    probabilities={"opt-a": 0.85, "opt-b": 0.10, "opt-c": 0.05},
)
value = row.model_dump(mode="json")
value["teacher"] = teacher.model_dump(mode="json")
validated = DecisionRecord.model_validate(value)
```

This shows the representation, not a measured teacher response. Probabilities
must cover exactly the candidate IDs, be finite/nonnegative and sum to one.
The hash binds state, question and option order. Changing any input invalidates
the cache. Keep independent labels even when the teacher disagrees.

## Validation and leakage

```bash
python -m summit.recipes.decision.dataset validate \
  --train data/decision-9b/train.jsonl \
  --validation data/decision-9b/validation.jsonl
```

IDs, groups and identical states cannot cross splits. Put all questions or
permutations from one case in one group. These checks do not detect semantic
paraphrases or pretraining contamination. Review rights, duplicates and label
quality separately. Encoding rejects overlength rows; do not silently truncate
policy evidence. Test rows must never become teacher-acquisition or training rows.

`candidate_ce` needs only independent reference labels. `candidate_distillation`
requires cached targets on every training row and minimizes
`(1-alpha) * CE + alpha * T² * KL(teacher_T || student_T)`. At `alpha=1`, it
uses only the distribution loss. At `alpha=0.5`, it mixes both. A one-hot native
answer is **hard-label distillation**, not a recovered teacher confidence.

## Rebuild the causal-parent data pipeline

Historical caches are not shipped. You can use your own canonical data, or
rebuild the CLINC150/ContractNLI pipeline below. A fresh remote teacher run can
produce different labels; this reproduces the method, not exact checkpoint bytes.
Keep source attribution and license files; review redistribution rights before
publishing datasets or weights. The fetcher preserves pinned source notices.

```bash
python -m summit.recipes.decision.fetch --output data/decision-raw
```

Create `runs/sources.yaml` with the following (paths assume that location):

```yaml
seed: 42
calibration_percent: 5
candidate_count: 8
overlap_policy: quarantine
sources:
  - source: clinc150
    path: ../data/decision-raw/clinc150/data_full.json
    revision: 828f8093932c8fe6ca7936c3d2e52903b1c523de
    max_train_rows: 3500
  - source: contractnli
    path: ../data/decision-raw/contractnli/contract-nli.zip
    member: contract-nli/train.json
    split: train
    revision: eced6528dd3c1d14d73f9a87df8f7bdbc03126f9
    max_train_rows: 1500
  - source: contractnli
    path: ../data/decision-raw/contractnli/contract-nli.zip
    member: contract-nli/dev.json
    split: dev
    revision: eced6528dd3c1d14d73f9a87df8f7bdbc03126f9
  - source: contractnli
    path: ../data/decision-raw/contractnli/contract-nli.zip
    member: contract-nli/test.json
    split: test
    revision: eced6528dd3c1d14d73f9a87df8f7bdbc03126f9
```

Run conversion, pinned-tokenizer length auditing and grouped sampling:

```bash
python -m summit.recipes.decision.sources \
  -f runs/sources.yaml --output data/decision-converted
python -m summit.recipes.decision.audit \
  --data-dir data/decision-converted --output data/decision-length-filtered \
  --tokenizer Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --max-length 2048
python -m summit.recipes.decision.pilot select \
  --data-dir data/decision-length-filtered --output data/decision-pilot
```

Token auditing needs `.[decision]` and may download tokenizer metadata. CLINC
shortlists use train-only retrieval; the correct answer is not artificially
inserted. Report this as converted candidate routing, not original 150-way CLINC
accuracy. The sampler selects 512 training / 256 development decisions without
reading final test or calibration inputs. Conversion/auditing preserve official
splits, including processing held-out inputs for overlap/length checks.

### Acquire native-answer hard targets

Create `runs/teacher.yaml`. This is a **template**, deliberately requiring you
to replace budget and current-rate placeholders before use:

```yaml
model: accounts/fireworks/models/qwen3p8-max
tokenizer: Qwen/Qwen3.8-2.4T-A95B
tokenizer_revision: 207bd685a7e3696cfaff12ded7c6a7ea0f88c996
prompt_mode: reasoning_cached
max_input_tokens: 8192
max_records: 512
max_requests: 512
max_usd: APPROVED_CAP
input_usd_per_million: VERIFIED_INPUT_RATE
output_usd_per_million: VERIFIED_OUTPUT_RATE
```

Verify endpoint identity, tokenizer compatibility, availability and current prices
before paid execution. The first command plans only; it may download tokenizer
metadata but sends no inference request:

```bash
python -m summit.recipes.decision.reasoning_reference \
  -f runs/teacher.yaml --data data/decision-pilot/train.jsonl \
  --output outputs/teacher-native --max-output-tokens 2048
# Repeat with --execute only after explicit authorization.
```

Preserve incomplete/truncated responses. An explicitly approved repair uses a
new output directory, `--retry-incomplete-from` and a reviewed output-token cap.
Track cumulative spending across directories; each local ledger is not a
provider-enforced wallet. Do not replace missing responses with reference labels.

After every answer is complete:

```bash
python -m summit.recipes.decision.native_targets \
  -f runs/teacher.yaml --data data/decision-pilot/train.jsonl \
  --reference outputs/teacher-native --output outputs/teacher-hard
python -m summit.recipes.decision.dataset attach-teacher \
  --data data/decision-pilot/train.jsonl --cache outputs/teacher-hard/cache.jsonl \
  --output data/decision-pilot/train-distilled.jsonl
mkdir -p data/decision-9b
python -m summit.recipes.decision.pilot augment-order \
  --data data/decision-pilot/train-distilled.jsonl --output data/decision-9b/train.jsonl
cp data/decision-pilot/validation.jsonl data/decision-9b/validation.jsonl
```

Pass repeated `--repair REPAIR_DIR` to `native_targets` when repairs were needed.
The permutation step produces 1,024 views from 512 decisions and transports
targets by semantic option ID; it is not another 512 teacher calls. All creation
commands should target fresh paths. Never overwrite a frozen dataset.

## Typed Decisions specialization

Install `.[benchmark]`. Download the pinned public **train file only** first:

```bash
hf download LocalLLaMA/typed-decisions --repo-type dataset \
  --revision 561333a8576d22875380b14d25a13065b046538c \
  --include 'all/train-00000-of-00001.parquet' \
  --local-dir data/typed-decisions-raw
python scripts/prepare_typed_decisions_training.py \
  --dataset data/typed-decisions-raw/all/train-00000-of-00001.parquet \
  --output data/typed-decisions
```

Expected source SHA-256:
`46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5`.
With default seed `20261001`, the split is 1,080 cases / 5,400 decisions for
training and 120 cases / 600 decisions for validation (30 cases per workflow).
Every case's five questions stay together. Targets use the supplied gold
distributions, averaged from three benchmark teacher samples. No paid Qwen Max
request is made. The script verifies the pinned train-file hash before creating
outputs, checks split isolation and records source hashes and split metadata.

After freezing the specialist, download the test file using the same command
with `--include 'all/test-00000-of-00001.parquet'`, then use the evaluator in
[recipes](recipes.md#evaluation-and-evidence). Never pass that test file to the
training-data preparation script. Dataset metadata/license review is required
before redistributing derived data; this repository includes neither parquet
files nor third-party text.
