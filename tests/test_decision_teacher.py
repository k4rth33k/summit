import json

import httpx
import pytest

from summit.recipes.decision.dataset import generate_records
from summit.recipes.decision.teacher import TeacherConfig, collect, parse_echo


class TinyTokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def config(**kwargs):
    return TeacherConfig(tokenizer_revision="pinned", max_usd=.1, input_usd_per_million=2,
                         output_usd_per_million=6, max_records=1, **kwargs)


def test_dry_run_never_writes_or_calls(tmp_path):
    result = collect(generate_records(policies=6)["train"], TinyTokenizer(), config(), tmp_path / "unused")
    assert result["status"] == "dry_run" and result["requests"] == 3
    assert not (tmp_path / "unused").exists()


def test_every_candidate_scored_and_resume_spends_nothing(tmp_path):
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        tokens = body["prompt"]
        return httpx.Response(200, json={"model": "test", "choices": [{"logprobs": {
            "token_ids": tokens + [0], "token_logprobs": [None] + [-1.] * (len(tokens) - 2) + [-float(len(calls)), -2.]}}],
            "usage": {"prompt_tokens": len(tokens), "completion_tokens": 1}})
    rows = generate_records(policies=6)["train"][:1]
    output = tmp_path / "cache"
    transport = httpx.MockTransport(handler)
    result = collect(rows, TinyTokenizer(), config(), output, execute=True, transport=transport)
    assert result["status"] == "completed" and len(calls) == 3
    cached = json.loads((output / "cache.jsonl").read_text())
    probs = cached["teacher"]["probabilities"]
    assert set(probs) == {o.id for o in rows[0].candidate_options}
    assert sum(probs.values()) == pytest.approx(1.)
    collect(rows, TinyTokenizer(), config(), output, execute=True, transport=transport)
    assert len(calls) == 3
    rows[0].question = "changed"
    with pytest.raises(ValueError, match="different inputs"):
        collect(rows, TinyTokenizer(), config(), output, execute=True, transport=transport)


def test_failed_attempt_is_reserved_and_never_auto_retried(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(429))
    rows = generate_records(policies=6)["train"][:1]
    with pytest.raises(ValueError, match="HTTP 429"):
        collect(rows, TinyTokenizer(), config(), tmp_path, execute=True, transport=transport)
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1
    assert not (tmp_path / "complete.json").exists()


def test_budget_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match="cap"):
        collect(generate_records(policies=6)["train"], TinyTokenizer(), config(max_requests=2), tmp_path, execute=True)


def test_echo_requires_exact_token_ids_and_usage():
    with pytest.raises(ValueError, match="exact requested"):
        parse_echo({"choices": [{"logprobs": {"token_logprobs": [None, -1]}}]}, [1, 2])


def test_reasoning_reference_is_hash_bound_and_final_answer_removed(tmp_path):
    from summit.recipes.decision.teacher import load_rationales
    from summit.recipes.decision.data import digest
    row = generate_records(policies=6)["train"][0]
    cfg = config(prompt_mode="reasoning_cached")
    (tmp_path / "complete.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps({"config": cfg.model_dump(), "inputs": {row.id: row.input_hash()}}))
    response = {"choices": [{"text": "Apply the policy carefully.</think>\n\nA", "finish_reason": "stop"}]}
    (tmp_path / f"response-{digest(row.id)}.json").write_text(json.dumps(response))
    traces = load_rationales(tmp_path, [row], cfg)
    assert traces[row.id] == "Apply the policy carefully.</think>"
    row.question = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        load_rationales(tmp_path, [row], cfg)


def test_reference_parser_rejects_truncation_and_unlisted_answer():
    from summit.recipes.decision.reasoning_reference import parse_answer
    row = generate_records(policies=6)["train"][0]
    assert parse_answer({"text": "thinking</think>\nA", "finish_reason": "stop"}, row.candidate_options) == row.candidate_options[0].id
    assert parse_answer({"text": "A", "finish_reason": "length"}, row.candidate_options) is None
    assert parse_answer({"text": "Z", "finish_reason": "stop"}, row.candidate_options) is None


def test_explicit_repair_only_replaces_invalid_trace(tmp_path):
    from summit.recipes.decision.teacher import load_rationales
    from summit.recipes.decision.data import digest
    row = generate_records(policies=6)['train'][0]
    cfg = config(prompt_mode='reasoning_cached')
    original, repair = tmp_path / 'original', tmp_path / 'repair'
    manifest = {'config':cfg.model_dump(), 'inputs':{row.id:row.input_hash()}}
    for path, meta, text, finish in [(original, manifest, 'unfinished', 'length'),
        (repair, {**manifest, 'repair_manifest_sha256':digest(manifest)}, 'complete</think>\nB', 'stop')]:
        path.mkdir()
        (path / 'complete.json').write_text('{}')
        (path / 'manifest.json').write_text(json.dumps(meta))
        (path / f'response-{digest(row.id)}.json').write_text(json.dumps({'choices':[{'text':text, 'finish_reason':finish}]}))
    with pytest.raises(ValueError, match='complete parseable'):
        load_rationales(original, [row], cfg)
    assert load_rationales(original, [row], cfg, repair=repair)[row.id] == 'complete</think>'
    (original / f'response-{digest(row.id)}.json').write_text(json.dumps({'choices':[{'text':'original</think>\nA','finish_reason':'stop'}]}))
    assert load_rationales(original, [row], cfg, repair=repair)[row.id] == 'original</think>'


def test_chained_repairs_bind_to_each_predecessor(tmp_path):
    from summit.recipes.decision.teacher import load_rationales
    from summit.recipes.decision.data import digest
    row = generate_records(policies=6)['train'][0]
    cfg = config(prompt_mode='reasoning_cached')
    paths = [tmp_path / name for name in ('original','repair1','repair2')]
    previous = None
    for i, path in enumerate(paths):
        path.mkdir()
        manifest = {'config':cfg.model_dump(),'inputs':{row.id:row.input_hash()}}
        if previous is not None:
            manifest['repair_manifest_sha256'] = digest(previous)
        (path/'manifest.json').write_text(json.dumps(manifest))
        (path/'complete.json').write_text('{}')
        (path/f'response-{digest(row.id)}.json').write_text(json.dumps({'choices':[{
            'text':'finished</think>\nA' if i==2 else 'incomplete', 'finish_reason':'stop' if i==2 else 'length'}]}))
        previous = manifest
    assert load_rationales(paths[0],[row],cfg,repair=paths[1:])[row.id]=='finished</think>'
    from summit.recipes.decision.native_targets import native_targets
    encoded = native_targets([row], cfg, paths[0], paths[1:])[0]['teacher']
    assert encoded['probabilities'][row.candidate_options[0].id] == 1
    assert sum(encoded['probabilities'].values()) == 1
    assert encoded['metadata']['label_encoding'] == 'one_hot_not_model_confidence'
    assert encoded['method'] == 'native_answer_hard_label'
    with pytest.raises(ValueError,match='not bound'):
        load_rationales(paths[0],[row],cfg,repair=[paths[2]])
