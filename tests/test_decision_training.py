"""Real CPU forward/backward/export/reload, with local random weights only."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from summit.recipes.decision.config import DecisionJobConfig, ModelConfig, ObjectiveConfig
from summit.recipes.decision.dataset import generate_records, write_dataset, attach_teacher
from summit.recipes.decision.models import prompt, CausalDecisionModel, CandidateScorer, resolve_adapter
from summit.recipes.decision.objective import decision_loss
from summit.recipes.decision.train import train, load_tokenizer, load_artifact, evaluate


@pytest.fixture
def tiny_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    torch.manual_seed(42)
    vocabulary = ["[PAD]", "[UNK]", "[EOS]"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
        "state", "question", "options", "description", "label", "evaluate_candidate", "Answer", "approve", "deny", "clarify"]
    tok = Tokenizer(WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    path = tmp_path / "tiny"
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=1024,
                                    n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    model.save_pretrained(path)
    return path


def test_loss_uses_reference_and_forward_kl():
    row = generate_records(policies=6)["train"][0]
    row.target.correct_option_id = row.candidate_options[0].id
    probs = {o.id: p for o, p in zip(row.candidate_options, [.8, .2, 0.0])}
    row = attach_teacher([row], [{"id": row.id, "teacher": {"input_sha256": row.input_hash(), "model": "test",
                           "method": "candidate_logprobs", "probabilities": probs}}])[0]
    values = torch.tensor([.2, -.1, .3], requires_grad=True)
    config = ObjectiveConfig(type="candidate_distillation", alpha=.5, temperature=2)
    actual = decision_loss([values], [row], config)
    target = torch.tensor([.8, .2, 0.0]).sqrt()
    target /= target.sum()
    expected = .5 * torch.nn.functional.cross_entropy(values[None], torch.tensor([0])) + 2 * torch.nn.functional.kl_div(
        (values / 2).log_softmax(-1), target, reduction="sum")
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(values.grad).all()
    config.temperature = .0001
    assert torch.isfinite(decision_loss([values], [row], config))


@pytest.mark.parametrize('publish_failure', [False, True])
def test_matched_runner_keeps_both_diagnostics_and_only_selected_published_weights(tmp_path, tiny_model, monkeypatch, publish_failure):
    splits = generate_records(policies=6)
    splits['train'] = attach_teacher(splits['train'], [
        {'id':r.id, 'teacher':{'input_sha256':r.input_hash(), 'model':'fixture',
         'method':'fixture', 'probabilities':{o.id:1/len(r.candidate_options) for o in r.candidate_options}}}
        for r in splits['train']])
    write_dataset(tmp_path/'data', splits, {})
    output = tmp_path/'matched'
    config = DecisionJobConfig.model_validate({
        'experiment_name':'matched', 'model':{'name':str(tiny_model),'dtype':'float32','gradient_checkpointing':False},
        'data':{'train':str(tmp_path/'data/train.jsonl'), 'validation':str(tmp_path/'data/validation.jsonl'), 'max_length':1024},
        'objective':{'type':'candidate_distillation'},
        'training':{'comparison':'ce_kd','device':'cpu','epochs':1,'max_steps':2,'batch_size':2,
                    'gradient_accumulation':2,'learning_rate':.001,'evaluate_reversed_options':True},
        'summitConfig':{'output':{'directory':str(output)}}})
    if publish_failure:
        from summit.recipes.decision.config import DecisionHFPushConfig
        config.summitConfig.output.hf_push = DecisionHFPushConfig(repo='fixture/private', private=True)
        def fail(*args, **kwargs):
            raise RuntimeError('fixture publication failure')
        monkeypatch.setattr('summit.recipes.decision.publication.publish_artifact', fail)
        with pytest.raises(RuntimeError, match='publication failure'):
            train(config)
        assert not (output/'complete.json').exists()
        assert (output/'failure.json').exists()
    else:
        complete = train(config)
        assert complete['comparison']=='ce_kd' and complete['total_optimizer_steps']==4
        model, tokenizer, metadata = load_artifact(output)
        assert evaluate(model,tokenizer,splits['validation'],metadata['max_length'])[0]['examples'] > 0
    comparison = json.loads((output/'comparison.json').read_text())
    assert comparison['checkpoint_selection']['selected'] in ('ce','kd')
    for arm in ('ce','kd'):
        assert (output/'comparisons'/arm/'predictions.jsonl').exists()
        assert not (output/'comparisons'/arm/'model').exists()
    assert (output/'model/model.safetensors').exists()
    with pytest.raises(FileExistsError):
        train(config)


@pytest.mark.parametrize("adapter", ["causal_lm", "candidate_scorer"])
@pytest.mark.parametrize("objective", ["candidate_ce", "candidate_distillation"])
def test_train_export_reload_and_no_overwrite(tmp_path, tiny_model, adapter, objective):
    splits = generate_records(policies=6)
    # A simple controlled overfit check, not a claim of task accuracy.
    for rows in splits.values():
        for row in rows:
            row.target.correct_option_id = row.candidate_options[0].id
    if objective == "candidate_distillation":
        splits["train"] = attach_teacher(splits["train"], [{"id": row.id, "teacher": {
            "input_sha256": row.input_hash(), "model": "test-teacher", "method": "candidate_logprobs",
            "probabilities": {o.id: p for o, p in zip(row.candidate_options, [.8, .1, .1])}}} for row in splits["train"]])
    write_dataset(tmp_path / "data", splits, {"fixture": True})
    config = DecisionJobConfig.model_validate({
        "experiment_name": f"tiny-{adapter}",
        "objective": {"type": objective},
        "model": {"name": str(tiny_model), "adapter": adapter, "dtype": "float32", "gradient_checkpointing": False},
        "data": {"train": str(tmp_path / "data/train.jsonl"), "validation": str(tmp_path / "data/validation.jsonl"), "max_length": 1024},
        "training": {"device": "cpu", "epochs": 3, "max_steps": 15, "learning_rate": .02,
                     "batch_size": 4, "gradient_accumulation": 2, "evaluate_reversed_options": True},
        "summitConfig": {"output": {"directory": str(tmp_path / "output")}},
    })
    complete = train(config)
    assert complete["reload_verified"] and complete["steps"] == 15
    output = tmp_path / "output"
    scores = json.loads((output / "evaluation.json").read_text())
    assert scores["after"]["nll"] < scores["before"]["nll"]
    reversed_scores = json.loads((output / "evaluation-reversed.json").read_text())
    assert reversed_scores["before"]["examples"] == len(splits["validation"])
    assert (output / "predictions-before.jsonl").exists()
    assert (output / "predictions-reversed.jsonl").exists()
    model, tokenizer, meta = load_artifact(output)
    _, predictions = evaluate(model, tokenizer, splits["test"], meta["max_length"])
    assert all(p["choice"] in p["probabilities"] for p in predictions)
    from summit.recipes.decision.data import DecisionInput
    from summit.recipes.decision.predict import predict
    row = splits["test"][0]
    request = DecisionInput.model_validate({"id": row.id, **row.payload()})
    response = predict(model, tokenizer, [request], meta["max_length"])[0]
    assert response["choice"] == predictions[0]["choice"]
    with pytest.raises(FileExistsError):
        train(config)
    assert (output / "complete.json").exists()


def test_prompt_hides_targets_and_causal_logits_match_full_lm(tiny_model):
    config = ModelConfig(name=str(tiny_model), dtype="float32", gradient_checkpointing=False)
    model = CausalDecisionModel(config).eval()
    tokenizer = load_tokenizer(tiny_model)
    row = generate_records(policies=6)["train"][0]
    text = prompt(row, tokenizer)
    assert row.id not in text and "correct_option_id" not in text and "provenance" not in text
    with torch.no_grad():
        scores = model.score([row], tokenizer, 1024)[0]
        ids = tokenizer.encode(text, add_special_tokens=False)
        full = model.lm(torch.tensor([ids])).logits[0, -1]
        answer_ids = [tokenizer.encode(text + letter, add_special_tokens=False)[-1] for letter in "ABC"]
        assert torch.allclose(scores, full[answer_ids], atol=1e-6)
    with pytest.raises(ValueError, match="no evidence is truncated"):
        model.score([row], tokenizer, 16)


def test_bad_adapter_and_failed_run_have_no_completion(tmp_path, tiny_model):
    with pytest.raises(ValueError, match="not callable"):
        resolve_adapter("json:does_not_exist")
    splits = generate_records(policies=6)
    write_dataset(tmp_path / "data", splits, {})
    config = DecisionJobConfig.model_validate({"experiment_name": "fail",
        "model": {"name": str(tiny_model), "dtype": "float32"},
        "data": {"train": str(tmp_path / "data/train.jsonl"), "validation": str(tmp_path / "data/validation.jsonl"), "max_length": 16},
        "training": {"device": "cpu"}, "summitConfig": {"output": {"directory": str(tmp_path / "failed")}}})
    with pytest.raises(ValueError, match="no evidence is truncated"):
        train(config)
    assert (tmp_path / "failed/failure.json").exists()
    assert not (tmp_path / "failed/complete.json").exists()


def test_publication_failure_keeps_local_artifact_without_success_marker(tmp_path, tiny_model, monkeypatch):
    from summit.recipes.decision import publication
    splits = generate_records(policies=6)
    write_dataset(tmp_path / 'data', splits, {})
    output = tmp_path / 'output'
    cfg = DecisionJobConfig.model_validate({'experiment_name':'publication-failure',
        'model':{'name':str(tiny_model),'dtype':'float32'},
        'data':{'train':str(tmp_path/'data/train.jsonl'),'validation':str(tmp_path/'data/validation.jsonl'),'max_length':1024},
        'training':{'device':'cpu','max_steps':1},
        'summitConfig':{'output':{'directory':str(output),'hf_push':{'repo':'fixture/slot','replace':True}}}})
    def fail(*args, **kwargs):
        raise ValueError('simulated storage quota failure')
    monkeypatch.setattr(publication, 'publish_artifact', fail)
    with pytest.raises(ValueError, match='storage quota'):
        train(cfg)
    assert not (output/'complete.json').exists()
    assert (output/'failure.json').exists()
    assert (output/'evaluation.json').exists()
    model, tokenizer, metadata = load_artifact(output)
    assert evaluate(model, tokenizer, splits['validation'], metadata['max_length'])[0]['examples'] > 0


@pytest.mark.parametrize("adapter", ["causal_lm", "candidate_scorer"])
def test_qwen35_hybrid_text_backbone_roundtrip(tmp_path, tiny_model, adapter):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    tokenizer = load_tokenizer(tiny_model)
    config = Qwen3_5TextConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=["linear_attention", "full_attention"], linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=1, linear_num_value_heads=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
                         "mrope_section": [1, 1, 2]})
    path = tmp_path / "tiny-qwen"
    Qwen3_5ForCausalLM(config).save_pretrained(path)
    cfg = ModelConfig(name=str(path), adapter=adapter, dtype="bfloat16", gradient_checkpointing=True)
    model = resolve_adapter(adapter)(cfg)
    row = generate_records(policies=6)["train"][0]
    model.train()
    loss = decision_loss(model.score([row], tokenizer, 1024), [row], ObjectiveConfig())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    output = tmp_path / "qwen-export"
    output.mkdir()
    model.eval()
    model.save(output)
    reloaded = resolve_adapter(adapter)(cfg, checkpoint=output).eval()
    with torch.no_grad():
        assert torch.allclose(model.score([row], tokenizer, 1024)[0], reloaded.score([row], tokenizer, 1024)[0], atol=1e-4)


def test_scorer_can_use_separate_backbone_and_head_rates(tiny_model):
    from summit.recipes.decision.config import TrainingConfig
    from summit.recipes.decision.train import optimizer_parameters
    cfg = ModelConfig(name=str(tiny_model), adapter='candidate_scorer', dtype='float32',
                      gradient_checkpointing=False)
    model = CandidateScorer(cfg)
    training = TrainingConfig(device='cpu', learning_rate=2e-7, head_learning_rate=1e-4)
    groups = optimizer_parameters(model, training)
    assert [group['lr'] for group in groups] == [2e-7, 1e-4]
    head_ids = {id(p) for p in model.head.parameters()}
    assert head_ids == {id(p) for p in groups[1]['params']}
    assert not head_ids & {id(p) for p in groups[0]['params']}


def test_qwen35_multimodal_checkpoint_loads_text_weights(tmp_path, tiny_model):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    tokenizer = load_tokenizer(tiny_model)
    config = Qwen3_5Config(text_config={"vocab_size": len(tokenizer), "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8, "layer_types": ["full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                            "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 16, "num_position_embeddings": 16})
    multimodal = Qwen3_5ForConditionalGeneration(config)
    path = tmp_path / "qwen-vlm"
    multimodal.save_pretrained(path)
    decision = CausalDecisionModel(ModelConfig(name=str(path), dtype="float32", gradient_checkpointing=False))
    assert torch.equal(decision.lm.get_input_embeddings().weight, multimodal.get_input_embeddings().weight)
    row = generate_records(policies=6)["train"][0]
    assert torch.isfinite(decision.score([row], tokenizer, 1024)[0]).all()


def test_custom_factory_is_bundled_and_reloads_in_clean_process(tmp_path, tiny_model):
    import subprocess
    import sys
    source = tmp_path / "local_adapter.py"
    source.write_text("from summit.recipes.decision.models import CandidateScorer\n"
                      "def create(config, checkpoint=None):\n"
                      "    return CandidateScorer(config, checkpoint=checkpoint)\n")
    splits = generate_records(policies=6)
    write_dataset(tmp_path / "data", splits, {})
    output = tmp_path / "custom-output"
    cfg = DecisionJobConfig.model_validate({"experiment_name": "custom", "code": [str(source)],
        "model": {"name": str(tiny_model), "adapter": "local_adapter:create", "dtype": "float32", "gradient_checkpointing": False},
        "data": {"train": str(tmp_path / "data/train.jsonl"), "validation": str(tmp_path / "data/validation.jsonl"), "max_length": 1024},
        "training": {"device": "cpu", "max_steps": 1}, "summitConfig": {"output": {"directory": str(output)}}})
    assert train(cfg)["reload_verified"]
    assert (output / "code/local_adapter.py").read_bytes() == source.read_bytes()
    result = subprocess.run([sys.executable, "-c", "from pathlib import Path; import sys; "
        "from summit.recipes.decision.train import load_artifact; load_artifact(Path(sys.argv[1]))", str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
