"""Explicit integration tests against locally built images; no image pulls.

SUMMIT_TEST_RUNTIME=1 pytest tests/test_simulation_container.py -q
SUMMIT_TEST_PHANTORA=1 pytest tests/test_simulation_container.py -q
"""
import json
import os

import pytest

from summit.config import load_config
from summit.render import render
from summit.simulation.runner import run_simulation


@pytest.fixture
def tiny_job(tmp_path):
    model = tmp_path / "tiny-model"
    model.mkdir()
    files = {
        "config.json": {
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "vocab_size": 4, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "max_position_embeddings": 128, "bos_token_id": 1, "eos_token_id": 2,
            "pad_token_id": 3, "tie_word_embeddings": False,
        },
        "tokenizer_config.json": {
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
            "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
        },
        "tokenizer.json": {
            "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
            "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None, "decoder": None,
            "model": {"type": "WordLevel", "unk_token": "<unk>",
                      "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}},
        },
    }
    for name, data in files.items():
        (model / name).write_text(json.dumps(data))
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    cfg.model_name = "summit-test-tiny-llama"
    cfg.summitConfig.resources.gpu = "A100-80GB:3"
    cfg.summitConfig.resources.rollout_gpus = 1
    cfg.data.batch_size = 1
    cfg.rollout.step_limit = 1
    cfg.data.max_prompt_length = cfg.data.max_response_length = 16
    cfg.data.max_sequence_length = 32
    return cfg, model


@pytest.mark.skipif(os.environ.get("SUMMIT_TEST_RUNTIME") != "1", reason="opt-in local Docker test")
def test_real_runtime_container(tiny_job, tmp_path):
    cfg, model = tiny_job
    result = run_simulation(cfg, render(cfg), mode="runtime", model_dir=model, output_dir=tmp_path)
    assert result["status"] == "completed", result
    assert result["coverage"]["model_metadata"] == "completed"
    assert result["coverage"]["patch_targets"] == "completed"


@pytest.mark.skipif(os.environ.get("SUMMIT_TEST_PHANTORA") != "1", reason="opt-in local Phantora test")
def test_real_two_rank_fsdp_update(tiny_job, tmp_path):
    cfg, model = tiny_job
    result = run_simulation(cfg, render(cfg), model_dir=model, output_dir=tmp_path, timeout=120)
    assert result["status"] == "completed", result
    assert result["completed_ranks"] == [0, 1]
    for rank in result["rank_results"]:
        assert rank["phase"] == "completed"
        assert rank["peak_allocated_bytes"] > 0


@pytest.mark.skipif(os.environ.get("SUMMIT_TEST_PHANTORA") != "1", reason="opt-in local Phantora test")
def test_real_virtual_vram_failure(tiny_job, tmp_path):
    cfg, model = tiny_job
    cfg.summitConfig.resources.gpu = "A100-1GB:3"
    config = json.loads((model / "config.json").read_text())
    # Embedding plus LM head alone exceed the virtual device capacity. The
    # weights are never downloaded and virtual CUDA allocates no real VRAM.
    config.update(vocab_size=65536, hidden_size=8192, intermediate_size=8192,
                  num_attention_heads=64, num_key_value_heads=64)
    (model / "config.json").write_text(json.dumps(config))
    result = run_simulation(cfg, render(cfg), model_dir=model, output_dir=tmp_path, timeout=120)
    assert result["status"] == "failed", result
    assert "SIM.CUDA_OOM" in {finding["code"] for finding in result["findings"]}
