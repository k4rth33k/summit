import pytest

from summit.config import GPUResource, load_config


def test_load_example_config():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    assert cfg.model_name == "Qwen/Qwen3.5-9B"
    assert cfg.teacher.backend == "fireworks"
    assert cfg.teacher.base_url == "https://api.fireworks.ai/inference"
    assert cfg.teacher.api_key_env == "FIREWORKS_API_KEY"
    assert cfg.summitConfig.output.hf_push.repo == "example-org/test-model"
    assert cfg.data.batch_size == 4
    assert cfg.data.shuffle is True
    assert cfg.use_wandb


def test_gpu_parse():
    g = GPUResource.parse("A100-80GB:8")
    assert (g.name, g.memory_gb, g.count) == ("A100", 80, 8)
    g = GPUResource.parse("H200:8")
    assert (g.name, g.memory_gb, g.count) == ("H200", None, 8)
    g = GPUResource.parse("H200,H200NVL-141GB:5")
    assert (g.name, g.memory_gb, g.count) == ("H200,H200NVL", 141, 5)


def test_friendli_defaults():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    cfg.teacher.backend = "friendli"
    cfg.teacher.base_url = None
    cfg.teacher.api_key_env = None
    cfg.teacher = type(cfg.teacher).model_validate(cfg.teacher.model_dump())
    assert cfg.teacher.base_url == "https://api.friendli.ai/serverless"
    assert cfg.teacher.api_key_env == "FRIENDLI_API_KEY"


def test_learning_qualification_config_is_budgeted_and_ordered():
    cfg = load_config("tests/fixtures/deepswe_opd_learning.yaml")
    assert len(cfg.data.tasks) == 16
    assert len(set(cfg.data.tasks)) == 12
    assert cfg.data.tasks[:4] == cfg.data.tasks[-4:]
    assert cfg.data.shuffle is False
    assert cfg.total_train_steps == 4
    assert cfg.rollout.num_workers == 4
    assert cfg.summitConfig.max_duration == "65m"
    assert cfg.summitConfig.retry_on_no_capacity is True
    assert cfg.summitConfig.retry_on_error is False
    assert cfg.summitConfig.retry_on_interruption is False
    assert cfg.summitConfig.retry_duration == "2h"
    assert cfg.summitConfig.resources.gpu_spec().name == "H200,H200NVL"


def test_budgeted_learning_config_uses_memory_efficient_optimizer():
    cfg = load_config("tests/fixtures/deepswe_opd_learning_96gb.yaml")
    assert cfg.optimizer == "adafactor"
    assert cfg.memory_efficient_fsdp is True
    assert cfg.summitConfig.resources.gpu_spec().name == "RTXPRO6000"
    assert cfg.summitConfig.resources.gpu_spec().count == 2
    assert cfg.summitConfig.resources.rollout_gpus == 1
    assert cfg.summitConfig.resources.disk == "300GB"
    assert cfg.summitConfig.regions == ["US-MO-2"]
    assert cfg.summitConfig.max_duration == "120m"
    assert cfg.data.tasks[:4] == cfg.data.tasks[-4:]
