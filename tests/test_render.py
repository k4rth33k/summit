import yaml

from summit.config import load_config
from summit.render import render


def test_render_produces_parseable_recipe():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    r = render(cfg)
    recipe = yaml.safe_load(r.recipe_yaml)  # must be valid YAML

    assert recipe["launch_mode"] == "local"
    assert recipe["trainer"]["type"] == "self_hosted_opd"
    assert "opd_trainer.py" in recipe["trainer"]["custom_trainer_module_path"]
    assert recipe["service"]["train_service"]["teacher"]["backend"] == "fireworks"
    assert recipe["service"]["train_service"]["teacher"]["model_path"] == (
        "accounts/fireworks/models/qwen3p8-2p4t-a95b"
    )
    assert recipe["rollout_worker"]["type"] == "deepswe"
    assert "deepswe_worker.py" in recipe["rollout_worker"]["custom_rollout_worker_module_path"]
    assert recipe["data"]["drop_last"] is False
    assert recipe["data"]["shuffle"] is True
    # (batch_size_reached alone trips NexRL's validate_config when
    # keep_batch_order is true; must AND with loaded_batch_finished)
    assert recipe["trajectory_pool"]["check_batch_ready_function"] == (
        "batch_size_reached_and_loaded_batch_finished"
    )
    assert recipe["trainer"]["total_train_steps"] == 2
    assert recipe["weight"]["sync_method"] == "disk"
    assert recipe["service"]["train_service"]["teacher"]["resource"]["world_size"] == 1

    # GPU split: 5 total, 1 for rollout (tp=1 for Qwen3.5 mamba) -> 4 training
    student = recipe["service"]["train_service"]["student"]
    assert student["resource"]["gpus_per_pod"] == 4
    assert recipe["service"]["inference_service"]["resource"]["gpus_per_replica"] == 1


def test_render_bootstrap_contains_key_steps():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    r = render(cfg)
    sh = r.bootstrap_sh
    for needle in [
        "sglang.launch_server",
        "api_server",
        "worker_process",
        "summit.nexrl_ext.boot",
        "convert_checkpoint_to_huggingface",
        "example-org/test-model",
        "accounts/fireworks/models/qwen3p8-2p4t-a95b",
        "abs-module-cache-flags,abs-stepped-slices",
        # in-flight fixes shipped via the bootstrap template
        "patched tracking.py: get_url guarded",
        "patched fsdp_workers.py: convert clears unshard ctx",
        'if [ "${N:-0}" -ge "$TRAIN_GPUS" ]',
        "run_setup_branch trainer setup_nexrl &",
        "run_setup_branch sglang setup_sglang &",
        "run_setup_branch model prefetch_model &",
        'wait -n -p FINISHED_SETUP_PID "${SETUP_PIDS[@]}"',
        '"sglang[all]==0.5.9"',
        '"sglang-router==0.3.2"',
    ]:
        assert needle in sh, needle


def test_bootstrap_starts_independent_setup_before_waiting():
    sh = render(load_config("tests/fixtures/deepswe_opd.yaml")).bootstrap_sh
    first_wait = sh.index('wait -n -p FINISHED_SETUP_PID')
    for branch in ("trainer setup_nexrl", "sglang setup_sglang",
                   "model prefetch_model", "tasks fetch_tasks"):
        assert sh.index(f"run_setup_branch {branch} &") < first_wait
    # Resolve directly to the qualified cuDNN version; never download an
    # older wheel in one command and replace it in another.
    assert sh.count('nvidia-cudnn-cu12==9.16.0.29') == 1
    assert '--overrides "$SUMMIT_REPO_DIR/summit/sglang-overrides.txt"' in sh


def test_render_rejects_bad_gpu_split():
    cfg = load_config("tests/fixtures/deepswe_opd.yaml")
    cfg.summitConfig.resources.rollout_gpus = 5
    import pytest

    with pytest.raises(ValueError):
        render(cfg)


def test_learning_qualification_preserves_task_order():
    cfg = load_config("tests/fixtures/deepswe_opd_learning.yaml")
    recipe = yaml.safe_load(render(cfg).recipe_yaml)
    assert recipe["data"]["shuffle"] is False
    assert recipe["data"]["batch_size"] == 4
    assert recipe["trainer"]["total_train_steps"] == 4


def test_adafactor_is_an_explicit_guarded_bootstrap_patch():
    cfg = load_config("tests/fixtures/deepswe_opd_learning_96gb.yaml")
    rendered = render(cfg)
    assert 'OPTIMIZER="adafactor"' in rendered.bootstrap_sh
    assert 'MEMORY_EFFICIENT_FSDP="true"' in rendered.bootstrap_sh
    assert 'if [ "$OPTIMIZER" = "adafactor" ]' in rendered.bootstrap_sh
    assert "AdamW -> Adafactor" in rendered.bootstrap_sh
    assert "memory-efficient FSDP prefetch" in rendered.bootstrap_sh
    assert "bind NCCL process group to local CUDA device" in rendered.bootstrap_sh
    assert "preserve response scoring mask" in rendered.bootstrap_sh
    assert '"scoring_attention_mask",' in rendered.bootstrap_sh
    assert "reverse-KL score-function surrogate" in rendered.bootstrap_sh
    assert "pseudo_ratio = torch.exp(student_log_probs - student_log_probs.detach())" in rendered.bootstrap_sh
    assert "skip zero-response microbatches" in rendered.bootstrap_sh
    assert "gradient_accumulation = len(micro_batches)" in rendered.bootstrap_sh
    assert "Entire optimizer " in rendered.bootstrap_sh
    assert "training incomplete: $SUCCESSFUL_UPDATES/4 updates succeeded" in rendered.bootstrap_sh
    assert "verified $SUCCESSFUL_UPDATES/4 successful training updates" in rendered.bootstrap_sh
    assert "emulate NCCL scatter with broadcasts" in rendered.bootstrap_sh
    assert "torch.distributed.broadcast(transfer, src=src_rank)" in rendered.bootstrap_sh
    assert "torch.cuda.synchronize()" in rendered.bootstrap_sh
    assert "https://download.pytorch.org/whl/cu128" in rendered.bootstrap_sh
    assert "--attention-backend triton" in rendered.bootstrap_sh
    assert "== [1b/7] NCCL collective gate ==" in rendered.bootstrap_sh
    assert "-m summit.nccl_probe" in rendered.bootstrap_sh
    assert "NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0" in rendered.bootstrap_sh
    assert 'if [ "$TRAIN_GPUS" -gt 1 ]; then' in rendered.bootstrap_sh
    assert '"${WORKER_NCCL_ENV[@]}"' in rendered.bootstrap_sh
    recipe = yaml.safe_load(rendered.recipe_yaml)
    mixed = recipe["service"]["train_service"]["student"]["actor"]["fsdp_config"]["mixed_precision"]
    assert mixed == {"param_dtype": "bf16", "reduce_dtype": "bf16", "buffer_dtype": "bf16"}
    model = recipe["service"]["train_service"]["student"]["actor"]["model"]
    assert model["enable_gradient_checkpointing"] is True
    actor = recipe["service"]["train_service"]["student"]["actor"]
    assert actor["ppo_mini_batch_size"] == 16
    assert actor["ppo_micro_batch_size"] == 1
    assert actor["optim"]["lr"] == 1.0e-6
    assert 'SUMMIT_ROLLOUT_RESULT_FILE="$EXPERIMENT_PATH/rollout-results"' in rendered.bootstrap_sh
    assert "rollout batch produced zero trainable trajectories" in rendered.bootstrap_sh
    assert "ROLLOUT_RESULTS % 4" in rendered.bootstrap_sh


def test_hopper_only_skips_blackwell_specific_runtime_workarounds():
    sh = render(load_config("tests/fixtures/deepswe_opd_learning.yaml")).bootstrap_sh
    assert "https://download.pytorch.org/whl/cu128" not in sh
    assert "--attention-backend triton" not in sh
    assert "NCCL_P2P_DISABLE=1" in sh
