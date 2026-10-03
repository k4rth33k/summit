"""One torchrun rank exercising NexRL v1.4.0's actual FSDP training code.

All adaptations are local to this process and guarded by PHANTORA=1. We keep
the loss/backward/gradient-clipping/AdamW implementations, but skip metrics
that branch on fake tensor values and force the finite-gradient optimizer
path. Loading checkpoints, numerical correctness, and export are not tested.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import copy
from pathlib import Path
import textwrap
import traceback


class UnsupportedSimulation(RuntimeError):
    """A path whose data-dependent behavior has not been adapted."""


def replace_function(function, old, new):
    """Fail on upstream drift rather than silently applying a partial patch."""
    source = textwrap.dedent(inspect.getsource(function))
    if source.count(old) != 1:
        raise RuntimeError(f"Simulator adapter anchor changed: {function.__qualname__}")
    namespace = {}
    exec(compile(source.replace(old, new), inspect.getfile(function), "exec"),
         function.__globals__, namespace)
    return namespace[function.__name__]


def replace_function_many(function, replacements):
    """Apply several guarded edits before compiling the function once."""
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Simulator adapter anchor changed: {function.__qualname__}")
        source = source.replace(old, new)
    namespace = {}
    exec(compile(source, inspect.getfile(function), "exec"), function.__globals__, namespace)
    return namespace[function.__name__]


def adapt_nexrl(optimizer="adamw", memory_efficient_fsdp=False):
    if os.environ.get("PHANTORA") != "1":
        raise RuntimeError("Simulation adaptations must never run outside Phantora")
    import torch
    import transformers
    from nexrl.train_service_backend.fsdp_worker.fsdp_workers import ModelWorker
    from nexrl.train_service_backend.fsdp_worker.fsdp_actor import DataParallelPPOActor
    from nexrl.train_service_backend.utils import core_algos, dist_utils

    model_worker_replacements = [
        ('attn_implementation="flash_attention_2"', 'attn_implementation="eager"'),
    ]
    if optimizer == "adafactor":
        from summit.nexrl_ext.optimizer import adafactor
        ModelWorker._build_model_optimizer.__globals__["summit_adafactor"] = adafactor
        model_worker_replacements.append(("optim.AdamW(", "summit_adafactor("))
    elif optimizer != "adamw":
        raise UnsupportedSimulation(f"Unsupported optimizer {optimizer!r}")
    if memory_efficient_fsdp:
        model_worker_replacements.extend([
            ("forward_prefetch=True", "forward_prefetch=False"),
            ("BackwardPrefetch.BACKWARD_PRE", "BackwardPrefetch.BACKWARD_POST"),
        ])
    ModelWorker._build_model_optimizer = replace_function_many(
        ModelWorker._build_model_optimizer, model_worker_replacements)
    # Keep __init__ intact: recompiling it as a standalone function would lose
    # the __class__ closure used by zero-argument super(). Intercept only its
    # exact allocator request, and preserve all other allocator settings.
    if inspect.getsource(ModelWorker.__init__).count('"expandable_segments:True"') != 1:
        raise RuntimeError("ModelWorker allocator adapter anchor changed")
    set_allocator_settings = torch.cuda.memory._set_allocator_settings

    def allocator_settings(settings):
        if settings == "expandable_segments:True":
            settings = "expandable_segments:False"
        return set_allocator_settings(settings)

    torch.cuda.memory._set_allocator_settings = allocator_settings

    function = core_algos.compute_reverse_kl_loss
    source = textwrap.dedent(inspect.getsource(function))
    marker = "    # Compute additional metrics (no gradient required)\n"
    if source.count(marker) != 1 or not source.rstrip().endswith("return reverse_kl_loss, metrics"):
        raise RuntimeError("Reverse KL metrics adapter anchor changed")
    namespace = {}
    exec(compile(source.split(marker)[0] + "    return reverse_kl_loss, {}\n",
                 inspect.getfile(function), "exec"), function.__globals__, namespace)
    core_algos.compute_reverse_kl_loss = namespace[function.__name__]
    DataParallelPPOActor._optimizer_step = replace_function(
        DataParallelPPOActor._optimizer_step,
        '''    if not torch.isfinite(grad_norm):
        print(f"WARN: grad_norm is not finite: {grad_norm}")
        self.actor_optimizer.zero_grad()
    else:
        self.actor_optimizer.step()''',
        "    self.actor_optimizer.step()")

    def from_metadata(cls, *args, config, torch_dtype, attn_implementation, **kwargs):
        if kwargs.get("trust_remote_code"):
            raise RuntimeError("Remote model code is outside the simulation adapter's supported scope")
        # FSDP's per-module meta materializer cannot preserve a parameter shared
        # by the input embedding and lm_head. Build a separate lm_head in the
        # simulator instead. This is conservative for memory and keeps the real
        # model unchanged; numerical equivalence is already outside scope.
        config = copy.deepcopy(config)
        config.tie_word_embeddings = False
        text_config = getattr(config, "text_config", config)
        text_config.tie_word_embeddings = False
        # Keep construction on meta. Putting the entire model on CUDA here
        # would overstate the initialization peak versus production's CPU
        # loader followed by FSDP's per-module transfers and sharding.
        with torch.device("meta"):
            return cls.from_config(config, torch_dtype=torch_dtype,
                                   attn_implementation=attn_implementation, trust_remote_code=False)

    for cls in {transformers.AutoModelForCausalLM, transformers.AutoModelForVision2Seq}:
        cls.from_pretrained = classmethod(from_metadata)

    def materialize_virtual_parameters(module):
        # Production initializes rank zero from checkpoint values on CPU;
        # other ranks already use this to_empty path. Simulated values are
        # meaningless, so materialize every rank as FSDP visits each module.
        module.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()

    dist_utils.init_fn = materialize_virtual_parameters
    return ModelWorker


def run():
    rank = int(os.environ["RANK"])
    result = {"rank": rank, "status": "inconclusive", "phase": "imports",
              "input_sha256": Path("/input/input.sha256").read_text()}
    tracer_enabled = False
    try:
        if os.environ.get("PHANTORA") != "1":
            raise RuntimeError("Start the worker with phantora_run")
        import torch
        from summit.nexrl_ext.compat import apply
        apply()
        from summit.simulation.container import load_recipe
        from nexrl.train_service_backend.utils.protocol import DataProto
        from transformers import AutoConfig
        manifest = json.loads(Path("/input/manifest.json").read_text())
        recipe = load_recipe()
        config = recipe.service.train_service.student.actor
        config.model.path = "/model"
        model_config = AutoConfig.from_pretrained("/model", local_files_only=True, trust_remote_code=False)
        if model_config.model_type not in {"llama", "qwen3", "qwen3_5", "qwen3_5_text"}:
            raise UnsupportedSimulation(f"Architecture {model_config.model_type!r} is outside the adapter's dense Llama/Qwen3/Qwen3.5 scope")
        if config.model.use_remove_padding or config.use_dynamic_bsz or config.ulysses_sequence_parallel_size != 1:
            raise UnsupportedSimulation("Adapter supports dense fixed-shape batches with sequence parallelism=1 only")
        if config.use_distillation_clipping or any(
            config.get(key, value) != value for key, value in {
                "train_token_start_pct": 0.0, "train_token_end_pct": 1.0,
                "advantage_start_pct": 0.0, "advantage_end_pct": 1.0,
                "entropy_start_pct": 0.0, "entropy_end_pct": 1.0,
            }.items()
        ):
            raise UnsupportedSimulation("Value-dependent filtering/clipping is outside the supported adapter path")
        ModelWorker = adapt_nexrl(
            manifest["workload"].get("optimizer", "adamw"),
            manifest["workload"].get("memory_efficient_fsdp", False),
        )
        torch.profiler.enable_function_tracer(os.environ["PHANTORA_SOCKET_PREFIX"] + ".simulator.sock")
        tracer_enabled = True
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        result["phase"] = "fsdp_initialization"
        worker = ModelWorker(config, role="actor")
        worker.init_model()
        result["initialization_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["initialization_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        result["phase"] = "synthetic_batch"
        world = manifest["virtual_cluster"]["gpus_per_host"]
        mini = config.ppo_mini_batch_size  # already normalized by ModelWorker
        rows = math.ceil(manifest["workload"]["trajectory_rows_upper_bound"] / (world * mini)) * mini
        seq = manifest["workload"]["max_sequence_length"]
        response = manifest["workload"]["max_response_length"]
        tensors = {
            "input_ids": torch.zeros((rows, seq), dtype=torch.long, device="cuda"),
            "attention_mask": torch.ones((rows, seq), dtype=torch.long, device="cuda"),
            "position_ids": torch.arange(seq, device="cuda").expand(rows, -1).contiguous(),
            "responses": torch.zeros((rows, response), dtype=torch.long, device="cuda"),
            "teacher_log_probs": torch.zeros((rows, response), dtype=torch.float32, device="cuda"),
            "scoring_attention_mask": torch.ones((rows, seq), dtype=torch.long, device="cuda"),
        }
        data = DataProto.from_dict(tensors=tensors, meta_info={
            "temperature": recipe.trainer.algorithm.temperature,
            "distillation_coeff": recipe.trainer.algorithm.distillation_coeff,
            "entropy_coeff": recipe.trainer.algorithm.entropy_coeff,
            "loss_agg_mode": "token-mean", "global_token_num": rows * seq * world,
        })
        result["scenario"] = {"rows_per_rank": rows, "sequence_length": seq, "response_length": response}
        result["phase"] = "distillation_update"
        worker.update_actor_with_distillation(data)
        torch.cuda.synchronize()
        result["update_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["update_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["peak_allocated_bytes"] = max(result["initialization_peak_allocated_bytes"], result["update_peak_allocated_bytes"])
        result["peak_reserved_bytes"] = max(result["initialization_peak_reserved_bytes"], result["update_peak_reserved_bytes"])
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        torch.profiler.disable_function_tracer()
        tracer_enabled = False
        result.update(status="completed", phase="completed")
    except Exception as exc:
        traceback.print_exc()
        result.update(code="SIM.WORKER_ERROR", message=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, UnsupportedSimulation):
            result["code"] = "SIM.UNSUPPORTED"
        # Only typed allocator failures are labelled as OOM. An arbitrary
        # RuntimeError may be an unsupported simulator call, not a bad model.
        if "torch" in locals() and isinstance(exc, torch.cuda.OutOfMemoryError):
            result.update(status="failed", code="SIM.CUDA_OOM")
    finally:
        # Write before tracer teardown: a failed collective can also stall the
        # tracer. The supervisor still requires a clean exit from every rank.
        Path(f"/results/rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        if tracer_enabled:
            torch.profiler.disable_function_tracer()
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
