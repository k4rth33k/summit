"""Two decision architectures and an explicit module:factory extension point.

Factories accept (ModelConfig, checkpoint: Path | None) and return an nn.Module
with score(records, tokenizer, max_length) -> list[Tensor[K]] and save(path).
The trainer owns loss, optimization, logging, and artifact lifecycle.
"""

import importlib
import json
from pathlib import Path
import string

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM
from safetensors.torch import load_file, save_file

from .prompts import prompt


def encode(texts, tokenizer, max_length, device):
    rows = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    if any(not row or len(row) > max_length for row in rows):
        raise ValueError(f"input exceeds max_length={max_length} (largest={max(map(len, rows))}); no evidence is truncated")
    # Right padding and explicit gather of the last real token, never logits at padding.
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
        mask[i, :len(row)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def last_hidden(backbone, inputs):
    hidden = backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
    ends = inputs["attention_mask"].sum(-1) - 1
    return hidden[torch.arange(len(ends), device=hidden.device), ends]


def load_lm(config, path=None):
    # Keep trainable weights and AdamW state in FP32. Pure BF16 AdamW can round
    # small updates away; dtype selects autocast compute, not optimizer storage.
    kwargs = {"dtype": torch.float32, "attn_implementation": "eager", "trust_remote_code": False}
    if path is None:
        kwargs["revision"] = config.revision
    return AutoModelForCausalLM.from_pretrained(str(path or config.name), **kwargs)


def answer_token_ids(records, texts, tokenizer):
    token_ids = []
    for row, text in zip(records, texts):
        prefix = tokenizer.encode(text, add_special_tokens=False)
        choices = []
        for letter in string.ascii_uppercase[:len(row.candidate_options)]:
            full = tokenizer.encode(text + letter, add_special_tokens=False)
            if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
                raise ValueError(f"answer {letter} is not a single token at the actual prompt boundary")
            choices.append(full[-1])
        if len(set(choices)) != len(choices):
            raise ValueError("answer letters do not map to distinct tokens")
        token_ids.append(choices)
    return token_ids


def causal_scores(lm, records, tokenizer, max_length, compute_dtype):
    texts = [prompt(row, tokenizer) for row in records]
    token_ids = answer_token_ids(records, texts, tokenizer)
    inputs = encode(texts, tokenizer, max_length, next(lm.parameters()).device)
    with torch.autocast(device_type=inputs["input_ids"].device.type, dtype=torch.bfloat16,
                        enabled=compute_dtype == "bfloat16"):
        hidden = last_hidden(lm.get_decoder(), inputs)
        head = lm.get_output_embeddings()
        return [torch.nn.functional.linear(
            h, head.weight[ids], head.bias[ids] if head.bias is not None else None
        ).float() for h, ids in zip(hidden, token_ids)]


class CausalDecisionModel(nn.Module):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.compute_dtype = config.dtype
        if config.adapter_kwargs:
            raise ValueError("builtin causal_lm accepts no adapter_kwargs")
        self.lm = load_lm(config, checkpoint / "model" if checkpoint else None)
        if not isinstance(self.lm.get_output_embeddings(), nn.Linear):
            raise ValueError("builtin causal_lm requires a linear vocabulary head; provide a custom adapter")
        if config.gradient_checkpointing:
            self.lm.gradient_checkpointing_enable()

    def score(self, records, tokenizer, max_length):
        # Equivalent to selecting these columns from the ordinary LM logits,
        # without allocating a sequence_length * vocabulary tensor.
        return causal_scores(self.lm, records, tokenizer, max_length, self.compute_dtype)

    def save(self, path):
        self.lm.save_pretrained(path / "model", safe_serialization=True)


class CandidateScorer(nn.Module):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.compute_dtype = config.dtype
        if config.adapter_kwargs:
            raise ValueError("builtin candidate_scorer accepts no adapter_kwargs")
        if checkpoint:
            self.backbone = AutoModel.from_pretrained(str(checkpoint / "model"), dtype=torch.float32,
                                                       attn_implementation="eager", trust_remote_code=False)
        else:
            lm = load_lm(config)
            self.backbone = lm.get_decoder()
        self.head = nn.Linear(self.backbone.config.hidden_size, 1, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)
        if checkpoint:
            self.head.load_state_dict(load_file(str(checkpoint / "head.safetensors")))
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def score(self, records, tokenizer, max_length):
        texts = [prompt(row, tokenizer, i) for row in records for i in range(len(row.candidate_options))]
        inputs = encode(texts, tokenizer, max_length, next(self.parameters()).device)
        with torch.autocast(device_type=inputs["input_ids"].device.type, dtype=torch.bfloat16,
                            enabled=self.compute_dtype == "bfloat16"):
            values = self.head(last_hidden(self.backbone, inputs)).squeeze(-1).float()
        return list(values.split([len(row.candidate_options) for row in records]))

    def save(self, path):
        self.backbone.save_pretrained(path / "model", safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()}, str(path / "head.safetensors"))


class QLoRACausalDecisionModel(nn.Module):
    """Four-bit frozen causal LM with an explicitly trainable LoRA adapter."""

    manages_device_placement = True

    def __init__(self, config, checkpoint=None):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise ValueError("qlora_causal requires CUDA")
        options = {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        unknown = set(config.adapter_kwargs) - set(options)
        if unknown:
            raise ValueError(f"unknown qlora_causal adapter_kwargs: {sorted(unknown)}")
        options.update(config.adapter_kwargs)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        source = Path(str(config.name))
        if source.is_dir() and (source / "model").is_dir():
            source = source / "model"
        base = AutoModelForCausalLM.from_pretrained(
            str(source),
            revision=None if checkpoint else config.revision,
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if checkpoint:
            self.lm = PeftModel.from_pretrained(base, checkpoint / "peft_adapter", is_trainable=False)
        else:
            lora = LoraConfig(
                r=int(options["rank"]),
                lora_alpha=int(options["alpha"]),
                lora_dropout=float(options["dropout"]),
                target_modules=list(options["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.lm = get_peft_model(base, lora)
        self.compute_dtype = config.dtype
        runtime_parameter_elements = sum(p.numel() for p in self.lm.parameters())
        logical_base_parameters = None
        source_manifest = source.parent / "manifest.json" if source.name == "model" else source / "manifest.json"
        if source_manifest.is_file():
            logical_base_parameters = json.loads(source_manifest.read_text()).get("parameters")
        self.training_metadata = {
            "adaptation": "qlora",
            "logical_base_parameters": logical_base_parameters,
            "runtime_parameter_elements": runtime_parameter_elements,
            "trainable_parameters": sum(p.numel() for p in self.lm.parameters() if p.requires_grad),
            "quantization": "bitsandbytes NF4 double-quantized",
            "compute": config.dtype,
            "optimizer": "AdamW FP32 LoRA parameters",
            "lora": options,
        }

    def score(self, records, tokenizer, max_length):
        return causal_scores(self.lm, records, tokenizer, max_length, self.compute_dtype)

    def save(self, path):
        self.lm.save_pretrained(path / "peft_adapter", safe_serialization=True)


def resolve_adapter(name):
    builtins = {
        "causal_lm": CausalDecisionModel,
        "candidate_scorer": CandidateScorer,
        "qlora_causal": QLoRACausalDecisionModel,
    }
    if name in builtins:
        return builtins[name]
    module, separator, symbol = name.partition(":")
    if not separator:
        raise ValueError(f"unknown adapter: {name}")
    factory = getattr(importlib.import_module(module), symbol, None)
    if not callable(factory):
        raise ValueError(f"adapter factory is not callable: {name}")
    return factory
