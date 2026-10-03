"""Compatibility shims: NexRL v1.4.0 with transformers>=5.

NexRL pins only a floor (`transformers>=4.56.2`), but its model_utils imports
`is_remote_url`, removed in transformers 5.x — while Qwen3.5-class student
architectures require the latest transformers. This shim restores the symbol.

Installed into the VM venv via a `.pth` file so it executes before any nexrl
import in every process (API server, FSDP workers, driver):

    echo 'import summit.nexrl_ext.compat; summit.nexrl_ext.compat.apply()' \
      > /opt/venv-nexrl/lib/python3.12/site-packages/00summit_compat.pth
"""

from __future__ import annotations


def apply() -> None:
    _patch_transformers()
    _stub_flash_attn()


def _patch_transformers() -> None:
    import transformers
    import transformers.utils

    if not hasattr(transformers.utils, "is_remote_url"):

        def is_remote_url(url) -> bool:
            return isinstance(url, str) and url.startswith(("http://", "https://"))

        transformers.utils.is_remote_url = is_remote_url

    # transformers 5.x renamed AutoModelForVision2Seq; NexRL (verl-derived)
    # still imports it when deciding the model class.
    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(
        transformers, "AutoModelForImageTextToText"
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText


def _stub_flash_attn() -> None:
    """fsdp_actor.py imports flash_attn.bert_padding unconditionally, but the
    functions are only exercised when use_remove_padding=true (we ship false in
    v0.1). flash-attn has no prebuilt wheel for many torch/cuda combos and
    builds take ~an hour, so we stub the module. If it ever *is* called, the
    error message points at the fix (build flash-attn or keep rmpad off)."""
    import sys
    import types

    if "flash_attn" in sys.modules:
        return  # already present (real or stubbed)
    try:
        import importlib.util

        if importlib.util.find_spec("flash_attn") is not None:
            return  # real flash-attn installed — nothing to do
    except (ImportError, ValueError):
        pass  # during site/.pth init find_spec can misbehave; stub anyway

    try:
        from einops import rearrange  # same function flash_attn re-exports
    except ImportError:
        rearrange = None

    def _disabled(*_args, **_kwargs):
        raise RuntimeError(
            "flash_attn stub called: this code path needs real flash-attn "
            "(set use_remove_padding=false — the v0.1 default — or build flash-attn)"
        )

    import importlib.machinery

    def _stub_module(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        # A valid __spec__ keeps importlib.util.find_spec(name) from raising
        # ValueError; the missing *distribution* metadata keeps
        # transformers' is_flash_attn_2_available() returning False.
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return m

    bert_padding = _stub_module("flash_attn.bert_padding")
    bert_padding.index_first_axis = lambda x, indices: x[indices]
    bert_padding.rearrange = rearrange or _disabled
    bert_padding.pad_input = _disabled
    bert_padding.unpad_input = _disabled

    flash_attn = _stub_module("flash_attn")
    flash_attn.bert_padding = bert_padding
    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.bert_padding"] = bert_padding
