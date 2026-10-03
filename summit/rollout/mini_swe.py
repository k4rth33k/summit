"""Glue between NexRL's inference client (the student policy) and
mini-swe-agent's DefaultAgent, with commands executed in a summit sandbox.

Design: we implement mini-swe-agent's `Model` protocol ourselves (rather than
using its litellm models) so every LLM call goes through NexRL's
`OpenAIInferenceServiceClient.generate()`, which returns `nexrl_train`
(prompt_tokens, response_tokens, response_logprobs) — the exact fields NexRL
needs to build OPD training datums. Prompt/action templates are loaded from
mini-swe-agent's own `mini_textbased.yaml` so behavior matches upstream
mini-swe-agent runs.
"""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from .sandbox.base import Sandbox, SandboxProvider

logger = logging.getLogger(__name__)

ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"


def _load_mswea_textbased_config() -> dict[str, Any]:
    import minisweagent.config

    cfg_path = Path(minisweagent.config.__file__).parent / "mini_textbased.yaml"
    return yaml.safe_load(cfg_path.read_text())


def repair_prompt_tokens(resp: dict[str, Any], chat_template_fn) -> dict[str, Any]:
    """Fill nexrl_train.prompt_tokens when the inference client returned none.

    NexRL's client guards `isinstance(x, list)`, but processor-style tokenizers
    (e.g. Qwen3.5 VLM) return a BatchEncoding from apply_chat_template — which
    silently empties prompt_tokens and would drop every OPD turn.
    """
    train = resp.get("nexrl_train")
    if train is None or train.get("prompt_tokens"):
        return resp
    ids = chat_template_fn()
    if hasattr(ids, "keys"):  # BatchEncoding / dict
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):  # batched -> take the single row
        ids = ids[0]
    train["prompt_tokens"] = list(ids) if ids else []
    return resp


class NexRLStudentModel:
    """mini-swe-agent Model protocol over a NexRL inference client's generate().

    `generate_fn(messages) -> completion dict` must return an OpenAI chat
    completion dict augmented with `nexrl_train` (NexRL's client does this).
    """

    def __init__(self, generate_fn: Callable[..., dict[str, Any]], model_cfg: dict[str, Any]):
        self._generate = generate_fn
        self._observation_template = model_cfg["observation_template"]
        self._format_error_template = model_cfg["format_error_template"]

    # --- Model protocol ---

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.actions_text import parse_regex_actions

        response = self._generate(messages)
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")

        message = {"role": "assistant", "content": content}
        try:
            actions = parse_regex_actions(
                content,
                action_regex=ACTION_REGEX,
                format_error_template=self._format_error_template,
                template_kwargs={"finish_reason": finish_reason},
            )
        except FormatError as e:
            # The agent persists cost from the failed call; we are free (local policy).
            e.messages[0].setdefault("extra", {})["cost"] = 0.0
            raise

        message["extra"] = {
            "actions": actions,
            "cost": 0.0,
            "timestamp": time.time(),
            # Payload for NexRL trajectory construction (consumed by the worker).
            "nexrl_train": response.get("nexrl_train", {}),
            "finish_reason": finish_reason,
        }
        return message

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        from minisweagent.models.utils.actions_text import format_observation_messages

        return format_observation_messages(
            outputs,
            observation_template=self._observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        u = platform.uname()
        return {"system": u.system, "release": u.release, "version": u.version, "machine": u.machine}

    def serialize(self) -> dict:
        return {"info": {"model": "nexrl-student-policy"}}


class SandboxEnvironment:
    """mini-swe-agent Environment protocol over a summit Sandbox."""

    def __init__(self, sandbox: Sandbox, env_vars: dict[str, str] | None = None):
        self._sandbox = sandbox
        self._env_vars = env_vars or {}

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        from minisweagent.exceptions import Submitted

        command = action.get("command", "")
        if self._env_vars:
            exports = " ".join(f'{k}="{v}"' for k, v in self._env_vars.items())
            command = f"{exports} {command}"
        try:
            returncode, out = self._sandbox.exec(command, cwd=cwd, timeout_s=timeout)
            output: dict[str, Any] = {"output": out, "returncode": returncode, "exception_info": ""}
        except Exception as e:  # noqa: BLE001
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
            }
        # Mirror LocalEnvironment's submission handling.
        lines = (output.get("output") or "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        u = platform.uname()
        return {"system": u.system, "release": u.release, "version": u.version, "machine": u.machine}

    def serialize(self) -> dict:
        return {"info": {"environment": "summit-sandbox"}}


def run_agent_task(
    task: dict[str, Any],
    generate_fn: Callable[..., dict[str, Any]],
    sandbox_provider: SandboxProvider,
    step_limit: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run one DeepSWE task with mini-swe-agent. Returns agent messages +
    exit info. Each assistant message carries extra.nexrl_train."""

    from minisweagent.agents.default import DefaultAgent

    cfg = _load_mswea_textbased_config()
    agent_cfg = cfg["agent"]
    model_cfg = cfg["model"]
    env_vars = cfg.get("environment", {}).get("env", {})

    sandbox = sandbox_provider.create(
        image=task.get("image"), setup_commands=task.get("setup_commands") or []
    )
    try:
        agent = DefaultAgent(
            NexRLStudentModel(generate_fn, model_cfg),
            SandboxEnvironment(sandbox, env_vars),
            system_template=agent_cfg["system_template"],
            instance_template=agent_cfg["instance_template"],
            step_limit=step_limit,
            cost_limit=0.0,  # disabled: local policy has no dollar cost
            output_path=output_path,
        )
        exit_info = agent.run(task=task.get("instruction", ""))
        return {"messages": agent.messages, "exit": exit_info}
    finally:
        sandbox.terminate()
