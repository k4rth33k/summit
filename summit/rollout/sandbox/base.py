"""Sandbox provider interface for rollout environments.

DeepSWE tasks ship a prebuilt container image per task; a sandbox provider
starts that image somewhere isolated and lets the agent execute bash commands.
v0.1 implements Modal; Daytona is an interface stub on the roadmap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Sandbox(ABC):
    @abstractmethod
    def exec(self, command: str, cwd: str = "", timeout_s: int | None = None) -> tuple[int, str]:
        """Run a bash command. Returns (returncode, combined output)."""

    @abstractmethod
    def terminate(self) -> None: ...


class SandboxProvider(ABC):
    @abstractmethod
    def create(self, image: str | None, setup_commands: list[str] | None = None) -> Sandbox:
        """Start a sandbox from a container image (registry reference)."""


def get_provider(name: str) -> SandboxProvider:
    if name == "modal":
        from .modal_impl import ModalProvider

        return ModalProvider()
    if name == "daytona":
        from .daytona_impl import DaytonaProvider

        return DaytonaProvider()
    raise ValueError(f"unknown sandbox provider: {name}")
