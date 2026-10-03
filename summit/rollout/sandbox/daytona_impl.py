"""Daytona sandbox provider — interface stub for the roadmap (v0.1 untested)."""

from __future__ import annotations

from .base import Sandbox, SandboxProvider


class DaytonaProvider(SandboxProvider):
    def create(self, image: str | None, setup_commands: list[str] | None = None) -> Sandbox:
        raise NotImplementedError(
            "Daytona sandbox support is on the summit roadmap; use sandbox: modal for v0.1"
        )
