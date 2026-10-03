"""Modal-backed sandboxes (the v0.1 tested path).

Auth via MODAL_TOKEN_ID / MODAL_TOKEN_SECRET env vars (standard Modal SDK).
"""

from __future__ import annotations

import logging

from .base import Sandbox, SandboxProvider

logger = logging.getLogger(__name__)

APP_NAME = "summit-deepswe"


class ModalSandbox(Sandbox):
    def __init__(self, sandbox):
        self._sb = sandbox

    def exec(self, command: str, cwd: str = "", timeout_s: int | None = None) -> tuple[int, str]:
        kwargs = {"workdir": cwd} if cwd else {}
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        p = self._sb.exec("bash", "-lc", command, **kwargs)
        p.wait()
        out = ""
        try:
            out = (p.stdout.read() or "") + (p.stderr.read() or "")
        except Exception:  # noqa: BLE001
            pass
        return p.returncode or 0, out

    def terminate(self) -> None:
        try:
            self._sb.terminate()
        except Exception:  # noqa: BLE001
            logger.warning("modal sandbox terminate failed", exc_info=True)


class ModalProvider(SandboxProvider):
    def create(self, image: str | None, setup_commands: list[str] | None = None) -> ModalSandbox:
        import modal  # imported lazily; only required on the VM

        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        img = modal.Image.from_registry(image) if image else modal.Image.debian_slim()
        sb = modal.Sandbox.create(image=img, app=app)
        for cmd in setup_commands or []:
            p = sb.exec("bash", "-lc", cmd)
            p.wait()
            if p.returncode:
                raise RuntimeError(f"sandbox setup command failed ({p.returncode}): {cmd}")
        return ModalSandbox(sb)
