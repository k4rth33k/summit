from __future__ import annotations

import sys
from types import SimpleNamespace

from summit import export


def test_export_declares_model_repo_type(monkeypatch, tmp_path, capsys) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "test-token"

        def create_repo(self, repo_id: str, **kwargs: object) -> None:
            calls.append(("create", {"repo_id": repo_id, **kwargs}))

        def upload_folder(self, **kwargs: object) -> str:
            calls.append(("upload", kwargs))
            return "https://huggingface.co/example/output"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit.export",
            "--hf-dir",
            str(tmp_path),
            "--repo",
            "example/output",
            "--private",
        ],
    )

    export.main()

    assert calls == [
        (
            "create",
            {
                "repo_id": "example/output",
                "repo_type": "model",
                "private": True,
                "exist_ok": True,
            },
        ),
        (
            "upload",
            {
                "repo_id": "example/output",
                "repo_type": "model",
                "folder_path": str(tmp_path),
            },
        ),
    ]
    assert "https://huggingface.co/example/output" in capsys.readouterr().out
