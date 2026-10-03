"""Guard the benchmark training converter against accidental test-set inputs."""

from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")
pytest.importorskip("transformers")


def test_unrecognized_parquet_fails_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import prepare_typed_decisions_training as preparation
    source = tmp_path / "test.parquet"
    source.write_bytes(b"not the pinned training dataset")
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", "--dataset", str(source), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        preparation.main()
    assert exc.value.code == 2
    assert not output.exists()
