"""Verify adapter-keyed cache directory: teacher retrain → fresh cache automatically.

Tests the pure helper `_adapter_sha8`. The full cache-redirect logic is
exercised by the integration smoke; this file isolates the hash determinism
+ change detection invariants that make the bridge step unnecessary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_v2 import _adapter_sha8  # noqa: E402


def _write_adapter(d: Path, content: bytes) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(content)


def test_sha_is_deterministic(tmp_path: Path):
    d = tmp_path / "adapter1"
    _write_adapter(d, b"weights-blob-A")
    a = _adapter_sha8(d)
    b = _adapter_sha8(d)
    assert a == b
    assert len(a) == 8


def test_different_weights_produce_different_sha(tmp_path: Path):
    """The whole point: a retrained teacher must yield a different cache key."""
    d1 = tmp_path / "old_adapter"
    d2 = tmp_path / "new_adapter"
    _write_adapter(d1, b"weights-blob-OLD")
    _write_adapter(d2, b"weights-blob-NEW")
    assert _adapter_sha8(d1) != _adapter_sha8(d2)


def test_falls_back_to_bin_format(tmp_path: Path):
    """Older PEFT adapters stored .bin; helper must still work."""
    d = tmp_path / "legacy_adapter"
    d.mkdir()
    (d / "adapter_model.bin").write_bytes(b"legacy-weights")
    sha = _adapter_sha8(d)
    assert len(sha) == 8


def test_missing_weights_raises(tmp_path: Path):
    d = tmp_path / "empty_adapter"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="No adapter weights"):
        _adapter_sha8(d)


def test_safetensors_takes_priority_over_bin(tmp_path: Path):
    """If both formats exist, the modern .safetensors wins (deterministic)."""
    d = tmp_path / "both_adapter"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"new-format")
    (d / "adapter_model.bin").write_bytes(b"old-format")
    sha = _adapter_sha8(d)
    expected = _adapter_sha8(_make_only(tmp_path / "only_safe", b"new-format"))
    assert sha == expected


def _make_only(d: Path, content: bytes) -> Path:
    _write_adapter(d, content)
    return d
