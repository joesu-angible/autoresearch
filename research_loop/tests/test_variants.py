"""Tests for research_loop.variants — JSONL variants file loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_loop.variants import load_variants, REQUIRED_FIELDS


def _valid_entry(**overrides) -> dict:
    base = {
        "hypothesis": "raise PRODUCTNESS_CLS_WEIGHT to 0.05",
        "expected_metric": "productness_neg_acc +0.02",
        "changed_files": ["student_finetune/train_v2.py"],
        "risks": ["may regress recall@1"],
        "rollback": "combined < incumbent - 0.005",
        "patch": "--- a/foo\n+++ b/foo\n@@\n+x\n",
    }
    base.update(overrides)
    return base


def _write_jsonl(tmp_path: Path, entries: list[dict | str]) -> Path:
    path = tmp_path / "variants.jsonl"
    lines = [e if isinstance(e, str) else json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_loads_valid_multi_entry_file(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [
        _valid_entry(hypothesis="weight=0.0"),
        _valid_entry(hypothesis="weight=0.05"),
        _valid_entry(hypothesis="weight=0.1"),
    ])
    specs = load_variants(path)
    assert len(specs) == 3
    assert [s["hypothesis"] for s in specs] == ["weight=0.0", "weight=0.05", "weight=0.1"]
    assert specs[0]["changed_files"] == ["student_finetune/train_v2.py"]


def test_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    path.write_text("\n" + json.dumps(_valid_entry()) + "\n\n" + json.dumps(_valid_entry()) + "\n\n")
    specs = load_variants(path)
    assert len(specs) == 2


def test_empty_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    path.write_text("")
    assert load_variants(path) == []


def test_only_blank_lines_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    path.write_text("\n\n   \n")
    assert load_variants(path) == []


def test_malformed_json_raises_with_line_number(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [
        _valid_entry(),
        "{not valid json",
    ])
    with pytest.raises(ValueError, match="line 2.*malformed JSON"):
        load_variants(path)


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_missing_required_field_raises(tmp_path: Path, missing_field: str) -> None:
    entry = _valid_entry()
    entry.pop(missing_field)
    path = _write_jsonl(tmp_path, [entry])
    with pytest.raises(ValueError, match=f"missing required fields.*{missing_field}"):
        load_variants(path)


def test_non_object_line_raises(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, ["[1, 2, 3]"])
    with pytest.raises(ValueError, match="expected JSON object"):
        load_variants(path)


def test_extra_fields_are_ignored(tmp_path: Path) -> None:
    entry = _valid_entry(extra_field="should be dropped", another=42)
    path = _write_jsonl(tmp_path, [entry])
    specs = load_variants(path)
    assert len(specs) == 1
    assert "extra_field" not in specs[0]
