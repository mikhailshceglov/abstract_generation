from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluate import evaluate

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_evaluate_returns_three_metrics():
    texts = json.loads((DATA_DIR / "extractive_texts.json").read_text(encoding="utf-8"))
    refs = json.loads((DATA_DIR / "extractive_references.json").read_text(encoding="utf-8"))
    m = evaluate(texts, refs)
    assert set(m.keys()) == {"rouge1", "rouge2", "rougeL"}
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_evaluate_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        evaluate(["a"], ["b", "c"])


def test_evaluate_single_pair():
    texts = [
        "Машинное обучение — область науки. Алгоритмы учатся на данных.",
    ]
    refs = [
        "Машинное обучение и алгоритмы на данных.",
    ]
    m = evaluate(texts, refs)
    assert m["rouge1"] > 0.0
