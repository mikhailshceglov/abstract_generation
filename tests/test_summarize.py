from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarizer import summarize

MAX_LEN = 300
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_empty_and_short_empty_returns_empty():
    assert summarize([""]) == [""]
    assert summarize(["   "]) == [""]


def test_batch_of_empty_strings():
    assert summarize(["", "", ""]) == ["", "", ""]


@pytest.mark.parametrize(
    "text",
    [
        "x " * 500,
        "Первое предложение. Другое второе важное. Третье тоже.",
        "Одно. Два. Три. Четыре. Пять.",
        "Краткий текст.",
    ],
)
def test_single_text_length_at_most_300(text):
    assert len(summarize([text])[0]) <= MAX_LEN


def test_batch_length_at_most_300():
    texts = [
        "x " * 500,
        "Первое предложение. Другое второе важное. Третье тоже.",
        "А. Б. В.",
    ]
    for s in summarize(texts):
        assert len(s) <= MAX_LEN


def test_not_trivial_prefix_long_text():
    filler = "Вводим слова для длины. " * 50
    important = "Ключевая идея: эксперимент подтвердил гипотезу."
    text = filler + important
    assert len(text) > MAX_LEN
    out = summarize([text])[0]
    trivial = text[:MAX_LEN]
    assert out != trivial


def test_multisentence_unique_included_and_not_prefix_when_long():
    s1 = " ".join(
        f"Вводное предложение номер {i} без уникальных маркеров." for i in range(25)
    )
    s2 = "Уникальное_слово_экспериментальное_12345 подтверждает результат."
    text = s1 + " " + s2
    assert len(text) > MAX_LEN
    out = summarize([text])[0]
    assert "Уникальное_слово_экспериментальное_12345" in out
    assert out != text[:MAX_LEN]


def test_output_length_matches_input_length():
    texts = json.loads((DATA_DIR / "extractive_texts.json").read_text(encoding="utf-8"))
    out = summarize(texts)
    assert len(out) == len(texts)


def test_golden_texts_and_references_same_count():
    texts = json.loads((DATA_DIR / "extractive_texts.json").read_text(encoding="utf-8"))
    refs = json.loads((DATA_DIR / "extractive_references.json").read_text(encoding="utf-8"))
    assert len(texts) == len(refs)
    assert len(texts) >= 6


def test_golden_data_files_exist():
    assert (DATA_DIR / "extractive_texts.json").is_file()
    assert (DATA_DIR / "extractive_references.json").is_file()
    assert (DATA_DIR / "abstractive_texts.json").is_file()
    assert (DATA_DIR / "abstractive_references.json").is_file()


def test_abstractive_files_are_json_arrays():
    at = json.loads((DATA_DIR / "abstractive_texts.json").read_text(encoding="utf-8"))
    ar = json.loads((DATA_DIR / "abstractive_references.json").read_text(encoding="utf-8"))
    assert isinstance(at, list) and isinstance(ar, list)


def test_abstractive_matches_extractive_corpus_size():
    et = json.loads((DATA_DIR / "extractive_texts.json").read_text(encoding="utf-8"))
    at = json.loads((DATA_DIR / "abstractive_texts.json").read_text(encoding="utf-8"))
    ar = json.loads((DATA_DIR / "abstractive_references.json").read_text(encoding="utf-8"))
    assert len(et) == len(at) == len(ar)


def test_summarize_is_deterministic():
    t = "Первое. Второе уникальное_слово_тест. Третье."
    a = summarize([t])
    b = summarize([t])
    assert a == b
