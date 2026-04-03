from __future__ import annotations

import argparse
import json
from pathlib import Path

from razdel import tokenize as razdel_tokenize
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from summarizer import summarize

DATA_DIR = Path(__file__).resolve().parent / "data"


class RazdelTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [t.text.lower() for t in razdel_tokenize(text)]


def load_json_list(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Ожидался JSON-массив, получено: {type(data)}")
    return [str(x) for x in data]


def evaluate(
    texts: list[str],
    references: list[str],
) -> dict[str, float]:
    if len(texts) != len(references):
        raise ValueError("Число текстов и эталонных рефератов должно совпадать")
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=RazdelTokenizer(),
    )
    preds = summarize(texts)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(preds)
    for hyp, ref in zip(preds, references):
        scores = scorer.score(ref, hyp)
        agg["rouge1"] += scores["rouge1"].fmeasure
        agg["rouge2"] += scores["rouge2"].fmeasure
        agg["rougeL"] += scores["rougeL"].fmeasure
    for k in agg:
        agg[k] /= max(n, 1)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--texts",
        type=Path,
        default=DATA_DIR / "extractive_texts.json",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=DATA_DIR / "extractive_references.json",
    )
    args = parser.parse_args()
    texts = load_json_list(args.texts)
    refs = load_json_list(args.references)
    metrics = evaluate(texts, refs)
    print("Средние ROUGE (F-score) по корпусу:")
    for name, val in sorted(metrics.items()):
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
