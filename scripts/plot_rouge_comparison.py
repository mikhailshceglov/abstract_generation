"""Группированный barplot: ROUGE по экстрактивному и абстрактивному эталону (одинаковые исходники)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rouge_comparison.png"

# Средние F-score по корпусу (25 документов)
METRICS = ("ROUGE-1", "ROUGE-2", "ROUGE-L")
EXTRACTIVE = (0.9141, 0.8185, 0.7445)
ABSTRACTIVE = (0.3986, 0.1209, 0.3177)


def main() -> None:
    x = np.arange(len(METRICS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.bar(
        x - width / 2,
        EXTRACTIVE,
        width,
        label="Экстрактивный эталон",
        color="#27ae60",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        ABSTRACTIVE,
        width,
        label="Абстрактивный эталон",
        color="#2980b9",
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_ylabel("ROUGE F-score (среднее по корпусу)")
    ax.set_title("Сравнение эталонов при одном и том же извлекательном summarize()")
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Сохранено: {OUT}")


if __name__ == "__main__":
    main()
