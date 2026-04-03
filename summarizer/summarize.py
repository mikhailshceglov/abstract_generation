from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
from razdel import sentenize, tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_ABSTRACT_LEN = 300
_CHUNK_WORDS = 48
_ALPHA_TR = 0.35

_RU_STOP = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да кто этот том
    быть был бы была были было бывают есть было есть ли лишь уже ещё
    для по из у к о об от до при над под за между перед
    который которая которое которые которой которым чем чего чему
    один одна одно одни одним одного этот эта это эти эту этой этим
    какой какая какое какие такой такая такое такие
    мне меня мной тебе тебя тобой ему его ей её им им них ней
    мы вы они ты он она оно
    здесь там тут где куда откуда когда пока если бы то ли же бы
    не ни нет да ну же ли вот даже ещё только лишь очень
    можно нужно надо будет былo быть
    """.split()
)


def _maybe_split_long_single_sentence(sents: list[str]) -> list[str]:
    if len(sents) != 1:
        return sents
    s = sents[0]
    if len(s) <= MAX_ABSTRACT_LEN:
        return sents
    words = s.split()
    if len(words) < _CHUNK_WORDS + 2:
        return sents
    chunks: list[str] = []
    step = _CHUNK_WORDS
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + step]))
    return chunks if len(chunks) > 1 else sents


def _sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    raw = [s.text.strip() for s in sentenize(text) if s.text.strip()]
    return _maybe_split_long_single_sentence(raw)


def _tokens_lower(sentence: str) -> list[str]:
    out: list[str] = []
    for t in tokenize(sentence):
        w = t.text.lower()
        if len(w) < 2 or not any(c.isalpha() for c in w):
            continue
        if w.isdigit():
            continue
        if w in _RU_STOP:
            continue
        out.append(w)
    return out


def _sentence_for_tfidf(sentence: str) -> str:
    toks = _tokens_lower(sentence)
    return " ".join(toks) if toks else sentence.lower()


def _textrank_scores(sentences: Sequence[str]) -> list[float]:
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    corpus = [_sentence_for_tfidf(s) for s in sentences]
    if not any(corpus):
        return [1.0 / n] * n
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(corpus)
    except ValueError:
        return [1.0 / n] * n
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    g = nx.from_numpy_array(sim, create_using=nx.Graph)
    for i in range(n):
        if g.degree(i) == 0:
            g.add_edge(i, (i + 1) % n, weight=1e-6)
    scores = nx.pagerank(g, weight="weight")
    return [float(scores[i]) for i in range(n)]


def _normalize_unit(scores: Sequence[float]) -> list[float]:
    a = np.asarray(scores, dtype=float)
    if a.size == 0:
        return []
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return [1.0 / len(scores)] * len(scores)
    return list((a - lo) / (hi - lo))


def _lexical_importance_scores(sentences: Sequence[str]) -> list[float]:
    n = len(sentences)
    if n == 0:
        return []
    corpus = [_sentence_for_tfidf(s) for s in sentences]
    if not any(corpus):
        return [1.0 / n] * n
    vectorizer = TfidfVectorizer()
    try:
        mat = vectorizer.fit_transform(corpus)
    except ValueError:
        return [1.0 / n] * n
    row_max = mat.max(axis=1)
    w = np.asarray(row_max.toarray() if hasattr(row_max, "toarray") else row_max).ravel()
    return [float(w[i]) for i in range(n)]


def _combined_scores(sentences: Sequence[str], tr: list[float]) -> list[float]:
    lex = _lexical_importance_scores(sentences)
    tr_u = _normalize_unit(tr)
    lex_u = _normalize_unit(lex)
    return [
        _ALPHA_TR * tr_u[i] + (1.0 - _ALPHA_TR) * lex_u[i] for i in range(len(sentences))
    ]


def _trim_to_limit(text: str, limit: int = MAX_ABSTRACT_LEN) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > limit * 3 // 5:
        return cut[:last_space].rstrip()
    return cut


def _build_abstract(sentences: list[str], scores: list[float]) -> str:
    order = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    parts: list[str] = []
    length = 0
    for i in order:
        s = sentences[i]
        add_len = len(s) + (1 if parts else 0)
        if length + add_len <= MAX_ABSTRACT_LEN:
            parts.append(s)
            length += add_len
        elif not parts:
            parts.append(_trim_to_limit(s, MAX_ABSTRACT_LEN))
            break
    if not parts:
        return ""
    result = " ".join(parts)
    return _trim_to_limit(result, MAX_ABSTRACT_LEN)


def summarize(texts: list[str]) -> list[str]:
    out: list[str] = []
    for text in texts:
        sents = _sentences(text)
        if not sents:
            out.append("")
            continue
        tr = _textrank_scores(sents)
        scores = _combined_scores(sents, tr)
        out.append(_build_abstract(sents, scores))
    return out
