"""Mini pipeline RAG 100 % hors-ligne (sans clé API) pour pouvoir tester l'évaluation.

`retrieve` classe les documents par recouvrement de mots ; `answer` est extractif
(il ne renvoie que des phrases du contexte). Pour utiliser un vrai LLM, passez
votre propre fonction `generate(question, context) -> str` à `RAG`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DATA = Path(__file__).resolve().parent.parent / "data"
STOP = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "est", "a", "au", "aux",
        "que", "qui", "quel", "quelle", "quels", "quelles", "pour", "par", "sur", "dans", "ce", "il", "elle"}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[\wéèêàâîôûùç'-]+", text.lower()) if t not in STOP]


def load_corpus() -> list[dict]:
    return json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))


def retrieve(question: str, corpus: list[dict], k: int = 2) -> list[dict]:
    q = set(tokens(question))
    scored = sorted(corpus, key=lambda d: len(q & set(tokens(d["text"]))), reverse=True)
    return scored[:k]


def extractive_generate(question: str, context: list[dict]) -> str:
    q = set(tokens(question))
    sentences = [s.strip() for d in context for s in re.split(r"(?<=[.!?])\s+", d["text"]) if s.strip()]
    return max(sentences, key=lambda s: len(q & set(tokens(s))))


@dataclass
class RAG:
    corpus: list[dict]
    generate: Callable[[str, list[dict]], str] = extractive_generate
    k: int = 2

    def ask(self, question: str) -> tuple[str, list[dict]]:
        context = retrieve(question, self.corpus, self.k)
        return self.generate(question, context), context
