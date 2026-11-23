"""Métriques d'évaluation RAG simples et transparentes.

Elles servent de garde-fou rapide en CI ; pour une évaluation plus fine, branchez
RAGAS ou DeepEval à la place (même interface : une valeur entre 0 et 1).
"""
from __future__ import annotations

from .rag import tokens


def faithfulness(answer: str, context: list[dict]) -> float:
    """Part des mots de la réponse que l'on retrouve dans le contexte récupéré."""
    ans = tokens(answer)
    if not ans:
        return 0.0
    ctx = set(tokens(" ".join(d["text"] for d in context)))
    return sum(t in ctx for t in ans) / len(ans)


def hallucination_rate(answer: str, context: list[dict]) -> float:
    return 1.0 - faithfulness(answer, context)


def context_precision(context: list[dict], expected_doc: str) -> float:
    """1.0 si le bon document est en tête du contexte, 0.5 s'il est présent mais pas premier."""
    ids = [d["id"] for d in context]
    if not ids or expected_doc not in ids:
        return 0.0
    return 1.0 if ids[0] == expected_doc else 0.5


def answer_relevance(answer: str, reference: str) -> float:
    a, r = set(tokens(answer)), set(tokens(reference))
    return len(a & r) / len(r) if r else 0.0
