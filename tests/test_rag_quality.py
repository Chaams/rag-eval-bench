import json
from pathlib import Path

import pytest

from src.metrics import answer_relevance, context_precision, faithfulness, hallucination_rate
from src.rag import RAG, load_corpus

EVAL = [json.loads(l) for l in (Path(__file__).parent.parent / "data" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

# Quality gate : si une seuil n'est pas respecté, la CI échoue et le déploiement est bloqué.
MIN_FAITHFULNESS = 0.90
MAX_HALLUCINATION = 0.05
MIN_CONTEXT_PRECISION = 0.90
MIN_RELEVANCE = 0.80


@pytest.fixture(scope="module")
def rag():
    return RAG(load_corpus())


def _run(rag):
    return [(c, *rag.ask(c["question"])) for c in EVAL]


def test_faithfulness_gate(rag):
    scores = [faithfulness(a, ctx) for _, a, ctx in _run(rag)]
    assert sum(scores) / len(scores) >= MIN_FAITHFULNESS


def test_hallucination_gate(rag):
    rates = [hallucination_rate(a, ctx) for _, a, ctx in _run(rag)]
    assert sum(rates) / len(rates) <= MAX_HALLUCINATION


def test_retrieval_gate(rag):
    scores = [context_precision(ctx, c["expected_doc"]) for c, _, ctx in _run(rag)]
    assert sum(scores) / len(scores) >= MIN_CONTEXT_PRECISION


def test_relevance_gate(rag):
    scores = [answer_relevance(a, c["reference"]) for c, a, _ in _run(rag)]
    assert sum(scores) / len(scores) >= MIN_RELEVANCE


def test_metric_catches_hallucination():
    """Le garde-fou doit détecter un modèle qui invente : on simule un LLM défaillant."""
    liar = RAG(load_corpus(), generate=lambda q, ctx: "Le remboursement est illimité et la livraison offerte partout dans le monde.")
    answer, ctx = liar.ask("Sous combien de jours puis-je demander un remboursement ?")
    assert hallucination_rate(answer, ctx) > 0.3
