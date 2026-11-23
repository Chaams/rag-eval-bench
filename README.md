# rag-eval-bench

Banc d'évaluation d'un pipeline RAG avec **quality gates** : la CI échoue si la fidélité, la pertinence
ou la récupération passent sous un seuil, ou si le taux d'hallucination dépasse le maximum toléré.

- Fonctionne **hors-ligne** (pipeline extractif de démonstration, aucune clé API).
- Pour tester un vrai LLM : passez votre fonction `generate(question, context)` à `RAG(...)`.
- Pour des métriques avancées, remplacez `src/metrics.py` par RAGAS ou DeepEval.

## Lancer

```bash
pip install -r requirements.txt
pytest -v
```

## Structure

| Fichier | Rôle |
|---|---|
| `src/rag.py` | Mini pipeline : récupération + génération |
| `src/metrics.py` | Fidélité, hallucination, précision du contexte, pertinence |
| `data/eval_set.jsonl` | Jeu de référence (questions + réponses attendues) |
| `tests/test_rag_quality.py` | Quality gates + test qui vérifie que les hallucinations sont détectées |
| `.github/workflows/ci.yml` | Exécution à chaque push |
