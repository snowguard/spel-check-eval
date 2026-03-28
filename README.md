# Domain-Aware Query Correction for Healthcare Information Retrieval

Code and data for the paper: **"Domain-Aware Query Correction for Healthcare Information Retrieval: A Framework with Safety Constraints and Empirical Evaluation"** — submitted to the Journal of Biomedical Informatics (JBI).

## What This Is

Healthcare search systems routinely receive misspelled queries from patients and caregivers. General-purpose spell checkers can silently introduce dangerous corrections — changing one valid medical term into another (e.g., "hypertension" ↔ "hypotension"). This repository contains:

- **MedSpellGuard** — a safety-constraint layer that blocks corrections between confusable medical term pairs
- **Six correction methods** evaluated head-to-head: conservative edit distance, standard edit distance, context-aware ranking, SymSpell, llama3.1 8B, and Claude Sonnet
- **Three retrieval backends**: BM25 (lexical), TF-IDF, and dense embedding (nomic-embed-text)
- **Error taxonomy** mapping healthcare query error types to correction strategies with clinical risk annotations
- **Statistical validation** via paired bootstrap resampling (10,000 iterations)

## Key Findings

| Method | BM25 MRR Δ | Latency |
|--------|-----------|---------|
| Conservative edit distance | +6.7% | <5ms |
| Standard edit distance | +6.7% | <5ms |
| Context-aware ranking | +9.2% | <5ms |
| SymSpell | +5.1% | <1ms |
| llama3.1 8B | +13.8% | ~1.4s |
| Claude Sonnet | +10.7% | ~1.7s |

Dense retrieval benefits even more from correction (+21.8% MRR) than lexical retrieval (+6.7%).

## Repository Structure

