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

```
code/
├── pipeline.py               # Main experiment pipeline (BM25 + TF-IDF)
├── dense_retrieval.py         # Dense embedding experiments (nomic-embed-text)
├── llm_correction.py          # LLM-based correction (Ollama + Claude)
├── bootstrap_tests.py         # Paired bootstrap resampling tests
├── requirements.txt           # Python dependencies
├── data/
│   ├── LiveQA_MedicalTask_TREC2017/  # TREC 2017 queries + judgements
│   └── MedQuAD/                      # Answer corpus passages
└── results/                   # Cached experiment outputs
```

## Datasets

- **TREC 2017 LiveQA Medical** — 104 consumer health questions with NIST paraphrases ([trec.nist.gov](https://trec.nist.gov))
- **HealthSearchQA** — 4,436 Google health queries ([Singhal et al., 2023](https://doi.org/10.1038/s41586-023-06291-2))
- **MedQuAD** — 1,935 passages from NLM-curated sources ([Ben Abacha & Demner-Fushman, 2019](https://github.com/abachaa/MedQuAD))

## Quick Start

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run main experiments (BM25 + TF-IDF)
python code/pipeline.py

# Run dense retrieval experiments (requires Ollama with nomic-embed-text)
python code/dense_retrieval.py

# Run LLM correction (requires Ollama running locally)
python code/llm_correction.py

# Run bootstrap significance tests
python code/bootstrap_tests.py
```

### LLM Correction Setup

For llama3.1:
```bash
ollama pull llama3.1:8b
ollama serve
```

For dense retrieval embeddings:
```bash
ollama pull nomic-embed-text
```

For Claude: set `ANTHROPIC_API_KEY` in your environment.

## Requirements

- Python 3.9+
- numpy, scipy
- Ollama (for local LLM correction and dense retrieval)

## Citation

```bibtex
@article{singh2026domainaware,
  title={Domain-Aware Query Correction for Healthcare Information Retrieval:
         A Framework with Safety Constraints and Empirical Evaluation},
  author={Singh, Saurabh K},
  journal={arXiv preprint arXiv:2603.19249},
  year={2026}
}
```

## License

MIT
