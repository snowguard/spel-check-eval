#!/usr/bin/env python3
"""
Bootstrap significance tests for all experimental comparisons.
"""

import json
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


def paired_bootstrap(scores_a, scores_b, n_iterations=10000, seed=42):
    """Paired bootstrap test comparing two sets of per-query scores.

    Returns: mean_diff, ci_lower, ci_upper, p_value
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert n == len(scores_b), f"Length mismatch: {n} vs {len(scores_b)}"

    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    diffs = []
    count_reversed = 0
    for _ in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        boot_a = np.array([scores_a[i] for i in indices])
        boot_b = np.array([scores_b[i] for i in indices])
        diff = np.mean(boot_a) - np.mean(boot_b)
        diffs.append(diff)
        if diff <= 0:
            count_reversed += 1

    diffs = sorted(diffs)
    ci_lower = diffs[int(0.025 * n_iterations)]
    ci_upper = diffs[int(0.975 * n_iterations)]
    p_value = count_reversed / n_iterations

    return {
        'mean_diff': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
    }


def main():
    print("=" * 60)
    print("Bootstrap Significance Tests")
    print("=" * 60)

    # Load per-query results
    pq_file = RESULTS_DIR / 'per_query_results.json'
    if not pq_file.exists():
        print("ERROR: Run pipeline.py first to generate per_query_results.json")
        sys.exit(1)

    with open(pq_file) as f:
        pq_data = json.load(f)

    # Extract MRR scores for each condition
    def get_mrr_scores(key, retriever='bm25'):
        data = pq_data.get(key, {})
        pq = data.get(f'{retriever}_per_query', [])
        # Sort by qid for consistent pairing
        pq_sorted = sorted(pq, key=lambda x: x['qid'])
        return [x['MRR'] for x in pq_sorted]

    # Also load LLM results if available
    llm_file = RESULTS_DIR / 'llm_results.json'
    llm_data = {}
    if llm_file.exists():
        with open(llm_file) as f:
            llm_data = json.load(f)

    # Also load dense retrieval results if available
    dense_file = RESULTS_DIR / 'dense_retrieval_results.json'
    dense_data = {}
    if dense_file.exists():
        with open(dense_file) as f:
            dense_data = json.load(f)

    results = {}

    # Key comparisons for BM25
    baseline_key = '---_False_False'
    comparisons = {
        'edit_dist_vs_baseline': ('edit_distance_True_True', baseline_key, 'bm25'),
        'conservative_vs_baseline': ('conservative_True_True', baseline_key, 'bm25'),
        'symspell_vs_baseline': ('symspell_True_True', baseline_key, 'bm25'),
        'edit_dist_vs_conservative': ('edit_distance_True_True', 'conservative_True_True', 'bm25'),
        'query_only_vs_corpus_only_ed': ('edit_distance_True_False', 'edit_distance_False_True', 'bm25'),
    }

    # TF-IDF comparisons
    tfidf_comparisons = {
        'tfidf_edit_dist_vs_baseline': ('edit_distance_True_True', baseline_key, 'tfidf'),
        'tfidf_conservative_vs_baseline': ('conservative_True_True', baseline_key, 'tfidf'),
    }

    # MedSpellGuard comparisons
    guard_comparisons = {
        'guard_vs_no_guard_ed': ('edit_distance_True_True_guard', 'edit_distance_True_True', 'bm25'),
        'guard_vs_no_guard_cons': ('conservative_True_True_guard', 'conservative_True_True', 'bm25'),
    }

    all_comparisons = {**comparisons, **tfidf_comparisons, **guard_comparisons}

    print("\n{:<40} {:>8} {:>15} {:>8} {:>5}".format(
        "Comparison", "Δ MRR", "95% CI", "p", "Sig?"))
    print("-" * 80)

    for name, (key_a, key_b, retriever) in all_comparisons.items():
        scores_a = get_mrr_scores(key_a, retriever)
        scores_b = get_mrr_scores(key_b, retriever)

        if not scores_a or not scores_b:
            print(f"{name:<40} -- data not found --")
            continue

        if len(scores_a) != len(scores_b):
            print(f"{name:<40} -- length mismatch: {len(scores_a)} vs {len(scores_b)} --")
            continue

        result = paired_bootstrap(scores_a, scores_b)
        results[name] = result

        sig = "✓" if result['significant'] else "✗"
        print(f"{name:<40} {result['mean_diff']:>+8.4f} "
              f"[{result['ci_lower']:>+.4f}, {result['ci_upper']:>+.4f}] "
              f"{result['p_value']:>8.4f} {sig:>5}")

    # LLM comparisons
    if llm_data:
        print("\n--- LLM Comparisons ---")
        baseline_scores = get_mrr_scores(baseline_key, 'bm25')

        for llm_name in ['ollama', 'claude']:
            if llm_name not in llm_data:
                continue
            llm_pq = llm_data[llm_name].get('bm25_per_query', [])
            if not llm_pq:
                continue
            llm_pq_sorted = sorted(llm_pq, key=lambda x: x['qid'])
            llm_scores = [x['MRR'] for x in llm_pq_sorted]

            if len(llm_scores) == len(baseline_scores):
                result = paired_bootstrap(llm_scores, baseline_scores)
                results[f'{llm_name}_vs_baseline'] = result
                sig = "✓" if result['significant'] else "✗"
                print(f"{llm_name}_vs_baseline{'':<20} {result['mean_diff']:>+8.4f} "
                      f"[{result['ci_lower']:>+.4f}, {result['ci_upper']:>+.4f}] "
                      f"{result['p_value']:>8.4f} {sig:>5}")

    # Dense retrieval comparisons
    if dense_data:
        print("\n--- Dense Retrieval Comparisons ---")
        if 'baseline' in dense_data and 'query_corrected_edit_distance' in dense_data:
            base_pq = sorted(dense_data['baseline']['per_query'], key=lambda x: x['qid'])
            corr_pq = sorted(dense_data['query_corrected_edit_distance']['per_query'], key=lambda x: x['qid'])

            if len(base_pq) == len(corr_pq):
                base_mrr = [x['MRR'] for x in base_pq]
                corr_mrr = [x['MRR'] for x in corr_pq]
                result = paired_bootstrap(corr_mrr, base_mrr)
                results['dense_edit_dist_vs_baseline'] = result
                sig = "✓" if result['significant'] else "✗"
                print(f"{'dense_edit_dist_vs_baseline':<40} {result['mean_diff']:>+8.4f} "
                      f"[{result['ci_lower']:>+.4f}, {result['ci_upper']:>+.4f}] "
                      f"{result['p_value']:>8.4f} {sig:>5}")

    # Save
    with open(RESULTS_DIR / 'bootstrap_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Bootstrap results saved to {RESULTS_DIR / 'bootstrap_results.json'}")


if __name__ == '__main__':
    main()
