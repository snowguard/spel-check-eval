#!/usr/bin/env python3
"""
LLM-based spelling correction for healthcare queries.
Uses Claude API (Anthropic) and Ollama (llama3.1:8b).
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# Load .env file for API key
def load_env():
    env_paths = [
        BASE_DIR / '.env',
        BASE_DIR.parent / '.env',
        Path.home() / '.env',
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()
            return True
    return False


def call_ollama(prompt, model="llama3.1:8b"):
    """Call Ollama API for completion."""
    url = "http://localhost:11434/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200}
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            latency = time.time() - start
            return result.get('response', '').strip(), latency
    except Exception as e:
        return None, time.time() - start


def call_claude(prompt, api_key):
    """Call Claude API (Anthropic) for completion."""
    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }).encode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    req = urllib.request.Request(url, data=payload, headers=headers)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            latency = time.time() - start
            text = result.get('content', [{}])[0].get('text', '').strip()
            return text, latency
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        print(f"  Claude API error {e.code}: {error_body[:200]}")
        return None, time.time() - start
    except Exception as e:
        print(f"  Claude API error: {e}")
        return None, time.time() - start


def correct_query_llm(query_text, llm_fn, **kwargs):
    """Send a query to an LLM for spelling correction."""
    prompt = (
        "You are a medical spelling correction assistant. "
        "Correct any spelling errors in the following healthcare query. "
        "Return ONLY the corrected query text, nothing else. "
        "If no corrections are needed, return the query unchanged. "
        "Do not add explanations, quotes, or formatting.\n\n"
        f"Query: {query_text}"
    )
    response, latency = llm_fn(prompt, **kwargs)

    if response:
        # Clean up response - remove any prefixes like "Corrected query:" etc.
        response = response.strip()
        for prefix in ['Query:', 'Corrected:', 'Corrected query:', 'Output:']:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        # Remove surrounding quotes if present
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]

    return response, latency


def main():
    # Import pipeline for data loading
    sys.path.insert(0, str(BASE_DIR))
    from pipeline import (load_queries, load_qrels, load_passages,
                          build_vocabulary, tokenize, BM25, TFIDF,
                          evaluate_retrieval)

    print("=" * 60)
    print("LLM-Based Spelling Correction Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    queries = load_queries()
    all_qrels = load_qrels()
    passages = load_passages()
    queries_with_qrels = [q for q in queries if q['qid'] in all_qrels]
    print(f"  {len(queries_with_qrels)} queries with relevance judgments")

    vocab = build_vocabulary(passages, min_freq=2)

    # Build retrievers
    print("Building retrievers...")
    bm25 = BM25(passages)
    tfidf = TFIDF(passages)

    # Load API key
    load_env()
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    results = {}

    # === Ollama (llama3.1:8b) ===
    print("\n--- Ollama llama3.1:8b ---")
    ollama_corrections = []
    ollama_latencies = []

    for i, q in enumerate(queries_with_qrels):
        corrected, latency = correct_query_llm(q['original'], call_ollama, model="llama3.1:8b")
        if corrected is None:
            corrected = q['original']
            print(f"  Query {q['qid']}: Ollama failed, using original")
        ollama_corrections.append(corrected)
        ollama_latencies.append(latency)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(queries_with_qrels)} queries "
                  f"(avg latency: {sum(ollama_latencies)/len(ollama_latencies):.1f}s)")

    # Evaluate Ollama corrections
    ollama_queries = []
    for q, corrected in zip(queries_with_qrels, ollama_corrections):
        eq = dict(q)
        eq['corrected'] = corrected
        ollama_queries.append(eq)

    print("\n  Evaluating Ollama corrections...")
    bm25_agg_ollama, bm25_pq_ollama = evaluate_retrieval(bm25, ollama_queries, all_qrels)
    tfidf_agg_ollama, tfidf_pq_ollama = evaluate_retrieval(tfidf, ollama_queries, all_qrels)

    results['ollama'] = {
        'bm25': bm25_agg_ollama,
        'tfidf': tfidf_agg_ollama,
        'avg_latency': sum(ollama_latencies) / len(ollama_latencies),
        'corrections': [{'qid': q['qid'], 'original': q['original'], 'corrected': c}
                       for q, c in zip(queries_with_qrels, ollama_corrections)],
        'bm25_per_query': bm25_pq_ollama,
        'tfidf_per_query': tfidf_pq_ollama,
    }
    print(f"\n  Ollama BM25 MRR: {bm25_agg_ollama.get('MRR', 0):.3f}")
    print(f"  Ollama TF-IDF MRR: {tfidf_agg_ollama.get('MRR', 0):.3f}")
    print(f"  Avg latency: {results['ollama']['avg_latency']:.2f}s")

    # === Claude API ===
    if api_key:
        print("\n--- Claude API ---")
        claude_corrections = []
        claude_latencies = []

        for i, q in enumerate(queries_with_qrels):
            corrected, latency = correct_query_llm(q['original'], call_claude, api_key=api_key)
            if corrected is None:
                corrected = q['original']
                print(f"  Query {q['qid']}: Claude failed, using original")
            claude_corrections.append(corrected)
            claude_latencies.append(latency)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(queries_with_qrels)} queries "
                      f"(avg latency: {sum(claude_latencies)/len(claude_latencies):.1f}s)")
            # Rate limiting
            time.sleep(0.5)

        # Evaluate Claude corrections
        claude_queries = []
        for q, corrected in zip(queries_with_qrels, claude_corrections):
            eq = dict(q)
            eq['corrected'] = corrected
            claude_queries.append(eq)

        print("\n  Evaluating Claude corrections...")
        bm25_agg_claude, bm25_pq_claude = evaluate_retrieval(bm25, claude_queries, all_qrels)
        tfidf_agg_claude, tfidf_pq_claude = evaluate_retrieval(tfidf, claude_queries, all_qrels)

        results['claude'] = {
            'bm25': bm25_agg_claude,
            'tfidf': tfidf_agg_claude,
            'avg_latency': sum(claude_latencies) / len(claude_latencies),
            'corrections': [{'qid': q['qid'], 'original': q['original'], 'corrected': c}
                           for q, c in zip(queries_with_qrels, claude_corrections)],
            'bm25_per_query': bm25_pq_claude,
            'tfidf_per_query': tfidf_pq_claude,
        }
        print(f"\n  Claude BM25 MRR: {bm25_agg_claude.get('MRR', 0):.3f}")
        print(f"  Claude TF-IDF MRR: {tfidf_agg_claude.get('MRR', 0):.3f}")
        print(f"  Avg latency: {results['claude']['avg_latency']:.2f}s")
    else:
        print("\n  No ANTHROPIC_API_KEY found, skipping Claude experiments")

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    try:
        import numpy as np
    except ImportError:
        pass

    with open(RESULTS_DIR / 'llm_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\n✓ LLM results saved to {RESULTS_DIR / 'llm_results.json'}")


if __name__ == '__main__':
    main()
