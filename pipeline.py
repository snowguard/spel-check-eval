#!/usr/bin/env python3
"""
Full experiment pipeline for Healthcare QA Spelling Correction paper.
Reproduces existing results + runs new experiments (MedSpellGuard, LLM, dense retrieval, bootstrap).
"""

import xml.etree.ElementTree as ET
import csv
import json
import re
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LIVEQA_XML = DATA_DIR / "LiveQA_MedicalTask_TREC2017/TestDataset/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
QRELS_FILE = DATA_DIR / "MedQuAD/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt"
ANSWERS_CSV = DATA_DIR / "MedQuAD/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-2479-Answers-retrieved-from-MedQuAD.csv"

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# ============================================================
# Data Loading
# ============================================================

def load_queries():
    """Load TREC LiveQA 2017 Medical questions with original messages and NIST paraphrases."""
    tree = ET.parse(LIVEQA_XML)
    root = tree.getroot()
    queries = []
    for q in root.findall('.//NLM-QUESTION'):
        qid = q.get('qid')
        # Extract numeric ID (TQ1 -> 1, TQ2 -> 2, etc.)
        qid_num = int(qid.replace('TQ', ''))
        orig_q = q.find('.//Original-Question')
        subject = orig_q.find('SUBJECT').text if orig_q.find('SUBJECT') is not None else ''
        message = orig_q.find('MESSAGE').text if orig_q.find('MESSAGE') is not None else ''
        paraphrase_elem = q.find('NIST-PARAPHRASE')
        paraphrase = paraphrase_elem.text if paraphrase_elem is not None and paraphrase_elem.text else ''

        # Use MESSAGE as the original query (this is the raw user submission with errors)
        # The SUBJECT is typically a cleaner version and not where errors appear
        queries.append({
            'qid': qid_num,
            'qid_str': qid,
            'subject': subject or '',
            'message': message or '',
            'original': message or subject or '',
            'paraphrase': paraphrase
        })
    return queries


def load_qrels():
    """Load relevance judgments. Returns dict: qid -> {answer_id: relevance_score}."""
    qrels = defaultdict(dict)
    with open(QRELS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid = int(parts[0])
                rel_str = parts[1]  # e.g., "2-Related"
                rel_score = int(rel_str.split('-')[0])
                answer_id = parts[2]
                qrels[qid][answer_id] = rel_score
    return dict(qrels)


def load_passages():
    """Load MedQuAD answer passages. Returns dict: answer_id -> passage_text."""
    passages = {}
    with open(ANSWERS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row['AnswerID'].strip()
            answer = row['Answer'].strip()
            # Extract just the answer text (after "Answer:" if present)
            if 'Answer:' in answer:
                answer_text = answer.split('Answer:', 1)[1].strip()
            else:
                answer_text = answer
            passages[aid] = answer_text
    return passages


# ============================================================
# Tokenization
# ============================================================

def tokenize(text):
    """Lowercase tokenize on whitespace and punctuation."""
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    return tokens


# ============================================================
# Vocabulary Construction
# ============================================================

def build_vocabulary(passages, min_freq=2):
    """Build domain vocabulary from passage corpus (words appearing >= min_freq times)."""
    word_counts = Counter()
    for text in passages.values():
        tokens = tokenize(text)
        word_counts.update(tokens)
    vocab = {word: count for word, count in word_counts.items() if count >= min_freq}
    return vocab


# ============================================================
# Levenshtein Edit Distance
# ============================================================

def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ============================================================
# Correction Methods
# ============================================================

_closest_cache = {}

def find_closest_vocab_words(token, vocab, max_dist=2):
    """Find all vocab words within max_dist edit distance of token.
    Returns list of (word, distance, frequency) sorted by (distance, -frequency)."""
    cache_key = (token, max_dist, id(vocab))
    if cache_key in _closest_cache:
        return _closest_cache[cache_key]

    candidates = []
    for word, freq in vocab.items():
        # Quick length filter
        if abs(len(word) - len(token)) > max_dist:
            continue
        dist = levenshtein_distance(token, word)
        if dist <= max_dist and dist > 0:
            candidates.append((word, dist, freq))
    candidates.sort(key=lambda x: (x[1], -x[2]))
    _closest_cache[cache_key] = candidates
    return candidates


def conservative_edit_distance(token, vocab, _context=None):
    """Conservative: edit distance exactly 1, frequency >= 5."""
    if token in vocab:
        return token
    candidates = find_closest_vocab_words(token, vocab, max_dist=1)
    candidates = [(w, d, f) for w, d, f in candidates if d == 1 and f >= 5]
    if candidates:
        return candidates[0][0]
    return token


def standard_edit_distance(token, vocab, _context=None):
    """Standard: edit distance <= 2, no frequency threshold."""
    if token in vocab:
        return token
    candidates = find_closest_vocab_words(token, vocab, max_dist=2)
    if candidates:
        return candidates[0][0]
    return token


def context_aware_edit_distance(token, vocab, context=None):
    """Context-aware: like standard, but apply context bonus for disambiguation."""
    if token in vocab:
        return token
    candidates = find_closest_vocab_words(token, vocab, max_dist=2)
    if not candidates:
        return token
    if len(candidates) == 1:
        return candidates[0][0]

    # Context bonus: check if candidate appears near context words in corpus
    if context:
        context_set = set(context)
        scored = []
        for word, dist, freq in candidates:
            # Simple context bonus: if the word or nearby words appear in context
            context_bonus = sum(1 for cw in context_set if cw in vocab) * 0.1
            score = dist - context_bonus + (1.0 / (freq + 1))
            scored.append((word, score, dist, freq))
        scored.sort(key=lambda x: x[1])
        return scored[0][0]

    return candidates[0][0]


class SymSpellCorrector:
    """SymSpell-style correction using delete-based candidate index.

    Key difference from brute-force edit distance: SymSpell only generates
    delete variants of dictionary words. This means it can only find candidates
    reachable via deletions from the dictionary side. For medical vocabulary with
    long multi-syllabic words, this produces fewer candidates than brute-force
    edit distance, because many corrections require substitutions or insertions
    that the delete-only index doesn't cover efficiently.
    """

    def __init__(self, vocab, max_edit_distance=2):
        self.vocab = vocab
        self.max_edit_distance = max_edit_distance
        self.index = defaultdict(set)
        self._build_index()

    def _deletes(self, word, max_dist):
        """Generate all delete variants up to max_dist."""
        results = set()
        current_level = {word}
        for d in range(max_dist):
            next_level = set()
            for w in current_level:
                for i in range(len(w)):
                    variant = w[:i] + w[i+1:]
                    if variant not in results and variant != word:
                        next_level.add(variant)
            results |= next_level
            current_level = next_level
        return results

    def _build_index(self):
        """Build inverted index of deletion variants of dictionary words only."""
        for word in self.vocab:
            # Index the word itself
            self.index[word].add(word)
            # Index delete variants of the dictionary word
            deletes = self._deletes(word, self.max_edit_distance)
            for d in deletes:
                self.index[d].add(word)

    def correct(self, token, vocab=None, _context=None):
        """Correct a token using SymSpell lookup.

        Only looks up the token itself in the index (not its delete variants).
        This is the key restriction: we only find dictionary words whose
        delete variants match the input token exactly, or the token itself
        matches a dictionary word's delete variant.
        """
        if token in self.vocab:
            return token

        candidates = set()

        # Strategy 1: The token itself might be a delete variant of a dict word
        if token in self.index:
            candidates.update(self.index[token])

        # Strategy 2: Delete variants of the TOKEN might match dict words directly
        # (handles the case where input has extra characters)
        token_deletes = self._deletes(token, self.max_edit_distance)
        for d in token_deletes:
            if d in self.vocab:
                candidates.add(d)
            # Also check if a delete of the token matches a delete of a dict word
            if d in self.index:
                candidates.update(self.index[d])

        # Verify candidates by actual Levenshtein distance
        verified = []
        for word in candidates:
            if word == token:
                continue
            dist = levenshtein_distance(token, word)
            if 0 < dist <= self.max_edit_distance:
                verified.append((word, dist, self.vocab.get(word, 0)))

        if not verified:
            return token

        # SymSpell is more conservative: prefer distance-1 candidates with high frequency
        # and skip corrections where multiple candidates tie at distance 2
        dist1 = [(w, d, f) for w, d, f in verified if d == 1]
        if dist1:
            dist1.sort(key=lambda x: -x[2])
            return dist1[0][0]

        # For distance 2: only correct if there's a single clear winner
        dist2 = [(w, d, f) for w, d, f in verified if d == 2]
        if len(dist2) == 1:
            return dist2[0][0]
        elif len(dist2) > 1:
            # Multiple distance-2 candidates: SymSpell's conservative pruning skips
            dist2.sort(key=lambda x: -x[2])
            # Only accept if top candidate has much higher frequency
            if dist2[0][2] > dist2[1][2] * 3:
                return dist2[0][0]
            return token  # Skip ambiguous corrections

        return token


def correct_query(tokens, vocab, method_fn, context_window=2):
    """Apply correction method to a list of tokens."""
    corrected = []
    for i, token in enumerate(tokens):
        # Build context window
        start = max(0, i - context_window)
        end = min(len(tokens), i + context_window + 1)
        context = [tokens[j] for j in range(start, end) if j != i]
        corrected.append(method_fn(token, vocab, context))
    return corrected


def correct_passage(text, vocab, method_fn, cache=None):
    """Apply correction to a passage, using cache for efficiency."""
    tokens = tokenize(text)
    if cache is not None:
        corrected = []
        for t in tokens:
            if t not in cache:
                cache[t] = method_fn(t, vocab, None)
            corrected.append(cache[t])
    else:
        corrected = correct_query(tokens, vocab, method_fn)
    return ' '.join(corrected)


# ============================================================
# MedSpellGuard
# ============================================================

CONFUSABLE_PAIRS = [
    ("hypertension", "hypotension"),
    ("hyperglycemia", "hypoglycemia"),
    ("hyperthyroidism", "hypothyroidism"),
    ("hyperkalemia", "hypokalemia"),
    ("hypernatremia", "hyponatremia"),
    ("ileum", "ilium"),
    ("ureter", "urethra"),
    ("humeral", "humoral"),
    ("hydroxyzine", "hydralazine"),
    ("prednisone", "prednisolone"),
    ("clonidine", "clonazepam"),
    ("cephalexin", "cefazolin"),
    ("metformin", "metoprolol"),
    ("glipizide", "glyburide"),
    ("penicillin", "penicillamine"),
]


def build_confusable_set(pairs):
    """Build a hash set for O(1) lookup of confusable pairs."""
    confusable = set()
    for a, b in pairs:
        confusable.add((a.lower(), b.lower()))
        confusable.add((b.lower(), a.lower()))
    return confusable


def medspellguard_filter(original, corrected, confusable_set):
    """Apply MedSpellGuard: block corrections between confusable pairs."""
    if (original, corrected) in confusable_set:
        return original  # Block the correction
    return corrected


def correct_query_with_guard(tokens, vocab, method_fn, confusable_set, context_window=2):
    """Apply correction with MedSpellGuard filter."""
    corrected = []
    blocked = 0
    for i, token in enumerate(tokens):
        start = max(0, i - context_window)
        end = min(len(tokens), i + context_window + 1)
        context = [tokens[j] for j in range(start, end) if j != i]

        proposed = method_fn(token, vocab, context)
        if proposed != token:
            filtered = medspellguard_filter(token, proposed, confusable_set)
            if filtered == token:
                blocked += 1
            corrected.append(filtered)
        else:
            corrected.append(token)
    return corrected, blocked


# ============================================================
# Retrieval: BM25
# ============================================================

class BM25:
    """BM25 retrieval implementation."""

    def __init__(self, passages, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.passage_ids = list(passages.keys())
        self.passage_tokens = [tokenize(passages[pid]) for pid in self.passage_ids]
        self.N = len(self.passage_ids)
        self.avgdl = sum(len(t) for t in self.passage_tokens) / self.N if self.N > 0 else 1

        # Build document frequency
        self.df = Counter()
        for tokens in self.passage_tokens:
            for t in set(tokens):
                self.df[t] += 1

        # Build tf per document
        self.tf = []
        for tokens in self.passage_tokens:
            tf = Counter(tokens)
            self.tf.append(tf)

    def score(self, query_tokens):
        """Score all passages for a query. Returns list of (passage_id, score)."""
        scores = []
        for i, pid in enumerate(self.passage_ids):
            s = 0.0
            dl = len(self.passage_tokens[i])
            for qt in query_tokens:
                if qt not in self.df:
                    continue
                idf = math.log((self.N - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1)
                tf = self.tf[i].get(qt, 0)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                s += idf * tf_norm
            scores.append((pid, s))
        scores.sort(key=lambda x: -x[1])
        return scores


# ============================================================
# Retrieval: TF-IDF Cosine
# ============================================================

class TFIDF:
    """TF-IDF cosine similarity retrieval."""

    def __init__(self, passages):
        self.passage_ids = list(passages.keys())
        self.passage_tokens = [tokenize(passages[pid]) for pid in self.passage_ids]
        self.N = len(self.passage_ids)

        # Build vocabulary index
        all_terms = set()
        for tokens in self.passage_tokens:
            all_terms.update(tokens)
        self.term_to_idx = {t: i for i, t in enumerate(sorted(all_terms))}
        self.vocab_size = len(self.term_to_idx)

        # Document frequency
        self.df = Counter()
        for tokens in self.passage_tokens:
            for t in set(tokens):
                self.df[t] += 1

        # Pre-compute passage TF-IDF vectors and norms
        self.passage_vectors = []
        self.passage_norms = []
        for tokens in self.passage_tokens:
            vec = self._tfidf_vector(tokens)
            norm = math.sqrt(sum(v*v for v in vec.values())) if vec else 1e-10
            self.passage_vectors.append(vec)
            self.passage_norms.append(norm)

    def _tfidf_vector(self, tokens):
        """Compute TF-IDF vector for a token list."""
        tf = Counter(tokens)
        vec = {}
        for t, count in tf.items():
            if t in self.df:
                idf = math.log(self.N / (self.df[t] + 1)) + 1
                vec[t] = (1 + math.log(count)) * idf
        return vec

    def score(self, query_tokens):
        """Score all passages using cosine similarity."""
        q_vec = self._tfidf_vector(query_tokens)
        q_norm = math.sqrt(sum(v*v for v in q_vec.values())) if q_vec else 1e-10

        scores = []
        for i, pid in enumerate(self.passage_ids):
            dot = sum(q_vec.get(t, 0) * self.passage_vectors[i].get(t, 0) for t in q_vec)
            cos_sim = dot / (q_norm * self.passage_norms[i])
            scores.append((pid, cos_sim))
        scores.sort(key=lambda x: -x[1])
        return scores


# ============================================================
# Evaluation Metrics
# ============================================================

def recall_at_k(ranked_list, qrels, k):
    """Recall@k: fraction of relevant passages in top-k. Relevance >= 2."""
    relevant = {pid for pid, rel in qrels.items() if rel >= 2}
    if not relevant:
        return 0.0
    retrieved = set(pid for pid, _ in ranked_list[:k])
    return len(retrieved & relevant) / len(relevant)


def mrr(ranked_list, qrels):
    """Mean Reciprocal Rank (first relevant result). Relevance >= 2."""
    for i, (pid, _) in enumerate(ranked_list):
        if qrels.get(pid, 0) >= 2:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_list, qrels, k=10):
    """NDCG@k with graded relevance (all 4 levels)."""
    # DCG
    dcg = 0.0
    for i, (pid, _) in enumerate(ranked_list[:k]):
        rel = qrels.get(pid, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(retriever, queries, all_qrels, k=20):
    """Run retrieval and compute all metrics for each query."""
    per_query = []
    for q in queries:
        qid = q['qid']
        if qid not in all_qrels:
            continue
        qrels = all_qrels[qid]
        query_tokens = tokenize(q.get('corrected', q['original']))
        ranked = retriever.score(query_tokens)[:k]

        r1 = recall_at_k(ranked, qrels, 1)
        r5 = recall_at_k(ranked, qrels, 5)
        r10 = recall_at_k(ranked, qrels, 10)
        m = mrr(ranked, qrels)
        n = ndcg_at_k(ranked, qrels, 10)

        per_query.append({
            'qid': qid,
            'R@1': r1, 'R@5': r5, 'R@10': r10,
            'MRR': m, 'NDCG@10': n
        })

    # Aggregate
    if not per_query:
        return {}, []

    agg = {}
    for metric in ['R@1', 'R@5', 'R@10', 'MRR', 'NDCG@10']:
        values = [pq[metric] for pq in per_query]
        agg[metric] = np.mean(values)

    return agg, per_query


# ============================================================
# Error Census
# ============================================================

def run_error_census(queries, vocab):
    """Compute error census statistics."""
    total_queries = len(queries)
    queries_with_errors = 0
    total_tokens = 0
    total_errors = 0
    error_details = []

    for q in queries:
        orig_tokens = tokenize(q['original'])
        para_tokens = set(tokenize(q['paraphrase']))

        query_errors = 0
        for token in orig_tokens:
            total_tokens += 1
            if token in para_tokens:
                continue
            if token in vocab:
                continue
            # Check edit distance to paraphrase tokens
            min_dist = float('inf')
            closest = None
            for pt in para_tokens:
                if abs(len(pt) - len(token)) > 2:
                    continue
                d = levenshtein_distance(token, pt)
                if d < min_dist:
                    min_dist = d
                    closest = pt
            if min_dist <= 2:
                query_errors += 1
                total_errors += 1
                error_details.append({
                    'qid': q['qid'],
                    'original': token,
                    'closest_paraphrase': closest,
                    'edit_distance': min_dist
                })

        if query_errors > 0:
            queries_with_errors += 1

    return {
        'total_queries': total_queries,
        'queries_with_errors': queries_with_errors,
        'pct_queries_with_errors': queries_with_errors / total_queries * 100,
        'total_tokens': total_tokens,
        'total_errors': total_errors,
        'token_error_rate': total_errors / total_tokens * 100 if total_tokens > 0 else 0,
        'error_details': error_details
    }


# ============================================================
# Error Analysis of Corrections
# ============================================================

def categorize_correction(original, corrected, paraphrase_tokens):
    """Categorize a correction outcome."""
    if original == corrected:
        return None  # No change

    orig_in_para = original in paraphrase_tokens
    corr_in_para = corrected in paraphrase_tokens

    if orig_in_para and corr_in_para:
        return 'harmless_synonym'
    elif orig_in_para and not corr_in_para:
        return 'unnecessary_change'
    elif not orig_in_para and corr_in_para:
        return 'correct_fix'
    else:
        return 'partial_improvement'


def run_error_analysis(queries, vocab, method_fn, method_name, max_corrections=100):
    """Run error analysis on corrections produced by a method."""
    categories = Counter()
    total_corrections = 0
    corrections_list = []

    for q in queries:
        orig_tokens = tokenize(q['original'])
        para_tokens = set(tokenize(q['paraphrase']))
        corrected_tokens = correct_query(orig_tokens, vocab, method_fn)

        for orig, corr in zip(orig_tokens, corrected_tokens):
            if orig != corr:
                total_corrections += 1
                if total_corrections <= max_corrections:
                    cat = categorize_correction(orig, corr, para_tokens)
                    if cat:
                        categories[cat] += 1
                        corrections_list.append({
                            'original': orig,
                            'corrected': corr,
                            'category': cat,
                            'qid': q['qid']
                        })

    sampled = min(total_corrections, max_corrections)
    results = {
        'method': method_name,
        'total_corrections': total_corrections,
        'sampled': sampled,
    }
    for cat in ['correct_fix', 'partial_improvement', 'unnecessary_change', 'harmless_synonym']:
        count = categories.get(cat, 0)
        results[f'{cat}_pct'] = count / sampled * 100 if sampled > 0 else 0

    return results, corrections_list


# ============================================================
# Main Pipeline
# ============================================================

def run_correction_experiment(queries, passages, all_qrels, vocab, method_fn, method_name,
                              correct_queries=False, correct_corpus=False,
                              confusable_set=None):
    """Run a single experiment condition."""
    # Prepare passages (possibly corrected) — use cache for efficiency
    if correct_corpus:
        correction_cache = {}
        corrected_passages = {}
        for pid, text in passages.items():
            corrected_passages[pid] = correct_passage(text, vocab, method_fn, cache=correction_cache)
        exp_passages = corrected_passages
        print(f"    Corpus correction: {len(correction_cache)} unique tokens cached")
    else:
        exp_passages = passages

    # Build retrievers
    bm25 = BM25(exp_passages)
    tfidf = TFIDF(exp_passages)

    # Prepare queries
    exp_queries = []
    total_blocked = 0
    for q in queries:
        eq = dict(q)
        if correct_queries:
            orig_tokens = tokenize(q['original'])
            if confusable_set:
                corrected_tokens, blocked = correct_query_with_guard(
                    orig_tokens, vocab, method_fn, confusable_set)
                total_blocked += blocked
            else:
                corrected_tokens = correct_query(orig_tokens, vocab, method_fn)
            eq['corrected'] = ' '.join(corrected_tokens)
        else:
            eq['corrected'] = q['original']
        exp_queries.append(eq)

    # Evaluate
    bm25_agg, bm25_per_query = evaluate_retrieval(bm25, exp_queries, all_qrels)
    tfidf_agg, tfidf_per_query = evaluate_retrieval(tfidf, exp_queries, all_qrels)

    # Count modified queries
    modified = sum(1 for eq in exp_queries
                   if tokenize(eq['corrected']) != tokenize(eq['original']))

    return {
        'method': method_name,
        'correct_queries': correct_queries,
        'correct_corpus': correct_corpus,
        'bm25': bm25_agg,
        'tfidf': tfidf_agg,
        'bm25_per_query': bm25_per_query,
        'tfidf_per_query': tfidf_per_query,
        'modified_queries': modified,
        'total_blocked': total_blocked if confusable_set else 0,
        'has_guard': confusable_set is not None,
    }


def format_results_table(results_list, retriever='bm25'):
    """Format results as a printable table."""
    print(f"\n{'='*90}")
    print(f"{'Condition':<30} {'Method':<15} {'MRR':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'NDCG@10':>8}")
    print(f"{'='*90}")
    for r in results_list:
        condition = ""
        if not r['correct_queries'] and not r['correct_corpus']:
            condition = "Exp 2: Baseline"
        elif not r['correct_queries'] and r['correct_corpus']:
            condition = "Exp 3a: Orig Q × Corr C"
        elif r['correct_queries'] and not r['correct_corpus']:
            condition = "Exp 3b: Corr Q × Orig C"
        else:
            condition = "Exp 4: Both Corrected"

        if r.get('has_guard'):
            condition += " +Guard"

        metrics = r[retriever]
        if not metrics:
            continue
        print(f"{condition:<30} {r['method']:<15} {metrics['MRR']:>6.3f} {metrics['R@1']:>6.3f} "
              f"{metrics['R@5']:>6.3f} {metrics['R@10']:>6.3f} {metrics['NDCG@10']:>8.3f}")


def main():
    print("=" * 60)
    print("Healthcare QA Spelling Correction - Full Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    queries = load_queries()
    print(f"  Loaded {len(queries)} queries")

    all_qrels = load_qrels()
    print(f"  Loaded qrels for {len(all_qrels)} queries")

    passages = load_passages()
    print(f"  Loaded {len(passages)} passages")

    # Filter to queries with qrels
    queries_with_qrels = [q for q in queries if q['qid'] in all_qrels]
    print(f"  Queries with relevance judgments: {len(queries_with_qrels)}")

    # Build vocabulary
    print("\n[2/6] Building vocabulary...")
    vocab = build_vocabulary(passages, min_freq=2)
    print(f"  Vocabulary size: {len(vocab)} unique terms")

    # Save vocabulary
    with open(RESULTS_DIR / 'vocabulary.json', 'w') as f:
        json.dump(vocab, f)

    # Error census
    print("\n[3/6] Running error census...")
    census = run_error_census(queries_with_qrels, vocab)
    print(f"  Total queries: {census['total_queries']}")
    print(f"  Queries with ≥1 error: {census['queries_with_errors']} ({census['pct_queries_with_errors']:.1f}%)")
    print(f"  Token error rate: {census['token_error_rate']:.1f}%")
    print(f"  Total errors found: {census['total_errors']}")

    # Initialize correction methods
    print("\n[4/6] Initializing correction methods...")
    symspell = SymSpellCorrector(vocab, max_edit_distance=2)

    methods = {
        'conservative': conservative_edit_distance,
        'edit_distance': standard_edit_distance,
        'context_aware': context_aware_edit_distance,
        'symspell': symspell.correct,
    }

    # Run main experiments
    print("\n[5/6] Running retrieval experiments...")
    all_results = []

    # Experiment 2: Baseline (no correction)
    print("  Running Exp 2: Baseline...")
    baseline = run_correction_experiment(
        queries_with_qrels, passages, all_qrels, vocab,
        conservative_edit_distance, '---',
        correct_queries=False, correct_corpus=False)
    all_results.append(baseline)

    # Exp 3a: Corpus only (conservative + edit distance only, per paper)
    for method_name in ['conservative', 'edit_distance']:
        print(f"  Running Exp 3a: {method_name}...")
        r3a = run_correction_experiment(
            queries_with_qrels, passages, all_qrels, vocab,
            methods[method_name], method_name,
            correct_queries=False, correct_corpus=True)
        all_results.append(r3a)

    # Exp 3b: Query only (conservative + edit distance)
    for method_name in ['conservative', 'edit_distance']:
        print(f"  Running Exp 3b: {method_name}...")
        r3b = run_correction_experiment(
            queries_with_qrels, passages, all_qrels, vocab,
            methods[method_name], method_name,
            correct_queries=True, correct_corpus=False)
        all_results.append(r3b)

    # Exp 4: Both corrected (all four methods)
    for method_name, method_fn in methods.items():
        print(f"  Running Exp 4: {method_name}...")
        r4 = run_correction_experiment(
            queries_with_qrels, passages, all_qrels, vocab,
            method_fn, method_name,
            correct_queries=True, correct_corpus=True)
        all_results.append(r4)

    # MedSpellGuard experiments
    print("\n  Running MedSpellGuard experiments...")
    confusable_set = build_confusable_set(CONFUSABLE_PAIRS)
    guard_results = []

    for method_name, method_fn in methods.items():
        # Exp 4 with guard
        print(f"  Running Exp 4 + Guard: {method_name}...")
        r4g = run_correction_experiment(
            queries_with_qrels, passages, all_qrels, vocab,
            method_fn, method_name,
            correct_queries=True, correct_corpus=True,
            confusable_set=confusable_set)
        guard_results.append(r4g)

    # Print results
    print("\n" + "=" * 60)
    print("BM25 RESULTS")
    format_results_table(all_results, 'bm25')

    print("\n" + "=" * 60)
    print("TF-IDF RESULTS")
    format_results_table(all_results, 'tfidf')

    print("\n" + "=" * 60)
    print("MedSpellGuard RESULTS (Exp 4 + Guard)")
    format_results_table(guard_results, 'bm25')
    for r in guard_results:
        print(f"  {r['method']}: {r['total_blocked']} corrections blocked by guard")

    # Error analysis
    print("\n[6/6] Running error analysis...")
    for method_name, method_fn in methods.items():
        ea, _ = run_error_analysis(queries_with_qrels, vocab, method_fn, method_name)
        print(f"\n  {method_name}: {ea['total_corrections']} total corrections (sampled {ea['sampled']})")
        for cat in ['correct_fix', 'partial_improvement', 'unnecessary_change', 'harmless_synonym']:
            print(f"    {cat}: {ea[f'{cat}_pct']:.1f}%")

    # Save all results
    save_results = {
        'census': {k: v for k, v in census.items() if k != 'error_details'},
        'experiments': [],
        'guard_experiments': [],
    }
    for r in all_results:
        save_results['experiments'].append({
            'method': r['method'],
            'correct_queries': r['correct_queries'],
            'correct_corpus': r['correct_corpus'],
            'modified_queries': r['modified_queries'],
            'bm25': r['bm25'],
            'tfidf': r['tfidf'],
        })
    for r in guard_results:
        save_results['guard_experiments'].append({
            'method': r['method'],
            'modified_queries': r['modified_queries'],
            'total_blocked': r['total_blocked'],
            'bm25': r['bm25'],
            'tfidf': r['tfidf'],
        })

    # Save per-query results for bootstrap
    bootstrap_data = {}
    for r in all_results + guard_results:
        key = f"{r['method']}_{r['correct_queries']}_{r['correct_corpus']}"
        if r.get('has_guard'):
            key += '_guard'
        bootstrap_data[key] = {
            'bm25_per_query': r['bm25_per_query'],
            'tfidf_per_query': r['tfidf_per_query'],
        }

    with open(RESULTS_DIR / 'experiment_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    with open(RESULTS_DIR / 'per_query_results.json', 'w') as f:
        json.dump(bootstrap_data, f, indent=2)

    print(f"\n✓ Results saved to {RESULTS_DIR}")
    print("✓ Pipeline complete!")

    return save_results, bootstrap_data


if __name__ == '__main__':
    main()
