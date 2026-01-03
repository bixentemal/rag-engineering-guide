# RAG Evaluation & Metrics

**Category**: Foundational Concepts
**Impact**: Critical - You can't improve what you can't measure
**Audience**: Engineers evaluating and improving RAG systems

---

## Overview

Evaluating RAG systems is challenging because there are two components to assess:
1. **Retrieval Quality**: Did we find the right documents?
2. **Generation Quality**: Did we generate the right answer?

Plus their interaction: great retrieval + poor generation = bad answers, and vice versa.

This document covers:
- Retrieval metrics (Recall, Precision, MRR, NDCG)
- Generation metrics (Faithfulness, Relevance, Correctness)
- End-to-end evaluation frameworks (RAGAS, ARES)
- Practical evaluation strategies

---

## Quick Reference: Metrics Overview

### Retrieval Metrics

| Metric | Question Answered | Range | Higher is Better |
|--------|------------------|-------|------------------|
| **Recall@k** | Did we retrieve the relevant docs? | [0, 1] | Yes |
| **Precision@k** | What fraction retrieved are relevant? | [0, 1] | Yes |
| **MRR** | How high is the first relevant result? | [0, 1] | Yes |
| **NDCG@k** | Are relevant docs ranked correctly? | [0, 1] | Yes |
| **Hit Rate@k** | Did we get at least one relevant doc? | [0, 1] | Yes |

### Generation Metrics

| Metric | Question Answered | Range | Higher is Better |
|--------|------------------|-------|------------------|
| **Faithfulness** | Is the answer supported by context? | [0, 1] | Yes |
| **Answer Relevance** | Does the answer address the question? | [0, 1] | Yes |
| **Context Relevance** | Is the retrieved context relevant? | [0, 1] | Yes |
| **Correctness** | Is the answer factually correct? | [0, 1] | Yes |

---

## Retrieval Metrics in Detail

### Recall@k

**The most important retrieval metric for RAG.**

"Out of all relevant documents, what fraction did we retrieve in top-k?"

```python
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@k: Fraction of relevant documents retrieved in top-k.

    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: List of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Recall score in [0, 1]
    """
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)

    if len(relevant_set) == 0:
        return 0.0

    hits = len(retrieved_at_k & relevant_set)
    return hits / len(relevant_set)


# Example
retrieved = ["doc1", "doc3", "doc5", "doc2", "doc7"]
relevant = ["doc1", "doc2", "doc4"]  # Ground truth

print(recall_at_k(retrieved, relevant, k=3))  # 0.33 (found doc1)
print(recall_at_k(retrieved, relevant, k=5))  # 0.67 (found doc1, doc2)
```

**Interpretation**:
- Recall@5 = 0.67 means we retrieved 2 out of 3 relevant documents in top 5
- Recall@20 is commonly used in RAG (Anthropic reports this metric)
- Low recall = missing relevant information

---

### Precision@k

"Out of retrieved documents, what fraction are relevant?"

```python
def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Precision@k: Fraction of top-k results that are relevant.

    High precision = less noise in context.
    """
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)

    if k == 0:
        return 0.0

    hits = len(retrieved_at_k & relevant_set)
    return hits / k


# Example
retrieved = ["doc1", "doc3", "doc5", "doc2", "doc7"]
relevant = ["doc1", "doc2", "doc4"]

print(precision_at_k(retrieved, relevant, k=3))  # 0.33 (1 relevant in top 3)
print(precision_at_k(retrieved, relevant, k=5))  # 0.40 (2 relevant in top 5)
```

**Interpretation**:
- Precision@5 = 0.40 means 40% of top-5 results are relevant
- Important when context window is limited
- Low precision = noise dilutes relevant information

---

### MRR (Mean Reciprocal Rank)

"On average, how high is the first relevant result?"

```python
def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Reciprocal Rank for a single query.

    Returns 1/rank of first relevant document (0 if none found).
    """
    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]]
) -> float:
    """
    MRR: Average reciprocal rank across all queries.
    """
    rr_sum = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    )
    return rr_sum / len(all_retrieved)


# Example
retrieved = ["doc3", "doc1", "doc5"]  # doc1 is relevant, at position 2
relevant = ["doc1", "doc2"]

print(reciprocal_rank(retrieved, relevant))  # 0.5 (1/2)
```

**Interpretation**:
- MRR = 1.0 means the relevant doc is always first
- MRR = 0.5 means it's typically second
- Good for "I only need one good result" scenarios

---

### NDCG@k (Normalized Discounted Cumulative Gain)

"Are relevant documents ranked in the right order?"

```python
import numpy as np

def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain at k.

    Gives more credit to relevant docs at higher ranks.
    """
    relevance_scores = np.array(relevance_scores[:k])
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return np.sum(relevance_scores / discounts)


def ndcg_at_k(
    retrieved: List[str],
    relevance: Dict[str, float],  # doc_id -> relevance score
    k: int
) -> float:
    """
    Normalized DCG: DCG / Ideal DCG

    Args:
        retrieved: Ranked list of retrieved doc IDs
        relevance: Mapping of doc_id -> relevance score (0, 1, 2, etc.)
        k: Cutoff

    Returns:
        NDCG score in [0, 1]
    """
    # Get relevance scores for retrieved docs
    retrieved_relevance = [relevance.get(doc, 0) for doc in retrieved[:k]]

    # Ideal ranking (all relevant docs first, sorted by relevance)
    ideal_relevance = sorted(relevance.values(), reverse=True)[:k]

    dcg = dcg_at_k(retrieved_relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# Example with graded relevance
retrieved = ["doc1", "doc3", "doc2", "doc5", "doc4"]
relevance = {
    "doc1": 3,  # Highly relevant
    "doc2": 2,  # Relevant
    "doc3": 0,  # Not relevant
    "doc4": 1,  # Somewhat relevant
    "doc5": 0,  # Not relevant
}

print(ndcg_at_k(retrieved, relevance, k=5))  # ~0.85
```

**Interpretation**:
- NDCG@k = 1.0 means perfect ranking
- Accounts for graded relevance (not just binary)
- Standard benchmark metric (BEIR, MTEB use this)

---

### Hit Rate@k

"Did we retrieve at least one relevant document in top-k?"

```python
def hit_rate_at_k(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]],
    k: int
) -> float:
    """
    Fraction of queries where at least one relevant doc is in top-k.
    """
    hits = 0
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        relevant_set = set(relevant)
        if any(doc in relevant_set for doc in retrieved[:k]):
            hits += 1

    return hits / len(all_retrieved)
```

**Interpretation**:
- Simpler than Recall@k
- Useful when you only need one good document
- Hit Rate@10 = 0.95 means 95% of queries get at least one relevant doc in top 10

---

## Generation Metrics

### Faithfulness (Groundedness)

"Is the generated answer supported by the retrieved context?"

This is the **most critical RAG metric**—it measures hallucination.

```python
def evaluate_faithfulness_llm(
    question: str,
    answer: str,
    context: str,
    evaluator_llm: LanguageModel
) -> float:
    """
    Use LLM to evaluate if answer is faithful to context.

    Returns score in [0, 1].
    """
    prompt = f"""You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Evaluate faithfulness:
1. Is every claim in the answer supported by the context?
2. Does the answer avoid adding information not in the context?

Score from 0 to 1:
- 1.0: Completely faithful, all claims supported
- 0.5: Partially faithful, some claims unsupported
- 0.0: Unfaithful, makes claims not in context

Provide your score as a single number:"""

    response = evaluator_llm.generate(prompt)
    return float(response.strip())
```

#### Claim-Level Faithfulness

More rigorous approach: decompose answer into claims and verify each.

```python
def claim_level_faithfulness(
    answer: str,
    context: str,
    llm: LanguageModel
) -> dict:
    """
    Decompose answer into claims and verify each against context.
    """
    # Step 1: Extract claims
    claims_prompt = f"""Extract all factual claims from this answer as a list:

Answer: {answer}

Claims (one per line):"""

    claims_response = llm.generate(claims_prompt)
    claims = [c.strip() for c in claims_response.split('\n') if c.strip()]

    # Step 2: Verify each claim
    verified_claims = 0
    claim_results = []

    for claim in claims:
        verify_prompt = f"""Is this claim supported by the context?

Context: {context}

Claim: {claim}

Answer only "yes" or "no":"""

        is_supported = llm.generate(verify_prompt).strip().lower() == "yes"
        claim_results.append({"claim": claim, "supported": is_supported})
        if is_supported:
            verified_claims += 1

    return {
        "faithfulness_score": verified_claims / len(claims) if claims else 1.0,
        "total_claims": len(claims),
        "supported_claims": verified_claims,
        "claim_details": claim_results
    }
```

---

### Answer Relevance

"Does the answer actually address the question?"

```python
def evaluate_answer_relevance(
    question: str,
    answer: str,
    evaluator_llm: LanguageModel
) -> float:
    """
    Evaluate if the answer is relevant to the question.
    """
    prompt = f"""Evaluate if this answer is relevant to the question.

Question: {question}

Answer: {answer}

Consider:
1. Does the answer address what was asked?
2. Is the answer complete?
3. Is the answer focused (not off-topic)?

Score from 0 to 1:
- 1.0: Perfectly relevant and complete
- 0.5: Partially relevant or incomplete
- 0.0: Not relevant to the question

Score:"""

    response = evaluator_llm.generate(prompt)
    return float(response.strip())
```

---

### Context Relevance

"Is the retrieved context relevant to the question?"

```python
def evaluate_context_relevance(
    question: str,
    context: str,
    evaluator_llm: LanguageModel
) -> float:
    """
    Evaluate if retrieved context is relevant to the question.

    Low context relevance = retrieval problem.
    """
    prompt = f"""Evaluate if this context contains information relevant to answering the question.

Question: {question}

Context:
{context}

Score from 0 to 1:
- 1.0: Context contains all needed information
- 0.5: Context is partially relevant
- 0.0: Context is not relevant

Score:"""

    response = evaluator_llm.generate(prompt)
    return float(response.strip())
```

---

### Correctness (with Ground Truth)

"Is the answer factually correct?"

```python
def evaluate_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    evaluator_llm: LanguageModel
) -> float:
    """
    Compare generated answer to ground truth.

    Requires labeled test data.
    """
    prompt = f"""Compare the generated answer to the ground truth.

Question: {question}

Generated Answer: {answer}

Ground Truth: {ground_truth}

Evaluate correctness:
- Are the key facts the same?
- Is the meaning equivalent (even if wording differs)?

Score from 0 to 1:
- 1.0: Correct, matches ground truth
- 0.5: Partially correct
- 0.0: Incorrect

Score:"""

    response = evaluator_llm.generate(prompt)
    return float(response.strip())
```

---

## End-to-End Evaluation Frameworks

### RAGAS (RAG Assessment)

RAGAS is the most popular RAG evaluation framework, computing four key metrics.

```python
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str]
) -> dict:
    """
    Evaluate RAG system using RAGAS framework.

    Returns scores for:
    - faithfulness: Is answer grounded in context?
    - answer_relevancy: Does answer address question?
    - context_precision: Is context relevant?
    - context_recall: Did we retrieve needed context?
    """
    # Prepare dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    return result
```

#### RAGAS Metrics Explained

| Metric | What It Measures | How It's Computed |
|--------|------------------|-------------------|
| **Faithfulness** | Hallucination | Claims in answer supported by context |
| **Answer Relevancy** | Answer quality | Generated questions from answer → similarity to original |
| **Context Precision** | Retrieval precision | Relevant items in context / total context |
| **Context Recall** | Retrieval recall | Claims in ground truth present in context |

---

### ARES (Automated RAG Evaluation System)

ARES trains lightweight judges for evaluation without LLM calls per query.

```python
# pip install ares-ai

from ares import ARES

def evaluate_with_ares(
    test_data: List[dict],  # [{question, answer, context, label}]
    model_path: str = "ares-judge-v1"
) -> dict:
    """
    ARES: Train judges on small labeled set, then evaluate at scale.

    More efficient than per-query LLM evaluation.
    """
    ares = ARES(model_path)

    results = ares.evaluate(
        test_data,
        metrics=["context_relevance", "answer_faithfulness", "answer_relevance"]
    )

    return results
```

---

### Custom Evaluation Pipeline

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class RAGEvalResult:
    """Complete evaluation result for a RAG query."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]

    # Retrieval metrics
    recall_at_k: float
    precision_at_k: float
    mrr: float

    # Generation metrics
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    correctness: Optional[float]


class RAGEvaluator:
    """Comprehensive RAG evaluation pipeline."""

    def __init__(
        self,
        evaluator_llm: LanguageModel,
        embedding_model = None
    ):
        self.evaluator_llm = evaluator_llm
        self.embedding_model = embedding_model

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        retrieved_ids: List[str],
        relevant_ids: List[str],
        ground_truth: Optional[str] = None,
        k: int = 5
    ) -> RAGEvalResult:
        """Evaluate a single RAG query."""

        # Retrieval metrics
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        mrr = reciprocal_rank(retrieved_ids, relevant_ids)

        # Generation metrics
        context_str = "\n\n".join(contexts)

        faithfulness = self._evaluate_faithfulness(question, answer, context_str)
        answer_rel = self._evaluate_answer_relevance(question, answer)
        context_rel = self._evaluate_context_relevance(question, context_str)

        correctness = None
        if ground_truth:
            correctness = self._evaluate_correctness(question, answer, ground_truth)

        return RAGEvalResult(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            recall_at_k=recall,
            precision_at_k=precision,
            mrr=mrr,
            faithfulness=faithfulness,
            answer_relevance=answer_rel,
            context_relevance=context_rel,
            correctness=correctness
        )

    def evaluate_batch(
        self,
        test_cases: List[dict]
    ) -> Dict[str, float]:
        """Evaluate multiple queries and aggregate metrics."""
        results = []

        for case in test_cases:
            result = self.evaluate_single(**case)
            results.append(result)

        # Aggregate
        return {
            "avg_recall@k": np.mean([r.recall_at_k for r in results]),
            "avg_precision@k": np.mean([r.precision_at_k for r in results]),
            "avg_mrr": np.mean([r.mrr for r in results]),
            "avg_faithfulness": np.mean([r.faithfulness for r in results]),
            "avg_answer_relevance": np.mean([r.answer_relevance for r in results]),
            "avg_context_relevance": np.mean([r.context_relevance for r in results]),
            "avg_correctness": np.mean([r.correctness for r in results if r.correctness]),
            "num_evaluated": len(results)
        }

    # ... implement _evaluate_* methods as shown above
```

---

## Practical Evaluation Strategy

### Building a Test Set

```python
def create_test_set(
    corpus: List[str],
    llm: LanguageModel,
    num_samples: int = 100
) -> List[dict]:
    """
    Generate synthetic test set from corpus.

    For each document:
    1. Generate question
    2. Generate answer (ground truth)
    3. Mark document as relevant
    """
    test_cases = []

    for doc_id, doc in enumerate(random.sample(corpus, num_samples)):
        # Generate question
        q_prompt = f"""Generate a specific question that can be answered by this document:

Document: {doc[:2000]}

Question:"""
        question = llm.generate(q_prompt)

        # Generate ground truth answer
        a_prompt = f"""Answer this question based on the document:

Document: {doc[:2000]}

Question: {question}

Answer:"""
        ground_truth = llm.generate(a_prompt)

        test_cases.append({
            "question": question,
            "ground_truth": ground_truth,
            "relevant_ids": [doc_id],
            "source_doc": doc
        })

    return test_cases
```

### A/B Testing RAG Configurations

```python
def ab_test_rag_configs(
    config_a: dict,  # e.g., {"chunk_size": 512, "top_k": 5}
    config_b: dict,  # e.g., {"chunk_size": 1024, "top_k": 10}
    test_set: List[dict],
    evaluator: RAGEvaluator
) -> dict:
    """
    Compare two RAG configurations.
    """
    # Build RAG systems
    rag_a = build_rag_system(**config_a)
    rag_b = build_rag_system(**config_b)

    results_a = []
    results_b = []

    for test_case in test_set:
        question = test_case["question"]

        # Get answers from both systems
        answer_a, contexts_a, retrieved_a = rag_a.query(question)
        answer_b, contexts_b, retrieved_b = rag_b.query(question)

        # Evaluate both
        eval_a = evaluator.evaluate_single(
            question=question,
            answer=answer_a,
            contexts=contexts_a,
            retrieved_ids=retrieved_a,
            relevant_ids=test_case["relevant_ids"],
            ground_truth=test_case["ground_truth"]
        )

        eval_b = evaluator.evaluate_single(
            question=question,
            answer=answer_b,
            contexts=contexts_b,
            retrieved_ids=retrieved_b,
            relevant_ids=test_case["relevant_ids"],
            ground_truth=test_case["ground_truth"]
        )

        results_a.append(eval_a)
        results_b.append(eval_b)

    # Compare
    return {
        "config_a": {
            "settings": config_a,
            "avg_faithfulness": np.mean([r.faithfulness for r in results_a]),
            "avg_recall": np.mean([r.recall_at_k for r in results_a]),
        },
        "config_b": {
            "settings": config_b,
            "avg_faithfulness": np.mean([r.faithfulness for r in results_b]),
            "avg_recall": np.mean([r.recall_at_k for r in results_b]),
        },
        "winner": "A" if ... else "B"
    }
```

---

## Metric Selection Guide

### By Use Case

| Use Case | Primary Metrics | Why |
|----------|----------------|-----|
| **General Q&A** | Faithfulness, Answer Relevance | Balance of grounding and helpfulness |
| **Factual lookup** | Recall@k, Correctness | Finding and verifying facts |
| **Research assistant** | Context Relevance, NDCG | Quality of retrieved sources |
| **Customer support** | Faithfulness, Answer Relevance | No hallucination, helpful answers |
| **High-stakes (legal/medical)** | Faithfulness (primary), Claim-level verification | Cannot hallucinate |

### Minimum Acceptable Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Recall@10 | 0.70 | 0.85 | 0.95 |
| Faithfulness | 0.80 | 0.90 | 0.98 |
| Answer Relevance | 0.70 | 0.85 | 0.95 |
| Context Relevance | 0.60 | 0.80 | 0.90 |

---

## Common Evaluation Mistakes

### 1. Only Measuring Retrieval OR Generation

```python
# WRONG: Only checking if retrieval "feels" good
if retrieved_docs:
    print("Retrieval working!")  # No actual measurement

# RIGHT: Measure both components
retrieval_metrics = evaluate_retrieval(retrieved, relevant)
generation_metrics = evaluate_generation(answer, context)
```

### 2. Using the Same LLM for Generation and Evaluation

```python
# PROBLEMATIC: Same model may not catch its own errors
answer = gpt4.generate(question, context)
score = gpt4.evaluate(answer)  # May be biased

# BETTER: Use different model or multiple evaluators
answer = gpt4.generate(question, context)
score = claude.evaluate(answer)  # Cross-model evaluation
```

### 3. Not Testing Edge Cases

```python
# Include in test set:
edge_cases = [
    {"type": "no_answer", "question": "Question with no answer in corpus"},
    {"type": "contradictory", "question": "Question where sources disagree"},
    {"type": "multi_hop", "question": "Question requiring multiple docs"},
    {"type": "out_of_scope", "question": "Question outside corpus scope"},
]
```

---

## References

1. **RAGAS**: [ragas.io](https://ragas.io) - RAG Assessment framework
2. **ARES**: [arxiv.org/abs/2311.09476](https://arxiv.org/abs/2311.09476) - Automated RAG Evaluation
3. **BEIR Benchmark**: [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir) - Retrieval benchmark
4. **MTEB**: [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding benchmark
5. Es, S., et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." [arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)
