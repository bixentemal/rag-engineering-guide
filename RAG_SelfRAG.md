# Self-RAG (Self-Reflective Retrieval Augmented Generation)

**Category**: Generator-centric
**Maturity**: Research
**Primary Source**: Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024 Oral (Top 1%)*. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

---

## Overview

Self-RAG represents a paradigm shift in RAG architecture: instead of treating retrieval and generation as separate systems, Self-RAG trains a single LLM that adaptively decides when to retrieve, evaluates retrieved passages, and critiques its own outputs—all through special **reflection tokens** generated as part of the model's output.

The key innovation is that the LLM learns to generate four types of reflection tokens that control the RAG process:
1. **[Retrieve]**: Should I retrieve information now?
2. **[ISREL]**: Is this retrieved passage relevant?
3. **[ISSUP]**: Is my output supported by the passage?
4. **[ISUSE]**: Is my output useful overall?

This self-critique mechanism dramatically reduces hallucination rates. In experiments, Self-RAG generates only 2% of correct predictions from outside provided passages, compared to 15-20% for other models (indicating reliance on potentially incorrect parametric knowledge).

---

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Training ["Training Phase (Offline)"]
        TRAIN_DATA[Training Corpus] --> CRITIC[Critic Model<br/>GPT-4 Labels]
        CRITIC --> ANNOTATED[Annotated Corpus<br/>with Reflection Tokens]
        ANNOTATED --> FINETUNE[Fine-tune Generator<br/>LLaMA-2 7B/13B]
    end

    subgraph Inference ["Inference Phase"]
        Q[Query] --> GEN[Generator LLM]

        GEN --> RET_TOK{[Retrieve]?}
        RET_TOK -->|Yes| RETRIEVER[Retrieve Passages]
        RET_TOK -->|No| DIRECT[Generate Directly]
        RET_TOK -->|Continue| CONTINUE[Continue Current]

        RETRIEVER --> PASSAGES[Top-K Passages]
        PASSAGES --> PARALLEL[Parallel Generation<br/>per Passage]

        PARALLEL --> REL_TOK[Generate [ISREL]]
        REL_TOK --> SEGMENTS[Segment Outputs]
        SEGMENTS --> SUP_TOK[Generate [ISSUP]]
        SUP_TOK --> USE_TOK[Generate [ISUSE]]

        USE_TOK --> BEAM[Segment-wise<br/>Beam Search]
        BEAM --> BEST[Best Output]

        DIRECT --> BEST
    end

    BEST --> RESPONSE[Final Response]
```

---

## How It Works

### Reflection Tokens

| Token | Question | Values | Purpose |
|-------|----------|--------|---------|
| **[Retrieve]** | Should I retrieve? | Yes, No, Continue | Controls when to retrieve |
| **[ISREL]** | Is passage relevant? | Relevant, Irrelevant | Filters retrieved content |
| **[ISSUP]** | Is output supported? | Fully, Partially, No Support | Ensures groundedness |
| **[ISUSE]** | Is output useful? | 5, 4, 3, 2, 1 | Overall quality assessment |

### Training Process

1. **Critic Training**: Train a critic model (using GPT-4 annotations) to generate reflection tokens
2. **Corpus Annotation**: Use critic to annotate training corpus with reflection tokens
3. **Generator Training**: Fine-tune LLM on corpus with interleaved reflection tokens

At inference, the generator produces reflection tokens itself—no separate critic needed.

### Inference with Segment-wise Beam Search

For each segment of generation:
1. Generate [Retrieve] token; if "Yes", retrieve passages
2. For each passage, generate output with [ISREL], [ISSUP], [ISUSE] tokens
3. Score candidates using reflection token probabilities
4. Select best segment via beam search
5. Repeat until complete

```
Score = w_rel * P(ISREL=Relevant) +
        w_sup * P(ISSUP=Fully) +
        w_use * P(ISUSE=5)
```

---

## Implementation

### Reflection Token Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple

class SelfRAGGenerator:
    """
    Self-RAG model that generates reflection tokens for self-critique.
    Based on fine-tuned LLaMA-2.
    """

    # Special token vocabulary
    RETRIEVE_TOKENS = ["[Retrieve=Yes]", "[Retrieve=No]", "[Retrieve=Continue]"]
    ISREL_TOKENS = ["[ISREL=Relevant]", "[ISREL=Irrelevant]"]
    ISSUP_TOKENS = ["[ISSUP=Fully]", "[ISSUP=Partially]", "[ISSUP=No]"]
    ISUSE_TOKENS = ["[ISUSE=5]", "[ISUSE=4]", "[ISUSE=3]", "[ISUSE=2]", "[ISUSE=1]"]

    def __init__(self, model_path: str = "selfrag/selfrag_llama2_7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Add special tokens
        special_tokens = (self.RETRIEVE_TOKENS + self.ISREL_TOKENS +
                         self.ISSUP_TOKENS + self.ISUSE_TOKENS)
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def should_retrieve(self, context: str) -> Tuple[bool, float]:
        """
        Generate [Retrieve] token to decide if retrieval is needed.

        Returns:
            should_retrieve: Boolean decision
            confidence: Probability of the decision
        """
        prompt = f"{context}[Retrieve="
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Get logits for next token
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]

        # Get probabilities for retrieve tokens
        yes_prob = self._get_token_prob(logits, "[Retrieve=Yes]")
        no_prob = self._get_token_prob(logits, "[Retrieve=No]")

        should_retrieve = yes_prob > no_prob
        confidence = max(yes_prob, no_prob) / (yes_prob + no_prob)

        return should_retrieve, confidence

    def evaluate_relevance(self, query: str, passage: str) -> Tuple[bool, float]:
        """
        Generate [ISREL] token to evaluate passage relevance.
        """
        prompt = f"Query: {query}\nPassage: {passage}\n[ISREL="
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]

        rel_prob = self._get_token_prob(logits, "[ISREL=Relevant]")
        irrel_prob = self._get_token_prob(logits, "[ISREL=Irrelevant]")

        is_relevant = rel_prob > irrel_prob
        confidence = rel_prob / (rel_prob + irrel_prob)

        return is_relevant, confidence

    def evaluate_support(self, passage: str, output: str) -> Tuple[str, float]:
        """
        Generate [ISSUP] token to check if output is supported.
        """
        prompt = f"Passage: {passage}\nOutput: {output}\n[ISSUP="
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]

        probs = {
            "Fully": self._get_token_prob(logits, "[ISSUP=Fully]"),
            "Partially": self._get_token_prob(logits, "[ISSUP=Partially]"),
            "No": self._get_token_prob(logits, "[ISSUP=No]")
        }

        support_level = max(probs, key=probs.get)
        confidence = probs[support_level] / sum(probs.values())

        return support_level, confidence

    def _get_token_prob(self, logits, token: str) -> float:
        """Get probability for a specific token."""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return torch.softmax(logits, dim=-1)[token_id].item()
```

### Full Self-RAG Pipeline

```python
from typing import List, Dict

def selfrag_generate(
    query: str,
    generator: SelfRAGGenerator,
    retriever: Retriever,
    max_segments: int = 5,
    beam_width: int = 3,
    top_k_passages: int = 5
) -> str:
    """
    Self-RAG generation with reflection-guided retrieval and critique.

    Steps:
    1. For each segment, decide whether to retrieve
    2. If retrieving, generate candidates per passage
    3. Score candidates using reflection tokens
    4. Select best via beam search
    """
    context = f"Question: {query}\nAnswer:"
    generated_segments = []

    for segment_idx in range(max_segments):
        # Step 1: Should we retrieve?
        should_retrieve, ret_conf = generator.should_retrieve(context)

        if should_retrieve:
            # Step 2: Retrieve passages
            passages = retriever.retrieve(query, top_k=top_k_passages)

            # Step 3: Generate candidates per passage
            candidates = []
            for passage in passages:
                # Check relevance
                is_relevant, rel_conf = generator.evaluate_relevance(query, passage.text)
                if not is_relevant:
                    continue

                # Generate output segment
                segment = generator.generate_segment(context, passage.text)

                # Check support
                support_level, sup_conf = generator.evaluate_support(passage.text, segment)

                # Compute score
                score = compute_segment_score(
                    rel_confidence=rel_conf,
                    support_level=support_level,
                    support_confidence=sup_conf
                )

                candidates.append({
                    "segment": segment,
                    "passage": passage.text,
                    "score": score
                })

            # Step 4: Select best candidate
            if candidates:
                best = max(candidates, key=lambda x: x["score"])
                generated_segments.append(best["segment"])
                context += f" {best['segment']}"
            else:
                # No good candidates, generate without retrieval
                segment = generator.generate_segment(context, None)
                generated_segments.append(segment)
                context += f" {segment}"
        else:
            # Generate directly without retrieval
            segment = generator.generate_segment(context, None)
            generated_segments.append(segment)
            context += f" {segment}"

        # Check for completion
        if generator.is_complete(context):
            break

    return " ".join(generated_segments)

def compute_segment_score(
    rel_confidence: float,
    support_level: str,
    support_confidence: float,
    w_rel: float = 1.0,
    w_sup: float = 1.0
) -> float:
    """Compute segment score from reflection token probabilities."""
    support_scores = {"Fully": 1.0, "Partially": 0.5, "No": 0.0}
    sup_score = support_scores[support_level] * support_confidence

    return w_rel * rel_confidence + w_sup * sup_score
```

---

## Use Cases

### Example 1: Medical Information Systems
- **Scenario**: Patient-facing health information system where accuracy is critical
- **Why this architecture**: [ISSUP] tokens ensure outputs are grounded in medical sources; 2% vs 15-20% unsupported claims
- **Expected outcome**: Dramatically reduced hallucination; better citation accuracy

### Example 2: Legal Research Assistant
- **Scenario**: Attorney research tool requiring precise case citations
- **Why this architecture**: Self-critique identifies when claims lack support; retrieval decisions are query-aware
- **Expected outcome**: Higher citation accuracy; flagging of uncertain claims

### Example 3: Fact-Checking System
- **Scenario**: Automated claim verification against knowledge base
- **Why this architecture**: [ISSUP] directly measures support level; 81% accuracy vs 71% for alternatives
- **Expected outcome**: Better true/false classification; confidence calibration

---

## Pros and Cons

### Advantages

- **Dramatically reduced hallucination**: 2% unsupported claims vs 15-20% for standard models (Asai et al., 2023)
- **Improved citation accuracy**: Significant gains in attribution to sources
- **Adaptive retrieval**: Retrieves only when beneficial, saving latency on simple queries
- **Self-contained**: No separate critic or evaluator needed at inference
- **Outperforms ChatGPT**: On Open-domain QA, reasoning, and fact verification (Asai et al., 2023)

### Limitations

- **Requires fine-tuning**: Cannot be applied to closed models (GPT-4, Claude); need access to weights
- **Training cost**: Fine-tuning 7B-13B model requires ~$500 in compute (8 A100 GPUs for ~24 hours)
- **Increased inference cost**: Segment-wise beam search with multiple passages is compute-intensive
- **Latency**: 2-5s per query due to multiple generation passes and beam search
- **Model-specific**: Trained models are specific to base architecture (LLaMA-2)

### Compared to Alternatives

- **vs. CRAG**: CRAG uses external evaluator; Self-RAG internalizes critique. Self-RAG is more accurate but requires fine-tuning.
- **vs. Traditional RAG**: Self-RAG adds adaptive retrieval + self-critique; much better accuracy but higher cost/complexity.
- **vs. Agentic RAG**: Agentic uses LLM reasoning for decisions; Self-RAG uses trained tokens. Self-RAG is faster at inference.

---

## Performance Benchmarks

| Task | Metric | Self-RAG 7B | Self-RAG 13B | ChatGPT | Llama2-chat 13B | Source |
|------|--------|-------------|--------------|---------|-----------------|--------|
| Open-domain QA | Accuracy | Higher | Highest | Baseline | Lower | Asai et al., 2023 |
| Fact Verification | Accuracy | 81% | - | - | 71% (Alpaca) | Asai et al., 2023 |
| Unsupported Claims | Rate | 2% | - | - | 15-20% | Asai et al., 2023 |

---

## Training Requirements

### Data Requirements
- **Critic training data**: ~100K query-passage-output triplets with GPT-4 annotations
- **Generator training data**: Existing corpus augmented with reflection tokens

### Compute Requirements
- **Critic training**: 1-2 A100 GPUs, 4-8 hours
- **Generator fine-tuning**: 8 A100 GPUs, 24 hours (~$500-800)

### Model Sizes
- **LLaMA-2 7B**: Faster, slightly lower quality
- **LLaMA-2 13B**: Higher quality, slower inference

---

## Available Resources

- **Paper**: [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- **Project Page**: [selfrag.github.io](https://selfrag.github.io/)
- **Models**: Available on Hugging Face (selfrag/selfrag_llama2_7b, selfrag/selfrag_llama2_13b)
- **Code**: GitHub repository with training and inference code

---

## References

1. Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024 Oral*. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
2. Project Page: [selfrag.github.io](https://selfrag.github.io/)
