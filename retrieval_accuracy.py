"""
retrieval_accuracy.py

Three retrieval-side fixes for the most common RAG accuracy failure:
the model is reasoning well over the wrong evidence.

  1. HyDE query expansion   -- close the gap between how a question is
     phrased and how the real answer is written.
  2. Hybrid retrieval + RRF -- combine dense (meaning-based) and sparse
     (exact-match) search instead of relying on either alone.
  3. Cross-encoder reranking -- re-score the fused candidate set jointly
     against the query, instead of trusting the first-pass ranking.

Companion module to the multilingual RAG blueprint in this repo. See the
LinkedIn article "Your RAG Accuracy Problem Isn't the Model. It's the
Retrieval." for the full writeup, real-world examples, and plain-language
explanation of each fix.

All model/LLM calls are injected as callables (same pattern as
agentic_rag.py in this repo), so the pipeline is fully testable without a
live API key or a downloaded reranker model.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from rank_bm25 import BM25Okapi
import numpy as np


# ---------------------------------------------------------------------------
# 1. HyDE query expansion
# ---------------------------------------------------------------------------

def hyde_expand(query: str, llm_fn: Callable[[str], str]) -> str:
    """
    Generate a hypothetical answer to the query and return it in place of
    the raw query for retrieval.

    llm_fn: a callable taking a prompt string and returning the model's
    text response. In production this wraps a real Claude API call; in
    tests/demos it can be a stub that returns a canned hypothetical answer.

    The hypothetical answer does not need to be factually correct -- it
    only needs to be stylistically close to a real document, so the
    embedding lands near the documents that actually matter.
    """
    prompt = (
        "Write a short, plausible answer to this question, in the style "
        "of a formal document. Don't hedge or caveat -- just state the "
        "answer directly, as if it were true.\n\n"
        f"Question: {query}"
    )
    return llm_fn(prompt)


def hyde_retrieve(
    query: str,
    llm_fn: Callable[[str], str],
    embed_fn: Callable[[str], np.ndarray],
    doc_embeddings: dict[str, np.ndarray],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Retrieve using the HyDE-expanded query instead of the raw query."""
    hypothetical = hyde_expand(query, llm_fn)
    hyde_vector = embed_fn(hypothetical)

    scores = {
        doc_id: _cosine_sim(hyde_vector, vec)
        for doc_id, vec in doc_embeddings.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# 2. Hybrid retrieval with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists of document IDs into a single ranking.

    RRF only uses rank *position* in each list, not raw scores -- which
    sidesteps the problem of dense-search cosine scores and BM25 scores
    living on completely different, non-comparable scales.

    k=60 is the constant used in the original RRF paper (Cormack et al.).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] += 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Combines a dense (embedding) retriever and a sparse (BM25) retriever,
    fused with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        corpus: dict[str, str],
        embed_fn: Callable[[str], np.ndarray],
        doc_embeddings: dict[str, np.ndarray],
    ):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.embed_fn = embed_fn
        self.doc_embeddings = doc_embeddings

        tokenized_corpus = [corpus[doc_id].lower().split() for doc_id in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _dense_search(self, query: str, top_k: int) -> list[str]:
        query_vec = self.embed_fn(query)
        scores = {
            doc_id: _cosine_sim(query_vec, self.doc_embeddings[doc_id])
            for doc_id in self.doc_ids
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]

    def _bm25_search(self, query: str, top_k: int) -> list[str]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [self.doc_ids[i] for i in ranked_idx]

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        dense_ids = self._dense_search(query, top_k)
        bm25_ids = self._bm25_search(query, top_k)
        fused = reciprocal_rank_fusion([dense_ids, bm25_ids])
        return fused[:top_k]


# ---------------------------------------------------------------------------
# 3. Cross-encoder reranking
# ---------------------------------------------------------------------------

@dataclass
class RerankedResult:
    doc_id: str
    text: str
    score: float


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    score_fn: Callable[[str, str], float],
    top_n: int = 5,
) -> list[RerankedResult]:
    """
    Re-score each (query, candidate_text) pair jointly and reorder.

    candidates: list of (doc_id, doc_text) tuples from the fused
    retrieval step (HybridRetriever.retrieve + a corpus lookup).

    score_fn: a callable taking (query, doc_text) and returning a
    relevance score. In production this wraps a real cross-encoder
    (e.g. BAAI/bge-reranker-large); in tests/demos it can be a stub.
    This only runs on the short fused candidate list, not the full
    corpus, since cross-encoders are far more expensive per pair than
    the first-pass retrieval methods above.
    """
    scored = [
        RerankedResult(doc_id=doc_id, text=text, score=score_fn(query, text))
        for doc_id, text in candidates
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_n]


def make_bge_reranker() -> Callable[[str, str], float]:
    """
    Returns a score_fn backed by a real BAAI/bge-reranker-large
    cross-encoder. Requires `sentence-transformers` and downloads model
    weights from Hugging Face on first call -- not used in the offline
    demo notebook, but this is the production entry point.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("BAAI/bge-reranker-large")

    def score_fn(query: str, doc_text: str) -> float:
        return float(model.predict([(query, doc_text)])[0])

    return score_fn


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def full_pipeline(
    query: str,
    corpus: dict[str, str],
    embed_fn: Callable[[str], np.ndarray],
    doc_embeddings: dict[str, np.ndarray],
    llm_fn: Callable[[str], str],
    rerank_score_fn: Callable[[str, str], float],
    fused_top_k: int = 20,
    final_top_n: int = 5,
    use_hyde: bool = True,
) -> list[RerankedResult]:
    """
    HyDE query expansion (optional) -> hybrid retrieval -> RRF fusion ->
    cross-encoder reranking -> final top-N ready to hand to the LLM for
    answer generation.
    """
    search_query = hyde_expand(query, llm_fn) if use_hyde else query

    retriever = HybridRetriever(corpus, embed_fn, doc_embeddings)
    fused = retriever.retrieve(search_query, top_k=fused_top_k)

    candidates = [(doc_id, corpus[doc_id]) for doc_id, _ in fused]
    return rerank(query, candidates, rerank_score_fn, top_n=final_top_n)


if __name__ == "__main__":
    # Quick smoke test with tiny synthetic data and stubbed model calls --
    # see retrieval_accuracy_demo.ipynb for the full walkthrough.
    corpus = {
        "doc_1": "revisions to section 4.2 leave entitlements effective q2",
        "doc_2": "employee handbook: general workplace conduct guidelines",
        "doc_3": "how to submit a reimbursement claim for travel expenses",
    }

    rng = np.random.default_rng(42)

    def fake_embed_fn(text: str) -> np.ndarray:
        # Deterministic pseudo-embedding for the smoke test only.
        h = abs(hash(text)) % (2**32)
        return np.random.default_rng(h).random(16)

    fake_embeddings = {doc_id: fake_embed_fn(text) for doc_id, text in corpus.items()}

    def fake_llm_fn(prompt: str) -> str:
        return "section 4.2 leave entitlements were revised effective q2"

    def fake_score_fn(query: str, doc_text: str) -> float:
        shared = len(set(query.lower().split()) & set(doc_text.lower().split()))
        return float(shared)

    results = full_pipeline(
        query="how many sick days do I get",
        corpus=corpus,
        embed_fn=fake_embed_fn,
        doc_embeddings=fake_embeddings,
        llm_fn=fake_llm_fn,
        rerank_score_fn=fake_score_fn,
        final_top_n=2,
    )

    print("Top results:")
    for r in results:
        print(f"  {r.doc_id} (score={r.score:.2f}): {r.text}")
