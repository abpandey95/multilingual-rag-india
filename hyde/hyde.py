"""
hyde.py

Hypothetical Document Embeddings (HyDE) — reference implementation.

HyDE addresses a specific retrieval failure mode: user queries and source
documents are often written in different linguistic registers (casual
question vs. formal documentation), which causes direct query-to-document
embedding similarity to underperform even when the correct document exists
in the index.

The fix: generate a short hypothetical answer to the query first, embed
THAT instead of the raw query, and retrieve using the hypothetical
document's embedding. Because the hypothetical answer is written in the
same register as real documents, it lands closer to the correct document
in vector space than the original question would.

This module is dependency-light and runs fully offline by default:
- Embeddings: TF-IDF vectors (scikit-learn) instead of a hosted embedding
  API, so the demo notebook runs without external credentials.
- Hypothetical document generation: a pluggable `Generator` interface.
  A `TemplateGenerator` (offline, deterministic) is provided for demos.
  An `AnthropicGenerator` is provided for production use with a real LLM
  — swap it in by setting ANTHROPIC_API_KEY and passing it to
  HydeRetriever(generator=AnthropicGenerator()).

Swapping TF-IDF for a real embedding model (OpenAI, Voyage, BGE, etc.) in
production only requires replacing the `Embedder` implementation; the
retrieval logic (hyde_retrieve, hyde_retrieve_ensemble) is unchanged.
"""

from __future__ import annotations

import abc
import os
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------
# Generator: produces a hypothetical answer passage for a given query
# --------------------------------------------------------------------------

class Generator(abc.ABC):
    """Abstract interface for hypothetical-document generation."""

    @abc.abstractmethod
    def generate(self, query: str, *, temperature: float = 0.7) -> str:
        """Return a plausible (not necessarily correct) answer passage."""
        raise NotImplementedError


class TemplateGenerator(Generator):
    """
    Offline, deterministic stand-in for an LLM call.

    A real LLM asked to write a hypothetical answer will naturally pull
    in domain-specific vocabulary related to the query's topic (e.g. a
    question about failed payments produces a passage using words like
    "transaction", "retry", "balance" — not generic filler). To keep
    this demo honest and offline-runnable, this generator does rule-
    based topic matching against a small set of domain vocabularies and
    produces a passage using that domain's real terminology. This is a
    reasonable proxy for what an LLM does implicitly through its
    training; it is NOT a substitute for one in production — a real LLM
    generalizes to topics and phrasings this keyword map has never seen.
    """

    _DOMAIN_MAP = [
        (
            ["payment", "pay", "charge", "transaction", "billing"],
            "Payment transaction failures are typically resolved through "
            "an automatic retry mechanism within a defined retry window, "
            "commonly triggered by insufficient balance, expired card "
            "credentials, or bank-side fraud flagging.",
        ),
        (
            ["lock", "login", "access", "authentication", "signin", "sign in"],
            "Account lockouts occur after repeated failed authentication "
            "attempts and are resolved by requesting an unlock through "
            "the verified recovery email or phone number on file.",
        ),
        (
            ["notification", "alert", "late", "delay", "push"],
            "Notification delivery delays can result from provider-side "
            "queuing, device connectivity issues, or notification "
            "preference settings that suppress non-critical alerts.",
        ),
        (
            ["export", "download", "data"],
            "Data export requests are processed asynchronously and made "
            "available through a time-limited download link, with "
            "processing time varying by file size and account "
            "subscription tier.",
        ),
        (
            ["sync", "integration", "third party", "third-party", "app", "connect"],
            "Integration synchronization failures are most commonly "
            "caused by an expired API token, a revoked permission scope, "
            "or a rate limit imposed by the external provider.",
        ),
        (
            ["refund", "money back", "return", "reimburse"],
            "Refund eligibility and processing time are determined by "
            "the policy in effect at the time of purchase and the "
            "original payment method used for the transaction.",
        ),
    ]

    _FALLBACK = (
        "This issue is addressed by reviewing the relevant configuration, "
        "confirming the affected component, and applying the documented "
        "resolution steps for this category of issue."
    )

    def generate(self, query: str, *, temperature: float = 0.7) -> str:
        query_lower = query.lower()
        for keywords, passage in self._DOMAIN_MAP:
            if any(kw in query_lower for kw in keywords):
                return passage
        return self._FALLBACK


class AnthropicGenerator(Generator):
    """
    Production generator using the Anthropic API.

    Requires the `anthropic` package and ANTHROPIC_API_KEY set in the
    environment. Not used by the offline demo notebook, but included as
    the drop-in production path referenced in the article.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        try:
            import anthropic  # noqa: F401 (import guarded intentionally)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnthropicGenerator requires the 'anthropic' package: "
                "pip install anthropic"
            ) from exc
        self._anthropic = __import__("anthropic")
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key=..."
            )
        self.client = self._anthropic.Anthropic(api_key=self.api_key)

    def generate(self, query: str, *, temperature: float = 0.7) -> str:
        prompt = textwrap.dedent(f"""\
            Write a short passage (2-4 sentences) that could plausibly
            answer the following question, in the style of formal
            technical documentation. Do not worry about factual
            accuracy — focus on matching the tone and vocabulary a real
            document on this topic would use.

            Question: {query}

            Passage:""")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# --------------------------------------------------------------------------
# Embedder: turns text into vectors
# --------------------------------------------------------------------------

class Embedder:
    """
    TF-IDF based embedder, fit once on the document corpus.

    This stands in for a hosted embedding model (OpenAI text-embedding-3,
    Voyage, BGE, etc.). TF-IDF is used here purely so the notebook is
    reproducible offline; in production, swap `embed`/`embed_batch` for
    calls to a real embedding API or local sentence-transformer model.
    The retrieval logic downstream does not care which embedder is used.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> None:
        self.vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() on a corpus first.")
        return self.vectorizer.transform([text]).toarray()[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() on a corpus first.")
        return self.vectorizer.transform(texts).toarray()


# --------------------------------------------------------------------------
# Vector store: in-memory nearest-neighbor search
# --------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float


class InMemoryVectorStore:
    """Minimal in-memory vector store using cosine similarity."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.documents: List[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def index(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        texts = [d.text for d in self.documents]
        self.embedder.fit(texts)
        self._matrix = self.embedder.embed_batch(texts)

    def similarity_search_by_vector(
        self, embedding: np.ndarray, k: int = 5
    ) -> List[RetrievalResult]:
        if self._matrix is None:
            raise RuntimeError("Vector store has not been indexed yet.")
        sims = cosine_similarity(embedding.reshape(1, -1), self._matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            RetrievalResult(
                doc_id=self.documents[i].doc_id,
                text=self.documents[i].text,
                score=float(sims[i]),
            )
            for i in top_idx
        ]


# --------------------------------------------------------------------------
# HyDE retrieval
# --------------------------------------------------------------------------

@dataclass
class HydeRetriever:
    """
    Orchestrates HyDE retrieval: generate hypothetical document(s),
    embed them, and search the real document store.
    """

    generator: Generator = field(default_factory=TemplateGenerator)

    def retrieve(
        self,
        query: str,
        vector_store: InMemoryVectorStore,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Single-hypothesis HyDE retrieval."""
        hypothetical_doc = self.generator.generate(query)
        hyde_embedding = vector_store.embedder.embed(hypothetical_doc)
        return vector_store.similarity_search_by_vector(hyde_embedding, k=top_k)

    def retrieve_ensemble(
        self,
        query: str,
        vector_store: InMemoryVectorStore,
        n_hypotheses: int = 4,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Multi-hypothesis HyDE retrieval: generate several hypothetical
        documents, embed each, average the vectors, then search. This
        smooths out noise from any single generation.
        """
        embeddings = []
        for _ in range(n_hypotheses):
            doc = self.generator.generate(query, temperature=0.9)
            embeddings.append(vector_store.embedder.embed(doc))
        averaged = np.mean(embeddings, axis=0)
        return vector_store.similarity_search_by_vector(averaged, k=top_k)

    def retrieve_direct(
        self,
        query: str,
        vector_store: InMemoryVectorStore,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Baseline: embed the raw query directly, no HyDE step. Used for
        side-by-side comparison against HyDE retrieval.
        """
        query_embedding = vector_store.embedder.embed(query)
        return vector_store.similarity_search_by_vector(query_embedding, k=top_k)


def compare_retrieval(
    query: str,
    vector_store: InMemoryVectorStore,
    retriever: HydeRetriever,
    top_k: int = 3,
) -> dict:
    """
    Convenience function: run both direct and HyDE retrieval for a query
    and return both result sets for side-by-side comparison.
    """
    return {
        "query": query,
        "direct": retriever.retrieve_direct(query, vector_store, top_k=top_k),
        "hyde": retriever.retrieve(query, vector_store, top_k=top_k),
    }
