"""
cache_invalidation.py

Dependency-tracked invalidation for RAG pipelines.

Solves three layered staleness problems:
  1. Document-level: has a source document changed at all?
  2. Chunk-level: which specific chunks inside it changed?
  3. Answer-cache-level: which cached query -> answer pairs depended on
     the chunks that changed, and therefore need to be purged?

Companion module to the multilingual RAG blueprint in this repo.
See the LinkedIn article "Your RAG System Is Right Today. Tomorrow
It's Confidently Wrong." for the full writeup.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional


def compute_hash(content: str) -> str:
    """Deterministic content hash used for both documents and chunks."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. Document-level tracking
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    doc_id: str
    content_hash: str
    last_verified: datetime
    chunk_ids: list[str] = field(default_factory=list)


def check_staleness(
    doc_id: str,
    current_content: str,
    doc_store: dict[str, DocumentRecord],
) -> bool:
    """Returns True if the document is new or has changed since last check."""
    record = doc_store.get(doc_id)
    if record is None:
        return True
    return compute_hash(current_content) != record.content_hash


# ---------------------------------------------------------------------------
# 2. Chunk-level diffing
# ---------------------------------------------------------------------------

def diff_chunks(
    old_chunks: dict[str, str],
    new_chunks: dict[str, str],
) -> dict[str, list[str]]:
    """
    Compares two {chunk_id: text} maps and classifies each chunk as
    added, removed, changed, or unchanged. Only added/changed chunks
    need re-embedding; only added/changed/removed chunks require any
    downstream cache action.
    """
    old_hashes = {cid: compute_hash(text) for cid, text in old_chunks.items()}
    new_hashes = {cid: compute_hash(text) for cid, text in new_chunks.items()}

    added = [cid for cid in new_hashes if cid not in old_hashes]
    removed = [cid for cid in old_hashes if cid not in new_hashes]
    changed = [
        cid for cid in new_hashes
        if cid in old_hashes and new_hashes[cid] != old_hashes[cid]
    ]
    unchanged = [
        cid for cid in new_hashes
        if cid in old_hashes and new_hashes[cid] == old_hashes[cid]
    ]
    return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}


# ---------------------------------------------------------------------------
# 3. Dependency-tracked answer cache
# ---------------------------------------------------------------------------

class AnswerCache:
    """
    A response cache that remembers which chunk_ids were used to
    generate each cached answer, so invalidation can target exactly
    the affected answers instead of clearing everything.
    """

    def __init__(self):
        self.cache: dict[str, dict] = {}
        self.chunk_to_answers: dict[str, set[str]] = defaultdict(set)

    def store(self, query_hash: str, answer: str, source_chunk_ids: list[str]) -> None:
        self.cache[query_hash] = {
            "answer": answer,
            "chunk_ids": list(source_chunk_ids),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        for cid in source_chunk_ids:
            self.chunk_to_answers[cid].add(query_hash)

    def get(self, query_hash: str) -> Optional[dict]:
        return self.cache.get(query_hash)

    def invalidate_by_chunks(self, changed_chunk_ids: list[str]) -> int:
        """Purges only cached answers that depended on the given chunks.
        Returns the number of cached answers purged."""
        affected_queries: set[str] = set()
        for cid in changed_chunk_ids:
            affected_queries |= self.chunk_to_answers.get(cid, set())

        for qhash in affected_queries:
            self.cache.pop(qhash, None)

        for cid in changed_chunk_ids:
            self.chunk_to_answers.pop(cid, None)

        return len(affected_queries)


# ---------------------------------------------------------------------------
# 4. Event-driven pipeline entry point
# ---------------------------------------------------------------------------

def on_document_updated(
    doc_id: str,
    new_content: str,
    doc_store: dict[str, DocumentRecord],
    chunker,          # object with .split(content) -> dict[str, str]
    embedder,         # object with .embed(text) -> vector
    vector_store,     # object with .upsert(id, vector) / .delete(id)
    answer_cache: AnswerCache,
) -> dict:
    """
    Call this from a webhook handler (Drive change notification, CMS
    publish hook, DB trigger) when a single document changes. Only the
    chunks that actually changed are re-embedded, and only the cached
    answers that depended on them are purged.
    """
    old_record = doc_store.get(doc_id)
    old_chunks: dict[str, str] = {}
    if old_record:
        # In a real system you'd fetch old chunk text from your chunk
        # store by id; simplified here for illustration.
        old_chunks = getattr(old_record, "_chunk_text_cache", {})

    new_chunks = chunker.split(new_content)
    diff = diff_chunks(old_chunks, new_chunks)

    to_embed = diff["added"] + diff["changed"]
    for cid in to_embed:
        vector_store.upsert(cid, embedder.embed(new_chunks[cid]))

    for cid in diff["removed"]:
        vector_store.delete(cid)

    purged = answer_cache.invalidate_by_chunks(diff["changed"] + diff["removed"])

    record = DocumentRecord(
        doc_id=doc_id,
        content_hash=compute_hash(new_content),
        last_verified=datetime.now(timezone.utc),
        chunk_ids=list(new_chunks.keys()),
    )
    record._chunk_text_cache = new_chunks  # type: ignore[attr-defined]
    doc_store[doc_id] = record

    return {
        "doc_id": doc_id,
        "chunks_re_embedded": len(to_embed),
        "chunks_removed": len(diff["removed"]),
        "answers_purged": purged,
    }


if __name__ == "__main__":
    # Minimal smoke test / usage example
    doc_store: dict[str, DocumentRecord] = {}
    cache = AnswerCache()

    class DummyChunker:
        def split(self, content: str) -> dict[str, str]:
            # naive split for demo purposes only
            parts = content.split("\n\n")
            return {f"chunk_{i}": p for i, p in enumerate(parts) if p.strip()}

    class DummyEmbedder:
        def embed(self, text: str):
            return [len(text)]  # placeholder vector

    class DummyVectorStore:
        def __init__(self):
            self.store = {}
        def upsert(self, cid, vec):
            self.store[cid] = vec
        def delete(self, cid):
            self.store.pop(cid, None)

    chunker, embedder, vstore = DummyChunker(), DummyEmbedder(), DummyVectorStore()

    v1 = "Refund window is 30 days.\n\nShipping takes 5-7 business days."
    result1 = on_document_updated("policy_doc", v1, doc_store, chunker, embedder, vstore, cache)
    cache.store("q_refund_window", "30 days", ["chunk_0"])
    cache.store("q_shipping_time", "5-7 business days", ["chunk_1"])
    print("Initial ingest:", result1)

    v2 = "Refund window is 45 days.\n\nShipping takes 5-7 business days."
    result2 = on_document_updated("policy_doc", v2, doc_store, chunker, embedder, vstore, cache)
    print("After edit:", result2)
    print("Refund answer still cached?", cache.get("q_refund_window") is not None)
    print("Shipping answer still cached?", cache.get("q_shipping_time") is not None)
