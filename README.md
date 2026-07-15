# Multilingual RAG for Indian Languages — A Practical Blueprint

A stage-by-stage, runnable blueprint for building Retrieval-Augmented Generation (RAG) pipelines
that are deliberately designed for Indian languages — Hindi, Bhojpuri, Maithili, and others —
instead of assuming a pipeline tuned on English data generalizes.

Most agentic AI and RAG systems today are built and benchmarked almost entirely on English,
internet-scale data. That works well inside that lane. Step outside it — into India's low-resource
languages — and each stage of a standard RAG pipeline degrades in a specific, traceable way. This
repo walks through why, and what to actually do about it, with working code for every stage.

## What's inside

`multilingual_rag_blueprint.ipynb` — a single notebook covering:

1. **Language-aware chunking** — sentence-boundary chunking for Indic scripts instead of naive
   punctuation splitting (the Devanagari danda `।` is not a period `.`)
2. **Embedding model benchmarking** — how to actually test whether a "multilingual" embedding
   model performs well in your target language, rather than trusting its stated language coverage
3. **Hybrid retrieval (BM25 + dense, fused with RRF)** — with stemming/morphological handling for
   inflection-heavy Indian languages
4. **Reranking** — cross-encoder reranking, and when to skip it for low-resource languages instead
5. **Query expansion (HyDE)** — and why it can hurt more than help in genuinely low-resource
   settings
6. **Evaluation** — MRR and Recall@k, and why translated English eval sets don't work
7. **`MultilingualRAGPipeline`** — all of the above assembled into one reusable class

The notebook runs end-to-end on an included synthetic Hindi corpus, so you can execute it without
needing your own documents first. Swap in your own corpus where marked.

## Quickstart

```bash
git clone https://github.com/<your-username>/multilingual-rag-india.git
cd multilingual-rag-india
pip install -r requirements.txt
jupyter notebook multilingual_rag_blueprint.ipynb
```

Model weights (embedding + reranker) download on first run of the "uncomment to execute" cells.

## Why this exists

Retrieval and reasoning systems perform in proportion to how well they're tuned to the actual data
distribution they run on. A pipeline that works beautifully on English, cloud-connected data isn't
wrong — it's answering a narrower question than "does this work for India's full linguistic
reality?" This repo is one attempt at closing that gap, stage by stage.

## Cache invalidation module

`cache_invalidation.py` addresses a problem the blueprint above doesn't cover:
what happens after a source document changes. It adds dependency-tracked
freshness handling on top of the retrieval pipeline:

1. **Document-level hashing** — detects whether a source document changed at
   all (`check_staleness`)
2. **Chunk-level diffing** — identifies exactly which chunks were added,
   removed, or changed, so only those get re-embedded (`diff_chunks`)
3. **Dependency-tracked answer cache** — tracks which cached query → answer
   pairs were generated from which chunks, so an edit purges only the
   affected cached answers, not the whole cache (`AnswerCache`,
   `on_document_updated`)

Run a quick smoke test directly:

```bash
python cache_invalidation.py
```

Or open `cache_invalidation_demo.ipynb` for a full walkthrough against a
synthetic Hindi policy document, contrasting chunk-level invalidation with
what naive whole-document invalidation would have cost.

## Agentic RAG — retrieval as a decision, not a fixed step

`agentic-rag/agentic_rag.py` addresses a different problem: most RAG chains
retrieve unconditionally on every query, even when retrieval isn't needed or
the first pass comes back weak. This module wires a small LangGraph graph
that makes retrieval a decision rather than a fixed step:

1. **Router node** — decides whether a query needs retrieval at all
   (`make_router_node`)
2. **Confidence-check node** — scores whether a retrieval pass is actually
   good enough to answer from, instead of trusting it blindly
   (`make_confidence_check_node`)
3. **Reformulate-and-retry node** — rewrites the query and re-retrieves when
   confidence is low, up to a configurable retry cap
   (`make_reformulate_node`)
4. **Honest fallback** — returns an explicit "I don't know" instead of
   guessing once retries are exhausted (`honest_fallback_node`)

All model/retriever calls are injected as callables, so the graph is testable
without a live model or vector store.

Open `agentic-rag/agentic_rag_demo.ipynb` for an executed walkthrough on
three queries that each take a different path through the graph: a direct
answer with no retrieval, a single successful retrieval pass, and a
reformulate-and-retry pass on a vaguely worded query.

## Related work

- A. K. Pandey and S. S. Roy, "Extractive Question Answering Over Ancient Scriptures Texts Using
  Generative AI and Natural Language Processing Techniques," *IEEE Access*, vol. 12, 2024.
  [DOI: 10.1109/ACCESS.2024.3431282](https://doi.org/10.1109/ACCESS.2024.3431282)
- A. K. Pandey and S. S. Roy, "Natural Language Generation Using Sequential Models: A Survey,"
  *Neural Processing Letters*, vol. 55, pp. 7709–7742, 2023.
  [DOI: 10.1007/s11063-023-11281-6](https://doi.org/10.1007/s11063-023-11281-6)

## License

MIT — use freely, contributions and issues welcome.

## Author

Abhishek Kumar Pandey — [LinkedIn](#) · Founder, Nuviq Technologies
