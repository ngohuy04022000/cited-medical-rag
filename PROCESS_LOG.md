# PROCESS_LOG

This log documents the key AI interactions while building this project: the prompts I used,
the raw output, and exactly what I changed and why.

---

## RAG System

### Prompt 1 — Architecture design

**My prompt:**
> "I need to build a minimal RAG system in Python over 5 short markdown documents
> (~20 lines each). Requirements: answers must cite the source passage, must refuse
> when the answer is not in the docs, and must handle a question whose answer spans
> two documents. What retrieval approach would you choose and why?"

**AI output (summary):**
Suggested three options: (1) TF-IDF with cosine similarity, (2) sentence-transformers
embeddings + FAISS, (3) a hosted embedding API. Recommended option 2 as most "production
realistic."

**What I changed and why:**
I chose option 1 (TF-IDF) instead. The corpus is 5 documents, each ~20 lines —
a neural embedding model adds ~80 MB of download, a separate model-load step, and GPU
dependency for a corpus where BM25-style retrieval will perform just as well. For this
project's purpose (demonstrating RAG understanding, not raw retrieval quality),
TF-IDF is cleaner and the README can note the upgrade path to embeddings. I also added
`max_per_source` capping (max 2 chunks per document) which the AI did not suggest, to
improve diversity on cross-document questions.

---

### Prompt 2 — System prompt for the generator

**My prompt:**
> "Write a system prompt for an LLM that must answer questions from retrieved context,
> cite the source document for each fact, and refuse with a consistent sentence when the
> answer isn't present."

**AI output (raw):**
```
You are a helpful assistant. Answer the user's question based on the provided context.
Always cite your sources when possible. If you are unsure, say so politely.
```

**What I changed and why:**
This is too vague to be testable. I rewrote it entirely with:
- A specific citation format: `[Source: filename]` — not "cite your sources"
- An exact refusal phrase quoted verbatim so it can be matched programmatically
- "Do NOT use any knowledge outside the provided context" — the AI's "if unsure, say so"
  leaves a loophole for partial hallucination
- A two-level refusal: check at retrieval level (no chunks → immediate refusal) AND
  instruct the LLM to refuse when chunks don't contain the answer

The distinction between the two levels matters: the AI's draft relied entirely on the
LLM to decide when to refuse. But if retrieval returns zero chunks, calling the LLM at
all wastes tokens and risks a hallucinated response.

---

### Prompt 3 — Unit test generation

**My prompt:**
> "Generate pytest unit tests for a TF-IDF retriever class that has a method
> `retrieve(query, top_k)`. The class is initialized with a list of chunk dicts
> (text, source, chunk_id)."

**AI output (summary):**
Generated three tests: (1) retrieves correct document for a known query, (2) returns
fewer than top_k results when corpus is small, (3) top result has highest score.

**What I changed and why:**
All three tests were happy-path. I added:
- `test_empty_query_does_not_crash` — empty string crashes naive TF-IDF vectorizers
  because `transform([""])` produces an all-zero sparse vector; the AI did not think of
  this
- `test_stopwords_only_query_does_not_crash` — all-stopword query also produces a zero
  vector (different path: words exist but are filtered), again not in the AI's output
- `test_per_source_cap_applied` — verifies the diversity logic I added; the AI didn't
  know about `max_per_source` because I hadn't described it in the prompt
- Changed scope: the AI generated tests with hard-coded chunk fixtures; I switched to
  loading real documents with `load_documents(DOCS_DIR)` so tests exercise the actual
  knowledge base

---

## Where AI Helped, and Where It Was Wrong

1. **Helped:** Outlined the retrieval architecture options quickly; saved ~15 min of
   comparison research.
2. **Helped:** Suggested the initial structure for chunker + retriever + generator as
   separate modules — good separation of concerns.
3. **Wrong:** System prompt was too vague ("cite your sources", "if unsure, say so") —
   I rewrote it to be specific and testable.
4. **Wrong:** Generated only happy-path tests; missed the empty-query and
   stopwords-only edge cases that are the most likely to cause silent failures.
