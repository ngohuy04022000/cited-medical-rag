# Cited Medical RAG

**Author:** Ngo Tri Huy

---

## How to Run

### 1. Clone and install

```bash
git clone <your-repo-url>
cd cited-medical-rag
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Open .env and set your ANTHROPIC_API_KEY
```

### 3. Run the RAG system

```bash
python -m src.main
```

Interactive CLI — type a question, get a cited answer. Type `quit` to exit.

#### Run offline — no API key needed

Set `RAG_BACKEND=local` to answer from the retrieved passages without calling the LLM.
This runs the full pipeline (chunk → retrieve → cite) for demos, dev, and CI:

```bash
# macOS / Linux
RAG_BACKEND=local python -m src.main
```

```powershell
# Windows PowerShell
$env:RAG_BACKEND="local"; python -m src.main
```

The same flag works for the eval (`python -m src.tests.eval`) and the API
(`uvicorn src.api:app`). The offline backend refuses on truly out-of-scope
questions (nothing retrieved) but cannot do the LLM's *semantic* refusal
("relevant docs retrieved but the specific fact is absent") — use the `anthropic`
backend for that.

### 4. Run the 5-question evaluation

```bash
python -m src.tests.eval
```

Runs 3 answerable + 2 must-refuse questions and prints PASS/FAIL for each.

### 5. Run as a REST API service (for delivery / demo)

```bash
uvicorn src.api:app --reload
```

Then open:
- **http://localhost:8000/** — web UI (chat-style, shows citations, confidence, latency)
- **http://localhost:8000/docs** — interactive Swagger UI

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Single-page web UI (`src/static/index.html`, no build step) |
| `/health` | GET | Liveness + index size + config (no API key needed) |
| `/query` | POST | `{"question": "...", "include_chunks": false}` → cited answer + confidence + latency |

The UI works with either backend — start with `RAG_BACKEND=local` to demo fully offline,
or set `ANTHROPIC_API_KEY` for LLM-generated answers. It polls `/health` every 15s to
show backend/readiness status in the header.

**Citation display:** each `[Source: filename]` in the answer is rendered as a small,
color-coded numbered badge (like a footnote), reused consistently for the same source.
A "Nguồn trích dẫn" (references) list below the answer shows, for every cited source:
a human-readable document label, the filename, section, relevance score, and the actual
excerpt backing the claim — so a claim can always be traced to the real passage, not just
a filename. Clicking a citation badge scrolls to and briefly highlights its reference card.

**Screenshots** (`docs/screenshots/`, captured running offline with `RAG_BACKEND=local`):

| Empty state | Answer with numbered citations | Citation click → highlighted source | Refusal state |
|---|---|---|---|
| ![Empty state](docs/screenshots/01_empty_state.png) | ![Answer with citation](docs/screenshots/02_answer_with_citation.png) | ![Citation click flash](docs/screenshots/citation_click_flash.png) | ![Refusal state](docs/screenshots/03_refusal_state.png) |

Example:

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is glaucoma and why is early detection important?"}'
```

### 6. Run with Docker

```bash
docker build -t cited-medical-rag .
docker run -e ANTHROPIC_API_KEY=sk-... -p 8000:8000 cited-medical-rag
```

The image includes a `HEALTHCHECK` that polls `/health`.

### 7. Run the unit tests

```bash
pytest src/tests -v
```

No API key required for unit tests — chunker, retriever, config, generation logic
(mocked LLM), and the HTTP layer (mocked pipeline) are all tested without calling the API.

---

## Configuration

All tunables are environment variables with safe defaults (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAG_BACKEND` | `anthropic` | `anthropic` (LLM) or `local` (offline, no key) |
| `ANTHROPIC_API_KEY` | — | Required when `RAG_BACKEND=anthropic` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | LLM model |
| `RAG_MAX_TOKENS` | `600` | Max answer tokens |
| `RAG_TEMPERATURE` | `0.0` | Determinism |
| `RAG_REQUEST_TIMEOUT` | `30` | Per-request timeout (s) |
| `RAG_MAX_RETRIES` | `3` | SDK auto-retry on 429/5xx/network |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |
| `RAG_MAX_PER_SOURCE` | `2` | Diversity cap per document |
| `RAG_MIN_SCORE` | `0.01` | Relevance floor for retrieval |
| `RAG_LOCAL_MIN_CONFIDENCE` | `0.12` | Offline backend refuses below this top score |
| `RAG_CHUNK_SIZE` | `700` | Target chunk size (chars) |
| `RAG_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Architecture

```
knowledge_base/   ← 5 knowledge-base markdown files
src/
  config.py      ← centralized Settings (env vars) + structured logging
  chunker.py     ← loads docs, section-aware chunking with metadata
  retriever.py   ← TF-IDF (1–2 gram, sublinear) + cosine similarity, top-k retrieval
  generator.py   ← Anthropic Claude API: cited answers, refusal, retries, error handling
  rag.py         ← orchestrator: chunker → retriever → generator (cached client)
  main.py        ← interactive CLI (shows sources, scores, confidence, latency)
  api.py         ← FastAPI service: /, /query, /health, /docs
  static/
    index.html   ← single-page web UI (chat-style, no build step)
  tests/
    test_rag.py  ← unit tests (mocks LLM; covers edge cases + config + errors)
    test_api.py  ← HTTP-layer tests (mocked pipeline; no API calls)
    eval.py      ← 5-question integration eval (requires API key)
Dockerfile       ← container image running the API
```

**Retrieval:** TF-IDF (scikit-learn) with 1–2 grams and sublinear TF (BM25-style term
saturation). The section heading is indexed alongside body text to sharpen document
routing. Lightweight, no embedding API needed for a 5-document corpus. Upgrade path:
replace `TFIDFRetriever` with `sentence-transformers` + FAISS.

**Generation:** `claude-haiku-4-5-20251001` by default (configurable). The system prompt
enforces citation format and an exact refusal phrase. The Anthropic client is created
once per pipeline (cached) with automatic retries and a request timeout; a final failure
is wrapped in `GenerationError` and surfaced cleanly to the CLI/API.

### What changed in this hardening pass

- **Efficiency:** the Anthropic client is now created once and reused (previously a new
  client was built on every query); the retriever indexes section headings for better
  routing; section-aware chunking merges one-line fragments into coherent context.
- **Reliability:** request timeout + SDK auto-retry on 429/5xx/network; clean error
  surfacing; structured logging; centralized, validated configuration.
- **Features:** FastAPI REST service (`/query`, `/health`, Swagger `/docs`), Docker image
  with healthcheck, per-answer confidence/latency/refusal flags, richer CLI output.
- **Tests:** expanded from 13 to 29 (chunker sections, config, generator error handling,
  and the HTTP layer) — all pass without an API key.

---

## Hard Constraints Met

| Requirement | How |
|-------------|-----|
| Answers cite source passage | System prompt enforces `[Source: filename]` format |
| Refuse when answer not present | Two-level: score=0 → empty retrieval → immediate refusal; score>0 → LLM instructed to refuse if context doesn't contain answer |
| Cross-document questions | top_k=5 with per-source diversity limit pulls from multiple docs |
| 5-question eval (3 answerable, 2 refuse) | `src/tests/eval.py` |
