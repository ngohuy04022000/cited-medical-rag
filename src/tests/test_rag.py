"""
Unit tests for the Cited Medical RAG pipeline.

Tests are split into:
- TestChunker: document loading (no API calls)
- TestRetriever: TF-IDF retrieval (no API calls)
- TestGenerator: generation logic (mocked API)

Run with: pytest src/tests/test_rag.py -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch

import anthropic
import httpx

from src.chunker import load_documents
from src.config import get_settings
from src.rag import RAGPipeline
from src.retriever import TFIDFRetriever
from src.generator import (
    REFUSAL_PHRASE,
    GenerationError,
    build_context,
    extractive_answer,
    generate_answer,
)

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chunks():
    return load_documents(DOCS_DIR)


@pytest.fixture(scope="module")
def retriever(chunks):
    return TFIDFRetriever(chunks)


# ---------------------------------------------------------------------------
# TestChunker
# ---------------------------------------------------------------------------

class TestChunker:
    def test_loads_all_five_documents(self, chunks):
        sources = {c["source"] for c in chunks}
        expected = {
            "01_glaucoma.md",
            "02_diabetic_retinopathy.md",
            "03_cataract.md",
            "04_amd.md",
            "05_screening_workflow.md",
        }
        assert expected == sources

    def test_chunks_have_required_fields(self, chunks):
        for c in chunks:
            assert "text" in c
            assert "source" in c
            assert "chunk_id" in c

    def test_all_chunks_are_non_empty(self, chunks):
        for c in chunks:
            assert c["text"].strip(), f"Empty chunk found: {c['chunk_id']}"

    def test_readme_is_excluded(self, chunks):
        sources = {c["source"] for c in chunks}
        assert "README.md" not in sources

    def test_chunks_carry_section_metadata(self, chunks):
        """Every document starts with an H1, so every chunk gets a section heading."""
        for c in chunks:
            assert c.get("section"), f"Missing section for {c['chunk_id']}"

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_small_paragraphs_are_merged(self, chunks):
        """Section-aware merging keeps chunk counts compact (<= a few per doc)."""
        from collections import Counter

        per_source = Counter(c["source"] for c in chunks)
        for source, count in per_source.items():
            assert 1 <= count <= 5, f"{source} produced {count} chunks"


# ---------------------------------------------------------------------------
# TestRetriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_glaucoma_query_retrieves_glaucoma_doc(self, retriever):
        results = retriever.retrieve("What is glaucoma?")
        sources = [r["source"] for r in results]
        assert "01_glaucoma.md" in sources

    def test_cross_document_retrieval(self, retriever):
        """Question about screening platform + disease risk factors spans multiple docs."""
        results = retriever.retrieve(
            "What conditions does the AI screening platform screen for and what are risk factors?"
        )
        sources = {r["source"] for r in results}
        # Must pull from at least 2 different documents
        assert len(sources) >= 2

    def test_out_of_scope_query_returns_no_results(self, retriever):
        """A query with zero vocabulary overlap returns an empty list."""
        results = retriever.retrieve("quantum entanglement superconductor")
        assert results == [] or all(r["score"] < 0.05 for r in results)

    # --- Edge case AI is likely to miss ---
    def test_empty_query_does_not_crash(self, retriever):
        """
        Edge case: empty string.
        Naive TF-IDF code often crashes here because transforming '' produces an
        all-zero sparse vector and cosine_similarity raises a division-by-zero warning
        or returns NaN. Our retriever must return [] without error.
        """
        results = retriever.retrieve("")
        assert results == []

    def test_stopwords_only_query_does_not_crash(self, retriever):
        """
        Edge case: query containing only English stop words ('the', 'a', 'is').
        TfidfVectorizer(stop_words='english') strips them all, leaving an all-zero
        vector. Without the nnz==0 guard this silently returns spurious top results.
        """
        results = retriever.retrieve("the a is are and or but")
        assert results == []

    def test_per_source_cap_applied(self, retriever):
        """No single document should contribute more than max_per_source=2 chunks."""
        results = retriever.retrieve("eye vision retina", top_k=10, max_per_source=2)
        from collections import Counter
        counts = Counter(r["source"] for r in results)
        for source, count in counts.items():
            assert count <= 2, f"{source} exceeded per-source cap: {count}"

    def test_results_carry_section_and_score(self, retriever):
        """Retrieved chunks expose section metadata and a numeric score."""
        results = retriever.retrieve("What is glaucoma?")
        assert results
        for r in results:
            assert isinstance(r["score"], float)
            assert "section" in r

    def test_phrase_query_ranks_correct_document_first(self, retriever):
        """Bigram indexing routes a 2-word phrase to the right document."""
        results = retriever.retrieve("optic nerve damage")
        assert results[0]["source"] == "01_glaucoma.md"


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self, monkeypatch):
        for var in ("RAG_TOP_K", "RAG_MIN_SCORE", "ANTHROPIC_MODEL"):
            monkeypatch.delenv(var, raising=False)
        settings = get_settings()
        assert settings.top_k == 5
        assert settings.min_score == 0.01
        assert settings.model == "claude-haiku-4-5-20251001"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RAG_TOP_K", "9")
        monkeypatch.setenv("RAG_TEMPERATURE", "0.3")
        settings = get_settings()
        assert settings.top_k == 9
        assert settings.temperature == 0.3

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("RAG_TOP_K", "not-a-number")
        settings = get_settings()
        assert settings.top_k == 5


# ---------------------------------------------------------------------------
# TestGenerator
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_empty_chunks_returns_refusal_without_api_call(self):
        """Generator must short-circuit and refuse when retrieval returns nothing."""
        mock_client = MagicMock()
        answer = generate_answer("What is glaucoma?", [], client=mock_client)
        assert answer == REFUSAL_PHRASE
        mock_client.messages.create.assert_not_called()

    def test_answer_includes_source_citation(self):
        """When the mocked LLM returns a cited answer, the source tag must be present."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(
                text="Glaucoma damages the optic nerve. [Source: 01_glaucoma.md]"
            )
        ]
        chunks = [{"text": "Glaucoma is...", "source": "01_glaucoma.md"}]
        answer = generate_answer("What is glaucoma?", chunks, client=mock_client)
        assert "01_glaucoma.md" in answer

    def test_refusal_phrase_passthrough(self):
        """If the LLM itself returns the refusal phrase, it is returned unchanged."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text=REFUSAL_PHRASE)
        ]
        chunks = [{"text": "Unrelated content.", "source": "03_cataract.md"}]
        answer = generate_answer("What is the price of surgery?", chunks, client=mock_client)
        assert answer == REFUSAL_PHRASE

    def test_build_context_includes_source_and_section(self):
        ctx = build_context(
            [{"text": "Glaucoma damages the optic nerve.", "source": "01_glaucoma.md", "section": "Glaucoma"}]
        )
        assert "[Source: 01_glaucoma.md]" in ctx
        assert "Section: Glaucoma" in ctx

    def test_build_context_without_section_still_works(self):
        ctx = build_context([{"text": "x", "source": "01_glaucoma.md"}])
        assert "[Source: 01_glaucoma.md]" in ctx
        assert "Section:" not in ctx

    def test_api_error_is_wrapped_in_generation_error(self):
        """A non-recoverable API failure surfaces as a clean GenerationError."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        chunks = [{"text": "Glaucoma is...", "source": "01_glaucoma.md"}]
        with pytest.raises(GenerationError):
            generate_answer("What is glaucoma?", chunks, client=mock_client)


# ---------------------------------------------------------------------------
# TestExtractiveBackend (offline, no API key)
# ---------------------------------------------------------------------------

class TestExtractiveBackend:
    def test_empty_chunks_refuses(self):
        assert extractive_answer("anything", []) == REFUSAL_PHRASE

    def test_answer_is_cited(self):
        chunks = [
            {
                "text": "Glaucoma damages the optic nerve. It is often painless early on.",
                "source": "01_glaucoma.md",
            }
        ]
        answer = extractive_answer("What does glaucoma damage?", chunks)
        assert "[Source: 01_glaucoma.md]" in answer
        assert "optic nerve" in answer

    def test_selects_query_relevant_sentence(self):
        chunks = [
            {
                "text": "Treatment is usually prescription eye drops. Vision lost cannot be recovered.",
                "source": "01_glaucoma.md",
            }
        ]
        answer = extractive_answer("What is the treatment for glaucoma?", chunks)
        assert "eye drops" in answer

    def test_no_overlap_falls_back_to_lead_sentence(self):
        chunks = [{"text": "First sentence here. Second one.", "source": "x.md"}]
        answer = extractive_answer("zzzzz qqqqq", chunks)
        assert "First sentence here." in answer
        assert "[Source: x.md]" in answer


# ---------------------------------------------------------------------------
# TestOfflinePipeline — full pipeline end-to-end without an API key
# ---------------------------------------------------------------------------

class TestOfflinePipeline:
    @pytest.fixture
    def offline_rag(self, monkeypatch):
        monkeypatch.setenv("RAG_BACKEND", "local")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        return RAGPipeline(settings=get_settings())

    def test_answerable_question_returns_cited_answer(self, offline_rag):
        result = offline_rag.query("What is glaucoma and why is early detection important?")
        assert not result["refused"]
        assert "01_glaucoma.md" in result["answer"]
        assert "01_glaucoma.md" in result["sources"]
        assert result["confidence"] > 0

    def test_out_of_scope_question_refuses(self, offline_rag):
        result = offline_rag.query("What medications are prescribed for clinical depression?")
        assert result["refused"]
        assert result["answer"] == REFUSAL_PHRASE

    def test_no_api_client_is_built_offline(self, offline_rag):
        offline_rag.query("What is glaucoma?")
        # Offline backend must never construct an Anthropic client.
        assert offline_rag._client is None
