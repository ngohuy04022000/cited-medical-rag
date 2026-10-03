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

from src.chunker import _chunk_markdown, load_documents
from src.config import get_settings, _normalize_backend
from src.rag import RAGPipeline
from src.retriever import TFIDFRetriever
from src.generator import (
    REFUSAL_PHRASE,
    GenerationError,
    build_context,
    extractive_answer,
    generate_answer,
    is_refusal,
    term_coverage,
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

    def test_heading_without_blank_line_is_still_detected(self):
        """
        Regression: a heading immediately followed by body text on the very next
        line (no blank line between them) must still be recognized as a section
        header. Before the fix, paragraph-block splitting on "\\n\\n" swallowed
        the "# Heading" line into the chunk's body text verbatim (leaking raw
        markdown into LLM context) and left `section` empty.
        """
        content = (
            "# Some New Disease\n"
            "This paragraph has no blank line after the heading above it.\n\n"
            "Second paragraph, normally separated."
        )
        chunks = _chunk_markdown(content, "test.md", chunk_size=700)
        assert chunks, "heading-only-no-blank-line content produced no chunks"
        for c in chunks:
            assert c["section"] == "Some New Disease"
            assert not c["text"].lstrip().startswith("#"), (
                f"raw heading markdown leaked into chunk text: {c['text']!r}"
            )


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

    def test_backend_local_aliases_normalize_to_local(self):
        for alias in ("local", "Local", "OFFLINE", "extractive", "mock"):
            assert _normalize_backend(alias) == "local"

    def test_backend_unrecognized_value_warns_and_falls_back(self, caplog):
        """
        Regression: a typo like RAG_BACKEND=lokal must not be silently treated
        as valid input — it should fall back to 'anthropic' AND log a warning,
        consistent with how _get_int/_get_float handle invalid values. Before
        the fix, the fallback ternary was a no-op dead branch: any unrecognized
        value silently became 'anthropic' with zero diagnostic.
        """
        with caplog.at_level("WARNING", logger="cmrag.config"):
            result = _normalize_backend("lokal")
        assert result == "anthropic"
        assert any("lokal" in rec.getMessage() for rec in caplog.records)

    def test_backend_anthropic_value_is_silent(self, caplog):
        with caplog.at_level("WARNING", logger="cmrag.config"):
            result = _normalize_backend("anthropic")
        assert result == "anthropic"
        assert caplog.records == []


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

    def test_low_term_coverage_refuses(self):
        chunks = [{"text": "Cataract surgery replaces the cloudy lens.", "source": "03_cataract.md"}]
        answer = extractive_answer(
            "What is the price of cataract surgery?", chunks, min_term_coverage=0.75
        )
        assert answer == REFUSAL_PHRASE

    def test_full_term_coverage_answers(self):
        chunks = [{"text": "Cataract surgery replaces the cloudy lens.", "source": "03_cataract.md"}]
        answer = extractive_answer(
            "What does cataract surgery replace?", chunks, min_term_coverage=0.75
        )
        assert "[Source: 03_cataract.md]" in answer


class TestTermCoverage:
    def test_inflections_match_by_prefix(self):
        chunks = [{"text": "Patients undergo screening.", "source": "x.md"}]
        assert term_coverage("Which patients are screened?", chunks) == 1.0

    def test_filler_words_are_ignored(self):
        chunks = [{"text": "Glaucoma damages the optic nerve.", "source": "x.md"}]
        assert term_coverage("Tell me what glaucoma does", chunks) == 1.0

    def test_missing_topic_lowers_coverage(self):
        chunks = [{"text": "Glaucoma damages the optic nerve.", "source": "x.md"}]
        assert term_coverage("glaucoma cost", chunks) == 0.5


class TestIsRefusal:
    def test_exact_phrase_is_refusal(self):
        assert is_refusal(REFUSAL_PHRASE)

    def test_cited_answer_is_not_refusal(self):
        assert not is_refusal("Glaucoma damages the optic nerve. [Source: 01_glaucoma.md]")

    def test_partial_answer_with_citation_is_not_refusal(self):
        answer = (
            "Glaucoma damages the optic nerve. [Source: 01_glaucoma.md] "
            + REFUSAL_PHRASE
        )
        assert not is_refusal(answer)


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

    def test_in_scope_question_with_absent_fact_refuses(self, offline_rag):
        result = offline_rag.query("What is the reported success rate of glaucoma surgery?")
        assert result["refused"]

    def test_no_api_client_is_built_offline(self, offline_rag):
        offline_rag.query("What is glaucoma?")
        # Offline backend must never construct an Anthropic client.
        assert offline_rag._client is None


# ---------------------------------------------------------------------------
# TestResilience — predicted failure scenarios and their hardening
# ---------------------------------------------------------------------------

class TestResilience:
    def test_missing_docs_dir_raises_clear_error(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="RAG documents directory not found"):
            load_documents(str(missing))

    def test_empty_corpus_raises_clear_runtime_error(self, tmp_path, monkeypatch):
        """A docs dir with only an excluded/empty file must fail fast with a
        readable message instead of sklearn's cryptic 'empty vocabulary' error."""
        (tmp_path / "README.md").write_text("excluded, not indexed", encoding="utf-8")
        monkeypatch.setenv("RAG_BACKEND", "local")
        with pytest.raises(RuntimeError, match="No documents found"):
            RAGPipeline(docs_dir=str(tmp_path), settings=get_settings())

    def test_top_k_zero_is_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv("RAG_TOP_K", "0")
        assert get_settings().top_k == 1

    def test_top_k_negative_is_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv("RAG_TOP_K", "-3")
        assert get_settings().top_k == 1

    def test_chunk_size_too_small_is_clamped(self, monkeypatch):
        monkeypatch.setenv("RAG_CHUNK_SIZE", "1")
        assert get_settings().chunk_size == 50

    def test_missing_api_key_raises_generation_error_not_raw_typeerror(self, monkeypatch):
        """
        Regression: anthropic.Anthropic(api_key=None) does NOT raise at
        construction time in this SDK version — it raises a bare TypeError
        deep inside .messages.create() instead, which is not an
        anthropic.APIError and was previously left uncaught. generate_answer
        must now catch the missing-key case proactively and raise a clean
        GenerationError before ever reaching the SDK call.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = get_settings()
        assert settings.backend == "anthropic"
        chunks = [{"text": "Glaucoma is...", "source": "01_glaucoma.md"}]
        with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY is not set"):
            generate_answer("What is glaucoma?", chunks, settings=settings)

    def test_pipeline_client_property_raises_generation_error_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("RAG_BACKEND", "anthropic")
        rag = RAGPipeline(settings=get_settings())
        with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY is not set"):
            rag.query("What is glaucoma?")

    def test_empty_llm_response_raises_generation_error(self):
        """A stop_reason with zero content blocks must not crash with a raw
        IndexError on response.content[0]."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = []
        mock_client.messages.create.return_value.stop_reason = "end_turn"
        chunks = [{"text": "Glaucoma is...", "source": "01_glaucoma.md"}]
        with pytest.raises(GenerationError, match="empty response"):
            generate_answer("What is glaucoma?", chunks, client=mock_client)
