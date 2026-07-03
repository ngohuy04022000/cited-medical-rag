"""Tests for the on-device 'hf' local-LLM backend.

These tests never load the real ~3 GB model. They cover:
- the citation safety net (_ensure_citations), which guarantees the hard
  constraint that a non-refusal answer always carries a [Source: ...] tag, and
- the empty-retrieval short-circuit, which must refuse without touching the model.

A real end-to-end check against the downloaded model is done manually (see
README); it is intentionally excluded from the unit suite so `pytest` stays fast
and needs no model files.
"""

from unittest.mock import patch

from src.config import _normalize_backend
from src.generator import REFUSAL_PHRASE
from src.local_llm import _ensure_citations, local_llm_answer


class TestCitationSafetyNet:
    def test_appends_sources_when_model_omits_citation(self):
        chunks = [
            {"source": "01_glaucoma.md", "text": "..."},
            {"source": "02_diabetic_retinopathy.md", "text": "..."},
        ]
        answer = _ensure_citations("Glaucoma damages the optic nerve.", chunks)
        assert "[Source: 01_glaucoma.md]" in answer
        assert "[Source: 02_diabetic_retinopathy.md]" in answer

    def test_keeps_existing_citation_untouched(self):
        original = "Glaucoma damages the optic nerve. [Source: 01_glaucoma.md]"
        chunks = [{"source": "01_glaucoma.md", "text": "..."}]
        assert _ensure_citations(original, chunks) == original

    def test_does_not_annotate_a_refusal(self):
        chunks = [{"source": "03_cataract.md", "text": "..."}]
        assert _ensure_citations(REFUSAL_PHRASE, chunks) == REFUSAL_PHRASE

    def test_no_sources_leaves_answer_unchanged(self):
        assert _ensure_citations("some answer", []) == "some answer"

    def test_dedupes_repeated_sources(self):
        chunks = [
            {"source": "01_glaucoma.md", "text": "a"},
            {"source": "01_glaucoma.md", "text": "b"},
        ]
        answer = _ensure_citations("Prose with no citation.", chunks)
        assert answer.count("[Source: 01_glaucoma.md]") == 1


class TestLocalLLMBackend:
    def test_empty_chunks_refuses_without_loading_model(self):
        # If retrieval returned nothing, we must refuse without ever loading the
        # multi-GB model. _load_pipeline is patched to explode if called.
        with patch("src.local_llm._load_pipeline", side_effect=AssertionError("model loaded!")):
            answer = local_llm_answer("What is glaucoma?", [])
        assert answer == REFUSAL_PHRASE

    def test_backend_aliases_route_to_hf(self):
        for alias in ("hf", "huggingface", "qwen", "local-llm"):
            assert _normalize_backend(alias) == "hf"
