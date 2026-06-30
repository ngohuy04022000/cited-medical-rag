"""RAG pipeline: chunker → retriever → generator."""

import logging
import time
from typing import Dict, Optional

import anthropic

from src.chunker import load_documents
from src.config import Settings, get_settings
from src.generator import REFUSAL_PHRASE, extractive_answer, generate_answer
from src.retriever import TFIDFRetriever

logger = logging.getLogger("cmrag.rag")


class RAGPipeline:
    def __init__(
        self,
        docs_dir: Optional[str] = None,
        top_k: Optional[int] = None,
        max_per_source: Optional[int] = None,
        client: Optional[anthropic.Anthropic] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.top_k = top_k if top_k is not None else self.settings.top_k
        self.max_per_source = (
            max_per_source if max_per_source is not None else self.settings.max_per_source
        )
        self._client = client

        resolved_dir = docs_dir or self.settings.docs_dir
        chunks = load_documents(resolved_dir, chunk_size=self.settings.chunk_size)
        self.num_chunks = len(chunks)
        self.retriever = TFIDFRetriever(chunks, min_score=self.settings.min_score)
        logger.info(
            "RAGPipeline ready: %d chunks from %s", self.num_chunks, resolved_dir
        )

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazily create and cache one Anthropic client for the pipeline's life.

        Previously a new client was built on every query; reusing one keeps the
        underlying HTTP connection pool warm and avoids per-request setup cost.
        """
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key,
                max_retries=self.settings.max_retries,
                timeout=self.settings.request_timeout,
            )
        return self._client

    def query(self, question: str) -> Dict:
        """
        Run the full RAG pipeline for a question.

        Returns:
            {
                "question": str,
                "answer": str,             # cited answer or refusal phrase
                "sources": List[str],      # unique source filenames used
                "confidence": float,       # top retrieval similarity (0..1)
                "refused": bool,           # True when the system declined to answer
                "latency_ms": float,
                "retrieved_chunks": List[Dict],
            }
        """
        start = time.perf_counter()

        retrieved = self.retriever.retrieve(
            question, top_k=self.top_k, max_per_source=self.max_per_source
        )

        if self.settings.backend == "local":
            # Offline extractive answer — no API key, no network call.
            answer = extractive_answer(
                question,
                retrieved,
                min_confidence=self.settings.local_min_confidence,
            )
        else:
            # Only spin up the API client when there is context worth sending.
            client = self.client if retrieved else None
            answer = generate_answer(
                question, retrieved, client=client, settings=self.settings
            )

        sources = list(dict.fromkeys(c["source"] for c in retrieved))  # dedupe, keep order
        confidence = max((c["score"] for c in retrieved), default=0.0)
        refused = REFUSAL_PHRASE.lower() in answer.lower()
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        logger.info(
            "query handled in %.1fms | sources=%s | confidence=%.3f | refused=%s",
            latency_ms,
            sources,
            confidence,
            refused,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 4),
            "refused": refused,
            "latency_ms": latency_ms,
            "retrieved_chunks": retrieved,
        }
