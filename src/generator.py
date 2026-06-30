"""Anthropic-powered answer generation with mandatory citation and refusal.

Production hardening:
- The Anthropic client is configured with automatic retries and a request
  timeout (exponential backoff on 429/5xx/connection errors is handled by the
  SDK), so transient failures don't surface to the user.
- A final, non-recoverable API failure is wrapped in `GenerationError` with a
  clean message instead of leaking a raw stack trace to callers (CLI / API).
- Section metadata from the retriever is passed into the context to help the
  model locate the relevant passage, while the citation format stays exactly
  `[Source: <filename>]` so refusals remain programmatically detectable.
"""

import logging
import os
import re
from typing import Dict, List, Optional

import anthropic
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import Settings, get_settings

logger = logging.getLogger("cmrag.generator")

REFUSAL_PHRASE = "The answer to this question is not in the provided documents."

SYSTEM_PROMPT = f"""You are a medical knowledge assistant for retinal-disease screening.
Answer questions using ONLY the context passages provided below.

Rules — follow all of them exactly:
1. Cite every piece of information with [Source: <filename>] immediately after the sentence that uses it.
2. If the answer spans multiple documents, cite each one where it is used.
3. If the context does not contain the answer, respond with exactly this sentence and nothing else:
   "{REFUSAL_PHRASE}"
4. Do NOT use any medical knowledge outside of the provided context.
5. Be concise and accurate."""


class GenerationError(RuntimeError):
    """Raised when the LLM call fails after the SDK exhausts its retries."""


def build_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        header = f"[Source: {c['source']}]"
        section = c.get("section")
        if section:
            header += f"\nSection: {section}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)


def _build_client(settings: Settings) -> anthropic.Anthropic:
    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(
        api_key=api_key,
        max_retries=settings.max_retries,
        timeout=settings.request_timeout,
    )


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    client: Optional[anthropic.Anthropic] = None,
    settings: Optional[Settings] = None,
) -> str:
    """
    Generate a cited answer using the provided chunks.

    Returns REFUSAL_PHRASE immediately (without calling the API) when
    retrieved_chunks is empty — this covers the zero-score retrieval case.
    Raises GenerationError if the API call fails after retries.
    """
    if not retrieved_chunks:
        return REFUSAL_PHRASE

    settings = settings or get_settings()
    if client is None:
        client = _build_client(settings)

    context = build_context(retrieved_chunks)
    user_message = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f'Answer (cite sources using [Source: filename], or say "{REFUSAL_PHRASE}" if not found):'
    )

    try:
        response = client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:  # network / status / timeout after retries
        logger.error("Anthropic API call failed: %s", exc)
        raise GenerationError(f"LLM generation failed: {exc}") from exc

    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Offline backend — no API key required
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _content_terms(text: str) -> set:
    """Lowercased non-stopword tokens used for sentence/query overlap scoring."""
    return {
        w.lower()
        for w in _WORD_RE.findall(text)
        if w.lower() not in ENGLISH_STOP_WORDS
    }


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def extractive_answer(
    question: str,
    retrieved_chunks: List[Dict],
    max_sentences: int = 3,
    min_confidence: float = 0.0,
) -> str:
    """
    Build a cited answer offline by selecting the retrieved sentences that
    overlap most with the question. No API key or network call required.

    Refuses (REFUSAL_PHRASE) when:
    - retrieval returned nothing, or
    - the best retrieval score is below `min_confidence` (weak match — the
      offline backend declines rather than answer from a barely-relevant chunk).

    It cannot judge "strongly-retrieved context that still doesn't contain the
    specific fact"; that semantic refusal is the Anthropic backend's job.
    """
    if not retrieved_chunks:
        return REFUSAL_PHRASE

    top_score = max((c.get("score", 1.0) for c in retrieved_chunks), default=0.0)
    if top_score < min_confidence:
        return REFUSAL_PHRASE

    q_terms = _content_terms(question)

    scored = []
    for rank, chunk in enumerate(retrieved_chunks):
        for sent in _sentences(chunk["text"]):
            overlap = len(q_terms & _content_terms(sent))
            if overlap > 0:
                # Prefer high query overlap, then higher-ranked (earlier) chunks.
                scored.append((overlap, -rank, chunk["source"], sent))

    if not scored:
        # No sentence-level overlap: fall back to the top chunk's first sentence.
        top = retrieved_chunks[0]
        sents = _sentences(top["text"])
        lead = sents[0] if sents else top["text"]
        return f"{lead} [Source: {top['source']}]"

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    seen = set()
    parts = []
    for _, _, source, sent in scored:
        if sent in seen:
            continue
        seen.add(sent)
        parts.append(f"{sent} [Source: {source}]")
        if len(parts) >= max_sentences:
            break

    return " ".join(parts)
