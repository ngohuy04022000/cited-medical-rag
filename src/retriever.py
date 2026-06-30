"""TF-IDF retriever with cosine similarity and per-source diversity.

Retrieval-quality improvements:
- `ngram_range=(1, 2)` captures short phrases ("optic nerve", "fundus camera"),
  which matters on a small corpus where single keywords are ambiguous.
- `sublinear_tf=True` dampens term-frequency, approximating BM25-style saturation
  so a word repeated many times in one chunk does not dominate the score.
- The section heading is indexed together with the body text, strengthening
  document routing without polluting the text shown to the user.
"""

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MIN_SCORE = 0.01  # discard chunks with no meaningful overlap with query
MIN_SCORE = DEFAULT_MIN_SCORE  # kept for backward compatibility


def _index_text(chunk: Dict) -> str:
    """Text used for vectorization: section heading + body when available."""
    section = chunk.get("section", "")
    return f"{section}\n{chunk['text']}" if section else chunk["text"]


class TFIDFRetriever:
    def __init__(self, chunks: List[Dict], min_score: float = DEFAULT_MIN_SCORE) -> None:
        self.chunks = chunks
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        texts = [_index_text(c) for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5, max_per_source: int = 2) -> List[Dict]:
        """
        Return up to top_k chunks most relevant to query.

        Edge cases handled:
        - Empty or whitespace-only query → returns []
        - Query whose terms are all stopwords → TF-IDF vector is all-zero → returns []
        - Queries with no vocabulary overlap → all scores 0 → returns []
        """
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])

        # All words were stopwords or unknown — zero vector, cosine is undefined
        if query_vec.nnz == 0:
            return []

        scores = cosine_similarity(query_vec, self.matrix)[0]

        # Rank by score descending, enforce per-source cap for diversity
        ranked_indices = np.argsort(scores)[::-1]
        source_count: Dict[str, int] = {}
        results = []

        for idx in ranked_indices:
            if scores[idx] < self.min_score:
                break
            source = self.chunks[idx]["source"]
            if source_count.get(source, 0) >= max_per_source:
                continue
            source_count[source] = source_count.get(source, 0) + 1
            results.append({**self.chunks[idx], "score": float(scores[idx])})
            if len(results) >= top_k:
                break

        return results
