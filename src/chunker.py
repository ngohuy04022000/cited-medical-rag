"""Load markdown documents and split them into section-aware chunks with metadata.

Improvements over naive paragraph splitting:
- Tracks the nearest markdown heading and attaches it as `section` metadata.
  The retriever indexes this alongside the body text, which sharply improves
  routing to the correct document on short corpora.
- Merges consecutive small paragraphs up to `chunk_size` characters so the LLM
  receives coherent, self-contained context instead of one-line fragments.
- Backward compatible: every chunk still has `text`, `source`, `chunk_id`.
"""

import os
import re
from typing import Dict, List

EXCLUDED_FILES = {"README.md"}
DEFAULT_CHUNK_SIZE = 700

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")


def load_documents(docs_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[Dict]:
    """
    Read all .md files in docs_dir (excluding README.md) and return a flat list
    of section-aware chunks, each with keys: text, source, section, chunk_id.
    """
    chunks: List[Dict] = []

    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".md") or filename in EXCLUDED_FILES:
            continue

        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks.extend(_chunk_markdown(content, filename, chunk_size))

    return chunks


def _chunk_markdown(content: str, filename: str, chunk_size: int) -> List[Dict]:
    """Split a single document into section-aware chunks.

    Headings are detected per-line (not per blank-line-separated paragraph
    block), so a heading immediately followed by body text on the very next
    line — with no blank line between them — is still recognized as a
    section header instead of being absorbed into the chunk text verbatim
    (which would leak raw "#" markdown into the LLM context and leave
    `section` empty for that part of the document).
    """
    results: List[Dict] = []
    current_section = ""
    buffer: List[str] = []
    buffer_len = 0
    next_id = 0
    paragraph_lines: List[str] = []

    def flush() -> None:
        nonlocal buffer, buffer_len, next_id
        if not buffer:
            return
        results.append(
            {
                "text": "\n\n".join(buffer),
                "source": filename,
                "section": current_section,
                "chunk_id": f"{filename}#{next_id}",
            }
        )
        next_id += 1
        buffer = []
        buffer_len = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, buffer, buffer_len
        para = "\n".join(paragraph_lines).strip()
        paragraph_lines = []
        if not para:
            return
        # Start a fresh chunk once the buffer would overflow the target size.
        if buffer and buffer_len + len(para) > chunk_size:
            flush()
        buffer.append(para)
        buffer_len += len(para)

    for line in content.split("\n"):
        heading = _HEADING_RE.match(line)
        if heading:
            # A heading starts a new section. Flush any pending paragraph and
            # chunk; the heading text becomes `section` metadata rather than
            # part of a chunk's body text.
            flush_paragraph()
            flush()
            current_section = heading.group(2).strip()
            continue

        if line.strip() == "":
            flush_paragraph()
        else:
            paragraph_lines.append(line)

    flush_paragraph()
    flush()

    # Fallback: a document that is only a heading (no body) still yields one chunk
    # so it is never silently dropped from the index.
    if not results and content.strip():
        results.append(
            {
                "text": content.strip(),
                "source": filename,
                "section": current_section,
                "chunk_id": f"{filename}#0",
            }
        )

    return results
