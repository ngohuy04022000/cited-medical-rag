"""Local LLM backend: run a small instruct model fully on this machine.

This is a third generation backend (alongside "anthropic" and the offline
"local" extractive one). It loads a Hugging Face causal-LM from a directory on
disk — no API key, no network — and reuses the exact same SYSTEM_PROMPT,
citation format, and REFUSAL_PHRASE as the Anthropic backend, so a small local
model can still cite sources and refuse out-of-scope questions.

Default model: Qwen2.5-1.5B-Instruct (multilingual, ~3 GB, runs on CPU or GPU).

The heavy imports (torch, transformers) and the model itself are loaded lazily
and cached process-wide, so importing this module — or running the rest of the
test suite — never pays the model-load cost unless a query actually needs it.
"""

import logging
import threading
from typing import Dict, List, Optional

from src.config import Settings, get_settings
from src.generator import (
    _CITATION_RE,
    GenerationError,
    REFUSAL_PHRASE,
    SYSTEM_PROMPT,
    build_context,
)

logger = logging.getLogger("cmrag.local_llm")

# A small (1.5B) model follows the citation format less reliably than a frontier
# model, so we (a) reinforce the format with a one-shot example in the prompt and
# (b) apply a deterministic safety net after generation. Together these keep the
# hard constraint — "answers must cite the source passage" — always satisfied.
_ONESHOT = (
    "Example of the required format:\n"
    "Context passages:\n"
    "[Source: 01_glaucoma.md]\nGlaucoma damages the optic nerve.\n\n"
    "Question: What does glaucoma damage?\n"
    "Answer: Glaucoma damages the optic nerve. [Source: 01_glaucoma.md]\n"
)

def _ensure_citations(answer: str, retrieved_chunks: List[Dict]) -> str:
    """Guarantee the hard constraint: a non-refusal answer must carry at least
    one [Source: ...] tag. If the small model produced prose without any
    citation, append the documents it was actually grounded on (the retrieved
    sources) so the claim is still traceable."""
    if REFUSAL_PHRASE.lower() in answer.lower():
        return answer
    if _CITATION_RE.search(answer):
        return answer
    sources = list(dict.fromkeys(c["source"] for c in retrieved_chunks))
    if not sources:
        return answer
    tags = " ".join(f"[Source: {s}]" for s in sources)
    logger.info("Local LLM omitted citations; appending retrieved sources: %s", sources)
    return f"{answer.rstrip()} {tags}"


class LocalLLMError(GenerationError):
    """Raised when the local model cannot be loaded or generation fails.

    Subclasses GenerationError so the CLI and API surface its actionable
    message (e.g. "model directory not found") instead of a generic 500.
    """


# Process-wide singleton so the ~3 GB model is loaded at most once, guarded for
# the case where a threaded server (uvicorn) races two first requests.
_pipeline = None
_load_lock = threading.Lock()


def _load_pipeline(settings: Settings):
    """Load tokenizer + model once and cache them. Returns (tokenizer, model)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _load_lock:
        if _pipeline is not None:
            return _pipeline

        import os

        if not os.path.isdir(settings.hf_model_dir):
            raise LocalLLMError(
                f"Local model directory not found: {settings.hf_model_dir!r}. "
                "Download it first (see README, 'Run with a local LLM'), or set "
                "RAG_HF_MODEL_DIR to an existing model directory."
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LocalLLMError(
                "The 'hf' backend needs torch + transformers. Install them with "
                "`pip install torch transformers accelerate`."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading local LLM from %s on %s…", settings.hf_model_dir, device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.hf_model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                settings.hf_model_dir,
                torch_dtype=dtype,
                device_map=device,
            )
        except Exception as exc:  # corrupt download, OOM, etc.
            raise LocalLLMError(f"Failed to load local model: {exc}") from exc

        model.eval()
        _pipeline = (tokenizer, model)
        logger.info("Local LLM ready (device=%s, dtype=%s)", device, dtype)
        return _pipeline


def local_llm_answer(
    question: str,
    retrieved_chunks: List[Dict],
    settings: Optional[Settings] = None,
) -> str:
    """
    Generate a cited answer with a small local model.

    Short-circuits to REFUSAL_PHRASE when retrieval returns nothing (no chunks),
    mirroring the other backends, so we never spin up the model just to refuse.
    """
    if not retrieved_chunks:
        return REFUSAL_PHRASE

    settings = settings or get_settings()
    tokenizer, model = _load_pipeline(settings)

    import torch

    context = build_context(retrieved_chunks)
    user_message = (
        f"{_ONESHOT}\n"
        f"Now answer this one the same way.\n\n"
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f'Answer (cite each fact with [Source: filename], or say exactly "{REFUSAL_PHRASE}" if the answer is not in the context):'
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=settings.hf_max_new_tokens,
                do_sample=False,  # greedy — reproducible, no temperature wobble
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Only decode the newly generated tokens, not the echoed prompt.
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as exc:
        logger.error("Local LLM generation failed: %s", exc)
        raise LocalLLMError(f"Local model generation failed: {exc}") from exc

    if not answer:
        return REFUSAL_PHRASE
    return _ensure_citations(answer, retrieved_chunks)
