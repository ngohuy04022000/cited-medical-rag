"""
5-question evaluation for the Cited Medical RAG system.

Runs against any backend (RAG_BACKEND=anthropic needs ANTHROPIC_API_KEY;
local and hf need no key). Exits non-zero if any question fails, so it can
gate CI.
Run with: python -m src.tests.eval

Questions:
  Q1 — answerable, single document (glaucoma)
  Q2 — answerable, single document (diabetic retinopathy)
  Q3 — answerable, CROSS-DOCUMENT (platform workflow + disease docs)
  Q4 — must refuse (surgery success rate not in docs)
  Q5 — must refuse (completely out of scope)
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.config import get_settings
from src.rag import RAGPipeline
from src.generator import is_refusal

# refuse_kind:
#   "out_of_scope"       — the question is unrelated to the knowledge base.
#   "in_scope_no_answer" — relevant docs are retrieved but lack the specific fact.
#                          The LLM backends judge this semantically; the offline
#                          backend approximates it with a query-term coverage check.
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is glaucoma and why is early detection important?",
        "expected": "answerable",
        "note": "Single-doc answer from 01_glaucoma.md",
    },
    {
        "id": "Q2",
        "question": "What are the stages of diabetic retinopathy and what treatments are available?",
        "expected": "answerable",
        "note": "Single-doc answer from 02_diabetic_retinopathy.md",
    },
    {
        "id": "Q3",
        "question": (
            "What conditions does the AI screening platform detect, "
            "and what are the main risk factors for those conditions?"
        ),
        "expected": "answerable",
        "note": "Cross-document: 05_screening_workflow.md + 04_amd.md + 02_diabetic_retinopathy.md",
    },
    {
        "id": "Q4",
        "question": "What is the reported success rate of glaucoma surgery?",
        "expected": "refuse",
        "refuse_kind": "in_scope_no_answer",
        "note": "Success rates are NOT mentioned in any document (needs LLM to refuse)",
    },
    {
        "id": "Q5",
        "question": "What medications are prescribed for clinical depression?",
        "expected": "refuse",
        "refuse_kind": "out_of_scope",
        "note": "Completely out of scope - no psychiatric content in any document",
    },
]


def run_eval() -> int:
    settings = get_settings()
    if settings.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("Run offline with no key:  RAG_BACKEND=local python -m src.tests.eval")
        sys.exit(1)

    rag = RAGPipeline(settings=settings)
    passed = 0

    print("=" * 60)
    print(f"Cited Medical RAG - 5-Question Evaluation  (backend: {settings.backend})")
    print("=" * 60)

    for q in EVAL_QUESTIONS:
        result = rag.query(q["question"])
        answer = result["answer"]
        refused = is_refusal(answer)
        ok = refused if q["expected"] == "refuse" else not refused
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        print(f"\n[{status}] {q['id']} ({q['note']})")
        print(f"  Q: {q['question']}")
        print(f"  Expected: {q['expected']}  |  Refused: {refused}")
        if result["sources"]:
            print(f"  Sources: {', '.join(result['sources'])}")
        preview = answer[:200] + ("..." if len(answer) > 200 else "")
        print(f"  A: {preview}")

    print("\n" + "=" * 60)
    print(f"Result: {passed}/{len(EVAL_QUESTIONS)} passed")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_eval() == len(EVAL_QUESTIONS) else 1)
