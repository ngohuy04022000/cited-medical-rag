"""Interactive CLI for the Cited Medical RAG system."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.config import get_settings, setup_logging
from src.generator import GenerationError
from src.rag import RAGPipeline


def _print_result(result: dict) -> None:
    print(f"\nAnswer:\n{result['answer']}")

    if result["retrieved_chunks"]:
        print("\nSources consulted:")
        seen = set()
        for chunk in result["retrieved_chunks"]:
            src = chunk["source"]
            if src in seen:
                continue
            seen.add(src)
            print(f"  - {src}  (relevance {chunk['score']:.2f})")

    flag = "  [refused]" if result["refused"] else ""
    print(
        f"\nconfidence={result['confidence']:.2f} | "
        f"latency={result['latency_ms']:.0f}ms{flag}\n"
    )


def main() -> None:
    setup_logging()
    settings = get_settings()

    if settings.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key,")
        print("or run offline with no key:  RAG_BACKEND=local python -m src.main")
        sys.exit(1)

    print("Cited Medical RAG - Retinal Disease Knowledge Assistant")
    print("Knowledge base: Glaucoma, Diabetic Retinopathy, Cataract, AMD, Screening Workflow")
    print(f"Backend: {settings.backend}" + ("  (offline, no API key)" if settings.backend == "local" else ""))
    print("Type 'quit' or press Ctrl+C to exit.\n")

    rag = RAGPipeline(settings=settings)

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if question.lower() in ("quit", "exit", "q", ""):
            break

        try:
            result = rag.query(question)
        except GenerationError as exc:
            print(f"\n[error] {exc}\nPlease try again in a moment.\n")
            continue

        _print_result(result)


if __name__ == "__main__":
    main()
