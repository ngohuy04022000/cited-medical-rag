# SKILL: Citation-Aware RAG Prompt Engineering

**Purpose:** A reusable recipe for writing LLM prompts that force the model to cite sources
and refuse gracefully when the answer is not present — the two hardest constraints in RAG
systems.

---

## When to Use

Any time you build a RAG system where:
- Answers must be traceable to source passages
- Hallucination is unacceptable (medical, legal, financial domains)
- The system must say "I don't know" rather than making something up

---

## The Recipe

### Step 1 — Define the exact refusal phrase

Pick ONE canonical sentence and embed it in both the system prompt and your evaluation
logic. Using a consistent phrase lets you detect refusals programmatically.

```
REFUSAL_PHRASE = "The answer to this question is not in the provided documents."
```

### Step 2 — Write the system prompt

Key elements, in order:
1. **Role + scope** — tell the model it only knows what's in the context
2. **Citation format** — be specific: `[Source: filename]` not "cite sources"
3. **Exact refusal instruction** — quote the refusal phrase verbatim
4. **Anti-hallucination rule** — "Do NOT use any knowledge outside the provided context"

```
SYSTEM_PROMPT = f"""You are a [domain] assistant.
Answer questions using ONLY the context passages provided below.

Rules — follow all of them exactly:
1. Cite every piece of information with [Source: <filename>] immediately after the
   sentence that uses it.
2. If the answer spans multiple documents, cite each one where it is used.
3. If the context does not contain the answer, respond with exactly this sentence
   and nothing else:
   "{REFUSAL_PHRASE}"
4. Do NOT use any knowledge outside of the provided context.
5. Be concise and accurate."""
```

### Step 3 — Two-level refusal enforcement

Don't rely on the LLM alone. Add a retrieval-level guard:

```python
if not retrieved_chunks:           # retrieval found nothing relevant
    return REFUSAL_PHRASE          # short-circuit — don't even call the LLM
```

The LLM handles "context retrieved but doesn't actually answer the question."
The retrieval guard handles "nothing was retrieved at all."

### Step 4 — Evaluate refusal behaviour

Always test refusal with at least 2 out-of-scope questions before shipping.
Check with a simple string match — don't use the LLM to judge its own refusals.

```python
refused = REFUSAL_PHRASE.lower() in answer.lower()
assert refused, f"Should have refused but said: {answer}"
```

---

## Common Mistakes AI Makes in First Draft

| Mistake | Fix |
|---------|-----|
| Vague "cite your sources" | Specify exact format: `[Source: filename]` |
| No refusal instruction | Add explicit "say EXACTLY: ..." |
| Single-level refusal only | Add retrieval-level guard for zero-score results |
| Refusal phrasing varies | Hardcode one constant, use it everywhere |
| Tests only happy path | Always add 2+ must-refuse test cases |

---

## Prompt I Used to Generate the Initial Draft

> "Write a Python system prompt for a RAG assistant that must cite source documents and
> refuse if the answer isn't present. The refusal must be a single consistent sentence
> so I can detect it programmatically."

**What I changed from the AI output:**
- Added the two-level refusal (retrieval + LLM) — the AI only put it in the system prompt
- Changed "cite sources" to the specific `[Source: filename]` format
- Added "Do NOT use any knowledge outside context" — the AI left this implicit
- Hardcoded the refusal phrase as a constant rather than leaving it inline in the prompt
