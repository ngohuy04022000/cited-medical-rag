# Knowledge base

Five short English documents the assistant answers from. Every answer must come from
these files, cite them, and be refused when they do not contain the answer.

- `01_glaucoma.md`
- `02_diabetic_retinopathy.md`
- `03_cataract.md`
- `04_amd.md`
- `05_screening_workflow.md` — a fictional AI-assisted screening workflow

This file is excluded from indexing (`src/chunker.py`, `EXCLUDED_FILES`). The content
is sample material for demonstrating grounded retrieval, not clinical guidance.
