# Cited Medical RAG API — container image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the knowledge base.
COPY src ./src
COPY knowledge_base ./knowledge_base

EXPOSE 8000

# ANTHROPIC_API_KEY is provided at runtime, e.g.:
#   docker run -e ANTHROPIC_API_KEY=sk-... -p 8000:8000 cited-medical-rag
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
