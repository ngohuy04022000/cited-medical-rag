"""Tests for the FastAPI service.

The RAG pipeline is replaced with a fake via dependency injection, so these
tests exercise the HTTP layer (routing, validation, error mapping) without
loading the index or calling the Anthropic API.
"""

import dataclasses

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import src.api as api_mod


class FakePipeline:
    num_chunks = 12

    def __init__(self):
        self.calls = []

    def query(self, question):
        self.calls.append(question)
        return {
            "question": question,
            "answer": "Glaucoma damages the optic nerve. [Source: 01_glaucoma.md]",
            "sources": ["01_glaucoma.md"],
            "confidence": 0.42,
            "refused": False,
            "latency_ms": 12.3,
            "retrieved_chunks": [
                {
                    "source": "01_glaucoma.md",
                    "section": "Glaucoma",
                    "chunk_id": "01_glaucoma.md#0",
                    "score": 0.42,
                    "text": "Glaucoma is a group of eye conditions...",
                }
            ],
        }


@pytest.fixture
def client(monkeypatch):
    fake = FakePipeline()
    # Pre-seed the singleton so lifespan startup doesn't build the real index.
    monkeypatch.setattr(api_mod, "_pipeline", fake)
    api_mod.app.dependency_overrides[api_mod.get_pipeline] = lambda: fake
    with TestClient(api_mod.app) as c:
        yield c, fake
    api_mod.app.dependency_overrides.clear()


def test_health(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["num_chunks"] == 12
    assert "model" in body


def test_query_success(client, monkeypatch):
    c, fake = client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    resp = c.post("/query", json={"question": "What is glaucoma?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "01_glaucoma.md" in body["answer"]
    assert body["sources"] == ["01_glaucoma.md"]
    assert body["refused"] is False
    assert body["retrieved_chunks"] is None  # not requested
    assert fake.calls == ["What is glaucoma?"]


def test_query_include_chunks(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    resp = c.post(
        "/query", json={"question": "What is glaucoma?", "include_chunks": True}
    )
    assert resp.status_code == 200
    chunks = resp.json()["retrieved_chunks"]
    assert isinstance(chunks, list) and len(chunks) == 1
    assert chunks[0]["section"] == "Glaucoma"


def test_query_missing_api_key_returns_503(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    no_key = dataclasses.replace(api_mod.settings, anthropic_api_key=None)
    monkeypatch.setattr(api_mod, "settings", no_key)
    resp = c.post("/query", json={"question": "What is glaucoma?"})
    assert resp.status_code == 503


def test_query_empty_question_returns_422(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    resp = c.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_local_backend_allows_query_without_key(client, monkeypatch):
    """With RAG_BACKEND=local the /query endpoint works without an API key."""
    c, _ = client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    local_settings = dataclasses.replace(
        api_mod.settings, backend="local", anthropic_api_key=None
    )
    monkeypatch.setattr(api_mod, "settings", local_settings)
    resp = c.post("/query", json={"question": "What is glaucoma?"})
    assert resp.status_code == 200
    assert "01_glaucoma.md" in resp.json()["answer"]


def test_health_reports_backend(client):
    c, _ = client
    body = c.get("/health").json()
    assert "backend" in body
    assert "generation_ready" in body


def test_unhandled_exception_returns_clean_500(client, monkeypatch):
    """
    Regression: an exception that isn't a GenerationError (e.g. a bug in the
    pipeline, or an SDK-internal TypeError) must not leak a raw traceback to
    the client — the global exception handler should map it to a clean,
    generic 500 JSON body instead.

    Uses a dedicated TestClient with raise_server_exceptions=False: that flag
    reproduces how a real ASGI server behaves (client only ever sees the HTTP
    response). The shared `client` fixture keeps the default True so that any
    *other* test hitting an unexpected 500 still fails loudly.
    """
    _, fake = client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def boom(question):
        raise TypeError("simulated unexpected SDK-internal failure")

    fake.query = boom
    with TestClient(api_mod.app, raise_server_exceptions=False) as c2:
        resp = c2.post("/query", json={"question": "What is glaucoma?"})
    assert resp.status_code == 500
    assert "detail" in resp.json()
    assert "TypeError" not in resp.text
    assert "Traceback" not in resp.text
