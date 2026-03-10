"""Connection smoke tests — verify API keys and services are reachable.

These are NOT unit tests. They make real API calls and cost a tiny
amount of money (~$0.000001 per run). Run them once to confirm your
environment is wired up correctly, then use the unit tests for TDD.

Usage:
    pytest tests/test_connections.py -v
    pytest tests/test_connections.py -v -k "openai"   # one test only
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skip_if_missing(env_var: str):
    """Skip a test if the required environment variable is not set."""
    if not os.getenv(env_var):
        pytest.skip(f"{env_var} not set in .env")


# ── OpenAI ────────────────────────────────────────────────────────────────────

def test_openai_key_is_set():
    """OPENAI_API_KEY must be present in environment."""
    key = os.getenv("OPENAI_API_KEY", "")
    assert key, "OPENAI_API_KEY is not set — check your .env file"
    assert key.startswith("sk-"), f"OPENAI_API_KEY looks malformed: {key[:10]}..."


def test_openai_embedding():
    """OpenAI embedding API must return a 1536-dim vector for a test string."""
    _skip_if_missing("OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        input="FedRAMP authorization boundary test",
    )

    embedding = response.data[0].embedding
    assert isinstance(embedding, list), "Expected a list"
    assert len(embedding) == 1536, f"Expected 1536 dims, got {len(embedding)}"
    print(f"\n  ✓ Embedding received: {len(embedding)} dimensions")


def test_openai_chat():
    """OpenAI chat API must return a non-empty response."""
    _skip_if_missing("OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=5,
    )

    answer = response.choices[0].message.content.strip().lower()
    assert answer, "Got empty response from OpenAI chat"
    print(f"\n  ✓ OpenAI chat responded: '{answer}'")


# ── Anthropic ─────────────────────────────────────────────────────────────────

def test_anthropic_key_is_set():
    """ANTHROPIC_API_KEY must be present if LLM_PROVIDER=anthropic."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider != "anthropic":
        pytest.skip("LLM_PROVIDER is not anthropic — skipping Anthropic key check")

    key = os.getenv("ANTHROPIC_API_KEY", "")
    assert key, "ANTHROPIC_API_KEY is not set — check your .env file"
    assert key.startswith("sk-ant-"), f"ANTHROPIC_API_KEY looks malformed: {key[:12]}..."


def test_anthropic_chat():
    """Anthropic chat API must return a non-empty response (only if provider=anthropic)."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider != "anthropic":
        pytest.skip("LLM_PROVIDER is not anthropic — skipping")
    _skip_if_missing("ANTHROPIC_API_KEY")

    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Use LLM_MODEL if set to an Anthropic model, otherwise fall back to a known good one.
    model = os.getenv("LLM_MODEL", "")
    if not model or not model.startswith("claude"):
        model = "claude-haiku-4-5-20251001"

    response = client.messages.create(
        model=model,
        max_tokens=5,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    )

    answer = response.content[0].text.strip().lower()
    assert answer, "Got empty response from Anthropic"
    print(f"\n  ✓ Anthropic chat responded: '{answer}'")


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def test_chromadb_local():
    """ChromaDB must be importable and able to create an in-memory collection."""
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.create_collection("smoke_test")
    assert collection is not None
    client.delete_collection("smoke_test")
    print("\n  ✓ ChromaDB in-memory client works")
