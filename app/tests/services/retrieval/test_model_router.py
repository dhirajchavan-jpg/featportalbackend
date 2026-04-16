import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.retrieval.model_router import (
    ModelRouter,
    get_model_router
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def fake_llm():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def router(monkeypatch, fake_llm):
    monkeypatch.setattr(
        "app.services.retrieval.model_router.llm_provider.get_router_llm",
        lambda: fake_llm
    )
    monkeypatch.setattr(
        "app.services.retrieval.model_router.llm_provider.get_simple_llm",
        lambda: "SIMPLE_MODEL"
    )
    monkeypatch.setattr(
        "app.services.retrieval.model_router.llm_provider.get_complex_llm",
        lambda: "COMPLEX_MODEL"
    )

    return ModelRouter()


# -----------------------------
# Rule-based retrieval
# -----------------------------

@pytest.mark.asyncio
async def test_complex_by_long_query(router):
    result = await router.route_query(
        query="this is a very long query " * 5,
        query_metadata={"token_count": 30}
    )
    assert result == "complex"


@pytest.mark.asyncio
async def test_complex_by_keywords(router):
    result = await router.route_query(
        query="Provide a comprehensive impact analysis",
        query_metadata={}
    )
    assert result == "complex"


@pytest.mark.asyncio
async def test_simple_by_short_query(router):
    result = await router.route_query(
        query="What is Python?",
        query_metadata={"token_count": 3}
    )
    assert result == "simple"


@pytest.mark.asyncio
async def test_simple_by_definition(router):
    result = await router.route_query(
        query="Define machine learning",
        query_metadata={}
    )
    assert result == "simple"


# -----------------------------
# AI-based retrieval
# -----------------------------

@pytest.mark.asyncio
async def test_ai_routes_to_simple(router, fake_llm):
    fake_llm.ainvoke.return_value = SimpleNamespace(content="simple")

    result = await router.route_query(
        query="Explain Python briefly",
        query_metadata={"token_count": 15}
    )

    assert result == "simple"
    fake_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_ai_routes_to_complex(router, fake_llm):
    fake_llm.ainvoke.return_value = SimpleNamespace(content="complex")

    result = await router.route_query(
        query="Explain distributed systems tradeoffs",
        query_metadata={"token_count": 15}
    )

    assert result == "complex"


@pytest.mark.asyncio
async def test_ai_ambiguous_response_defaults_complex(router, fake_llm):
    fake_llm.ainvoke.return_value = SimpleNamespace(content="maybe")

    result = await router.route_query(
        query="Some unclear query",
        query_metadata={"token_count": 12}
    )

    assert result == "complex"


@pytest.mark.asyncio
async def test_ai_exception_defaults_complex(router, fake_llm):
    fake_llm.ainvoke.side_effect = RuntimeError("LLM failure")

    result = await router.route_query(
        query="Some query",
        query_metadata={"token_count": 12}
    )

    assert result == "complex"


# -----------------------------
# Project-level router override
# -----------------------------

@pytest.mark.asyncio
async def test_project_router_override(monkeypatch, router):
    fake_override_llm = AsyncMock()
    fake_override_llm.ainvoke.return_value = SimpleNamespace(content="simple")

    monkeypatch.setattr(
        "app.services.retrieval.model_router.Ollama",
        lambda **kwargs: fake_override_llm
    )

    result = await router.route_query(
        query="Explain REST APIs",
        query_metadata={"token_count": 12},
        router_model_name="custom-router"
    )

    assert result == "simple"
    fake_override_llm.ainvoke.assert_called_once()


# -----------------------------
# Model getters
# -----------------------------

def test_get_model_simple(router):
    assert router.get_model("simple") == "SIMPLE_MODEL"


def test_get_model_complex(router):
    assert router.get_model("complex") == "COMPLEX_MODEL"


def test_get_model_info_simple():
    router = ModelRouter()
    info = router.get_model_info("simple")
    assert "model_name" in info


# -----------------------------
# Singleton behavior
# -----------------------------

def test_get_model_router_singleton(monkeypatch):
    from app.services.retrieval import model_router
    model_router._model_router = None

    r1 = get_model_router()
    r2 = get_model_router()

    assert r1 is r2
