import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from app.middleware.monitoring import get_analytics_summary
from unittest.mock import patch


from app.middleware.monitoring import (
    MonitoringMiddleware,
    analytics_data,
    ACTIVE_REQUESTS,
)

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(autouse=True)
def clear_analytics():
    """
    Summary:
        Ensures a clean analytics state before and after each test.

    Explanation:
        This fixture automatically clears the global analytics store
        before and after every test, preventing cross-test contamination
        and ensuring deterministic metric assertions.
    """
    analytics_data.clear()
    yield
    analytics_data.clear()


@pytest.fixture
def app():
    """
    Summary:
        Creates a FastAPI application instrumented with MonitoringMiddleware.

    Explanation:
        The app includes:
        - A successful endpoint (/ok)
        - A failing endpoint (/error)
        This allows validation of both success and error metric paths.
    """
    app = FastAPI()
    app.add_middleware(MonitoringMiddleware)

    @app.get("/ok")
    async def ok():
        return {"msg": "ok"}

    @app.get("/error")
    async def error():
        raise RuntimeError("boom")

    return app


@pytest.fixture
def client(app):
    """
    Summary:
        Provides a test client for API requests.

    Explanation:
        Uses FastAPI's TestClient to simulate HTTP requests
        through the monitoring middleware.
    """
    return TestClient(app)



def test_success_request_updates_analytics(client):
    """
    Summary:
        Successful requests must update analytics metrics correctly.

    Explanation:
        This test validates that:
        - The request is counted
        - Success count increments
        - Error count remains zero
        - Latency and last_request timestamps are recorded
    """
    response = client.get("/ok")

    assert response.status_code == 200

    key = "GET:/ok"
    assert key in analytics_data

    data = analytics_data[key]

    assert data["total_requests"] == 1
    assert data["success_count"] == 1
    assert data["error_count"] == 0
    assert len(data["latencies"]) == 1
    assert data["avg_latency"] > 0
    assert data["last_request"] is not None


def test_error_request_updates_error_metrics(app):
    """
    Summary:
        Failed requests must be tracked as errors in analytics.

    Explanation:
        This test ensures that when an endpoint raises an exception:
        - The request is recorded
        - Error count increments
        - Success count remains zero
        - Latency is still captured
    """
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/error")
    assert response.status_code == 500  # FastAPI returns 500 for RuntimeError

    key = "GET:/error"
    assert key in analytics_data  # now the middleware recorded it

    data = analytics_data[key]

    assert data["total_requests"] == 1
    assert data["success_count"] == 0
    assert data["error_count"] == 1
    assert len(data["latencies"]) == 1
    assert data["avg_latency"] > 0


    data = analytics_data[key]

    assert data["total_requests"] == 1
    assert data["success_count"] == 0
    assert data["error_count"] == 1


def test_active_requests_decrement_on_success(client):
    """
    Summary:
        Active request gauge must decrement after successful requests.

    Explanation:
        This test validates that the ACTIVE_REQUESTS metric
        does not leak counts after a normal request lifecycle.
    """
    before = ACTIVE_REQUESTS.collect()[0].samples[0].value

    client.get("/ok")

    after = ACTIVE_REQUESTS.collect()[0].samples[0].value

    assert before == after


def test_active_requests_decrement_on_exception(client):
    """
    Summary:
        Active request gauge must decrement even when exceptions occur.

    Explanation:
        Ensures that in case of unhandled exceptions, the middleware
        still decrements the active request counter, preventing
        metric inflation.
    """
    before = ACTIVE_REQUESTS.collect()[0].samples[0].value

    with pytest.raises(RuntimeError):
        client.get("/error")

    after = ACTIVE_REQUESTS.collect()[0].samples[0].value

    assert before == after


def test_metrics_route_is_skipped(app):
    """
    Summary:
        The /metrics endpoint must not be monitored.

    Explanation:
        This test confirms that internal observability endpoints
        are excluded from analytics tracking to avoid recursive
        or polluted metrics.
    """
    client = TestClient(app)

    client.get("/metrics")

    # No analytics should be created
    assert analytics_data == {}



def test_get_analytics_summary(client):
    """
    Summary:
        Aggregated analytics summary must reflect collected metrics.

    Explanation:
        This test verifies that the summary API:
        - Aggregates metrics per endpoint
        - Computes error rate correctly
        - Reports average latency in milliseconds
    """
    client.get("/ok")
    client.get("/ok")

    summary = get_analytics_summary()

    assert len(summary) == 1
    item = summary[0]

    assert item["endpoint"] == "/ok"
    assert item["total_requests"] == 2
    assert item["error_rate"] == 0
    assert item["avg_latency_ms"] > 0
