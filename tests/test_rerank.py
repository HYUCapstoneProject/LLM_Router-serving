from __future__ import annotations


def test_scores_are_positionally_aligned_with_candidates(rerank_client, auth_headers):
    """The gateway maps scores back to rows by index, so order is the contract."""
    response = rerank_client.post(
        "/router/rerank",
        json={
            "query": "what is the capital of australia",
            "candidates": [
                "unrelated text about cooking",
                "which city is the capital of australia",
            ],
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["scores"]) == 2
    # The second candidate is the paraphrase; the endpoint must not reorder.
    assert body["scores"][1] > body["scores"][0]


def test_model_version_is_returned_for_threshold_pinning(rerank_client, auth_headers):
    """A threshold is only valid for the weights it was measured on."""
    response = rerank_client.post(
        "/router/rerank",
        json={"query": "q", "candidates": ["c"]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["modelVersion"] == "cross-encoder/quora-distilroberta-base@f62e7a4"


def test_returns_503_when_reranker_is_not_configured(client, auth_headers):
    """Without the model the gateway must fail closed to a cache miss."""
    response = client.post(
        "/router/rerank",
        json={"query": "q", "candidates": ["c"]},
        headers=auth_headers,
    )
    assert response.status_code == 503
    assert response.json()["code"] == "RERANK_MODEL_NOT_READY"


def test_requires_authentication(rerank_client):
    response = rerank_client.post(
        "/router/rerank", json={"query": "q", "candidates": ["c"]}
    )
    assert response.status_code == 401


def test_rejects_empty_candidate_list(rerank_client, auth_headers):
    response = rerank_client.post(
        "/router/rerank",
        json={"query": "q", "candidates": []},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_rejects_blank_candidate(rerank_client, auth_headers):
    response = rerank_client.post(
        "/router/rerank",
        json={"query": "q", "candidates": ["ok", "   "]},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_rejects_too_many_candidates(rerank_client, auth_headers, settings):
    response = rerank_client.post(
        "/router/rerank",
        json={
            "query": "q",
            "candidates": ["c"] * (settings.rerank_max_candidates + 1),
        },
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "TOO_MANY_CANDIDATES"


def test_rejects_unknown_fields(rerank_client, auth_headers):
    """StrictModel keeps a typo from silently becoming a default."""
    response = rerank_client.post(
        "/router/rerank",
        json={"query": "q", "candidates": ["c"], "threshold": 0.9},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_health_reports_reranker_without_gating_readiness(client, rerank_client):
    """The reranker is optional; its absence must not take the service down."""
    disabled = client.get("/health").json()
    assert disabled["status"] == "ok"
    assert disabled["rerankModel"] == "disabled"

    enabled = rerank_client.get("/health").json()
    assert enabled["status"] == "ok"
    assert enabled["rerankModel"] == "loaded"
