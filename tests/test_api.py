# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from cine_bert.api.main import app

client = TestClient(app)


def test_root():
    """
    GET / should return a welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert isinstance(data["message"], str)
    assert "Welcome" in data["message"]


def test_predict_valid():
    """
    POST /predict with valid input should return scores list.
    """
    payload = {"text": "This movie was fantastic!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check input echoed
    assert data.get("input") == payload["text"]

    # Check scores format
    assert "scores" in data
    scores = data["scores"]
    assert isinstance(scores, list)
    assert len(scores) >= 2  # at least two labels
    for entry in scores:
        assert "label" in entry and "score" in entry
        assert isinstance(entry["label"], str)
        assert isinstance(entry["score"], float)


def test_predict_invalid():
    """
    POST /predict with invalid payload should return 422 or 500.
    """
    # Missing 'text' field
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity

    # Malformed JSON
    response = client.post("/predict", data="not a json")
    assert response.status_code in (400, 422)