import pytest
from fastapi.testclient import TestClient
from ..core import config

def test_predict_success(client: TestClient, mock_predict, sample_texts):
    """Test prediction success with mocked output of the ml.predict function"""
    mock_predict.return_value = {
        "same_author_probability": 0.75
    }

    response = client.post("/predict/", json=sample_texts)
    assert response.status_code == 200

    data = response.json()
    assert data["same_author_probability"] == 0.75

    mock_predict.assert_called_once_with(sample_texts["text1"], sample_texts["text2"])

def test_predict_text1_too_short(client, mock_predict, sample_text):
    """Test prediction fails when text1 is too short."""
    short_text = "Short text"

    response = client.post("/predict/", json={
        "text1": short_text,
        "text2": sample_text["text"]
    })

    assert response.status_code == 422  # Unprocessable content error
    mock_predict.assert_not_called()

def test_predict_text2_too_short(client, mock_predict, sample_text):
    """Test prediction fails when text1 is too short."""
    short_text = "Short text"

    response = client.post("/predict/", json={
        "text1": sample_text["text"],
        "text2": short_text
    })

    assert response.status_code == 422  # Unprocessable content error
    mock_predict.assert_not_called()

def test_predict_both_texts_too_short(client, mock_predict, sample_text):
    """Test prediction fails when both texts are too short."""
    short_text = "Short text"
    response = client.post("/predict/", json={
        "text1": short_text,
        "text2": short_text
    })
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_predict_text1_too_long(client, mock_predict, sample_text):
    """Test prediction fails when text1 is too long."""
    long_text = sample_text["text"] * 10
    response = client.post("/predict/", json={
        "text1": long_text,
        "text2": sample_text["text"]
    })

    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_predict_text2_too_long(client, mock_predict, sample_text):
    """Test prediction fails when text2 is too long."""
    long_text = sample_text["text"] * 10
    response = client.post("/predict/", json={
        "text1": sample_text["text"],
        "text2": long_text
    })
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_predict_both_texts_too_long(client, mock_predict, sample_text):
    """Test prediction fails when both texts are too long."""
    long_text = sample_text["text"] * 10
    response = client.post("/predict/", json={
        "text1": long_text,
        "text2": long_text
    })
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_predict_missing_text1(client, mock_predict, sample_text):
    """Test prediction fails when text1 is missing."""
    response = client.post("/predict/", json={
        "text1": sample_text["text"]
    })
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_predict_missing_text2(client, mock_predict, sample_text):
    """Test prediction fails when text2 is missing."""
    response = client.post("/predict/", json={
        "text2": sample_text["text"]
    })
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_predict_missing_both_texts(client, mock_predict):
    """Test prediction fails when both texts are missing."""
    response = client.post("/predict/", json={})
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_predict_texts_with_null_values(client, mock_predict):
    response = client.post("/predict/", json={
        "text1": None,
        "text2": None
    })
    assert response.status_code == 422
    mock_predict.assert_not_called()


def test_predict_unicode_characters(client, mock_predict):
    """Test prediction with unicode characters."""
    text1 = "这是一个测。😂✌️。😂" * 2_500
    text2 = "这是另一🐻💀️。😂️" * 2_500

    mock_predict.return_value = {
        "same_author_probability": 0.65
    }

    response = client.post("/predict/", json={
        "text1": text1,
        "text2": text2
    })

    assert response.status_code == 200
    data = response.json()
    assert data["same_author_probability"] == 0.65

    mock_predict.assert_called_once_with(text1, text2)

def test_predict_special_characters(client, mock_predict):
    """Text predicti on with text samples made of special characters only"""
    text = "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?" * 1000
    mock_predict.return_value = {
        "same_author_probability": 0.45
    }

    response = client.post("/predict/", json={
        "text1": text,
        "text2": text
    })

    assert response.status_code == 200
    data = response.json()
    assert data["same_author_probability"] == 0.45
    mock_predict.assert_called_once_with(text, text)

def test_predict_newlines_and_tabs(client, mock_predict):
    """Test prediction with texts containing newlines and tabs."""
    text = "\n\t\n\t\n\t\n\t\n\t\n\t\n\t\n\t\n\t" * 1000

    mock_predict.return_value = {
        "same_author_probability": 0.45
    }

    response = client.post("/predict/", json={
        "text1": text,
        "text2": text
    })

    assert response.status_code == 200
    data = response.json()
    assert data["same_author_probability"] == 0.45

    mock_predict.assert_called_once_with(text, text)


def test_predict_whitespace(client, mock_predict):
    """Test prediction with texts containing newlines and tabs."""
    text = "            " * 1000

    mock_predict.return_value = {
        "same_author_probability": 0.45
    }

    response = client.post("/predict/", json={
        "text1": text,
        "text2": text
    })

    assert response.status_code == 200
    data = response.json()
    assert data["same_author_probability"] == 0.45

    mock_predict.assert_called_once_with(text, text)

def test_predict_numeric_values(client, mock_predict):
    """Test prediction with texts containing newlines and tabs."""
    text = "1234567890" * 1000

    mock_predict.return_value = {
        "same_author_probability": 0.45
    }

    response = client.post("/predict/", json={
        "text1": text,
        "text2": text
    })

    assert response.status_code == 200
    data = response.json()
    assert data["same_author_probability"] == 0.45

    mock_predict.assert_called_once_with(text, text)


def test_empty_json_body(client, mock_predict):
    """Test prediction with empty json body."""
    response = client.post("/predict/", json={})
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_invalid_json_body(client, mock_predict):
    """Test prediction with invalid json body."""
    response = client.post("/predict/", data="This is not JSON")
    assert response.status_code == 422
    mock_predict.assert_not_called()

def test_invalid_predict_http_method(client):
    """Test that GET is not allowed on predict endpoint."""
    response = client.get("/predict/")
    assert response.status_code == 405 #method not allowed code

