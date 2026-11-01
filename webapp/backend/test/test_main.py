from fastapi.testclient import TestClient
from fastapi import status


def test_root_endpoint(client: TestClient):
    """
    Test that root endpoint returns expected data.
    """
    response = client.get("/")
    data = response.json()

    assert response.status_code == status.HTTP_200_OK
    assert data.get("message") == "Authorship Verification DT - backend"
    assert data.get("version") == "1.0.0"
    assert data.get("docs") == "/docs"


def test_return_health_check(client: TestClient):
    """
    Test that health endpoint returns expected data.
    """
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

def test_root_endpoint_method_not_allowed(client):
    """Test that POST is not allowed on root endpoint."""
    response = client.post("/")
    assert response.status_code == 405

def test_health_endpoint_method_not_allowed(client):
    """Test that POST is not allowed on health endpoint."""
    response = client.post("/health")
    assert response.status_code == 405