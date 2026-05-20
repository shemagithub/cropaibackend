import os
import pytest
import requests

BASE_URL = os.environ.get("EXPO_PUBLIC_BACKEND_URL", "https://cropcare-ai-10.preview.emergentagent.com").rstrip("/")


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture
def api_client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


@pytest.fixture(scope="session")
def farmer_token():
    r = requests.post(f"{BASE_URL}/api/auth/login",
                      json={"email": "farmer@cropdoctor.ai", "password": "Farmer@123"}, timeout=30)
    assert r.status_code == 200, f"farmer login failed: {r.status_code} {r.text}"
    return r.json()["token"]


@pytest.fixture(scope="session")
def admin_token():
    r = requests.post(f"{BASE_URL}/api/auth/login",
                      json={"email": "admin@cropdoctor.ai", "password": "Admin@123"}, timeout=30)
    assert r.status_code == 200, f"admin login failed: {r.status_code} {r.text}"
    return r.json()["token"]


@pytest.fixture(scope="session")
def test_image_b64():
    with open("/tmp/leaf_b64.txt") as f:
        return f.read().strip()
