"""Backend regression tests for CropDoctor AI."""
import time
import uuid
import requests
import pytest


# ---------- Health ----------
def test_root_health(base_url):
    r = requests.get(f"{base_url}/api/", timeout=15)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


# ---------- Auth ----------
class TestAuth:
    def test_login_seeded_farmer(self, base_url):
        r = requests.post(f"{base_url}/api/auth/login",
                          json={"email": "farmer@cropdoctor.ai", "password": "Farmer@123"}, timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert "token" in data and "user" in data
        assert data["user"]["email"] == "farmer@cropdoctor.ai"
        assert data["user"]["role"] == "farmer"

    def test_login_invalid(self, base_url):
        r = requests.post(f"{base_url}/api/auth/login",
                          json={"email": "farmer@cropdoctor.ai", "password": "wrong"}, timeout=20)
        assert r.status_code == 401

    def test_register_new_user(self, base_url):
        email = f"test_{uuid.uuid4().hex[:8]}@cropdoctor.ai"
        r = requests.post(f"{base_url}/api/auth/register",
                          json={"email": email, "password": "Pass@1234",
                                "name": "TEST User", "role": "farmer", "language": "en"}, timeout=20)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["user"]["email"] == email
        # duplicate
        r2 = requests.post(f"{base_url}/api/auth/register",
                           json={"email": email, "password": "Pass@1234", "name": "TEST"}, timeout=20)
        assert r2.status_code == 400

    def test_me_with_token(self, base_url, farmer_token):
        r = requests.get(f"{base_url}/api/auth/me",
                         headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r.status_code == 200
        assert r.json()["email"] == "farmer@cropdoctor.ai"

    def test_me_without_token(self, base_url):
        r = requests.get(f"{base_url}/api/auth/me", timeout=10)
        assert r.status_code == 401

    def test_patch_me(self, base_url, farmer_token):
        new_loc = f"TEST_loc_{uuid.uuid4().hex[:5]}"
        r = requests.patch(f"{base_url}/api/auth/me",
                           headers={"Authorization": f"Bearer {farmer_token}"},
                           json={"location": new_loc, "language": "en"}, timeout=20)
        assert r.status_code == 200
        # GET to verify persistence
        r2 = requests.get(f"{base_url}/api/auth/me",
                          headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r2.json()["location"] == new_loc


# ---------- Disease Scan (LLM) ----------
class TestScans:
    @pytest.fixture(scope="class")
    def created_scan(self, base_url, farmer_token, test_image_b64):
        r = requests.post(f"{base_url}/api/scans",
                          headers={"Authorization": f"Bearer {farmer_token}"},
                          json={"image_base64": test_image_b64, "crop_name": "Tomato",
                                "model": "gemini", "notes": "TEST_scan"}, timeout=120)
        assert r.status_code == 200, f"scan failed: {r.status_code} {r.text[:300]}"
        return r.json()

    def test_scan_gemini_response_shape(self, created_scan):
        s = created_scan
        for f in ["scan_id", "disease_name", "severity", "confidence",
                  "symptoms", "causes", "treatments", "preventive_measures", "next_actions"]:
            assert f in s, f"missing field {f}"
        assert s["severity"] in ["healthy", "mild", "moderate", "severe"]
        assert isinstance(s["treatments"], list)
        assert isinstance(s["preventive_measures"], list)

    def test_scan_openai(self, base_url, farmer_token, test_image_b64):
        r = requests.post(f"{base_url}/api/scans",
                          headers={"Authorization": f"Bearer {farmer_token}"},
                          json={"image_base64": test_image_b64, "crop_name": "Potato",
                                "model": "openai", "notes": "TEST_openai"}, timeout=120)
        assert r.status_code == 200, r.text[:400]
        d = r.json()
        assert "disease_name" in d and "scan_id" in d

    def test_list_scans(self, base_url, farmer_token, created_scan):
        r = requests.get(f"{base_url}/api/scans",
                         headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r.status_code == 200
        scans = r.json()
        assert isinstance(scans, list)
        assert any(s["scan_id"] == created_scan["scan_id"] for s in scans)

    def test_get_scan_by_id(self, base_url, farmer_token, created_scan):
        sid = created_scan["scan_id"]
        r = requests.get(f"{base_url}/api/scans/{sid}",
                         headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r.status_code == 200
        assert r.json()["scan_id"] == sid

    def test_delete_scan(self, base_url, farmer_token, created_scan):
        sid = created_scan["scan_id"]
        r = requests.delete(f"{base_url}/api/scans/{sid}",
                            headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r.status_code == 200
        assert r.json().get("deleted") == 1
        # verify gone
        r2 = requests.get(f"{base_url}/api/scans/{sid}",
                          headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r2.status_code == 404


# ---------- Dashboard ----------
def test_dashboard(base_url, farmer_token):
    r = requests.get(f"{base_url}/api/dashboard",
                     headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
    assert r.status_code == 200
    d = r.json()
    for f in ["total_scans", "healthy_scans", "diseased_scans", "health_score",
              "severity_breakdown", "recent_scans", "top_diseases"]:
        assert f in d
    assert isinstance(d["recent_scans"], list)
    assert isinstance(d["severity_breakdown"], dict)


# ---------- Weather (mock) ----------
def test_weather(base_url, farmer_token):
    r = requests.get(f"{base_url}/api/weather",
                     headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
    assert r.status_code == 200
    d = r.json()
    assert "location" in d and "current" in d
    assert len(d["forecast"]) == 7
    assert isinstance(d["alerts"], list)


# ---------- Knowledge ----------
class TestKnowledge:
    def test_list(self, base_url):
        r = requests.get(f"{base_url}/api/knowledge", timeout=15)
        assert r.status_code == 200
        items = r.json()
        assert len(items) >= 5
        assert any(i["id"] == "kb_late_blight" for i in items)

    def test_get_one(self, base_url):
        r = requests.get(f"{base_url}/api/knowledge/kb_late_blight", timeout=15)
        assert r.status_code == 200
        assert r.json()["id"] == "kb_late_blight"

    def test_get_missing(self, base_url):
        r = requests.get(f"{base_url}/api/knowledge/kb_does_not_exist", timeout=15)
        assert r.status_code == 404


# ---------- Chatbot ----------
def test_chatbot(base_url, farmer_token):
    r = requests.post(f"{base_url}/api/chatbot",
                      headers={"Authorization": f"Bearer {farmer_token}"},
                      json={"message": "How to treat tomato late blight?"}, timeout=120)
    assert r.status_code == 200, r.text[:300]
    d = r.json()
    assert "reply" in d and len(d["reply"]) > 5
    assert "session_id" in d


def test_chatbot_history(base_url, farmer_token):
    r = requests.get(f"{base_url}/api/chatbot/history",
                     headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------- Experts ----------
class TestExperts:
    def test_list_experts(self, base_url):
        r = requests.get(f"{base_url}/api/experts", timeout=15)
        assert r.status_code == 200
        experts = r.json()
        assert len(experts) >= 3
        assert any(e["user_id"] == "expert_1" for e in experts)

    def test_send_message_and_thread(self, base_url, farmer_token):
        r = requests.post(f"{base_url}/api/expert/messages",
                          headers={"Authorization": f"Bearer {farmer_token}"},
                          json={"expert_id": "expert_1", "message": "TEST_Hi"}, timeout=20)
        assert r.status_code == 200
        time.sleep(1.2)
        r2 = requests.get(f"{base_url}/api/expert/messages/expert_1",
                          headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r2.status_code == 200
        msgs = r2.json()
        assert any(m["content"] == "TEST_Hi" for m in msgs)
        # auto-reply present
        assert any(m["from_user"] == "expert_1" for m in msgs)


# ---------- Notifications ----------
def test_notifications(base_url, farmer_token):
    r = requests.get(f"{base_url}/api/notifications",
                     headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
    assert r.status_code == 200
    arr = r.json()
    assert isinstance(arr, list) and len(arr) >= 1


# ---------- Admin ----------
class TestAdmin:
    def test_admin_stats_ok(self, base_url, admin_token):
        r = requests.get(f"{base_url}/api/admin/stats",
                         headers={"Authorization": f"Bearer {admin_token}"}, timeout=20)
        assert r.status_code == 200
        d = r.json()
        for f in ["total_users", "farmers", "experts", "total_scans", "diseased_scans"]:
            assert f in d
        assert d["farmers"] >= 1

    def test_admin_stats_denied_for_farmer(self, base_url, farmer_token):
        r = requests.get(f"{base_url}/api/admin/stats",
                         headers={"Authorization": f"Bearer {farmer_token}"}, timeout=20)
        assert r.status_code == 403
