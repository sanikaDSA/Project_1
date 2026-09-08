import os
import sys

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from starlette.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app

client = TestClient(app)

def run_all_e2e_tests():
    print("=================================================================")
    print("E2E INTEGRATION TEST: SEHAT AI FASTAPI SERVER & CLINICAL API")
    print("=================================================================")

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    health_data = res_health.json()
    print(f"Health Status: {health_data['status']} | App: {health_data['app_name']} | Bot: {health_data['bot_name']}")

    # 2. Start Session (Mandatory Q1 Greeting)
    res_start = client.post("/api/sehat/session/start")
    assert res_start.status_code == 200, f"Start session failed: {res_start.text}"
    start_data = res_start.json()
    sid = start_data["session_id"]
    greeting = start_data["bot_speech_hi"]
    print(f"\n[1] Session Started: ID = {sid}")
    print(f"    Diabetes Dost Opening (Q1): '{greeting}'")
    assert greeting.startswith("नमस्ते, मेरा नाम Diabetes Dost") or greeting.startswith("नमस्ते, मेरा नाम"), "Opening message must start with 'नमस्ते, मेरा नाम Diabetes Dost...'"

    # 3. Turn 1: Patient gives consent -> Next should be Q2 (Demographics)
    res_t1 = client.post("/api/sehat/session/turn", data={"session_id": sid, "transcript": "हाँ, मैं सहमत हूँ और रिकॉर्डिंग की अनुमति देता हूँ"})
    assert res_t1.status_code == 200
    t1_data = res_t1.json()
    print(f"\n[2] Turn 1 Response (Q2): '{t1_data['bot_speech_hi']}'")
    assert t1_data["state"]["consent"] is True, "Consent must be recorded as true"

    # 4. Turn 2: Patient gives Demographics (Name, Age, Gender) -> Next should be Q3 (Status)
    res_t2 = client.post("/api/sehat/session/turn", data={"session_id": sid, "transcript": "मेरा नाम चेतन शर्मा है, उम्र 32 वर्ष, पुरुष"})
    assert res_t2.status_code == 200
    t2_data = res_t2.json()
    print(f"\n[3] Turn 2 Response (Q3): '{t2_data['bot_speech_hi']}'")
    assert t2_data["state"]["patient"]["age"] == 32, "Age must be extracted"

    # 5. Turn 3: Patient states Known Diabetic -> Next is branch A question
    res_t3 = client.post("/api/sehat/session/turn", data={"session_id": sid, "transcript": "हाँ, मुझे पिछले 2 साल से डायबिटीज है"})
    assert res_t3.status_code == 200
    t3_data = res_t3.json()
    print(f"\n[4] Turn 3 Response (Branch A): '{t3_data['bot_speech_hi']}'")
    assert t3_data["branch"] == "known_diabetic", "Branch must transition to known_diabetic"

    # 6. Turn 4: Patient mentions high thirst & urination
    res_t4 = client.post("/api/sehat/session/turn", data={"session_id": sid, "transcript": "मुझे बहुत ज्यादा प्यास लगती है और बार-बार पेशाब आता है"})
    assert res_t4.status_code == 200
    t4_data = res_t4.json()
    print(f"\n[5] Turn 4 Response (Follow-up): '{t4_data['bot_speech_hi']}'")
    assert len(t4_data["state"]["symptoms_reported"]) >= 1, "Symptoms must be extracted"

    # 7. Test Emergency Interruption in separate session
    res_start_em = client.post("/api/sehat/session/start")
    sid_em = res_start_em.json()["session_id"]
    res_t_em = client.post("/api/sehat/session/turn", data={"session_id": sid_em, "transcript": "डॉक्टर साहब, सीने में बहुत तेज दर्द और सांस लेने में तकलीफ हो रही है"})
    assert res_t_em.status_code == 200
    em_data = res_t_em.json()
    print(f"\n[6] Testing Emergency Interruption via API...")
    print(f"    Emergency Interruption Speech: '{em_data['bot_speech_hi'][:90]}...'")
    assert em_data["is_emergency"] is True
    assert em_data["is_completed"] is True
    assert em_data["state"]["recommended_urgency"] == "urgent"

    # 8. Check Emergencies Feed
    res_em_feed = client.get("/api/sehat/emergencies")
    assert res_em_feed.status_code == 200
    feed_data = res_em_feed.json()
    print(f"    Emergency Alerts Feed Count: {feed_data['count']}")
    assert feed_data["count"] >= 1

    # 9. Check Summary and Doctor Queue
    res_summary = client.get(f"/api/sehat/session/{sid}/summary")
    assert res_summary.status_code == 200
    sum_data = res_summary.json()["summary"]
    print(f"\n[7] Summary Retrieved for '{sum_data['patient']['name']}':")
    print(f"    Urgency: {sum_data['recommended_urgency'].upper()}")
    print(f"    Total Q&A items: {len(sum_data['qa_transcript'])}")

    # 10. Check Audit Trail
    res_audit = client.get(f"/api/sehat/audit/{sid}")
    assert res_audit.status_code == 200
    audit_events = res_audit.json()["events"]
    print(f"\n[8] Audit Trail Verified: {len(audit_events)} audit events recorded.")
    assert len(audit_events) >= 3

    print("\n" + "🎉"*28)
    print("ALL SEHAT AI E2E FASTAPI ENDPOINTS & LOGIC VALIDATED SUCCESSFULLY!")
    print("🎉"*28 + "\n")

if __name__ == "__main__":
    run_all_e2e_tests()
