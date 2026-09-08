import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend root is in path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding='utf-8')

from app import app, GLOBAL_CURA_SESSIONS, EMERGENCY_ALERTS_FEED

client = TestClient(app)

def test_e2e_cura_workflow():
    print("\n" + "=" * 65)
    print("E2E INTEGRATION TEST: CURA FASTAPI SERVER & CLINICAL API")
    print("=" * 65)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    print(f"Health Status: {data['status']} | App: {data['app_name']}")
    assert data["status"] == "online"

    # 2. Start new session
    res = client.post("/api/cura/session/start")
    assert res.status_code == 200
    data = res.json()
    session_id = data["session_id"]
    print(f"\n[1] Session Started: ID = {session_id}")
    print(f"    Cura Opening (Q1): '{data['bot_speech_hi']}'")
    assert "Namaste" in data['bot_speech_hi'] or "नमस्ते" in data['bot_speech_hi']
    assert "Diabetes" in data['bot_speech_hi'] or "Cura" in data['bot_speech_hi'] or "Sehat" in data['bot_speech_hi'] or "सेहत" in data['bot_speech_hi']
    assert data["questions_asked_count"] == 1

    # 3. Turn 1 -> Patient consents (Q1 -> Q2 Identity)
    res = client.post("/api/cura/session/turn", data={
        "session_id": session_id,
        "transcript": "हाँ, मैं सहमत हूँ और आप बातचीत रिकॉर्ड कर सकते हैं।"
    })
    assert res.status_code == 200
    data = res.json()
    print(f"\n[2] Turn 1 Response (Q2): '{data['bot_speech_hi']}'")
    assert data["question_id"] == "Q2_IDENTITY"
    assert data["state"]["demographics"]["consent_given"] is True

    # 4. Turn 2 -> Patient gives Name, Age, Gender (Q2 -> Q3 Branch)
    res = client.post("/api/cura/session/turn", data={
        "session_id": session_id,
        "transcript": "मेरा नाम महेश कुमार है, मेरी उम्र 50 साल है, पुरुष।"
    })
    assert res.status_code == 200
    data = res.json()
    print(f"\n[3] Turn 2 Response (Q3): '{data['bot_speech_hi']}'")
    assert data["question_id"] == "Q3_DIABETIC_STATUS"
    assert data["state"]["demographics"]["name"] == "महेश"
    assert data["state"]["demographics"]["age"] == 50

    # 5. Turn 3 -> Patient confirms Known Diabetic (Branch A)
    res = client.post("/api/cura/session/turn", data={
        "session_id": session_id,
        "transcript": "हाँ, मुझे 4 साल से डायबिटीज है।"
    })
    assert res.status_code == 200
    data = res.json()
    print(f"\n[4] Turn 3 Response (Branch A): '{data['bot_speech_hi']}'")
    assert data["branch"] == "BRANCH_A"

    # 6. Check summary endpoint
    res = client.get(f"/api/cura/session/{session_id}/summary")
    assert res.status_code == 200
    summary_data = res.json()["summary"]
    print(f"\n[5] Mid-session Summary Retrieved: Branch={summary_data['branch']}, Risk={summary_data['risk_tier']}")
    assert "Branch A" in summary_data["clinical_summary_en"]
    assert len(summary_data["qa_transcript"]) >= 3

    # 7. Test Emergency Interruption in separate session
    print("\n[6] Testing Emergency Interruption via API...")
    res_em_start = client.post("/api/cura/session/start")
    em_session_id = res_em_start.json()["session_id"]
    
    res_em_turn = client.post("/api/cura/session/turn", data={
        "session_id": em_session_id,
        "transcript": "मेरे सीने में बहुत तेज दर्द हो रहा है और सांस फूल रही है!"
    })
    em_turn_data = res_em_turn.json()
    print(f"    Emergency Interruption Speech: '{em_turn_data['bot_speech_hi'][:90]}...'")
    assert em_turn_data["is_emergency"] is True
    assert em_turn_data["is_completed"] is True

    # 8. Check Emergency Alerts Feed for Doctor Dashboard
    res_alerts = client.get("/api/cura/emergencies")
    assert res_alerts.status_code == 200
    alerts_data = res_alerts.json()
    print(f"    Emergency Alerts Feed Count: {alerts_data['count']}")
    assert alerts_data["count"] >= 1
    assert alerts_data["emergencies"][0]["session_id"] == em_session_id

    # 9. Check All Sessions Queue
    res_all = client.get("/api/cura/sessions")
    assert res_all.status_code == 200
    all_sessions = res_all.json()
    print(f"\n[7] Total Doctor Queue Sessions: {all_sessions['total']}")
    assert all_sessions["total"] >= 2

    print("\n" + "🎉" * 25)
    print("ALL CURA E2E FASTAPI ENDPOINTS & LOGIC VALIDATED SUCCESSFULLY!")
    print("🎉" * 25)

if __name__ == "__main__":
    test_e2e_cura_workflow()
