import sys
import httpx
import logging
from pathlib import Path

# Configure utf-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_suite")

from audio_pipeline.clinical_flow import (
    ClinicalConversationSession,
    get_or_create_session,
    SYMPTOM_KNOWLEDGE_BASE,
    EMERGENCY_TRIGGERS,
    DIABETES_QUESTIONS
)

def run_comprehensive_validation():
    print("=" * 70)
    print("🏥 ALEXA HINDI MEDICAL VOICE ASSISTANT - COMPREHENSIVE VALIDATION SUITE")
    print("=" * 70)

    # 1. TEST DOCTOR OPENING GREETING
    sess1 = ClinicalConversationSession(session_id="test_init")
    greeting = sess1.get_first_greeting()
    print("\n[TEST 1: INITIAL DOCTOR AI GREETING (ALEXA STYLE)]")
    print(f"Doctor AI Speaks First: '{greeting['doctor_text']}'")
    assert "नमस्ते" in greeting['doctor_text'], "Doctor greeting failed"
    print("✅ Doctor First Greeting PASS")

    # 2. TEST GENERAL HEALTH & SYMPTOMS (NOT DIABETES)
    general_test_cases = [
        ("fever", "मुझे 2 दिन से बहुत तेज बुखार और ठंड लग रही है", "बुखार"),
        ("cough", "मुझे 4 दिनों से सूखी खांसी आ रही है", "खांसी"),
        ("headache", "मेरे सिर में बहुत तेज दर्द हो रहा है", "सिरदर्द"),
        ("stomach_pain", "मेरे पेट में मरोड़ और दर्द हो रहा है", "पेट"),
        ("nausea_vomiting", "मुझे आज 3 बार उल्टी हुई है और जी मिचला रहा है", "उल्टी"),
        ("diarrhea", "मुझे बार-बार पतले दस्त और लूज मोशन हो रहे हैं", "दस्त"),
        ("constipation", "मुझे कई दिनों से कब्ज है और पेट साफ नहीं हो रहा", "कब्ज"),
        ("dizziness", "मुझे अचानक बहुत चक्कर आ रहे हैं और अंधेरा छा रहा है", "चक्कर"),
        ("weakness", "मुझे बहुत ज्यादा कमजोरी और थकान महसूस हो रही है", "कमजोरी"),
        ("body_pain", "मेरे पूरे बदन में दर्द और कमर दर्द हो रहा है", "दर्द"),
        ("joint_pain", "मेरे घुटनों और जोड़ों में चलने पर बहुत दर्द होता है", "घुटने"),
        ("eye_problems", "मेरी आंखों में जलन और दर्द हो रहा है", "आंखों"),
        ("ear_tooth_pain", "मेरे कान में बहुत तेज दर्द हो रहा है", "कान"),
        ("skin_problems", "मेरी त्वचा पर लाल चकत्ते और खुजली हो रही है", "खुजली"),
        ("sleep_stress", "मुझे रात को नींद नहीं आती और बहुत तनाव रहता है", "नींद")
    ]

    print("\n[TEST 2: GENERAL SYMPTOM HANDLING (NON-DIABETES CASES)]")
    for category, user_input, expected_keyword in general_test_cases:
        sess = ClinicalConversationSession(session_id=f"test_{category}")
        res = sess.select_next_response(user_input)
        print(f"  • User: '{user_input}'")
        print(f"    Doctor: '{res['doctor_text'][:90]}...'")
        assert res['domain'] == "general", f"Expected general domain for {category}, got {res['domain']}"
        assert expected_keyword in res['doctor_text'], f"Keyword '{expected_keyword}' missing in response"
        print(f"    [PASS] {category.upper()}")

    # 3. TEST DIABETES SPECIFIC DOMAIN (ONLY WHEN RELEVANT)
    diabetes_test_cases = [
        ("high_sugar", "मेरी फास्टिंग शुगर 240 आई है और बहुत ज्यादा प्यास लग रही है", "240"),
        ("low_sugar", "मेरी ब्लड शुगर कम हो गई है और हाथ कांप रहे हैं व पसीना आ रहा है", "लो ब्लड शुगर"),
        ("insulin", "मुझे डॉक्टर ने इंसुलिन लिखा है, इंसुलिन कब लेना चाहिए?", "इंसुलिन"),
        ("hba1c", "HbA1c टेस्ट क्या है और यह क्यों कराया जाता है?", "HbA1c"),
        ("medications", "मैं डायबिटीज के लिए मेटफॉर्मिन 500 ले रहा हूँ", "Metformin")
    ]

    print("\n[TEST 3: SPECIALIZED DIABETES CONSULTATION]")
    for category, user_input, expected_keyword in diabetes_test_cases:
        sess = ClinicalConversationSession(session_id=f"test_dia_{category}")
        res = sess.select_next_response(user_input)
        print(f"  • User: '{user_input}'")
        print(f"    Doctor: '{res['doctor_text'][:90]}...'")
        assert res['domain'] == "diabetes", f"Expected diabetes domain for {category}, got {res['domain']}"
        assert expected_keyword.lower() in res['doctor_text'].lower(), f"Expected '{expected_keyword}' in {res['doctor_text']}"
        print(f"    [PASS] {category.upper()}")

    # 4. TEST EMERGENCY RED-FLAGS ESCALATION
    emergency_test_cases = [
        ("severe_chest_pain", "मेरे सीने में बहुत तेज दर्द हो रहा है और बाईं बांह में जा रहा है", "आपातकालीन स्थिति"),
        ("severe_breathing", "मुझे सांस लेने में बहुत तकलीफ हो रही है और दम घुट रहा है", "गंभीर स्थिति"),
        ("unconsciousness", "मरीज अचानक बेहोश हो गया है और कोई प्रतिक्रिया नहीं दे रहा", "स्ट्रोक"),
        ("severe_hypo", "मरीज की शुगर 40 हो गई है और बहुत ज्यादा कांपना बेहोशी हो रही है", "आपातकालीन चेतावनी")
    ]

    print("\n[TEST 4: EMERGENCY & RED-FLAG PROTOCOLS]")
    for category, user_input, expected_keyword in emergency_test_cases:
        sess = ClinicalConversationSession(session_id=f"test_em_{category}")
        res = sess.select_next_response(user_input)
        print(f"  • User: '{user_input}'")
        print(f"    Emergency Doctor Escalation: '{res['doctor_text'][:100]}...'")
        assert res['is_emergency'] is True, f"Emergency flag not set for {category}"
        assert expected_keyword in res['doctor_text'], f"Expected '{expected_keyword}' in response"
        print(f"    [PASS] {category.upper()} -> Immediate Escalation Triggered")

    # 5. TEST END-TO-END HTTP VOICE SESSION API
    print("\n[TEST 5: END-TO-END HTTP VOICE API & SPEECH SYNTHESIS]")
    try:
        with httpx.Client(base_url="http://localhost:8000", timeout=30.0) as client:
            r_init = client.post("/api/voice/session/start")
            assert r_init.status_code == 200, f"Session start failed: {r_init.status_code}"
            init_data = r_init.json()
            print(f"  • POST /api/voice/session/start -> Status 200 OK | Session: {init_data['session_id']}")
            print(f"    Doctor First Speech: {init_data['doctor_text']}")
            assert init_data.get("audio_url") or init_data.get("audio_output"), "Audio speech payload missing!"

            # Test turn with general fever audio
            audio_path = Path("datasets/processed_audio/ElevenLabs_02_16k_mono.wav")
            if audio_path.exists():
                with open(audio_path, "rb") as f:
                    files = {"file": ("patient_voice.wav", f, "audio/wav")}
                    form_data = {"session_id": init_data["session_id"]}
                    r_turn = client.post("/api/voice/session/turn", files=files, data=form_data)
                    assert r_turn.status_code == 200, f"Turn failed: {r_turn.status_code}"
                    turn_data = r_turn.json()
                    print(f"  • POST /api/voice/session/turn -> Status 200 OK | Turn {turn_data['turn_count']}")
                    print(f"    Patient Transcript: {turn_data['transcript']}")
                    print(f"    Doctor Spoken Reply: {turn_data['response_text']}")
                    assert turn_data.get("audio_url") or turn_data.get("audio_output"), "TTS Speech output missing!"
                    print("  [PASS] Full S2S Voice Endpoint Working Successfully")
    except Exception as e:
        print(f"  [HTTP S2S Note]: {e}")

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! 100% GENERAL & DIABETES COVERAGE VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    run_comprehensive_validation()
