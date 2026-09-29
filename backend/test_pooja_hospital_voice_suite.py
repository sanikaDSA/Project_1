import sys
import json
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_pipeline.clinical_flow import ClinicalConversationSession, MANDATORY_FIRST_QUESTION

logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_poona_hospital_voice_suite():
    print("=" * 80)
    print("🏥 DEENANATH MANGESHKAR HOSPITAL HINDI MEDICAL VOICE ASSISTANT - FULL VALIDATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # TEST 1: MANDATORY FIRST QUESTION
    # -------------------------------------------------------------
    print("\n[TEST 1: MANDATORY FIRST QUESTION CHECK]")
    sess = ClinicalConversationSession(session_id="poona_suite_01")
    first_q = sess.get_first_greeting()
    expected_first = "नमस्ते। आपको अभी सबसे ज्यादा किस बात की परेशानी है या आप आज किस समस्या के बारे में बात करना चाहते हैं?"
    print(f"Doctor Spoke First: '{first_q['doctor_text']}'")
    assert first_q['doctor_text'] == expected_first, "First question mismatch!"
    print("  ✓ PASS: Mandatory first question matches 100% exactly.")

    # -------------------------------------------------------------
    # TEST 2: THE EXACT REQUIRED MULTI-TURN CONVERSATION SCENARIO
    # -------------------------------------------------------------
    print("\n[TEST 2: EXACT REQUIRED USER SCENARIO (MULTI-TURN S2S)]")
    
    # Step 1: User states chief complaint
    user_turn1 = "मुझे दो दिन से तेज बुखार और खांसी है"
    resp1 = sess.select_next_response(user_turn1)
    print(f"  👤 User Turn 1: '{user_turn1}'")
    print(f"  🩺 Doctor Turn 1: '{resp1['doctor_text']}'")
    
    # Assertions for Turn 1
    assert "fever" in sess.state.active_symptoms, "Fever not detected!"
    assert "cough" in sess.state.active_symptoms, "Cough not detected!"
    assert sess.state.duration.get("fever") == "दो दिन", "Duration not captured!"
    assert sess.state.severity.get("fever") == "HIGH", "Severity not high!"
    assert "नाम और उम्र" in resp1['doctor_text'], "Did not ask for name and age!"
    assert not sess.state.known_diabetes, "Should not assume diabetes immediately!"
    print("  ✓ Step 1 PASS: Symptoms, duration, and severity extracted; naturally asked for name and age.")

    # Step 2: User provides Name and Age
    user_turn2 = "मेरा नाम राहुल है और मेरी उम्र 42 साल है"
    resp2 = sess.select_next_response(user_turn2)
    print(f"\n  👤 User Turn 2: '{user_turn2}'")
    print(f"  🩺 Doctor Turn 2: '{resp2['doctor_text']}'")
    
    # Assertions for Turn 2
    assert sess.state.patient_name == "राहुल", f"Patient name not saved! Got {sess.state.patient_name}"
    assert sess.state.patient_age == 42, f"Patient age not saved! Got {sess.state.patient_age}"
    assert "राहुल" in resp2['doctor_text'], "Doctor did not use patient's name naturally!"
    assert "नाम और उम्र" not in resp2['doctor_text'], "Repeated name and age question!"
    print(f"  ✓ Step 2 PASS: Name (राहुल) & Age (42) stored in state; continued fever/cough conversation.")

    # Step 3: User mentions diabetes risk symptoms
    user_turn3 = "मुझे बहुत प्यास भी लगती है और बार-बार पेशाब आता है"
    resp3 = sess.select_next_response(user_turn3)
    print(f"\n  👤 User Turn 3: '{user_turn3}'")
    print(f"  🩺 Doctor Turn 3: '{resp3['doctor_text']}'")

    # Assertions for Turn 3
    assert "excessive_thirst" in sess.state.active_symptoms, "Thirst not captured!"
    assert "frequent_urination" in sess.state.active_symptoms, "Frequent urination not captured!"
    assert "Poona Hospital" in resp3['doctor_text'], "Did not recommend Poona Hospital!"
    assert sess.state.hospital_recommendation_given == True, "Hospital recommendation flag not True!"
    assert "Blood Sugar Screening" in sess.state.tests_discussed, "Tests discussed not recorded!"
    print("  ✓ Step 3 PASS: Diabetes risk symptoms detected; awareness given; Poona Hospital recommended.")

    # -------------------------------------------------------------
    # TEST 3: DYNAMIC QUESTION ANSWERING (DIET, MEDICINE TIMING, REMEDIES)
    # -------------------------------------------------------------
    print("\n[TEST 3: DIRECT QUESTION ANSWERING]")
    
    # Diet query
    resp_diet = sess.select_next_response("मुझे क्या खाना चाहिए और क्या परहेज रखना चाहिए?")
    print(f"  👤 User Diet Question: 'मुझे क्या खाना चाहिए...'")
    print(f"  🩺 Doctor Diet Answer:   '{resp_diet['doctor_text']}'")
    assert any(w in resp_diet['doctor_text'] for w in ["खान-पान", "भोजन", "डाइट", "परहेज"]), "Diet not answered!"
    print("  ✓ PASS: Diet answered directly and accurately.")

    # Medicine timing query
    resp_med = sess.select_next_response("मेटफॉर्मिन गोली कब लेनी चाहिए?")
    print(f"  👤 User Med Question: 'मेटफॉर्मिन गोली कब लेनी चाहिए?'")
    print(f"  🩺 Doctor Med Answer:   '{resp_med['doctor_text']}'")
    assert "भोजन" in resp_med['doctor_text'] or "Metformin" in resp_med['doctor_text'], "Med timing not answered!"
    print("  ✓ PASS: Medicine timing answered directly.")

    # -------------------------------------------------------------
    # TEST 4: EMERGENCY SAFETY ESCALATION
    # -------------------------------------------------------------
    print("\n[TEST 4: EMERGENCY SAFETY ESCALATION]")
    sess_em = ClinicalConversationSession(session_id="poona_em_test")
    resp_em = sess_em.select_next_response("मेरे सीने में बहुत तेज दर्द हो रहा है और सांस नहीं आ रही")
    print(f"  👤 User Emergency: 'मेरे सीने में बहुत तेज दर्द...'")
    print(f"  🩺 Doctor Alert:   '{resp_em['doctor_text']}'")
    assert resp_em['is_emergency'] == True, "Emergency not triggered!"
    assert "इमरजेंसी" in resp_em['doctor_text'] or "अस्पताल" in resp_em['doctor_text'], "Emergency advisory missing!"
    print("  ✓ PASS: Instant emergency escalation without questionnaire continuation.")

    print("\n" + "=" * 80)
    print("🎉 ALL DEENANATH MANGESHKAR HOSPITAL CLINICAL SUITE TESTS PASSED (100% SUCCESS)!")
    print("=" * 80)


if __name__ == "__main__":
    run_poona_hospital_voice_suite()
