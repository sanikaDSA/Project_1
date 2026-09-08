import sys
import logging

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from audio_pipeline.clinical_flow import ClinicalConversationSession

def run_dynamic_intent_tests():
    print("=" * 75)
    print("🧠 DYNAMIC CONVERSATION ENGINE - ADVANCED SCENARIO VALIDATION SUITE")
    print("=" * 75)

    # -------------------------------------------------------------
    # SCENARIO 1: HEADACHE -> FEVER -> VOMITING (RISK RECALCULATION & NO REPEATS)
    # -------------------------------------------------------------
    print("\n[SCENARIO 1: SYMPTOM CHAIN & MULTI-SYMPTOM RISK ESCALATION]")
    sess1 = ClinicalConversationSession(session_id="scen1")
    
    # Turn 1: User introduces Headache with severity & duration
    t1_in = "मुझे दो दिन से बहुत तेज सिरदर्द हो रहा है"
    t1_out = sess1.select_next_response(t1_in)
    print(f"Turn 1: User: '{t1_in}'")
    print(f"        Doctor AI: '{t1_out['doctor_text']}'")
    assert "दो दिन" in sess1.state.known_information.get("duration", ""), "Duration not extracted!"
    assert sess1.state.severity.get("headache") == "HIGH", "Severity not extracted!"
    # Ensure Doctor AI does NOT ask "कब से है?"
    assert "कितने दिनों से" not in t1_out['doctor_text'], "Repeated duration question asked!"
    print("  ✓ Turn 1 PASS: Duration & Severity extracted; avoided redundant duration question.")

    # Turn 2: User adds Fever
    t2_in = "हाँ, बुखार भी है"
    t2_out = sess1.select_next_response(t2_in)
    print(f"\nTurn 2: User: '{t2_in}'")
    print(f"        Doctor AI: '{t2_out['doctor_text']}'")
    assert "fever" in sess1.state.active_symptoms, "Fever not added to active symptoms!"
    assert "headache" in sess1.state.active_symptoms, "Headache lost from active symptoms!"
    assert sess1.state.current_risk_level == "MODERATE", f"Expected MODERATE risk, got {sess1.state.current_risk_level}"
    print("  ✓ Turn 2 PASS: Multi-symptom merged (Headache + Fever), risk escalated to MODERATE.")

    # Turn 3: User adds Vomiting
    t3_in = "और मुझे उल्टी भी हो रही है"
    t3_out = sess1.select_next_response(t3_in)
    print(f"\nTurn 3: User: '{t3_in}'")
    print(f"        Doctor AI: '{t3_out['doctor_text']}'")
    assert "nausea_vomiting" in sess1.state.active_symptoms, "Vomiting not added!"
    assert sess1.state.current_risk_level == "HIGH", f"Expected HIGH risk, got {sess1.state.current_risk_level}"
    assert "संक्रमण" in t3_out['doctor_text'] or "जांच" in t3_out['doctor_text'], "Risk combination advisory missing!"
    print("  ✓ Turn 3 PASS: High-Risk Combination Detected (Headache + Fever + Vomiting) -> Immediate High-Risk Advisory.")

    # -------------------------------------------------------------
    # SCENARIO 2: TOPIC SWITCHING (HEADACHE -> STOMACH PAIN)
    # -------------------------------------------------------------
    print("\n[SCENARIO 2: DYNAMIC TOPIC SWITCHING & CONTEXT PRESERVATION]")
    sess2 = ClinicalConversationSession(session_id="scen2")
    
    # Turn 1: Headache
    sess2.select_next_response("मुझे सिरदर्द है")
    assert "headache" in sess2.state.active_symptoms
    
    # Turn 2: User suddenly switches topic to Stomach pain
    t2_in = "वैसे मुझे पेट में भी बहुत दर्द हो रहा है"
    t2_out = sess2.select_next_response(t2_in)
    print(f"Turn 2 (Topic Switch): User: '{t2_in}'")
    print(f"                       Doctor AI: '{t2_out['doctor_text']}'")
    assert "stomach_pain" in sess2.state.active_symptoms, "New symptom stomach_pain not captured!"
    assert "headache" in sess2.state.active_symptoms, "Previous symptom headache lost!"
    assert sess2.state.current_primary_intent == "stomach_pain", "Primary intent not switched to new topic!"
    print("  ✓ Scenario 2 PASS: Topic switched smoothly to Stomach Pain while preserving Headache in context.")

    # -------------------------------------------------------------
    # SCENARIO 3: GENERAL PATIENT TURNS OUT TO HAVE DIABETES + DIZZINESS
    # -------------------------------------------------------------
    print("\n[SCENARIO 3: DYNAMIC DIABETES CONTEXT ACTIVATION]")
    sess3 = ClinicalConversationSession(session_id="scen3")
    
    # User starts with general symptom but informs diabetes + low sugar reading
    t_in = "मुझे बहुत चक्कर आ रहे हैं और मेरी शुगर 60 आई है, मैं मेटफॉर्मिन लेता हूँ"
    t_out = sess3.select_next_response(t_in)
    print(f"Turn 1: User: '{t_in}'")
    print(f"        Doctor AI: '{t_out['doctor_text']}'")
    assert sess3.state.diabetes_context["active"] is True, "Diabetes context not activated!"
    assert sess3.state.known_information["sugar_values"] == "60", "Sugar reading 60 not extracted!"
    assert "Metformin" in sess3.state.medications, "Medication Metformin not captured!"
    assert "लो" in t_out['doctor_text'] or "चीनी" in t_out['doctor_text'] or "ग्लूकोज" in t_out['doctor_text'], "Hypoglycemia emergency guidance missing!"
    print("  ✓ Scenario 3 PASS: Diabetes Context Activated + Low Sugar Detected + Instant Safety Guidance.")

    # -------------------------------------------------------------
    # SCENARIO 4: DIRECT USER QUESTIONS (HOME REMEDY, DIET, TIMING)
    # -------------------------------------------------------------
    print("\n[SCENARIO 4: DIRECT CONVERSATIONAL QUESTIONS]")
    sess4 = ClinicalConversationSession(session_id="scen4")
    sess4.select_next_response("मुझे खांसी और गले में दर्द है")
    
    # User asks home remedies for existing symptom
    r_remedy = sess4.select_next_response("इसके लिए घरेलू उपाय क्या हैं?")
    print("User: 'इसके लिए घरेलू उपाय क्या हैं?'")
    print(f"Doctor AI: '{r_remedy['doctor_text']}'")
    assert "शहद" in r_remedy['doctor_text'] or "अदरक" in r_remedy['doctor_text'], "Cough home remedy missing!"
    
    # User asks diet
    r_diet = sess4.select_next_response("मुझे क्या खाना चाहिए और क्या नहीं?")
    print("\nUser: 'मुझे क्या खाना चाहिए और क्या नहीं?'")
    print(f"Doctor AI: '{r_diet['doctor_text']}'")
    assert "सूप" in r_diet['doctor_text'] or "पानी" in r_diet['doctor_text'], "Diet advice missing!"

    # User asks medicine timing
    r_timing = sess4.select_next_response("मेटफॉर्मिन गोळी कधी घ्यावी?")
    print("\nUser: 'मेटफॉर्मिन गोळी कधी घ्यावी?'")
    print(f"Doctor AI: '{r_timing['doctor_text']}'")
    assert "भोजन" in r_timing['doctor_text'] or "खाने" in r_timing['doctor_text'], "Timing advice missing!"
    print("  ✓ Scenario 4 PASS: Direct user questions answered with exact contextual guidance.")

    print("\n" + "=" * 75)
    print("🎉 ALL DYNAMIC CONVERSATIONAL SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == "__main__":
    run_dynamic_intent_tests()
