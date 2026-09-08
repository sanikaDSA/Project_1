import os
import sys
import unittest
from datetime import datetime, timezone

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from question_selector.question_selector import QuestionSelector, GLOBAL_QUESTION_SELECTOR
from emergency.emergency_detector import EmergencyDetector, GLOBAL_EMERGENCY_DETECTOR
from urgency.urgency_engine import UrgencyEngine, GLOBAL_URGENCY_ENGINE
from session_store.session_store import SehatSessionState, SessionStore, GLOBAL_SESSION_STORE
from dialogue.dialogue_manager import DialogueManager, GLOBAL_DIALOGUE_MANAGER
from summary.summary_generator import SummaryGenerator, GLOBAL_SUMMARY_GENERATOR
from orchestrator.orchestrator import SehatOrchestrator, GLOBAL_ORCHESTRATOR
from audit.audit_logger import AuditLogger, GLOBAL_AUDIT_LOGGER


class TestSehatAISuite(unittest.TestCase):

    def setUp(self):
        self.selector = GLOBAL_QUESTION_SELECTOR
        self.emergency = GLOBAL_EMERGENCY_DETECTOR
        self.urgency = GLOBAL_URGENCY_ENGINE
        self.dialogue = GLOBAL_DIALOGUE_MANAGER
        self.orchestrator = GLOBAL_ORCHESTRATOR

    # ========================================================
    # 1. QUESTION SELECTION TESTS
    # ========================================================

    def test_mandatory_common_order(self):
        """Verify Q1 consent, Q2 demographics, and Q3 diabetic status."""
        q1 = self.selector.select_next_question("common", [], [], 0, {}, [], [])
        self.assertEqual(q1["question_id"], "COM_CONSENT")
        self.assertTrue(q1["question_hi"].startswith("नमस्ते, मेरा नाम Diabetes Dost") or q1["question_hi"].startswith("नमस्ते, मेरा नाम"))

        q2 = self.selector.select_next_question("common", ["COM_CONSENT"], ["consent"], 1, {}, [], [])
        self.assertEqual(q2["question_id"], "COM_DEMOGRAPHICS")

        q3 = self.selector.select_next_question("common", ["COM_CONSENT", "COM_DEMOGRAPHICS"], ["consent", "demographics"], 2, {}, [], [])
        self.assertEqual(q3["question_id"], "COM_STATUS")

    def test_known_diabetic_branch_selection(self):
        """Verify dynamic branch question selection for known diabetics (including Type 1 / Type 2)."""
        asked = ["COM_CONSENT", "COM_DEMOGRAPHICS", "COM_STATUS"]
        completed = ["consent", "demographics", "diabetic_status"]
        q_next = self.selector.select_next_question("known_diabetic", asked, completed, 3, {"gender": "Male"}, [], [])
        self.assertIsNotNone(q_next)
        self.assertEqual(q_next["branch"], "known_diabetic")
        self.assertIn(q_next["question_id"], ["KD_DIABETES_TYPE", "KD_DURATION", "KD_MEDICATIONS", "KD_SUGAR_LEVELS"])

    def test_unsure_branch_selection(self):
        """Verify unsure branch questions."""
        asked = ["COM_CONSENT", "COM_DEMOGRAPHICS", "COM_STATUS"]
        completed = ["consent", "demographics", "diabetic_status"]
        q_next = self.selector.select_next_question("unsure", asked, completed, 3, {"gender": "Male"}, [], [])
        self.assertIsNotNone(q_next)
        self.assertEqual(q_next["branch"], "unsure")

    def test_not_diabetic_branch_selection(self):
        """Verify not_diabetic branch questions."""
        asked = ["COM_CONSENT", "COM_DEMOGRAPHICS", "COM_STATUS"]
        completed = ["consent", "demographics", "diabetic_status"]
        q_next = self.selector.select_next_question("not_diabetic", asked, completed, 3, {"gender": "Male"}, [], [])
        self.assertIsNotNone(q_next)
        self.assertEqual(q_next["branch"], "not_diabetic")

    def test_female_age_under_18_skips_pregnancy_and_gestational(self):
        """Verify that female patients under 18 years old are NEVER asked pregnancy or gestational diabetes questions."""
        asked = ["COM_CONSENT", "COM_DEMOGRAPHICS", "COM_STATUS", "UN_CLASSIC_SYMPTOMS", "UN_ENERGY_VISION", "UN_HEALING_NUMBNESS", "UN_FAMILY_HISTORY", "UN_PREV_TESTING"]
        completed = ["consent", "demographics", "diabetic_status", "classic_symptoms", "energy_and_vision", "healing_and_numbness", "family_history", "previous_testing"]
        
        # Female patient age 16 -> MUST NOT get pregnancy or gestational questions
        q_under18 = self.selector.select_next_question("unsure", asked, completed, 8, {"gender": "महिला (Female)", "age": 16}, [], [])
        self.assertNotIn(q_under18.get("question_id"), ["UN_PREGNANCY_HISTORY", "UN_GESTATIONAL"])
        self.assertEqual(q_under18.get("question_id"), "CLS_FINAL_FEEDBACK")

    def test_female_adult_pregnancy_flow(self):
        """Verify that adult female patients are first asked pregnancy history, and gestational diabetes only follows if positive."""
        asked = ["COM_CONSENT", "COM_DEMOGRAPHICS", "COM_STATUS", "UN_CLASSIC_SYMPTOMS", "UN_ENERGY_VISION", "UN_HEALING_NUMBNESS", "UN_FAMILY_HISTORY", "UN_PREV_TESTING"]
        completed = ["consent", "demographics", "diabetic_status", "classic_symptoms", "energy_and_vision", "healing_and_numbness", "family_history", "previous_testing"]
        
        # Step 1: Adult Female (age 28) is asked Pregnancy History first (NOT directly gestational diabetes)
        q_adult_f = self.selector.select_next_question("unsure", asked, completed, 8, {"gender": "महिला (Female)", "age": 28}, [], [])
        self.assertEqual(q_adult_f.get("question_id"), "UN_PREGNANCY_HISTORY")
        self.assertTrue("Pregnancy" in q_adult_f.get("question_hi", "") or "गर्भावस्था" in q_adult_f.get("question_hi", ""))

        # Case A: If pregnancy history is negative, gestational diabetes is skipped
        s_neg = SehatSessionState(session_id="test_preg_neg")
        s_neg.current_question_id = "UN_PREGNANCY_HISTORY"
        self.dialogue._extract_clinical_slots(s_neg, "नहीं, मेरी कोई प्रेगनेंसी नहीं रही है")
        self.assertFalse(s_neg.pregnancy_history)
        self.assertIn("pregnancy_history", s_neg.completed_topics)
        self.assertIn("gestational_diabetes", s_neg.completed_topics) # skipped

        # Case B: If pregnancy history is positive, gestational diabetes is asked next
        s_pos = SehatSessionState(session_id="test_preg_pos")
        s_pos.current_question_id = "UN_PREGNANCY_HISTORY"
        self.dialogue._extract_clinical_slots(s_pos, "हाँ, 2 साल पहले डिलीवरी हुई थी")
        self.assertTrue(s_pos.pregnancy_history)
        self.assertIn("pregnancy_history", s_pos.completed_topics)
        self.assertNotIn("gestational_diabetes", s_pos.completed_topics) # ready for follow up

        # Now question selector selects gestational diabetes follow-up
        asked_with_preg = asked + ["UN_PREGNANCY_HISTORY"]
        completed_with_preg = completed + ["pregnancy_history"]
        q_gest = self.selector.select_next_question("unsure", asked_with_preg, completed_with_preg, 9, {"gender": "महिला (Female)", "age": 28}, [], [])
        self.assertEqual(q_gest.get("question_id"), "UN_GESTATIONAL")

    def test_duplicate_prevention_and_closing_question(self):
        """Verify no duplicate questions asked and closing question triggered at budget limit."""
        asked = ["COM_CONSENT", "COM_DEMOGRAPHICS", "COM_STATUS", "KD_DURATION", "KD_MEDICATIONS", "KD_SUGAR_LEVELS", "KD_THIRST_URINATION", "KD_VISION_FATIGUE", "KD_NUMBNESS_WOUNDS", "KD_HYPO_EPISODES", "KD_LIFESTYLE_DIET"]
        completed = ["consent", "demographics", "diabetic_status", "diabetes_duration", "medications", "blood_sugar_readings", "thirst_and_urination", "vision_and_fatigue", "neuropathy_and_wounds", "hypoglycemia_symptoms", "lifestyle_factors"]
        
        # At question budget limit (11), next must be Closing Question
        q_close = self.selector.select_next_question("known_diabetic", asked, completed, 11, {"gender": "Male"}, [], [], max_budget=12)
        self.assertEqual(q_close["question_id"], "CLS_FINAL_FEEDBACK")
        self.assertTrue(q_close["question_hi"].startswith("क्या आप Doctor को अपनी Health") or q_close["question_hi"].startswith("क्या आप"))

    # ========================================================
    # 2. EMERGENCY DETECTION TESTS
    # ========================================================

    def test_emergency_chest_pain(self):
        is_em, ev = self.emergency.scan("मुझे छाती में बहुत तेज दर्द हो रहा है और दिल दब रहा है", "test_s1")
        self.assertTrue(is_em)
        self.assertEqual(ev["event_type"], "EMERGENCY")
        self.assertIn("cardiac", ev["categories"])

    def test_emergency_breathlessness(self):
        is_em, ev = self.emergency.scan("मुझे सांस लेने में बहुत तकलीफ हो रही है, दम घुट रहा है", "test_s2")
        self.assertTrue(is_em)
        self.assertIn("respiratory", ev["categories"])

    def test_emergency_fainting(self):
        is_em, ev = self.emergency.scan("मरीज चक्कर खाकर गिर गया और बेहोश हो गया", "test_s3")
        self.assertTrue(is_em)
        self.assertIn("fainting_consciousness", ev["categories"])

    def test_emergency_seizure(self):
        is_em, ev = self.emergency.scan("अचानक दौरा पड़ गया और शरीर अकड़ गया", "test_s4")
        self.assertTrue(is_em)
        self.assertIn("seizure", ev["categories"])

    def test_emergency_severe_vomiting_and_distress(self):
        is_em, ev = self.emergency.scan("अभी बहुत तकलीफ हो रही है और पानी भी नहीं पच रहा, उल्टी रुक नहीं रही", "test_s5")
        self.assertTrue(is_em)
        self.assertIn("severe_distress_vomiting", ev["categories"])

    def test_emergency_severe_hypoglycemia_glucose_low(self):
        is_em, ev = self.emergency.scan("मेरी शुगर 40 आ गई है और हाथ बहुत कांप रहे हैं", "test_s6")
        self.assertTrue(is_em)
        self.assertIn("severe_hypoglycemia_confusion", ev["categories"])

    def test_emergency_self_harm(self):
        is_em, ev = self.emergency.scan("मैं आत्महत्या करना चाहता हूँ", "test_s7")
        self.assertTrue(is_em)
        self.assertIn("self_harm", ev["categories"])

    def test_non_emergency_normal_statement(self):
        is_em, ev = self.emergency.scan("मुझे कभी-कभी मीठा खाने का मन करता है", "test_s8")
        self.assertFalse(is_em)
        self.assertIsNone(ev)

    # ========================================================
    # 3. URGENCY RULE ENGINE TESTS
    # ========================================================

    def test_urgency_emergency_is_always_urgent(self):
        res = self.urgency.evaluate(emergency_escalation=True, risk_signals=[], symptoms_reported=[], blood_sugar_readings=[])
        self.assertEqual(res["recommended_urgency"], "urgent")

    def test_urgency_extreme_hyperglycemia(self):
        res = self.urgency.evaluate(
            emergency_escalation=False,
            risk_signals=["High Sugar"],
            symptoms_reported=["fatigue"],
            blood_sugar_readings=[{"value": 380}]
        )
        self.assertEqual(res["recommended_urgency"], "urgent")

    def test_urgency_neuropathy_with_wound(self):
        res = self.urgency.evaluate(
            emergency_escalation=False,
            risk_signals=["Foot Ulcer"],
            symptoms_reported=["numbness in feet", "non healing ulcer on toe"],
            blood_sugar_readings=[]
        )
        self.assertEqual(res["recommended_urgency"], "urgent")

    def test_urgency_soon_tier(self):
        res = self.urgency.evaluate(
            emergency_escalation=False,
            risk_signals=["Frequent Urination"],
            symptoms_reported=["polyuria"],
            blood_sugar_readings=[{"value": 220}]
        )
        self.assertEqual(res["recommended_urgency"], "soon")

    def test_urgency_routine_tier(self):
        res = self.urgency.evaluate(
            emergency_escalation=False,
            risk_signals=[],
            symptoms_reported=[],
            blood_sugar_readings=[]
        )
        self.assertEqual(res["recommended_urgency"], "routine")

    # ========================================================
    # 4. SESSION STATE, CONSENT & DIALOGUE LIFECYCLE
    # ========================================================

    def test_consent_rejection_terminates_session(self):
        sid = "test_consent_rej"
        self.dialogue.start_session(sid)
        res = self.dialogue.process_turn(sid, "नहीं, मैं रिकॉर्डिंग के लिए सहमत नहीं हूँ")
        self.assertTrue(res["is_completed"])
        self.assertTrue(res.get("consent_rejected"))
        self.assertTrue("Consent के बिना" in res["bot_speech_hi"] or "सहमति के बिना" in res["bot_speech_hi"])

    def test_consent_acceptance_proceeds_to_demographics(self):
        sid = "test_consent_acc"
        self.dialogue.start_session(sid)
        res = self.dialogue.process_turn(sid, "हाँ, मैं सहमत हूँ")
        self.assertFalse(res["is_completed"])
        self.assertEqual(res["next_question_id"], "COM_DEMOGRAPHICS")

    def test_full_orchestrator_conversation_and_emergency_interruption(self):
        sid = "test_e2e_orch"
        
        # Turn 1: Start
        s1 = self.orchestrator.start_session(sid)
        self.assertEqual(s1["event"], "SESSION_STARTED")
        self.assertTrue(s1["bot_speech_hi"].startswith("नमस्ते, मेरा नाम Diabetes Dost") or s1["bot_speech_hi"].startswith("नमस्ते, मेरा नाम"))

        # Turn 2: Consent
        s2 = self.orchestrator.process_patient_turn(sid, transcript_text="हाँ, बिल्कुल सहमत हूँ")
        self.assertEqual(s2["state"]["current_question_id"], "COM_DEMOGRAPHICS")

        # Turn 3: Demographics
        s3 = self.orchestrator.process_patient_turn(sid, transcript_text="मेरा नाम राहुल शर्मा है, उम्र 35 वर्ष, पुरुष")
        self.assertEqual(s3["state"]["patient"]["name"], "राहुल शर्मा")
        self.assertEqual(s3["state"]["patient"]["age"], 35)

        # Turn 4: Emergency Trigger in Mid-Session
        s4 = self.orchestrator.process_patient_turn(sid, transcript_text="डॉक्टर साहब, अचानक सीने में बहुत तेज दर्द हो रहा है और दिल दब रहा है")
        self.assertEqual(s4["event"], "EMERGENCY_DETECTED")
        self.assertTrue(s4["is_emergency"])
        self.assertTrue(s4["is_completed"])
        self.assertEqual(s4["state"]["recommended_urgency"], "urgent")

    def test_hinglish_and_english_demographics_extraction(self):
        """Verify extraction of name, age, and gender across Hindi, Hinglish, and English."""
        # 1. Hindi with Devanagari transliterated 'मेल'
        s1 = SehatSessionState(session_id="test_demo_1")
        self.dialogue._extract_demographics(s1, "मेरा नाम गोविंद है मैं 22 साल का हूं और मैं मेल हूं")
        self.assertEqual(s1.demographics.get("name"), "गोविंद")
        self.assertEqual(s1.demographics.get("age"), 22)
        self.assertEqual(s1.demographics.get("gender"), "पुरुष (Male)")

        # 2. Hindi with Devanagari transliterated 'फीमेल'
        s2 = SehatSessionState(session_id="test_demo_2")
        self.dialogue._extract_demographics(s2, "मेरा नाम प्रिया है, उम्र 28 वर्ष और मैं फीमेल हूँ")
        self.assertEqual(s2.demographics.get("name"), "प्रिया")
        self.assertEqual(s2.demographics.get("age"), 28)
        self.assertEqual(s2.demographics.get("gender"), "महिला (Female)")

        # 3. Pure English sentence
        s3 = SehatSessionState(session_id="test_demo_3")
        self.dialogue._extract_demographics(s3, "My name is Amit, I am 45 years old and I am male")
        self.assertEqual(s3.demographics.get("name"), "Amit")
        self.assertEqual(s3.demographics.get("age"), 45)
        self.assertEqual(s3.demographics.get("gender"), "पुरुष (Male)")

        # 4. English with female
        s4 = SehatSessionState(session_id="test_demo_4")
        self.dialogue._extract_demographics(s4, "I am Sunita, 50 years old, female")
        self.assertEqual(s4.demographics.get("name"), "Sunita")
        self.assertEqual(s4.demographics.get("age"), 50)
        self.assertEqual(s4.demographics.get("gender"), "महिला (Female)")

    def test_inaudible_response_triggers_retry_once(self):
        """Verify that inaudible or empty speech triggers a polite retry once."""
        sid = "test_inaudible_sess"
        self.dialogue.start_session(sid)
        
        # Turn 1: Patient gives silence or filler
        res1 = self.dialogue.process_turn(sid, "...")
        self.assertTrue("Voice स्पष्ट नहीं आई" in res1["bot_speech_hi"] or "आवाज स्पष्ट नहीं आई" in res1["bot_speech_hi"])
        self.assertEqual(res1["next_question_id"], "COM_CONSENT")
        self.assertFalse(res1["is_completed"])

    def test_mandatory_demographics_reask_when_field_missing(self):
        """Verify that if name, age, or gender is missing, the bot asks for the missing field before proceeding."""
        sid = "test_demo_reask_sess"
        self.dialogue.start_session(sid)
        self.dialogue.process_turn(sid, "हाँ, सहमत हूँ") # Consent given -> now at COM_DEMOGRAPHICS

        # Patient only gives name: "मेरा नाम गोविंद है" (missing age and gender)
        res1 = self.dialogue.process_turn(sid, "मेरा नाम गोविंद है")
        self.assertEqual(res1["next_question_id"], "COM_DEMOGRAPHICS")
        self.assertTrue("Age" in res1["bot_speech_hi"] or "उम्र" in res1["bot_speech_hi"])
        self.assertTrue("Gender" in res1["bot_speech_hi"] or "लिंग" in res1["bot_speech_hi"])

        # Patient supplies remaining age and gender: "मेरी उम्र 24 वर्ष है और मैं पुरुष हूँ"
        res2 = self.dialogue.process_turn(sid, "मेरी उम्र 24 वर्ष है और मैं पुरुष हूँ")
        self.assertEqual(res2["next_question_id"], "COM_STATUS") # Successfully proceeded
        self.assertEqual(res2["state"]["patient"]["name"], "गोविंद")
        self.assertEqual(res2["state"]["patient"]["age"], 24)
        self.assertEqual(res2["state"]["patient"]["gender"], "पुरुष (Male)")


    def test_family_history_positive_and_negative_recording(self):
        """Verify that positive and negative family history answers are accurately recorded in session state."""
        # Case 1: Explicit negative ("नहीं है")
        s1 = SehatSessionState(session_id="test_fam_neg_1")
        s1.current_question_id = "UN_FAMILY_HISTORY"
        self.dialogue._extract_clinical_slots(s1, "नहीं है")
        self.assertFalse(s1.family_history)
        self.assertIn("family_history", s1.completed_topics)

        # Case 2: Mention of family with negation ("मेरे फैमिली में किसी को भी डायबिटीज नहीं है")
        s2 = SehatSessionState(session_id="test_fam_neg_2")
        s2.current_question_id = "ND_FAMILY_HISTORY"
        self.dialogue._extract_clinical_slots(s2, "मेरे फैमिली में किसी को भी डायबिटीज नहीं है")
        self.assertFalse(s2.family_history)
        self.assertIn("family_history", s2.completed_topics)

        # Case 3: Positive mention ("मेरी मम्मी को डायबिटीज है")
        s3 = SehatSessionState(session_id="test_fam_pos_1")
        s3.current_question_id = "ND_FAMILY_HISTORY"
        self.dialogue._extract_clinical_slots(s3, "मेरी मम्मी को डायबिटीज है")
        self.assertTrue(s3.family_history)
        self.assertIn("family_history", s3.completed_topics)

        # Case 4: Simple affirmative during family history question ("हां है")
        s4 = SehatSessionState(session_id="test_fam_pos_2")
        s4.current_question_id = "UN_FAMILY_HISTORY"
        self.dialogue._extract_clinical_slots(s4, "हां है")
        self.assertTrue(s4.family_history)
        self.assertIn("family_history", s4.completed_topics)

    def test_pregnancy_and_gestational_recording(self):
        """Verify that positive and negative pregnancy and gestational answers are accurately recorded."""
        # Case 1: Female negative pregnancy ("नहीं, मेरी शादी नहीं हुई है")
        s1 = SehatSessionState(session_id="test_preg_neg_1")
        s1.current_question_id = "UN_PREGNANCY_HISTORY"
        self.dialogue._extract_clinical_slots(s1, "नहीं, मेरी शादी नहीं हुई है")
        self.assertFalse(s1.pregnancy_history)
        self.assertFalse(s1.gestational_history)
        self.assertIn("pregnancy_history", s1.completed_topics)
        self.assertIn("gestational_diabetes", s1.completed_topics)

        # Case 2: Female positive pregnancy, negative gestational
        s2 = SehatSessionState(session_id="test_preg_pos_gest_neg")
        s2.current_question_id = "UN_PREGNANCY_HISTORY"
        self.dialogue._extract_clinical_slots(s2, "हाँ, 2 बच्चे हैं")
        self.assertTrue(s2.pregnancy_history)
        self.assertIn("pregnancy_history", s2.completed_topics)
        self.assertNotIn("gestational_diabetes", s2.completed_topics)

        s2.current_question_id = "UN_GESTATIONAL"
        self.dialogue._extract_clinical_slots(s2, "नहीं, शुगर बिल्कुल नॉर्मल थी")
        self.assertFalse(s2.gestational_history)
        self.assertIn("gestational_diabetes", s2.completed_topics)

        # Case 3: Female positive pregnancy, positive gestational
        s3 = SehatSessionState(session_id="test_preg_pos_gest_pos")
        s3.current_question_id = "ND_PREGNANCY_HISTORY"
        self.dialogue._extract_clinical_slots(s3, "हाँ, मेरी डिलीवरी हुई थी")
        self.assertTrue(s3.pregnancy_history)

        s3.current_question_id = "ND_GESTATIONAL"
        self.dialogue._extract_clinical_slots(s3, "हाँ, उस समय शुगर बढ़ गई थी (गेस्टेशनल डायबिटीज हुई थी)")
        self.assertTrue(s3.gestational_history)
        self.assertIn("Risk: Gestational Diabetes History", s3.risk_signals)

        # Test summary generation reflects pregnancy and family history
        summ = GLOBAL_SUMMARY_GENERATOR.generate(s3)
        self.assertTrue(summ["pregnancy_history"])
        self.assertTrue(summ["gestational_history"])


    def test_non_diabetic_patient_age_not_confused_with_diabetes_duration(self):
        """Verify that a patient's age (e.g., 8 years old) is NOT recorded as diabetes duration when non-diabetic."""
        sid = "test_child_age_8"
        self.dialogue.start_session(sid)
        
        # Turn 1: Consent
        self.dialogue.process_turn(sid, "हाँ, मैं सहमत हूँ")
        
        # Turn 2: Demographics with age 8 years
        res_demo = self.dialogue.process_turn(sid, "मेरा नाम राहुल है, मेरी उम्र 8 साल है, पुरुष")
        self.assertEqual(res_demo["state"]["patient"]["name"], "राहुल")
        self.assertEqual(res_demo["state"]["patient"]["age"], 8)
        # Ensure diabetes_history was NOT erroneously populated from the age phrase
        self.assertIsNone(res_demo["state"]["diabetes_history"])

        # Turn 3: Status is non-diabetic
        res_status = self.dialogue.process_turn(sid, "मुझे कोई डायबिटीज नहीं है")
        self.assertEqual(res_status["state"]["branch"], "not_diabetic")
        self.assertIsNone(res_status["state"]["diabetes_history"])

        # Generate summary and verify
        state = GLOBAL_SESSION_STORE.get(sid)
        summ = GLOBAL_SUMMARY_GENERATOR.generate(state)
        self.assertIsNone(summ["diabetes_history"])
        self.assertIn("Diabetes History / Duration: N/A (Non-Diabetic)", summ["clinical_summary_en"])
        self.assertIn("डायबिटीज अवधि / इतिहास: लागू नहीं (डायबिटीज नहीं है)", summ["clinical_summary_hi"])
    def test_save_different_patient_summaries(self):
        """Verify that multiple different patients have their clinical summaries saved separately without collision."""
        import os

        # Patient 1: Ramesh Kumar (Diabetic Male, 52 yrs)
        sid1 = "sess_patient_ramesh_001"
        self.dialogue.start_session(sid1)
        self.dialogue.process_turn(sid1, "हाँ, मैं सहमत हूँ")
        self.dialogue.process_turn(sid1, "मेरा नाम रमेश कुमार है, उम्र 52 वर्ष, पुरुष")
        self.dialogue.process_turn(sid1, "हाँ, मुझे 5 साल से टाइप 2 डायबिटीज है")
        state1 = GLOBAL_SESSION_STORE.get(sid1)
        summ1 = GLOBAL_SUMMARY_GENERATOR.generate(state1)

        # Patient 2: Sunita Sharma (Non-diabetic Female, 29 yrs)
        sid2 = "sess_patient_sunita_002"
        self.dialogue.start_session(sid2)
        self.dialogue.process_turn(sid2, "हाँ, मैं सहमत हूँ")
        self.dialogue.process_turn(sid2, "मेरा नाम सुनीता शर्मा है, उम्र 29 साल, महिला")
        self.dialogue.process_turn(sid2, "मुझे कोई डायबिटीज नहीं है, बस थोड़ा सिरदर्द और थकान है")
        state2 = GLOBAL_SESSION_STORE.get(sid2)
        summ2 = GLOBAL_SUMMARY_GENERATOR.generate(state2)

        # Check paths returned
        self.assertIn("summary_file_txt", summ1)
        self.assertIn("summary_file_json", summ1)
        self.assertIn("summary_file_txt", summ2)
        self.assertIn("summary_file_json", summ2)

        # Verify files exist on filesystem
        self.assertTrue(os.path.exists(summ1["summary_file_txt"]))
        self.assertTrue(os.path.exists(summ1["summary_file_json"]))
        self.assertTrue(os.path.exists(summ2["summary_file_txt"]))
        self.assertTrue(os.path.exists(summ2["summary_file_json"]))

        # Verify file contents are separate and accurate per patient
        with open(summ1["summary_file_txt"], "r", encoding="utf-8") as f:
            txt1 = f.read()
            self.assertIn("रमेश कुमार", txt1)
            self.assertIn("Age: 52", txt1)
            self.assertIn("KNOWN_DIABETIC", txt1)
            self.assertNotIn("सुनीता शर्मा", txt1)

        with open(summ2["summary_file_txt"], "r", encoding="utf-8") as f:
            txt2 = f.read()
            self.assertIn("सुनीता शर्मा", txt2)
            self.assertIn("Age: 29", txt2)
            self.assertIn("NOT_DIABETIC", txt2)
            self.assertNotIn("रमेश कुमार", txt2)


if __name__ == "__main__":
    unittest.main()


