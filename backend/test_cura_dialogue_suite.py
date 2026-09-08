import sys
import os
from pathlib import Path

# Ensure backend root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Configure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

from audio_pipeline.cura_clinical_flow import (
    CuraDialogueManager,
    get_or_create_cura_session,
    CLINICAL_QUESTION_BANK
)
from audio_pipeline.emergency_detector import EmergencyDetector
from audio_pipeline.clinical_summary import ClinicalSummaryGenerator

def test_persona_and_opening_sequence():
    print("\n" + "=" * 60)
    print("TEST 1: CURA PERSONA & MANDATORY OPENING SEQUENCE (Q1 -> Q2 -> Q3)")
    print("=" * 60)

    mgr = CuraDialogueManager(session_id="test_persona_01")
    
    # Q1: Start session
    turn1 = mgr.start_session()
    print(f"Q1 Prompt: {turn1['bot_speech_hi']}")
    assert "Namaste" in turn1['bot_speech_hi'] or "नमस्ते" in turn1['bot_speech_hi'], "Greeting missing"
    assert "Diabetes" in turn1['bot_speech_hi'] or "Cura" in turn1['bot_speech_hi'], "Persona name missing"
    assert "रिकॉर्ड" in turn1['bot_speech_hi'] or "record" in turn1['bot_speech_hi'].lower(), "Consent to record missing"
    assert turn1['question_id'] == "Q1_GREETING", f"Expected Q1_GREETING, got {turn1['question_id']}"
    print("✅ Q1 Cura Introduction & Consent PASS")

    # Q2: Patient consents -> Should ask Identity
    turn2 = mgr.process_patient_turn("हाँ बिल्कुल, हम आगे बढ़ सकते हैं और आप रिकॉर्ड कर सकते हैं।")
    print(f"Q2 Prompt: {turn2['bot_speech_hi']}")
    assert turn2['question_id'] == "Q2_IDENTITY", f"Expected Q2_IDENTITY, got {turn2['question_id']}"
    assert "नाम" in turn2['bot_speech_hi'] and "उम्र" in turn2['bot_speech_hi'], "Identity fields prompt missing"
    assert mgr.state.consent_given is True, "Consent slot not captured"
    print("✅ Q2 Demographics Request PASS")

    # Q3: Patient provides identity -> Should ask Diabetic Status (Branch point)
    turn3 = mgr.process_patient_turn("मेरा नाम राजेश शर्मा है, मेरी उम्र 48 साल है और मैं पुरुष हूँ।")
    print(f"Q3 Prompt: {turn3['bot_speech_hi']}")
    assert turn3['question_id'] == "Q3_DIABETIC_STATUS", f"Expected Q3_DIABETIC_STATUS, got {turn3['question_id']}"
    assert mgr.state.name == "राजेश", f"Expected name राजेश, got {mgr.state.name}"
    assert mgr.state.age == 48, f"Expected age 48, got {mgr.state.age}"
    assert "पुरुष" in mgr.state.gender, f"Expected male gender, got {mgr.state.gender}"
    print("✅ Q3 Diabetic Status Branch Point PASS")


def test_female_vs_male_gestational_rule():
    print("\n" + "=" * 60)
    print("TEST 2: GESTATIONAL DIABETES RULE (FEMALE ONLY)")
    print("=" * 60)

    # 1. Male Patient -> Must NOT be asked gestational diabetes
    mgr_male = CuraDialogueManager(session_id="test_male_gest")
    mgr_male.start_session()
    mgr_male.process_patient_turn("हाँ ठीक है।")
    mgr_male.process_patient_turn("मेरा नाम विकास है, उम्र 35 साल, पुरुष।") # Male
    mgr_male.process_patient_turn("हाँ, मुझे डायबिटीज है।")

    asked_male_qids = []
    for _ in range(12):
        if mgr_male.state.is_completed:
            break
        asked_male_qids.append(mgr_male.current_question_id)
        mgr_male.process_patient_turn("सामान्य है")

    assert not any("gestational" in q.lower() for q in asked_male_qids), f"Gestational diabetes asked to male patient: {asked_male_qids}"
    print(f"✅ Male Patient: Gestational diabetes correctly skipped (Total asked: {len(asked_male_qids)})")

    # 2. Female Patient -> CAN be asked gestational diabetes
    mgr_female = CuraDialogueManager(session_id="test_female_gest")
    mgr_female.start_session()
    mgr_female.process_patient_turn("हाँ ठीक है।")
    mgr_female.process_patient_turn("मेरा नाम सुनीता है, उम्र 34 साल, महिला।") # Female
    mgr_female.process_patient_turn("मुझे डायबिटीज नहीं है।")

    asked_female_qids = []
    for _ in range(12):
        if mgr_female.state.is_completed:
            break
        asked_female_qids.append(mgr_female.current_question_id)
        mgr_female.process_patient_turn("नहीं, कुछ नहीं")

    assert any("gestational" in q.lower() for q in asked_female_qids), f"Gestational diabetes question missing for female: {asked_female_qids}"
    print("✅ Female Patient: Gestational diabetes question correctly included")


def test_branch_a_known_diabetic():
    print("\n" + "=" * 60)
    print("TEST 3: BRANCH A — KNOWN DIABETIC (COMPREHENSIVE SCREENING)")
    print("=" * 60)

    mgr = CuraDialogueManager(session_id="test_branch_a")
    mgr.start_session()
    mgr.process_patient_turn("हाँ, आगे बढ़ें।")
    mgr.process_patient_turn("मेरा नाम सुरेश वर्मा है, उम्र 52 साल है, पुरुष।")
    
    # Q3: Confirms Known Diabetic
    res = mgr.process_patient_turn("हाँ, मुझे पिछले 5 साल से डायबिटीज की बीमारी है।")
    assert mgr.state.branch == "BRANCH_A", f"Expected BRANCH_A, got {mgr.state.branch}"

    conversation_inputs = [
        "मुझे 5 साल पहले पता चला था।", # Duration
        "मैं रोजाना सुबह-शाम Metformin 500mg लेता हूँ, दवा नियमित है।", # Meds
        "मेरी पिछली फास्टिंग शुगर 195 आई थी।", # Last sugar
        "हाँ, 3 महीने वाला HbA1c टेस्ट 7.8% आया था।", # HbA1c
        "हाँ, मुझे अत्यधिक थकान और बार-बार पेशाब आता है।", # Complications
        "पैरों में सुन्नपन और झनझनाहट रहती है।", # Numbness
        "नहीं, कोई खास पसीना या कांपना नहीं हुआ।", # Hypo episodes
        "मैं रोज 20 मिनट टहलता हूँ लेकिन खाने में कभी-कभी मीठा खा लेता हूँ।", # Diet
        "मेरे पिताजी को भी डायबिटीज की बीमारी थी।", # Family history
        "बस डॉक्टर साहब को बताइएगा कि मुझे पैरों में दर्द रहता है।" # Closing
    ]

    for user_inp in conversation_inputs:
        if mgr.state.is_completed:
            break
        curr_q = mgr.current_question_id
        res = mgr.process_patient_turn(user_inp)
        print(f"  [Q:{mgr.state.questions_asked_count}/{mgr.MAX_QUESTIONS}] Asked: {curr_q} -> Next: {res['question_id']}")

    print(f"Total Questions Asked: {mgr.state.questions_asked_count}")
    assert 6 <= mgr.state.questions_asked_count <= 13, f"Question budget violated: {mgr.state.questions_asked_count}"
    assert "Q_CLOSING" in mgr.state.questions_asked_ids, "Closing question was not asked"
    assert mgr.state.is_completed is True, "Interview was not marked completed"
    assert "Metformin" in mgr.state.medications, "Metformin not captured"
    assert "numbness_tingling" in mgr.state.symptoms_present, "Neuropathy symptom not captured"
    print("✅ Branch A Complete Flow & Summary PASS")


def test_fainting_heart_chest_pain_emergencies():
    print("\n" + "=" * 60)
    print("TEST 4: FAINTING, SEVERE HEART/CHEST PAIN EMERGENCY DETECTION")
    print("=" * 60)

    # 1. Fainting (English & Hindi)
    em_faint_1 = EmergencyDetector.evaluate_utterance("I am fainting and feel completely dizzy")
    assert em_faint_1 is not None and em_faint_1["is_emergency"] is True, "Fainting English trigger failed"
    print("✅ Fainting (English) trigger PASS")

    em_faint_2 = EmergencyDetector.evaluate_utterance("मरीज अचानक बेहोश होकर गिर पड़ा है")
    assert em_faint_2 is not None and em_faint_2["is_emergency"] is True, "Fainting Hindi trigger failed"
    print("✅ Fainting (Hindi) trigger PASS")

    # 2. Severe Heart / Chest Pain
    em_heart_1 = EmergencyDetector.evaluate_utterance("मेरे सीने में बहुत तेज दर्द हो रहा है और दिल में भारीपन है")
    assert em_heart_1 is not None and em_heart_1["is_emergency"] is True, "Heart pain Hindi trigger failed"
    print("✅ Severe Heart/Chest Pain (Hindi) trigger PASS")

    em_heart_2 = EmergencyDetector.evaluate_utterance("I have severe chest pain radiating to left arm")
    assert em_heart_2 is not None and em_heart_2["is_emergency"] is True, "Chest pain English trigger failed"
    print("✅ Severe Chest Pain (English) trigger PASS")

    em_heart_3 = EmergencyDetector.evaluate_utterance("heart me bahut tez dard ho raha hai")
    assert em_heart_3 is not None and em_heart_3["is_emergency"] is True, "Heart pain Hinglish trigger failed"
    print("✅ Heart Pain (Hinglish) trigger PASS")


if __name__ == "__main__":
    test_persona_and_opening_sequence()
    test_female_vs_male_gestational_rule()
    test_branch_a_known_diabetic()
    test_fainting_heart_chest_pain_emergencies()
    print("\n" + "🎉" * 20)
    print("ALL 7 CURA ENHANCEMENT VALIDATIONS PASSED PERFECTLY!")
    print("🎉" * 20)
