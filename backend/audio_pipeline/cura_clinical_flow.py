import re
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from audio_pipeline.emergency_detector import EmergencyDetector

logger = logging.getLogger("cura_clinical_flow")

# =====================================================================
# 1. MANDATORY OPENING PROMPTS & CLINICAL QUESTION BANK
# =====================================================================

MANDATORY_Q1_GREETING = "नमस्ते, मेरा नाम Diabetes Dost है। मैं डॉक्टर से मिलने से पहले आपकी प्राथमिक स्वास्थ्य जानकारी एकत्र करने में मदद करूँगा। क्या हम आगे बढ़ सकते हैं और क्या मैं आपकी बातचीत रिकॉर्ड कर सकता हूँ?"
MANDATORY_Q2_IDENTITY = "धन्यवाद। आपकी सही पहचान और रिकॉर्ड के लिए, कृपया अपना पूरा नाम, उम्र और लिंग बताइए।"
MANDATORY_Q3_DIABETES_STATUS = "क्या आपको पहले से डायबिटीज (शुगर की बीमारी) है, या आपको इसके लक्षण महसूस हो रहे हैं, या आपको डायबिटीज नहीं है?"

CLOSING_QUESTION = "धन्यवाद। डॉक्टर को दिखाने से पहले क्या आप अपनी तरफ से कुछ और जरूरी बात या लक्षण बताना चाहते हैं?"

CLINICAL_QUESTION_BANK: Dict[str, Dict[str, Any]] = {
    # Mandatory Sequence
    "Q1_GREETING": {
        "id": "Q1_GREETING",
        "category": "consent_and_greeting",
        "text_hi": MANDATORY_Q1_GREETING,
        "target_slot": "consent_given"
    },
    "Q2_IDENTITY": {
        "id": "Q2_IDENTITY",
        "category": "patient_identity",
        "text_hi": MANDATORY_Q2_IDENTITY,
        "target_slot": "demographics"
    },
    "Q3_DIABETIC_STATUS": {
        "id": "Q3_DIABETIC_STATUS",
        "category": "branch_resolver",
        "text_hi": MANDATORY_Q3_DIABETES_STATUS,
        "target_slot": "diabetic_status"
    },

    # --- BRANCH A: KNOWN DIABETIC ---
    "A_DURATION": {
        "id": "A_DURATION",
        "branch": "BRANCH_A",
        "category": "diabetes_history",
        "text_hi": "आपको डायबिटीज का पता कितने समय पहले चला था (जैसे कितने महीने या साल से है)?",
        "target_slot": "diagnosis_duration"
    },
    "A_MEDICATIONS": {
        "id": "A_MEDICATIONS",
        "branch": "BRANCH_A",
        "category": "medication_adherence",
        "text_hi": "क्या आप रोजाना नियमित रूप से डायबिटीज की गोलियां (जैसे Metformin) या इंसुलिन ले रहे हैं, और क्या कभी दवा छूटती है?",
        "target_slot": "medication_adherence"
    },
    "A_LAST_SUGAR_READING": {
        "id": "A_LAST_SUGAR_READING",
        "branch": "BRANCH_A",
        "category": "glycemic_control",
        "text_hi": "हाल ही में कराई गई आपकी फास्टिंग या पीपी ब्लड शुगर की कोई ताजा रिपोर्ट या संख्या याद है?",
        "target_slot": "last_sugar_reading"
    },
    "A_HBA1C": {
        "id": "A_HBA1C",
        "branch": "BRANCH_A",
        "category": "glycemic_control",
        "text_hi": "क्या आपने पिछले 3 महीनों में अपना 3 महीने वाला शुगर टेस्ट (HbA1c) कराया है, उसका परिणाम क्या था?",
        "target_slot": "hba1c"
    },
    "A_COMPLICATIONS_SYMPTOMS": {
        "id": "A_COMPLICATIONS_SYMPTOMS",
        "branch": "BRANCH_A",
        "category": "symptom_screening",
        "text_hi": "क्या हाल ही में आपको अत्यधिक थकान, बार-बार पेशाब आना, या आंखों में धुंधलापन महसूस हुआ है?",
        "target_slot": "complication_symptoms"
    },
    "A_NUMBNESS_HEALING": {
        "id": "A_NUMBNESS_HEALING",
        "branch": "BRANCH_A",
        "category": "symptom_screening",
        "text_hi": "क्या आपके पैरों के पंजों में सुन्नपन, झनझनाहट, जलन रहती है या कोई चोट व घाव देर से भरता है?",
        "target_slot": "numbness_and_wounds"
    },
    "A_HYPO_HYPER_EPISODES": {
        "id": "A_HYPO_HYPER_EPISODES",
        "branch": "BRANCH_A",
        "category": "acute_episodes",
        "text_hi": "क्या पिछले कुछ हफ्तों में आपको अचानक बहुत ज्यादा पसीना आना, हाथ कांपना (लो शुगर) या अत्यधिक प्यास लगना महसूस हुआ?",
        "target_slot": "hypo_hyper_episodes"
    },
    "A_DIET_LIFESTYLE": {
        "id": "A_DIET_LIFESTYLE",
        "branch": "BRANCH_A",
        "category": "lifestyle_risk",
        "text_hi": "आपकी दैनिक शारीरिक सक्रियता कैसी है और क्या खान-पान में मीठा या चावल नियमित नियंत्रित रहता है?",
        "target_slot": "diet_lifestyle"
    },
    "A_FAMILY_HISTORY": {
        "id": "A_FAMILY_HISTORY",
        "branch": "BRANCH_A",
        "category": "genetic_risk",
        "text_hi": "क्या आपके परिवार में माता-पिता या भाई-बहन में भी किसी को डायबिटीज की बीमारी है?",
        "target_slot": "family_history"
    },
    "A_GESTATIONAL_DIABETES": {
        "id": "A_GESTATIONAL_DIABETES",
        "branch": "BRANCH_A",
        "category": "gestational_risk",
        "female_only": True,
        "text_hi": "क्या गर्भावस्था के दौरान आपको कभी ब्लड शुगर बढ़ने (Gestational Diabetes) की समस्या रही थी?",
        "target_slot": "gestational_diabetes"
    },

    # --- BRANCH B: UNSURE ---
    "B_THIRST_URINATION": {
        "id": "B_THIRST_URINATION",
        "branch": "BRANCH_B",
        "category": "symptom_screening",
        "text_hi": "क्या आपको सामान्य से बहुत ज्यादा प्यास लगती है या रात में बार-बार पेशाब के लिए उठना पड़ता है?",
        "target_slot": "thirst_urination"
    },
    "B_WEIGHT_FATIGUE": {
        "id": "B_WEIGHT_FATIGUE",
        "branch": "BRANCH_B",
        "category": "symptom_screening",
        "text_hi": "क्या बिना किसी कारण अचानक वजन कम हुआ है या दिनभर बहुत ज्यादा कमजोरी व थकान महसूस होती है?",
        "target_slot": "weight_fatigue"
    },
    "B_VISION_HEALING": {
        "id": "B_VISION_HEALING",
        "branch": "BRANCH_B",
        "category": "symptom_screening",
        "text_hi": "क्या कभी आंखों के सामने धुंधलापन आता है या चोट व घाव भरने में काफी लंबा समय लगता है?",
        "target_slot": "vision_healing"
    },
    "B_NUMBNESS_FEET": {
        "id": "B_NUMBNESS_FEET",
        "branch": "BRANCH_B",
        "category": "symptom_screening",
        "text_hi": "क्या आपके हाथ या पैरों के पंजों में सुन्नपन, चींटियां रेंगने जैसा या जलन का अहसास होता है?",
        "target_slot": "numbness_tingling"
    },
    "B_FAMILY_HISTORY": {
        "id": "B_FAMILY_HISTORY",
        "branch": "BRANCH_B",
        "category": "genetic_risk",
        "text_hi": "क्या आपके परिवार में माता-पिता या भाई-बहन में किसी को डायबिटीज की बीमारी है?",
        "target_slot": "family_history"
    },
    "B_RECENT_TEST": {
        "id": "B_RECENT_TEST",
        "branch": "BRANCH_B",
        "category": "test_history",
        "text_hi": "क्या आपने कभी पहले ग्लूकोज या फास्टिंग ब्लड शुगर टेस्ट कराया है?",
        "target_slot": "test_history"
    },
    "B_DIET_ACTIVITY": {
        "id": "B_DIET_ACTIVITY",
        "branch": "BRANCH_B",
        "category": "lifestyle_risk",
        "text_hi": "आपकी दिनचर्या में खान-पान और शारीरिक सक्रियता (जैसे वॉक या व्यायाम) का स्तर कैसा रहता है?",
        "target_slot": "diet_activity"
    },
    "B_GESTATIONAL_DIABETES": {
        "id": "B_GESTATIONAL_DIABETES",
        "branch": "BRANCH_B",
        "category": "gestational_risk",
        "female_only": True,
        "text_hi": "क्या कभी गर्भावस्था के दौरान आपका ब्लड शुगर बढ़ने की शिकायत हुई थी?",
        "target_slot": "gestational_diabetes"
    },

    # --- BRANCH C: NOT DIABETIC (RISK FACTOR SCREEN) ---
    "C_FAMILY_HISTORY": {
        "id": "C_FAMILY_HISTORY",
        "branch": "BRANCH_C",
        "category": "genetic_risk",
        "text_hi": "बहुत अच्छा। क्या आपके परिवार (माता-पिता या भाई-बहन) में किसी को शुगर की समस्या रही है?",
        "target_slot": "family_history"
    },
    "C_PHYSICAL_ACTIVITY": {
        "id": "C_PHYSICAL_ACTIVITY",
        "branch": "BRANCH_C",
        "category": "lifestyle_risk",
        "text_hi": "क्या आपकी दिनचर्या में रोजाना 30 मिनट टहलना या कोई शारीरिक व्यायाम शामिल है?",
        "target_slot": "physical_activity"
    },
    "C_KEY_SYMPTOMS": {
        "id": "C_KEY_SYMPTOMS",
        "branch": "BRANCH_C",
        "category": "symptom_screening",
        "text_hi": "क्या हाल के दिनों में आपको अत्यधिक थकान, बार-बार प्यास या घाव देर से भरने जैसी कोई परेशानी तो नहीं है?",
        "target_slot": "subtle_symptoms"
    },
    "C_WEIGHT_METABOLIC": {
        "id": "C_WEIGHT_METABOLIC",
        "branch": "BRANCH_C",
        "category": "metabolic_risk",
        "text_hi": "क्या आपका वजन हाल ही में तेजी से बढ़ा है या पेट के आसपास भारीपन महसूस होता है?",
        "target_slot": "weight_metabolic"
    },
    "C_GESTATIONAL_DIABETES": {
        "id": "C_GESTATIONAL_DIABETES",
        "branch": "BRANCH_C",
        "category": "gestational_risk",
        "female_only": True,
        "text_hi": "क्या महिलाओं में गर्भावस्था के दौरान आपको कभी शुगर (Gestational Diabetes) की समस्या रही थी?",
        "target_slot": "gestational_diabetes"
    },

    # Closing
    "Q_CLOSING": {
        "id": "Q_CLOSING",
        "category": "closing",
        "text_hi": CLOSING_QUESTION,
        "target_slot": "additional_complaints"
    }
}

# =====================================================================
# 2. PATIENT STRUCTURED CLINICAL STATE
# =====================================================================

class PatientClinicalState:
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.created_at: float = time.time()
        
        # Demographics
        self.name: Optional[str] = None
        self.age: Optional[int] = None
        self.gender: Optional[str] = None
        self.consent_given: bool = False
        
        # Branch Status: 'BRANCH_A' (Known), 'BRANCH_B' (Unsure), 'BRANCH_C' (Not Diabetic)
        self.branch: Optional[str] = None
        self.diabetic_status_raw: Optional[str] = None
        
        # Clinical Slots
        self.diagnosis_duration: Optional[str] = None
        self.medications: List[str] = []
        self.medication_adherence: Optional[str] = None
        self.last_sugar_reading: Optional[str] = None
        self.hba1c: Optional[str] = None
        
        # Symptom Tracker
        self.symptoms_present: List[str] = []
        self.symptoms_denied: List[str] = []
        
        # Risk factors
        self.family_history: Optional[bool] = None
        self.physical_activity_level: Optional[str] = None
        self.gestational_diabetes: Optional[bool] = None
        self.weight_change: Optional[str] = None
        self.hypo_episodes: Optional[bool] = None
        self.additional_notes: Optional[str] = None
        
        # Conversation & Triage
        self.qa_transcript: List[Dict[str, Any]] = []
        self.questions_asked_ids: List[str] = []
        self.questions_asked_count: int = 0
        self.is_completed: bool = False
        self.is_emergency: bool = False
        self.emergency_details: Optional[Dict[str, Any]] = None
        self.risk_tier: str = "LOW"

    def is_female(self) -> bool:
        if not self.gender:
            return False
        g_lower = self.gender.lower()
        return "महिला" in g_lower or "female" in g_lower or "स्त्री" in g_lower or "woman" in g_lower

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "demographics": {
                "name": self.name,
                "age": self.age,
                "gender": self.gender,
                "consent_given": self.consent_given,
                "is_female": self.is_female()
            },
            "branch": self.branch,
            "diabetic_status_raw": self.diabetic_status_raw,
            "clinical_data": {
                "diagnosis_duration": self.diagnosis_duration,
                "medications": self.medications,
                "medication_adherence": self.medication_adherence,
                "last_sugar_reading": self.last_sugar_reading,
                "hba1c": self.hba1c,
                "symptoms_present": self.symptoms_present,
                "symptoms_denied": self.symptoms_denied,
                "family_history": self.family_history,
                "physical_activity_level": self.physical_activity_level,
                "gestational_diabetes": self.gestational_diabetes,
                "weight_change": self.weight_change,
                "hypo_episodes": self.hypo_episodes,
                "additional_notes": self.additional_notes
            },
            "conversation": {
                "questions_asked_count": self.questions_asked_count,
                "questions_asked_ids": self.questions_asked_ids,
                "qa_transcript": self.qa_transcript,
                "is_completed": self.is_completed
            },
            "triage": {
                "risk_tier": self.risk_tier,
                "is_emergency": self.is_emergency,
                "emergency_details": self.emergency_details
            }
        }


# =====================================================================
# 3. CURA DIALOGUE MANAGER & ORCHESTRATOR
# =====================================================================

GLOBAL_CURA_SESSIONS: Dict[str, 'CuraDialogueManager'] = {}

def get_or_create_cura_session(session_id: Optional[str] = None) -> 'CuraDialogueManager':
    if not session_id:
        session_id = f"cura_{str(uuid.uuid4())[:8]}"
    if session_id not in GLOBAL_CURA_SESSIONS:
        GLOBAL_CURA_SESSIONS[session_id] = CuraDialogueManager(session_id=session_id)
    return GLOBAL_CURA_SESSIONS[session_id]


class CuraDialogueManager:
    """
    Cura AI Voice Pre-Screener Orchestrator for Diabetes.
    - Persona: Cura (Hindi speech in/out, male voice)
    - Comprehensive clinical question extraction (budget: max 12-13 questions)
    - Non-negotiable opening: Q1 Consent -> Q2 Identity -> Q3 Diabetic Status (Branching)
    - Gestational diabetes asked ONLY for female patients
    - Parallel Emergency Detection on every turn
    - Convergence on closing question
    """

    MIN_QUESTIONS = 6
    MAX_QUESTIONS = 13

    # Ordered question sequences per branch
    BRANCH_QUESTION_FLOWS = {
        "BRANCH_A": [
            "A_DURATION",
            "A_MEDICATIONS",
            "A_LAST_SUGAR_READING",
            "A_HBA1C",
            "A_COMPLICATIONS_SYMPTOMS",
            "A_NUMBNESS_HEALING",
            "A_HYPO_HYPER_EPISODES",
            "A_DIET_LIFESTYLE",
            "A_FAMILY_HISTORY",
            "A_GESTATIONAL_DIABETES"
        ],
        "BRANCH_B": [
            "B_THIRST_URINATION",
            "B_WEIGHT_FATIGUE",
            "B_VISION_HEALING",
            "B_NUMBNESS_FEET",
            "B_FAMILY_HISTORY",
            "B_RECENT_TEST",
            "B_DIET_ACTIVITY",
            "B_GESTATIONAL_DIABETES"
        ],
        "BRANCH_C": [
            "C_FAMILY_HISTORY",
            "C_PHYSICAL_ACTIVITY",
            "C_KEY_SYMPTOMS",
            "C_WEIGHT_METABOLIC",
            "C_GESTATIONAL_DIABETES"
        ]
    }

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = PatientClinicalState(session_id=session_id)
        self.current_question_id: Optional[str] = None
        self.turn_number: int = 0
        self.last_bot_prompt: str = ""

    def start_session(self) -> Dict[str, Any]:
        self.turn_number = 1
        self.state.questions_asked_count = 1
        q_data = CLINICAL_QUESTION_BANK["Q1_GREETING"]
        self.current_question_id = q_data["id"]
        self.state.questions_asked_ids.append(self.current_question_id)
        self.last_bot_prompt = q_data["text_hi"]

        return {
            "session_id": self.session_id,
            "turn": self.turn_number,
            "question_id": self.current_question_id,
            "bot_speech_hi": q_data["text_hi"],
            "questions_asked_count": self.state.questions_asked_count,
            "total_budget": self.MAX_QUESTIONS,
            "branch": self.state.branch,
            "is_emergency": False,
            "is_completed": False
        }

    def process_patient_turn(self, patient_utterance: str) -> Dict[str, Any]:
        self.turn_number += 1
        utterance = (patient_utterance or "").strip()

        # Record Q&A pair in history
        if self.current_question_id:
            self.state.qa_transcript.append({
                "turn": self.turn_number - 1,
                "question_id": self.current_question_id,
                "question_hi": self.last_bot_prompt,
                "patient_response_hi": utterance
            })

        # =====================================================================
        # 1. PARALLEL EMERGENCY DETECTION
        # =====================================================================
        emergency = EmergencyDetector.evaluate_utterance(utterance, session_id=self.session_id)
        if emergency:
            self.state.is_emergency = True
            self.state.emergency_details = emergency
            self.state.risk_tier = "EMERGENCY_ESCALATED"
            self.state.is_completed = True
            self.last_bot_prompt = emergency["response_hi"]
            
            logger.warning(f"🚨 EMERGENCY TRIGGERED in session {self.session_id}: {emergency['category']}")

            return {
                "session_id": self.session_id,
                "turn": self.turn_number,
                "question_id": emergency["rule_id"],
                "bot_speech_hi": emergency["response_hi"],
                "is_emergency": True,
                "emergency_details": emergency,
                "questions_asked_count": self.state.questions_asked_count,
                "is_completed": True,
                "state": self.state.to_dict()
            }

        # =====================================================================
        # 2. EXTRACT SLOTS ACCORDING TO CURRENT QUESTION & UTTERANCE
        # =====================================================================
        self._extract_entities_and_slots(utterance)

        # =====================================================================
        # 3. DETERMINE NEXT QUESTION IN STATE MACHINE
        # =====================================================================
        next_q_data = self._select_next_question()
        self.current_question_id = next_q_data["id"]
        if next_q_data["id"] != "INTERVIEW_COMPLETED":
            self.state.questions_asked_ids.append(self.current_question_id)
            self.state.questions_asked_count += 1
        else:
            self.state.is_completed = True

        self.last_bot_prompt = next_q_data["text_hi"]

        # Update clinical risk score
        self._update_risk_scoring()

        return {
            "session_id": self.session_id,
            "turn": self.turn_number,
            "question_id": self.current_question_id,
            "bot_speech_hi": next_q_data["text_hi"],
            "questions_asked_count": self.state.questions_asked_count,
            "total_budget": self.MAX_QUESTIONS,
            "branch": self.state.branch,
            "is_emergency": False,
            "is_completed": self.state.is_completed,
            "state": self.state.to_dict()
        }

    def _select_next_question(self) -> Dict[str, Any]:
        """
        Selects next question adhering to:
        - Mandatory sequence Q1 -> Q2 -> Q3
        - Branch flow (A / B / C)
        - Female-only filtering for gestational diabetes
        - Convergence on Q_CLOSING before budget limit (12-13 questions)
        """
        # Step 1: Identity
        if "Q2_IDENTITY" not in self.state.questions_asked_ids:
            return CLINICAL_QUESTION_BANK["Q2_IDENTITY"]

        # Step 2: Diabetic Status
        if "Q3_DIABETIC_STATUS" not in self.state.questions_asked_ids:
            return CLINICAL_QUESTION_BANK["Q3_DIABETIC_STATUS"]

        # Step 3: Check completion
        if "Q_CLOSING" in self.state.questions_asked_ids:
            self.state.is_completed = True
            closing_farewell = "धन्यवाद। आपकी सभी जानकारियां सुरक्षित दर्ज कर ली गई हैं और डॉक्टर को सारांश सौंप दिया गया है। कृपया प्रतीक्षालय में आराम करें।"
            return {
                "id": "INTERVIEW_COMPLETED",
                "category": "farewell",
                "text_hi": closing_farewell
            }

        branch = self.state.branch or "BRANCH_B"
        candidate_questions = self.BRANCH_QUESTION_FLOWS.get(branch, self.BRANCH_QUESTION_FLOWS["BRANCH_B"])

        # Filter unasked questions and check gender constraints
        unasked = []
        is_patient_female = self.state.is_female()

        for qid in candidate_questions:
            if qid in self.state.questions_asked_ids:
                continue
            q_spec = CLINICAL_QUESTION_BANK.get(qid, {})
            # If question is female-only and patient is not female, skip it
            if q_spec.get("female_only") and not is_patient_female:
                continue
            unasked.append(qid)

        # If no unasked questions left OR we hit budget limit (saving 1 for closing), ask Q_CLOSING
        if not unasked or (self.state.questions_asked_count >= self.MAX_QUESTIONS - 1):
            return CLINICAL_QUESTION_BANK["Q_CLOSING"]

        next_qid = unasked[0]
        return CLINICAL_QUESTION_BANK[next_qid]

    # =====================================================================
    # 4. ROBUST CLINICAL ENTITY & SLOT EXTRACTION
    # =====================================================================
    def _extract_entities_and_slots(self, text: str):
        if not text:
            return
        lower = text.lower()

        # A. Q1 Consent extraction
        if self.current_question_id == "Q1_GREETING":
            if any(w in lower for w in ["हाँ", "हां", "ठीक", "yes", "sure", "आगे बढ़ें", "कर सकते हैं", "चलेगा", "bilkul", "बिल्कुल"]):
                self.state.consent_given = True
            elif any(w in lower for w in ["नहीं", "रुकिए", "no", "nahi"]):
                self.state.consent_given = False

        # B. Q2 Identity extraction (Name, Age, Gender)
        if self.current_question_id == "Q2_IDENTITY" or not self.state.name or not self.state.age:
            # 1. Name
            name_pats = [
                r"(?:मेरा\s*नाम|नाम\s*है|naam\s*hai|i\s*am)\s*[:=]?\s*([A-Za-z\u0900-\u097F]+)",
                r"^([A-Za-z\u0900-\u097F]{3,15})\s*(?:हूँ|हूं|here|bol\s*raha)$"
            ]
            for pat in name_pats:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    cand = m.group(1).strip()
                    if cand.lower() not in ["मुझे", "हाँ", "नहीं", "डॉक्टर", "साल", "पुरुष", "महिला", "sugar", "diabetic"]:
                        self.state.name = cand
                        break

            # 2. Age
            age_pats = [
                r"(?:मेरी\s*उम्र|उम्र|age\s*is|age)\s*[:=]?\s*(\d{1,2})",
                r"(\d{1,2})\s*(?:साल|वर्ष|years|yrs|varsh)",
                r"\b(?:उम्र|वय)\s*(\d{1,2})\b"
            ]
            for pat in age_pats:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        val = int(m.group(1))
                        if 1 <= val <= 110:
                            self.state.age = val
                            break
                    except ValueError:
                        pass

            # 3. Gender
            if any(w in lower for w in ["पुरुष", "male", "लड़का", "आदमी", "man", "boy", "purush"]):
                self.state.gender = "पुरुष (Male)"
            elif any(w in lower for w in ["महिला", "female", "स्त्री", "लड़की", "औरत", "woman", "girl", "mahila"]):
                self.state.gender = "महिला (Female)"

        # C. Q3 Diabetic Status & Branch Resolution
        if self.current_question_id == "Q3_DIABETIC_STATUS" or self.state.branch is None:
            if any(w in lower for w in ["हाँ", "हां", "शुगर है", "डायबिटीज है", "मधुमेह है", "yes", "i have diabetes", "पहले से है", "इंसुलिन", "मेटफॉर्मिन"]):
                self.state.branch = "BRANCH_A"
                self.state.diabetic_status_raw = "Known Diabetic (हाँ, डायबिटीज है)"
            elif any(w in lower for w in ["पता नहीं", "पक्का नहीं", "शायद", "unsure", "not sure", "don't know", "डाउट है", "संदेह", "लक्षण लग रहे"]):
                self.state.branch = "BRANCH_B"
                self.state.diabetic_status_raw = "Unsure / Suspected (संदेह / लक्षण मौजूद)"
            elif any(w in lower for w in ["नहीं", "नहीं है", "no", "nahi", "शुगर नहीं", "डायबिटीज नहीं"]):
                self.state.branch = "BRANCH_C"
                self.state.diabetic_status_raw = "Not Diabetic (डायबिटीज नहीं है)"

        # D. Duration of Diabetes
        if any(w in lower for w in ["साल", "महीने", "वर्ष", "years", "months"]) and ("डायबिटीज" in lower or "से है" in lower or self.current_question_id == "A_DURATION"):
            dur_m = re.findall(r'((?:\d+|एक|दो|तीन|चार|पांच|दस)\s*(?:साल|वर्ष|महीने|महीनों|years|months))', text)
            if dur_m:
                self.state.diagnosis_duration = dur_m[0]

        # E. Blood Sugar Readings
        sugar_vals = re.findall(r'\b(1\d\d|2\d\d|3\d\d|4\d\d|[6-9]\d)\b', text)
        if sugar_vals and any(w in lower for w in ["शुगर", "sugar", "फास्टिंग", "fasting", "pp", "पीपी", "रीडिंग", "रैंडम", "rbs", "fbs"]):
            self.state.last_sugar_reading = f"{sugar_vals[0]} mg/dL"

        # F. HbA1c
        hba1c_vals = re.findall(r'\b([5-9]\.\d|1[0-4]\.\d)\b', text)
        if any(w in lower for w in ["hba1c", "एचबीए1सी", "3 महीने वाला", "तीन महीने"]) and hba1c_vals:
            self.state.hba1c = f"{hba1c_vals[0]}%"

        # G. Medications Extraction
        med_dict = {
            "metformin": "Metformin", "मेटफॉर्मिन": "Metformin", "ग्लूकोफेज": "Metformin",
            "glimepiride": "Glimepiride", "ग्लिमेपिराइड": "Glimepiride", "amaryl": "Glimepiride",
            "insulin": "Insulin", "इंसुलिन": "Insulin", "इन्सुलिन": "Insulin",
            "vildagliptin": "Vildagliptin", "sitagliptin": "Sitagliptin", "dapagliflozin": "Dapagliflozin",
            "telmisartan": "Telmisartan", "amlodipine": "Amlodipine"
        }
        for med_k, med_v in med_dict.items():
            if med_k in lower and med_v not in self.state.medications:
                self.state.medications.append(med_v)

        # Medication adherence
        if self.current_question_id == "A_MEDICATIONS":
            if any(w in lower for w in ["रोज", "नियमित", "regular", "समय पर", "लेता हूँ", "लेती हूँ", "कंटिन्यू", "नहीं छूटती"]):
                self.state.medication_adherence = "Good / Regular (नियमित)"
            elif any(w in lower for w in ["भूल जाता", "छूट जाती", "कभी-कभी", "irregular", "छोड़ दी"]):
                self.state.medication_adherence = "Poor / Irregular (अनियमित / छूटती है)"

        # H. Specific Symptom Detection
        symptom_triggers = {
            "excessive_thirst": [r"प्यास.*ज्यादा", r"ज्यादा.*प्यास", r"गला.*सूख", r"thirst", r"बार-बार.*प्यास"],
            "frequent_urination": [r"बार-बार.*पेशाब", r"पेशाब.*ज्यादा", r"frequent.*urination", r"रात.*पेशाब"],
            "unexplained_weight_loss": [r"वजन.*कम", r"weight.*loss", r"दुबला"],
            "fatigue_weakness": [r"थकान", r"कमजोरी", r"fatigue", r"सुस्ती", r"कमज़ोरी"],
            "blurred_vision": [r"धुंधला", r"vision", r"आंखों.*कमजोरी", r"आंखों.*धुंधला"],
            "slow_healing_wounds": [r"घाव.*नहीं.*भर", r"चोट.*देर", r"wound", r"घाव.*देर", r"घाव.*सूखता.*नहीं"],
            "numbness_tingling": [r"सुन्न", r"झनझनाहट", r"चींटियां", r"tingling", r"numbness", r"पैरों.*जलन", r"पंजों.*सुन्न"]
        }

        is_negated_response = bool(re.search(r'(?:कोई\s*(?:भी\s*)?(?:परेशानी|समस्या|तकलीफ|दिक्कत|लक्षण)?\s*नहीं|बिलकुल\s*नहीं|ऐसा\s*कुछ\s*नहीं|nahi|no|नहीं\s*है)', lower))

        for sym_key, patterns in symptom_triggers.items():
            for pat in patterns:
                if re.search(pat, lower):
                    if is_negated_response and not any(aff in lower for aff in ["हाँ", "हां", "yes", "रहती है", "होता है", "लगता है"]):
                        if sym_key not in self.state.symptoms_denied:
                            self.state.symptoms_denied.append(sym_key)
                    else:
                        if sym_key not in self.state.symptoms_present:
                            self.state.symptoms_present.append(sym_key)
                    break

        # I. Family History
        if any(w in lower for w in ["पिता", "माता", "पापा", "मम्मी", "भाई", "बहन", "परिवार", "family"]):
            if any(w in lower for w in ["है", "था", "थी", "yes", "डायबिटीज है", "शुगर थी"]):
                self.state.family_history = True
            elif any(w in lower for w in ["नहीं", "किसी को नहीं", "no"]):
                self.state.family_history = False

        # J. Physical Activity
        if self.current_question_id in ["C_PHYSICAL_ACTIVITY", "A_DIET_LIFESTYLE", "B_DIET_ACTIVITY"]:
            if any(w in lower for w in ["वॉक", "टहलता", "दौड़", "योग", "exercise", "जिम", "हाँ", "घूमने"]):
                self.state.physical_activity_level = "Active (सक्रिय - नियमित वॉक/व्यायाम)"
            elif any(w in lower for w in ["नहीं", "बैठे रहना", "समय नहीं", "sedentary", "कम"]):
                self.state.physical_activity_level = "Sedentary (अक्रिय / कम शारीरिक हलचल)"

        # K. Gestational Diabetes
        if "gestational" in self.current_question_id.lower():
            if any(w in lower for w in ["हाँ", "हां", "yes", "हुई थी", "बढ़ी थी"]):
                self.state.gestational_diabetes = True
            elif any(w in lower for w in ["नहीं", "no", "कभी नहीं"]):
                self.state.gestational_diabetes = False

        # L. Closing notes
        if self.current_question_id == "Q_CLOSING":
            if text and not any(w in lower for w in ["नहीं", "कुछ नहीं", "बस यही", "no", "nothing"]):
                self.state.additional_notes = text

    # =====================================================================
    # 5. CLINICAL RISK SCORING ENGINE
    # =====================================================================
    def _update_risk_scoring(self):
        if self.state.is_emergency:
            self.state.risk_tier = "EMERGENCY_ESCALATED"
            return

        score = 0
        
        # Branch A risk factors
        if self.state.branch == "BRANCH_A":
            score += 3
            if self.state.medication_adherence and "Poor" in self.state.medication_adherence:
                score += 2
            if self.state.last_sugar_reading:
                try:
                    val = int(re.sub(r'\D', '', self.state.last_sugar_reading))
                    if val >= 250:
                        score += 3
                    elif val >= 180:
                        score += 2
                except Exception:
                    pass
        elif self.state.branch == "BRANCH_B":
            score += 2
        elif self.state.branch == "BRANCH_C":
            score += 0

        # Symptoms
        symptom_count = len(self.state.symptoms_present)
        score += (symptom_count * 1.5)

        if "blurred_vision" in self.state.symptoms_present or "slow_healing_wounds" in self.state.symptoms_present:
            score += 2
        if "numbness_tingling" in self.state.symptoms_present:
            score += 1.5

        if self.state.family_history:
            score += 1.5
        if self.state.gestational_diabetes:
            score += 2.0
        if self.state.physical_activity_level and "Sedentary" in self.state.physical_activity_level:
            score += 1

        if self.state.age and self.state.age >= 45:
            score += 1

        if score >= 6.5:
            self.state.risk_tier = "HIGH"
        elif score >= 3.0:
            self.state.risk_tier = "MODERATE"
        else:
            self.state.risk_tier = "LOW"
