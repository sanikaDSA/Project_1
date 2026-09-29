import re
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from question_selector.question_selector import GLOBAL_QUESTION_SELECTOR, QuestionSelector
from urgency.urgency_engine import GLOBAL_URGENCY_ENGINE, UrgencyEngine
from session_store.session_store import SehatSessionState, GLOBAL_SESSION_STORE
from audit.audit_logger import GLOBAL_AUDIT_LOGGER
from audio_pipeline.hinglish_formatter import format_patient_speech_hinglish

logger = logging.getLogger("sehat_dialogue")

class DialogueManager:
    """
    Core Dialogue Manager for Sehat AI.
    Coordinates context-aware question selection, slot extraction, branch transitions,
    consent enforcement, and termination conditions.
    """
    def __init__(self, question_selector: Optional[QuestionSelector] = None, urgency_engine: Optional[UrgencyEngine] = None):
        self.selector = question_selector or GLOBAL_QUESTION_SELECTOR
        self.urgency_engine = urgency_engine or GLOBAL_URGENCY_ENGINE

    def start_session(self, session_id: Optional[str] = None) -> Tuple[SehatSessionState, str]:
        """Initialize new Sehat AI pre-screening session with mandatory Sehat greeting."""
        sid = session_id or f"sehat_{uuid.uuid4().hex[:8]}"
        state = GLOBAL_SESSION_STORE.get_or_create(sid)
        state.current_question_id = "COM_CONSENT"
        state.asked_question_ids = ["COM_CONSENT"]
        state.questions_asked_count = 1

        first_prompt = (
            "नमस्ते, मेरा नाम Diabetes Dost है। मैं Doctor से मिलने से पहले आपकी Diabetes से जुड़ी कुछ Health "
            "जानकारी समझने में help करूंगी। क्या आप बातचीत और Audio Recording के लिए सहमत हैं?"
        )

        GLOBAL_AUDIT_LOGGER.log_event(sid, "SESSION_STARTED", {"initial_question_id": "COM_CONSENT"})
        GLOBAL_SESSION_STORE.save(state)
        return state, first_prompt

    def process_turn(self, session_id: str, patient_utterance: str) -> Dict[str, Any]:
        """
        Process a patient's voice turn.
        Extracts clinical entities, determines consent / branch, evaluates risk signals,
        acknowledges the patient's specific reply empathetically, and selects the next context-aware question.
        Only at the end after listening to all turns does it deliver the comprehensive clinical opinion.
        """
        state = GLOBAL_SESSION_STORE.get_or_create(session_id)
        current_qid = state.current_question_id

        # 0. Check for inaudible / unclear / empty response (Retry once safeguard)
        clean_text = (patient_utterance or "").strip()
        is_inaudible = (
            len(clean_text) < 2 or
            clean_text in ["...", "हम्म", "hmm", "uh", "um", "क्या", "?", "voice"]
        )
        if is_inaudible and current_qid != "COMPLETED":
            retries = state.question_retry_count.get(current_qid, 0)
            if retries < 1:
                state.question_retry_count[current_qid] = retries + 1
                q_text = self._lookup_question_text(current_qid)
                retry_speech = f"माफ कीजिए, आपकी Voice स्पष्ट नहीं आई। {q_text}"
                GLOBAL_AUDIT_LOGGER.log_event(session_id, "QUESTION_RETRY", {
                    "question_id": current_qid,
                    "reason": "inaudible_response"
                })
                GLOBAL_SESSION_STORE.save(state)
                return {
                    "bot_speech_hi": retry_speech,
                    "is_completed": False,
                    "state": state.to_dict(),
                    "next_question_id": current_qid,
                    "questions_asked_count": state.questions_asked_count,
                    "max_budget": state.max_budget
                }

        # 1. Format Patient Input (English words in English script, Hindi words in Hindi Devanagari)
        formatted_utterance = format_patient_speech_hinglish(patient_utterance)

        state.record_answer(
            question_id=current_qid,
            question_hi=self._lookup_question_text(current_qid),
            patient_answer_hi=formatted_utterance
        )
        GLOBAL_AUDIT_LOGGER.log_event(session_id, "PATIENT_RESPONSE", {
            "question_id": current_qid,
            "patient_utterance": formatted_utterance,
            "raw_utterance": patient_utterance
        })

        # 2. Process Q1: Mandatory Consent Check
        if current_qid == "COM_CONSENT":
            res = self._handle_consent_turn(state, formatted_utterance)
            res["formatted_patient_transcript"] = formatted_utterance
            return res

        # 3. Process Q2: Demographics Extraction & Mandatory Completeness Check
        if current_qid == "COM_DEMOGRAPHICS":
            self._extract_demographics(state, formatted_utterance)
            
            missing_fields = []
            if not state.demographics.get("name"):
                missing_fields.append("Name")
            if not state.demographics.get("age"):
                missing_fields.append("Age")
            if not state.demographics.get("gender"):
                missing_fields.append("Gender")

            if missing_fields and state.demographic_retry_count < 2:
                state.demographic_retry_count += 1
                p_name = state.demographics.get("name", "")
                
                if len(missing_fields) == 1:
                    field = missing_fields[0]
                    if "Gender" in field:
                        bot_speech = f"धन्यवाद {p_name} जी, कृपया अपना Gender भी बताइए।" if p_name else "धन्यवाद, कृपया अपना Gender भी बताइए।"
                    elif "Age" in field:
                        bot_speech = f"धन्यवाद {p_name} जी, कृपया अपनी Age भी बताइए।" if p_name else "धन्यवाद, कृपया अपनी Age भी बताइए।"
                    else:
                        bot_speech = "धन्यवाद, कृपया अपना पूरा Name भी बताइए।"
                else:
                    bot_speech = f"धन्यवाद, कृपया अपना {' और '.join(missing_fields)} भी स्पष्ट रूप से बताइए।"

                GLOBAL_AUDIT_LOGGER.log_event(session_id, "DEMOGRAPHIC_INCOMPLETE_RETRY", {
                    "missing_fields": missing_fields,
                    "retry_count": state.demographic_retry_count
                })
                GLOBAL_SESSION_STORE.save(state)
                return {
                    "bot_speech_hi": bot_speech,
                    "formatted_patient_transcript": formatted_utterance,
                    "is_completed": False,
                    "state": state.to_dict(),
                    "next_question_id": "COM_DEMOGRAPHICS",
                    "questions_asked_count": state.questions_asked_count,
                    "max_budget": state.max_budget
                }

            if "demographics" not in state.completed_topics:
                state.completed_topics.append("demographics")

        # 4. Mid-session branch re-alignment or Q3 branch selection
        u_lower = formatted_utterance.lower()
        if any(w in u_lower for w in ["मुझे डायबिटीज नहीं है", "मुझे शुगर नहीं है", "डायबिटीज ही नहीं है", "no diabetes", "nahi hai diabetes", "diabetes nahi hai", "diabetes nahi", "not diabetic"]):
            if state.branch != "not_diabetic":
                state.branch = "not_diabetic"
                if "diabetic_status" not in state.completed_topics:
                    state.completed_topics.append("diabetic_status")
        elif current_qid == "COM_STATUS":
            self._determine_branch(state, formatted_utterance)
            if "diabetic_status" not in state.completed_topics:
                state.completed_topics.append("diabetic_status")

        # 5. Extract Clinical Entities & Symptoms from formatted English+Hindi context
        self._extract_clinical_slots(state, formatted_utterance)

        # 6. Check for Closing Question Response (Normal Completion with Complete Clinical Opinion)
        if current_qid == "CLS_FINAL_FEEDBACK" or (state.current_question_id and state.current_question_id.startswith("CLS_")):
            state.is_completed = True
            self._finalize_urgency(state)
            closing_speech = self._build_final_opinion_speech(state)
            GLOBAL_AUDIT_LOGGER.log_event(session_id, "SESSION_COMPLETED", {"status": "normal_completion"})
            GLOBAL_SESSION_STORE.save(state)
            return {
                "bot_speech_hi": closing_speech,
                "formatted_patient_transcript": formatted_utterance,
                "is_completed": True,
                "state": state.to_dict(),
                "next_question_id": "COMPLETED",
                "questions_asked_count": state.questions_asked_count,
                "max_budget": state.max_budget
            }

        # 7. Select Context-Aware Next Question based on what the user said
        next_q = self.selector.select_next_question(
            current_branch=state.branch,
            asked_question_ids=state.asked_question_ids,
            completed_topics=state.completed_topics,
            questions_asked_count=state.questions_asked_count,
            demographics=state.demographics,
            symptoms_reported=state.symptoms_reported,
            risk_signals=state.risk_signals,
            max_budget=state.max_budget
        )

        if not next_q:
            # Reached natural conclusion -> Provide full clinical opinion & summary
            state.is_completed = True
            self._finalize_urgency(state)
            closing_speech = self._build_final_opinion_speech(state)
            GLOBAL_AUDIT_LOGGER.log_event(session_id, "SESSION_COMPLETED", {"status": "questions_exhausted"})
            GLOBAL_SESSION_STORE.save(state)
            return {
                "bot_speech_hi": closing_speech,
                "formatted_patient_transcript": formatted_utterance,
                "is_completed": True,
                "state": state.to_dict(),
                "next_question_id": "COMPLETED",
                "questions_asked_count": state.questions_asked_count,
                "max_budget": state.max_budget
            }

        # Advance Question State
        state.questions_asked_count += 1
        state.current_question_id = next_q["question_id"]
        state.asked_question_ids.append(next_q["question_id"])
        if next_q.get("topic") and next_q["topic"] not in state.completed_topics:
            state.completed_topics.append(next_q["topic"])

        # Frame next question with empathetic acknowledgment based on previous answer
        bot_speech = self._frame_next_question(state, current_qid, formatted_utterance, next_q)

        GLOBAL_AUDIT_LOGGER.log_event(session_id, "QUESTION_SELECTED", {
            "question_id": next_q["question_id"],
            "question_text": bot_speech,
            "turn_number": state.questions_asked_count
        })

        self._finalize_urgency(state)
        GLOBAL_SESSION_STORE.save(state)

        return {
            "bot_speech_hi": bot_speech,
            "formatted_patient_transcript": formatted_utterance,
            "is_completed": False,
            "state": state.to_dict(),
            "next_question_id": next_q["question_id"],
            "questions_asked_count": state.questions_asked_count,
            "max_budget": state.max_budget
        }

    def _frame_next_question(self, state: SehatSessionState, prev_qid: str, patient_utterance: str, next_q: Dict[str, Any]) -> str:
        """
        Dynamically builds a personalized conversational bridge directly acknowledging
        ONLY what the user said in the immediate previous turn.
        Never repeats the patient's name repeatedly on every turn.
        """
        p_name = state.demographics.get("name")
        next_raw_q = next_q.get("question_hi", "")
        u_lower = patient_utterance.lower()

        # 1. Patient reaffirmed they do not have diabetes
        if any(w in u_lower for w in ["डायबिटीज नहीं है", "शुगर नहीं है", "डायबिटीज ही नहीं है", "no diabetes", "nahi hai diabetes"]):
            return f"बिल्कुल, समझ गया कि आपको Diabetes नहीं है। {next_raw_q}"

        # 2. Acknowledgment after Demographics (Turn 2 -> 3 ONLY: use name once here)
        if prev_qid == "COM_DEMOGRAPHICS":
            if p_name:
                return f"धन्यवाद {p_name} जी, आपकी Details नोट कर ली गई हैं। {next_raw_q}"
            return f"धन्यवाद, आपकी Details नोट कर ली गई हैं। {next_raw_q}"

        # 3. Acknowledgment after Diabetic Status / Type
        if prev_qid == "COM_STATUS":
            if state.branch == "known_diabetic":
                return f"जी, समझ गया। {next_raw_q}"
            elif state.branch == "not_diabetic":
                return f"यह अच्छी बात है कि आपको Diabetes नहीं है। {next_raw_q}"
            else:
                return f"जी ठीक है, आइए आपकी Health और Symptoms को समझ लेते हैं। {next_raw_q}"

        if prev_qid == "KD_DIABETES_TYPE":
            if getattr(state, "diabetes_type", None):
                return f"जी, आपका {state.diabetes_type} का Record नोट कर लिया गया है। {next_raw_q}"
            return f"जी ठीक है। {next_raw_q}"

        # 4. Acknowledgment for Medications / Insulin (Turn for medications ONLY)
        if prev_qid == "KD_MEDICATIONS":
            if any(w in u_lower for w in ["नहीं", "nahin", "no", "nahi leta", "नहीं लेता", "नहीं लेती", "कोई नहीं"]):
                return f"समझ गया, आप कोई Medicine नहीं ले रहे हैं। {next_raw_q}"
            elif "इंसुलिन" in u_lower or "insulin" in u_lower:
                return f"ठीक है, Insulin का Record नोट कर लिया गया है। {next_raw_q}"
            else:
                return f"ठीक है, आपकी Regular Medicines का Record नोट कर लिया गया है। {next_raw_q}"

        # 5. Acknowledgment for Blood Sugar Values (Turn for sugar levels ONLY)
        if prev_qid in ["KD_SUGAR_LEVELS", "UN_PREV_TESTING", "ND_RECENT_SCREENING"]:
            recent_reading = state.blood_sugar_readings[-1]["value"] if state.blood_sugar_readings else None
            if recent_reading:
                return f"जी, आपका {recent_reading} mg/dL का Blood Sugar Level नोट कर लिया गया है। {next_raw_q}"
            elif any(w in u_lower for w in ["नहीं", "nahin", "no", "याद नहीं"]):
                return f"कोई बात नहीं। {next_raw_q}"
            else:
                return f"जी ठीक है। {next_raw_q}"

        # 6. Acknowledgment for Symptoms
        if prev_qid in ["KD_THIRST_URINATION", "UN_CLASSIC_SYMPTOMS"]:
            if any(w in u_lower for w in ["प्यास", "पेशाब", "thirst", "urine", "हाँ", "हां", "yes", "होती है", "लगती है"]):
                return f"जी, यह जानकारी नोट कर ली गई है। {next_raw_q}"
            return f"ठीक है। {next_raw_q}"

        if prev_qid in ["KD_VISION_FATIGUE", "UN_ENERGY_VISION"]:
            if any(w in u_lower for w in ["थकान", "धुंधला", "कमजोरी", "fatigue", "vision", "blur", "हाँ", "हां", "yes"]):
                return f"जी, यह Details नोट कर ली गई हैं। {next_raw_q}"
            return f"ठीक है। {next_raw_q}"

        if prev_qid in ["KD_NUMBNESS_WOUNDS", "UN_HEALING_NUMBNESS"]:
            if any(w in u_lower for w in ["घाव", "सुन्न", "झनझनाहट", "wound", "numb", "tingling", "हाँ", "हां", "yes"]):
                return f"जी, हाथ-पैरों की Condition नोट कर ली गई है। {next_raw_q}"
            return f"ठीक है। {next_raw_q}"

        if prev_qid in ["KD_HYPO_EPISODES"]:
            return f"ठीक है। {next_raw_q}"

        if prev_qid in ["KD_LIFESTYLE_DIET", "ND_PHYSICAL_ACTIVITY", "ND_WEIGHT_LIFESTYLE"]:
            return f"ठीक है। {next_raw_q}"

        # 7. Acknowledgment for Family History
        if prev_qid in ["UN_FAMILY_HISTORY", "ND_FAMILY_HISTORY", "KD_FAMILY_HISTORY"] or (prev_qid and "family" in prev_qid.lower()):
            if state.family_history is True:
                return f"जी, परिवार में Diabetes का इतिहास नोट कर लिया गया है। {next_raw_q}"
            elif state.family_history is False:
                return f"जी, समझ गया, परिवार में Diabetes की History नहीं है। {next_raw_q}"
            return f"जी, Family History नोट कर ली गई है। {next_raw_q}"

        # 8. Acknowledgment for Pregnancy History
        if prev_qid in ["UN_PREGNANCY_HISTORY", "ND_PREGNANCY_HISTORY", "KD_PREGNANCY_HISTORY"] or (prev_qid and "pregnancy" in prev_qid.lower()):
            if state.pregnancy_history is True:
                return f"जी, आपकी Pregnancy History नोट कर ली गई है। {next_raw_q}"
            elif state.pregnancy_history is False:
                return f"ठीक है, Pregnancy History में कोई पूर्व समस्या दर्ज नहीं है। {next_raw_q}"
            return f"ठीक है। {next_raw_q}"

        if prev_qid in ["UN_GESTATIONAL", "ND_GESTATIONAL", "KD_GESTATIONAL"] or (prev_qid and "gestational" in prev_qid.lower()):
            if state.gestational_history is True:
                return f"जी, गर्भावस्था में Sugar बढ़ने (Gestational Diabetes) का रिकॉर्ड नोट कर लिया गया है। {next_raw_q}"
            elif state.gestational_history is False:
                return f"ठीक है, गर्भावस्था के समय Sugar Normal रहने का रिकॉर्ड दर्ज है। {next_raw_q}"
            return f"ठीक है। {next_raw_q}"

        # 9. General fallback: Do NOT repeat old acknowledgements or patient name
        return f"{next_raw_q}"

    def _build_final_opinion_speech(self, state: SehatSessionState) -> str:
        """
        Builds a comprehensive, personalized clinical opinion and triage recommendation
        only after listening fully to the user's entire consultation.
        """
        p_name = state.demographics.get("name")
        greeting_prefix = f"{p_name} जी, " if p_name else ""
        
        # Clinical synthesis components
        findings = []
        if state.branch == "known_diabetic":
            if getattr(state, "diabetes_type", None):
                findings.append(f"आपको {state.diabetes_type} Diabetes है")
            else:
                findings.append("आपको पहले से Diabetes है")
        elif state.branch == "unsure" and state.symptoms_reported:
            findings.append("आपमें Diabetes से मिलते-जुलते Symptoms देखे गए हैं")
        elif state.branch == "not_diabetic":
            findings.append("आपको पहले से Diabetes नहीं है")

        if state.family_history is True:
            findings.append("Family में Diabetes की Positive History है")
        elif state.family_history is False:
            findings.append("Family में Diabetes की कोई History नहीं है")

        if state.pregnancy_history is True:
            if state.gestational_history is True:
                findings.append("Pregnancy के दौरान Gestational Diabetes (GDM) की History रही है")
            elif state.gestational_history is False:
                findings.append("Pregnancy के दौरान Sugar सामान्य (Normal) रही है")
            else:
                findings.append("Pregnancy की History दर्ज है")

        if state.symptoms_reported:
            sym_names = []
            if "polydipsia_excessive_thirst" in state.symptoms_reported:
                sym_names.append("Excessive Thirst (प्यास)")
            if "polyuria_frequent_urination" in state.symptoms_reported:
                sym_names.append("Frequent Urine (पेशाब)")
            if "fatigue_weakness" in state.symptoms_reported:
                sym_names.append("Fatigue (थकान)")
            if "blurred_vision" in state.symptoms_reported:
                sym_names.append("Blurry Vision")
            if "delayed_wound_healing" in state.symptoms_reported:
                sym_names.append("Delayed Wound Healing (घाव भरने में देरी)")
            if "numbness_tingling" in state.symptoms_reported:
                sym_names.append("Numbness (हाथ-पैरों में सुन्नपन)")
            if sym_names:
                findings.append(f"आपके बताए मुख्य Symptoms ({', '.join(sym_names)}) हैं")

        if state.blood_sugar_readings:
            recent_sugar = state.blood_sugar_readings[-1]["value"]
            findings.append(f"हालिया Blood Sugar Level लगभग {recent_sugar} mg/dL दर्ज है")

        findings_summary = " और ".join(findings) if findings else "आपकी Health Details नोट कर ली गई हैं"

        # Triage and advice synthesis
        advice = (
            "Diabetes के proper management के लिए Regular Blood Sugar Test (Fasting व PP) और हर 3 months में HbA1c Test कराना जरूरी है। "
            "Healthy Diet रखें और पर्याप्त पानी पिएं।"
        )

        closing_opinion = (
            f"धन्यवाद। {greeting_prefix}मैंने आपकी पूरी बात ध्यानपूर्वक सुन ली है। {findings_summary}। "
            f"{advice} Doctor के लिए आपकी संपूर्ण Clinical Summary (Poona Hospital Triage Report) तैयार कर दी गई है। "
            f"कृपया कुछ देर Wait करें, Poona Hospital के Doctor जल्द ही आपसे Consult करेंगे।"
        )
        return closing_opinion

    def _handle_consent_turn(self, state: SehatSessionState, utterance: str) -> Dict[str, Any]:
        cleaned = utterance.lower()
        
        # 1. Check affirmative patterns (Marathi, Hindi, English)
        affirmative_patterns = [
            r"\b(yes|yeah|yep|sure|ok|okay|agree|agreed|consent|proceed|start|continue|no\s*problem|fine|ready)\b",
            r"(हाँ|हां|बिल्कुल|सहमत|सहमति|ठीक\s*है|अवश्य|जरूर|चालेल|हो|होय|नक्की|माझी\s*संमती\s*आहे|सुरू\s*करा|चालू\s*करा|सांगा|तयार\s*आहे|काही\s*हरकत\s*नाही)"
        ]
        has_affirmative = any(re.search(pat, cleaned, re.IGNORECASE) for pat in affirmative_patterns)

        # 2. Explicit Rejection patterns
        rejection_patterns = [
            r"\b(disagree|don\'?t\s*agree|do\s*not\s*agree|not\s*agree|no\s*agree|reject|asahmat|mana|deny|cancel|don\'?t\s*consent)\b",
            r"(डिसअग्री|सहमति\s*नहीं|सहमत\s*नहीं|असहमत|मना|रिकॉर्ड\s*मत\s*करो|अनुमति\s*नहीं|संमती\s*नाही|नको|रेकॉर्ड\s*करू\s*नका)"
        ]
        is_rejection = any(re.search(pat, cleaned, re.IGNORECASE) for pat in rejection_patterns)

        # Strict rejection only if explicitly rejected without affirmative override
        if is_rejection and not has_affirmative:
            state.consent = False
            state.consent_rejected = True
            state.is_completed = True
            exit_speech = "समझ गया। आपकी Consent के बिना हम यह Pre-Screening आगे नहीं बढ़ाएंगे। आपका धन्यवाद।"
            GLOBAL_AUDIT_LOGGER.log_event(state.session_id, "CONSENT_REJECTED", {"utterance": utterance})
            GLOBAL_SESSION_STORE.save(state)
            return {
                "bot_speech_hi": exit_speech,
                "is_completed": True,
                "consent_rejected": True,
                "state": state.to_dict(),
                "next_question_id": "CONSENT_REJECTED"
            }

        state.consent = True
        state.audio_recording_consent = True
        state.consent_timestamp = datetime.now(timezone.utc).isoformat()
        state.questions_asked_count += 1
        state.current_question_id = "COM_DEMOGRAPHICS"
        state.asked_question_ids.append("COM_DEMOGRAPHICS")
        if "consent" not in state.completed_topics:
            state.completed_topics.append("consent")

        q2_speech = "धन्यवाद। आपकी सही पहचान और Record के लिए, कृपया अपना पूरा Name, Age और Gender बताइए।"
        GLOBAL_AUDIT_LOGGER.log_event(state.session_id, "CONSENT_GRANTED", {"timestamp": state.consent_timestamp})
        GLOBAL_SESSION_STORE.save(state)

        return {
            "bot_speech_hi": q2_speech,
            "is_completed": False,
            "state": state.to_dict(),
            "next_question_id": "COM_DEMOGRAPHICS",
            "questions_asked_count": state.questions_asked_count,
            "max_budget": state.max_budget
        }

    def _determine_branch(self, state: SehatSessionState, utterance: str):
        cleaned = utterance.lower()
        
        # 1. First check explicit negation (NOT diabetic - Marathi, Hindi, English)
        negations = [
            "नहीं है", "नहीं", "नही", "ना", "डायबिटीज नहीं", "शुगर नहीं", "nahi hai", "nahin", "nahi",
            "no diabetes", "no", "not diabetic", "normal", "नॉर्मल", "kuch nahi", "कुछ नहीं", "नेगेटिव", "negative",
            "i don't have diabetes", "i dont have diabetes", "i do not have diabetes", "no sugar", "non diabetic",
            "never had diabetes", "healthy", "don't have sugar", "dont have sugar", "no sugar problem",
            "डायबिटीस नाही", "शुगर नाही", "नाहीये", "नाही मला नाही", "काही त्रास नाही", "नॉर्मल"
        ]
        if any(w in cleaned for w in negations):
            state.branch = "not_diabetic"
            state.diabetes_history = None
            state.diabetes_type = None
            return

        # 2. Check unsure / symptoms / suspecting (Marathi, Hindi, English)
        unsure_words = [
            "पता नहीं", "शायद", "मालूम नहीं", "unsure", "not sure", "laksahn", "लक्षण", "doubt", "हो सकता",
            "symptoms", "doubtful", "borderline", "चेक कराना है", "check karana hai", "maybe", "i think so",
            "i don't know", "i dont know", "suspecting", "possible", "not tested",
            "माहित नाही", "शंका आहे", "तपासायचे आहे", "लक्षणे आहेत", "त्रास होतो", "चेक करायचे आहे"
        ]
        if any(w in cleaned for w in unsure_words):
            state.branch = "unsure"
            state.diabetes_history = None
            state.diabetes_type = None
            return

        # 3. Known diabetic positive indicators (Marathi, Hindi, English)
        positive_words = [
            "हाँ", "हां", "yes", "diabetic", "diabetes", "डायबिटीज", "डायबिटीस", "शुगर", "sugar", "sugar hai",
            "diabetes hai", "type 2", "type 1", "टाइप 2", "टाइप 1", "yes i have", "haan", "ha", "हो", "होय",
            "पहले से है", "साल से", "महीने से", "i have diabetes", "i have sugar", "diagnosed with diabetes",
            "i am diabetic", "i take insulin", "i take metformin", "suffering from diabetes", "taking medicines",
            "गोळ्या चालू आहेत", "औषध चालू आहे", "इन्सुलिन चालू आहे", "वर्ष झाले"
        ]
        if any(w in cleaned for w in positive_words):
            state.branch = "known_diabetic"
            return

        state.branch = "unsure"

    def _extract_demographics(self, state: SehatSessionState, utterance: str):
        u_lower = utterance.lower()

        # 1. Age extraction (Digits + Marathi/Hindi words + English words)
        age_val = None
        
        # Check direct integer match with optional age/years prefix or suffix
        age_match = re.search(r"\b(?:age|age\s*is|aged|i\s*am|i\'?m|उम्र|आयु|वय|माझे\s*वय|माझं\s*वय)?\s*(\d{1,3})\s*(?:वर्ष|वर्षे|साल|years?|saal|yr|yrs|sal|old)?\b", utterance, re.IGNORECASE)
        if age_match:
            try:
                v = int(age_match.group(1))
                if 1 <= v <= 120:
                    age_val = v
            except ValueError:
                pass

        if not age_val:
            word_num_map = {
                "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाच": 5, "छह": 6, "सहा": 6, "सात": 7, "आठ": 8, "नौ": 9, "नऊ": 9, "दस": 10, "दहा": 10,
                "ग्यारह": 11, "अकरा": 11, "बारह": 12, "बारा": 12, "तेरह": 13, "तेरा": 13, "चौदह": 14, "चौदा": 14, "पंद्रह": 15, "पंधरा": 15, "सोलह": 16, "सोळा": 16, "सत्रह": 17, "सतरा": 17, "अट्ठारह": 18, "अठरा": 18, "उन्नीस": 19, "एकोणीस": 19, "बीस": 20, "वीस": 20,
                "तीस": 30, "चालीस": 40, "चाळीस": 40, "पचास": 50, "पन्नास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "ऐंशी": 80,
                "twenty": 20, "twenty five": 25, "thirty": 30, "thirty five": 35, "forty": 40, "forty five": 45, "fifty": 50, "fifty five": 55, "sixty": 60, "sixty five": 65, "seventy": 70, "eighty": 80
            }
            for w, num in word_num_map.items():
                if re.search(r"(?:^|[^\w\u0900-\u097F])" + re.escape(w) + r"(?:[^\w\u0900-\u097F]|$)", u_lower):
                    age_val = num
                    break

        if age_val:
            state.demographics["age"] = age_val

        # 2. Gender extraction (Devanagari + English + Marathi + Hinglish)
        u_clean_gender = re.sub(r"[।,;:\.\?!()\"'/\-_।॥]", " ", u_lower)
        gender_words_in_utterance = set(u_clean_gender.strip().split())

        male_tokens = [
            "पुरुष", "मेल", "लड़का", "मुलगा", "आदमी", "मर्द", "जेंट्स", "जेंटलमैन", "बॉय", "पुरूष",
            "male", "man", "men", "boy", "guy", "gents", "gentleman", "purush", "aadmi", "admi", "ladka", "mard", "mr"
        ]
        female_tokens = [
            "महिला", "स्त्री", "फीमेल", "फिमेल", "लड़की", "मुलगी", "औरत", "बाई", "लेडी", "गर्ल", "नारी", "सुश्री",
            "female", "woman", "women", "lady", "girl", "aurat", "stri", "mahila", "ladki", "nari", "mrs", "ms", "miss"
        ]

        is_male = any(tok in gender_words_in_utterance or tok in u_clean_gender for tok in male_tokens)
        is_female = any(tok in gender_words_in_utterance or tok in u_clean_gender for tok in female_tokens)

        if is_female and not is_male:
            state.demographics["gender"] = "महिला (Female)"
        elif is_male and not is_female:
            state.demographics["gender"] = "पुरुष (Male)"
        elif is_female and is_male:
            state.demographics["gender"] = "महिला (Female)" if any(w in u_clean_gender for w in ["महिला", "female", "फीमेल", "woman", "स्त्री", "lady"]) else "पुरुष (Male)"

        # 3. Clean Name extraction
        name_already_set = bool(state.demographics.get("name"))
        explicit_name_intro = any(re.search(r"(?:^|[^\w\u0900-\u097F])" + re.escape(w) + r"(?:[^\w\u0900-\u097F]|$)", u_lower) for w in ["मेरा नाम", "माझे नाव", "माझं नाव", "माझ नाव", "my name is", "my name", "naam", "नाव", "नाम", "i am", "i'm", "this is", "name is"])
        
        if not name_already_set or explicit_name_intro:
            name_clean = utterance
            stop_words = [
                "मेरा नाम", "मेरी उम्र", "माझे नाव", "माझं नाव", "माझ नाव", "माझे वय", "माझं वय", "नाव", "नाम", "my name is", "my name", "i am", "name is", "name", "i'm", "this is", "naam",
                "माय नेम इज़", "माय नेम इज", "माय नेम इस", "माय नेम", "माय एज इज़", "माय एज इज", "माय एज इस", "माय एज", "माय", "आई एम", "आय एम",
                "हूँ", "हूं", "हुं", "हु", "हू", "आहे", "साल", "वर्ष", "वर्षे", "का हूँ", "की हूँ", "का हूं", "की हूं",
                "male", "female", "मेल", "फीमेल", "फिमेल", "पुरुष", "महिला", "स्त्री", "मुलगा", "मुलगी", "लड़का", "लड़की", "आदमी", "औरत",
                "years old", "years", "old", "age", "saal", "varsh", "sal", "umra", "umar", "vay", "ayu", "ka", "ki", "ke", "hai", "hain", "hoon", "hun", "aahe",
                "और", "आणि", "तथा", "एवं", "मैं", "मै", "मी", "है", "हैं", "उम्र", "आयु", "वय", "my", "is", "and", "a", "an", "the", "please", "here",
                "लिंग", "ling", "gender", "सेक्स", "sex", "स्त्री", "मर्द", "जेंट्स", "gentleman", "पुरूष",
                "मेरा", "मेरी", "मेरे", "माझा", "माझी", "माझे", "mera", "meri", "mere", "पुढे", "आगे", "एज", "जेंडर", "ईयर्स", "ओल्ड", "एंड", "इस", "इज़", "इज", "जी", "ji", "sahab", "mahila", "purush"
            ]
            
            name_clean = re.sub(r"[,;:\.\?!()\"'/\-_]", " ", name_clean)
            
            for rem in stop_words:
                name_clean = re.sub(r"(?:^|[^\w\u0900-\u097F])" + re.escape(rem) + r"(?:[^\w\u0900-\u097F]|$)", " ", name_clean, flags=re.IGNORECASE)

            invalid_name_particles = {
                "है", "हैं", "हूँ", "हूं", "हुं", "हु", "हू", "आहे", "का", "की", "के", "और", "आणि", "मैं", "मै", "मी", "से", "को",
                "is", "am", "are", "and", "i", "my", "me", "he", "she", "male", "female", "मेल", "फीमेल", "फिमेल",
                "लिंग", "ling", "gender", "पुरुष", "महिला", "स्त्री", "सेक्स", "sex", "उम्र", "age", "वय", "साल", "वर्ष", "वर्षे", "नाम", "नाव", "name",
                "मेरा", "मेरी", "मेरे", "माझा", "माझी", "माझे", "mera", "meri", "mere", "माय", "नेम", "आगे", "एज", "जेंडर", "ईयर्स", "ओल्ड", "एंड",
                "इस", "इज़", "इज", "details", "umra", "umar", "vay", "sal", "saal", "varsh", "mahila", "purush", "hun", "hoon", "hai", "hain", "aahe", "naam", "ji", "ayu", "आयु"
            }
            name_words = [
                w.strip(",.?!:;()\"'") for w in name_clean.strip().split()
                if not w.strip(",.?!:;()\"'").isdigit() and len(w.strip(",.?!:;()\"'")) > 1 and w.strip(",.?!:;()\"'").lower() not in invalid_name_particles
            ]
            
            if name_words:
                state.demographics["name"] = " ".join(name_words[:2])

    def _extract_clinical_slots(self, state: SehatSessionState, utterance: str):
        u_lower = utterance.lower()
        
        # Symptoms (Hindi + English + Transliterated keywords)
        symptom_map = {
            "polydipsia_excessive_thirst": [
                "प्यास", "thirst", "thirsty", "pyas", "gala sukh", "सूख", "water", "पानी", "excessive thirst", "बहुत प्यास",
                "drinking a lot of water", "dry mouth", "extreme thirst"
            ],
            "polyuria_frequent_urination": [
                "पेशाब", "urine", "peshab", "toilet", "mutra", "टॉयलेट", "peeing", "frequent urine", "washroom", "बार-बार पेशाब",
                "frequent urination", "peeing a lot", "wake up to pee", "night urination"
            ],
            "fatigue_weakness": [
                "थकान", "कमजोरी", "fatigue", "tired", "tiredness", "kamjori", "weakness", "सुस्ती", "lethargy", "weak", "कमज़ोरी",
                "exhausted", "low energy", "feeling tired"
            ],
            "blurred_vision": [
                "धुंधला", "धुंधलापन", "blur", "blurred", "blurry", "vision", "nazar", "aankh", "आँख", "eye", "eyes", "नजर", "नज़र",
                "cannot see clearly", "vision problem", "blurred vision"
            ],
            "weight_loss_gain": [
                "वजन", "वेट", "weight", "wajan", "patla", "mota", "weight loss", "weight gain", "loss", "vajan",
                "losing weight", "gaining weight"
            ],
            "delayed_wound_healing": [
                "घाव", "चोट", "wound", "healing", "ulcer", "chot", "ghav", "theek nahi", "नासूर", "injury", "infection", "cuts",
                "slow healing", "delayed healing", "wound not healing", "ulcers"
            ],
            "numbness_tingling": [
                "सुन्न", "झनझनाहट", "numbness", "tingling", "sunn", "jhanjhanahat", "पैरों में जलन", "burning", "pins and needles", "numb", "सुन्नपन",
                "numb feet", "tingling sensation", "burning feet"
            ],
            "hypoglycemia_episodes": [
                "पसीना", "कांप", "घबराहट", "चक्कर", "low sugar", "hypo", "शुगर कम", "sweating", "shaking", "trembling", "dizziness", "giddiness",
                "shivering", "fainting feeling", "shaky"
            ]
        }

        for sym, keywords in symptom_map.items():
            if any(k in u_lower for k in keywords):
                if sym not in state.symptoms_reported:
                    state.symptoms_reported.append(sym)
                    if sym in ["polydipsia_excessive_thirst", "polyuria_frequent_urination", "delayed_wound_healing", "hypoglycemia_episodes"]:
                        if sym not in state.risk_signals:
                            state.risk_signals.append(f"Risk: {sym}")

        # Mark symptom-related topics complete if matched
        if any(k in u_lower for k in ["प्यास", "पेशाब", "thirst", "urine", "water", "peeing"]):
            if "thirst_and_urination" not in state.completed_topics:
                state.completed_topics.append("thirst_and_urination")
            if "classic_symptoms" not in state.completed_topics:
                state.completed_topics.append("classic_symptoms")

        if any(k in u_lower for k in ["थकान", "धुंधला", "कमजोरी", "fatigue", "vision", "tired", "blur", "नजर", "eyes", "eye"]):
            if "vision_and_fatigue" not in state.completed_topics:
                state.completed_topics.append("vision_and_fatigue")
            if "energy_and_vision" not in state.completed_topics:
                state.completed_topics.append("energy_and_vision")

        if any(k in u_lower for k in ["घाव", "सुन्न", "झनझनाहट", "wound", "numb", "tingling", "healing"]):
            if "neuropathy_and_wounds" not in state.completed_topics:
                state.completed_topics.append("neuropathy_and_wounds")
            if "healing_and_numbness" not in state.completed_topics:
                state.completed_topics.append("healing_and_numbness")

        # Blood Sugar Readings
        sugar_match = re.search(r"\b(sugar|शुगर|fasting|pp|sugar\s*level|glucose|blood\s*sugar|hba1c)?\s*([6-9]\d|[1-5]\d{2})\b", u_lower)
        if sugar_match:
            val_str = sugar_match.group(2)
            try:
                val = int(val_str)
                state.blood_sugar_readings.append({"value": val, "raw": sugar_match.group(0)})
                if val > 200:
                    state.risk_signals.append(f"Elevated Blood Sugar: {val} mg/dL")
                if "blood_sugar_readings" not in state.completed_topics:
                    state.completed_topics.append("blood_sugar_readings")
                if "previous_testing" not in state.completed_topics:
                    state.completed_topics.append("previous_testing")
            except ValueError:
                pass

        # Type 1 / Type 2 Diabetes Classification
        if state.current_question_id == "KD_DIABETES_TYPE" or any(w in u_lower for w in ["टाइप 1", "type 1", "type1", "type one", "टाइप 2", "type 2", "type2", "type two"]):
            if any(w in u_lower for w in ["टाइप 1", "type 1", "type1", "टाइप1", "type one"]) or (state.current_question_id == "KD_DIABETES_TYPE" and "1" in u_lower and "2" not in u_lower):
                state.diabetes_type = "टाइप 1 (Type 1)"
                if "diabetes_type" not in state.completed_topics:
                    state.completed_topics.append("diabetes_type")
            elif any(w in u_lower for w in ["टाइप 2", "type 2", "type2", "टाइप2", "type two"]) or (state.current_question_id == "KD_DIABETES_TYPE" and "2" in u_lower and "1" not in u_lower):
                state.diabetes_type = "टाइप 2 (Type 2)"
                if "diabetes_type" not in state.completed_topics:
                    state.completed_topics.append("diabetes_type")
            elif any(w in u_lower for w in ["पता नहीं", "मालूम नहीं", "not sure", "unsure", "याद नहीं", "don't know", "dont know"]):
                state.diabetes_type = "अनिर्धारित (Unspecified)"
                if "diabetes_type" not in state.completed_topics:
                    state.completed_topics.append("diabetes_type")

        # Duration Extraction (Strictly only for known diabetics, never for demographics or non-diabetics)
        if state.branch == "known_diabetic" or state.current_question_id in ["KD_DURATION", "A_DURATION"]:
            is_age_context = bool(re.search(r"\b(उम्र|आयु|age|aged|old|साल\s*का|साल\s*की|साल\s*के|years\s*old)\b", u_lower))
            if state.current_question_id not in ["COM_DEMOGRAPHICS", "Q2_IDENTITY", "COM_CONSENT"] and not is_age_context:
                dur_match = re.search(r"(\d{1,2})\s*(साल|वर्ष|महीने|months?|years?|saal|varsh|mahine|yrs?|yr)", u_lower)
                if dur_match:
                    state.diabetes_history = dur_match.group(0)
                    if "diabetes_duration" not in state.completed_topics:
                        state.completed_topics.append("diabetes_duration")
        else:
            # For non-diabetic or unsure branch, diabetes duration must not be populated
            state.diabetes_history = None

        # Medications (English + Hindi + Brand names)
        if any(w in u_lower for w in ["दवा", "tablet", "tablets", "medicine", "medicines", "metformin", "gliclazide", "glimepiride", "गोली", "meds", "pills", "पिल्स", "टैबलेट", "dawa", "dawai", "oral meds"]):
            if "Oral Anti-Diabetic Medication" not in state.medications:
                state.medications.append("Oral Anti-Diabetic Medication")
            if "medications" not in state.completed_topics:
                state.completed_topics.append("medications")
        if any(w in u_lower for w in ["इंसुलिन", "insulin", "injection", "सुई", "insulin pen"]):
            if "Insulin Therapy" not in state.medications:
                state.medications.append("Insulin Therapy")
            if "medications" not in state.completed_topics:
                state.completed_topics.append("medications")

        # Family History
        is_family_q = (
            state.current_question_id in ["UN_FAMILY_HISTORY", "ND_FAMILY_HISTORY", "KD_FAMILY_HISTORY"]
            or "family" in (state.current_question_id or "").lower()
        )
        has_family_words = any(w in u_lower for w in [
            "माता", "पिता", "family", "father", "mother", "brother", "sister", "parents", "खानदान", "परिवार",
            "पापा", "मम्मी", "भाई", "बहन", "दादी", "दादा", "नानी", "नाना", "genetics", "hereditary",
            "khandan", "parivar", "family history", "मदर", "फादर", "डैडी"
        ])

        if is_family_q or has_family_words:
            is_neg = any(re.search(pat, u_lower) for pat in [
                r"\b(no|nahi|nahin|na|never|none|nobody|negative|not)\b",
                r"(नहीं|नही|नो|ना|कभी\s*नहीं|किसी\s*को\s*नहीं|किसी\s*को\s*भी\s*नहीं|कोई\s*नहीं|नहीं\s*है|नहीं\s*थी|नहीं\s*रही|समस्या\s*नहीं)"
            ])
            is_pos = any(re.search(pat, u_lower) for pat in [
                r"\b(yes|yeah|yup|haan|ha|hai|positive)\b",
                r"(हाँ|हां|है|थी|था|बिल्कुल|माता\s*को|पिता\s*को|मम्मी\s*को|पापा\s*को|भाई\s*को|बहन\s*को|परिवार\s*में\s*है|फैमिली\s*में\s*है)"
            ])

            if is_neg and not is_pos:
                state.family_history = False
                if "family_history" not in state.completed_topics:
                    state.completed_topics.append("family_history")
            elif is_pos and not is_neg:
                state.family_history = True
                if "family_history" not in state.completed_topics:
                    state.completed_topics.append("family_history")
            elif is_neg and is_pos:
                if any(w in u_lower for w in ["पापा को है", "मम्मी को है", "पिता को", "माता को", "भाई को", "बहन को", "फादर को", "मदर को", "family me hai", "father has", "mother has", "yes"]):
                    state.family_history = True
                else:
                    state.family_history = False
                if "family_history" not in state.completed_topics:
                    state.completed_topics.append("family_history")
            elif is_family_q:
                if any(w in u_lower for w in ["हाँ", "हां", "yes", "ha", "haan", "था", "थी", "है"]):
                    state.family_history = True
                else:
                    state.family_history = False
                if "family_history" not in state.completed_topics:
                    state.completed_topics.append("family_history")

        # Pregnancy History (Step 1: check if ever pregnant)
        is_preg_q = (
            state.current_question_id in ["UN_PREGNANCY_HISTORY", "ND_PREGNANCY_HISTORY", "KD_PREGNANCY_HISTORY"]
            or "pregnancy" in (state.current_question_id or "").lower()
        )
        has_preg_words = any(w in u_lower for w in [
            "माँ बनने", "गर्भावस्था", "प्रेगनेंसी", "pregnancy", "डिलीवरी", "pregnant", "प्रेग्नेंट",
            "बच्चा", "बच्चे", "डिलीवर", "डिलिवरी", "माँ हूँ", "मां हूं", "mother of"
        ])

        if is_preg_q or has_preg_words:
            is_preg_neg = any(re.search(pat, u_lower) for pat in [
                r"\b(no|nahi|nahin|na|never|not|none|unmarried|single)\b",
                r"(नहीं|नही|नो|ना|कभी\s*नहीं|नहीं\s*रही|नहीं\s*हुई|शादी\s*नहीं|अविवाहित|कुंवारी|बच्चा\s*नहीं|कोई\s*नहीं|प्रेग्नेंट\s*नहीं|प्रेगनेंसी\s*नहीं|ऐसी\s*कोई\s*बात\s*नहीं|कुछ\s*नहीं)"
            ])
            is_preg_pos = any(re.search(pat, u_lower) for pat in [
                r"\b(yes|yeah|yup|haan|ha|delivery|pregnant|baby|child|children)\b",
                r"(हाँ|हां|थी|था|है|हुई\s*थी|हुई\s*है|बच्चा|बच्चे|बेटी|बेटा|डिलीवरी|प्रेग्नेंट\s*थी|प्रेगनेंसी\s*रही|माँ\s*हूँ|मां\s*हूं|दो\s*बच्चे|एक\s*बच्चा)"
            ])

            if is_preg_neg and not is_preg_pos:
                state.pregnancy_history = False
                state.gestational_history = False
                if "pregnancy_history" not in state.completed_topics:
                    state.completed_topics.append("pregnancy_history")
                if "gestational_diabetes" not in state.completed_topics:
                    state.completed_topics.append("gestational_diabetes")
            elif is_preg_pos and not is_preg_neg:
                state.pregnancy_history = True
                if "pregnancy_history" not in state.completed_topics:
                    state.completed_topics.append("pregnancy_history")
            elif is_preg_neg and is_preg_pos:
                if any(w in u_lower for w in ["नहीं रही", "नहीं हुई", "शादी नहीं हुई", "बच्चा नहीं", "प्रेग्नेंट नहीं", "never pregnant", "not pregnant", "कभी नहीं रही"]):
                    state.pregnancy_history = False
                    state.gestational_history = False
                    if "pregnancy_history" not in state.completed_topics:
                        state.completed_topics.append("pregnancy_history")
                    if "gestational_diabetes" not in state.completed_topics:
                        state.completed_topics.append("gestational_diabetes")
                elif any(w in u_lower for w in ["हाँ", "हां", "yes", "2 साल पहले", "बच्चा है", "डिलीवरी हुई थी", "दो बच्चे"]):
                    state.pregnancy_history = True
                    if "pregnancy_history" not in state.completed_topics:
                        state.completed_topics.append("pregnancy_history")
                else:
                    state.pregnancy_history = False
                    state.gestational_history = False
                    if "pregnancy_history" not in state.completed_topics:
                        state.completed_topics.append("pregnancy_history")
                    if "gestational_diabetes" not in state.completed_topics:
                        state.completed_topics.append("gestational_diabetes")
            elif is_preg_q:
                if any(w in u_lower for w in ["हाँ", "हां", "yes", "ha", "haan", "था", "थी", "है"]):
                    state.pregnancy_history = True
                    if "pregnancy_history" not in state.completed_topics:
                        state.completed_topics.append("pregnancy_history")
                else:
                    state.pregnancy_history = False
                    state.gestational_history = False
                    if "pregnancy_history" not in state.completed_topics:
                        state.completed_topics.append("pregnancy_history")
                    if "gestational_diabetes" not in state.completed_topics:
                        state.completed_topics.append("gestational_diabetes")

        # Gestational Diabetes Follow-up (Step 2: check if blood sugar elevated during pregnancy)
        is_gest_q = (
            state.current_question_id in ["UN_GESTATIONAL", "ND_GESTATIONAL", "KD_GESTATIONAL"]
            or "gestational" in (state.current_question_id or "").lower()
        )
        has_gest_words = any(w in u_lower for w in [
            "गेस्टेशनल", "gestational", "गर्भावस्था के दौरान शुगर", "गर्भावस्था में शुगर",
            "प्रेगनेंसी में शुगर", "प्रेगनेंसी के दौरान शुगर", "pregnancy diabetes", "gdm"
        ])

        if is_gest_q or has_gest_words:
            is_gest_neg = any(re.search(pat, u_lower) for pat in [
                r"\b(no|nahi|nahin|na|never|normal|not|none)\b",
                r"(नहीं|नही|नो|ना|नॉर्मल|कभी\s*नहीं|सामान्य|ठीक\s*था|नहीं\s*थी|नहीं\s*बढ़ी|नहीं\s*बढ़ा|कोई\s*समस्या\s*नहीं)"
            ])
            is_gest_pos = any(re.search(pat, u_lower) for pat in [
                r"\b(yes|yeah|yup|haan|ha|high|elevated|gdm)\b",
                r"(हाँ|हां|थी|था|है|बढ़ी\s*थी|बढ़ी\s*थी|बढ़\s*गई\s*थी|हाई\s*थी|शुगर\s*हुई\s*थी|गेस्टेशनल|डायबिटीज\s*थी)"
            ])

            if is_gest_neg and not is_gest_pos:
                state.gestational_history = False
            elif is_gest_pos and not is_gest_neg:
                state.gestational_history = True
                if "Risk: Gestational Diabetes History" not in state.risk_signals:
                    state.risk_signals.append("Risk: Gestational Diabetes History")
            elif is_gest_neg and is_gest_pos:
                if any(w in u_lower for w in ["नहीं", "नॉर्मल", "normal", "सामान्य"]):
                    state.gestational_history = False
                else:
                    state.gestational_history = True
                    if "Risk: Gestational Diabetes History" not in state.risk_signals:
                        state.risk_signals.append("Risk: Gestational Diabetes History")
            elif is_gest_q:
                if any(w in u_lower for w in ["हाँ", "हां", "yes", "ha", "haan", "था", "थी", "बढ़ी"]):
                    state.gestational_history = True
                    if "Risk: Gestational Diabetes History" not in state.risk_signals:
                        state.risk_signals.append("Risk: Gestational Diabetes History")
                else:
                    state.gestational_history = False

            if "gestational_diabetes" not in state.completed_topics:
                state.completed_topics.append("gestational_diabetes")

    def _finalize_urgency(self, state: SehatSessionState):
        urg_res = self.urgency_engine.evaluate(
            emergency_escalation=state.emergency_escalation,
            risk_signals=state.risk_signals,
            symptoms_reported=state.symptoms_reported,
            blood_sugar_readings=state.blood_sugar_readings,
            family_history=state.family_history,
            gestational_history=state.gestational_history
        )
        state.recommended_urgency = urg_res.get("recommended_urgency", "routine")
        state.urgency_reasons = urg_res.get("urgency_reasons", [])

    def _lookup_question_text(self, qid: str) -> str:
        for branch_list in self.selector.bank.values():
            if isinstance(branch_list, list):
                for q in branch_list:
                    if isinstance(q, dict) and q.get("question_id") == qid:
                        return q.get("question_hi", "")
        if qid == "COM_CONSENT":
            return "नमस्ते, मेरा नाम Diabetes Dost है। मैं Doctor से मिलने से पहले आपकी Diabetes से जुड़ी कुछ Health जानकारी समझने में help करूंगी। क्या आप बातचीत और Audio Recording के लिए सहमत हैं?"
        elif qid == "COM_DEMOGRAPHICS":
            return "धन्यवाद। आपकी सही पहचान और रिकॉर्ड के लिए, कृपया अपना पूरा नाम, उम्र और लिंग बताइए।"
        elif qid == "COM_STATUS":
            return "क्या आपको पहले से डायबिटीज (शुगर की बीमारी) है, या आपको इसके कोई लक्षण महसूस हो रहे हैं, या आपको डायबिटीज नहीं है?"
        return qid


# Global instance
GLOBAL_DIALOGUE_MANAGER = DialogueManager()

