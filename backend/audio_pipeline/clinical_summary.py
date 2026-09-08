import time
from typing import Dict, Any, List, Optional
from audio_pipeline.cura_clinical_flow import PatientClinicalState

class ClinicalSummaryGenerator:
    """
    Generates structured, auditable clinical pre-screening summaries for doctors:
    - Side-by-side Hindi (चिकित्सीय सारांश) and English clinical summary
    - Clean Q&A transcript with timestamps & question IDs
    - Triage risk score (LOW, MODERATE, HIGH, EMERGENCY_ESCALATED)
    - Zero diagnosis guarantee: solely organizes patient-provided data for clinician review.
    """

    SYMPTOM_NAME_MAP = {
        "excessive_thirst": {"hi": "अत्यधिक प्यास (Polydipsia)", "en": "Excessive Thirst (Polydipsia)"},
        "frequent_urination": {"hi": "बार-बार पेशाब (Polyuria)", "en": "Frequent Urination (Polyuria)"},
        "unexplained_weight_loss": {"hi": "अस्पष्ट वजन में कमी", "en": "Unexplained Weight Loss"},
        "fatigue_weakness": {"hi": "अत्यधिक थकान व कमजोरी", "en": "Fatigue & Generalized Weakness"},
        "blurred_vision": {"hi": "धुंधलापन (Blurred Vision)", "en": "Blurred Vision / Visual Disturbances"},
        "slow_healing_wounds": {"hi": "घाव देर से भरना", "en": "Delayed Wound Healing"},
        "numbness_tingling": {"hi": "हाथ-पैरों में सुन्नपन/झनझनाहट (Neuropathy)", "en": "Peripheral Tingling / Numbness (Neuropathy risk)"}
    }

    @classmethod
    def generate_summary(
        cls, 
        state: PatientClinicalState, 
        default_view: str = "side_by_side"
    ) -> Dict[str, Any]:
        """
        Builds structured clinical summary payload in Hindi & English.
        """
        # Demographics
        demo = {
            "name": state.name or "अज्ञात / Not Specified",
            "age": state.age if state.age else "उल्लेखित नहीं / Unspecified",
            "gender": state.gender or "उल्लेखित नहीं / Unspecified",
            "consent_recorded": "हाँ (Consent Granted)" if state.consent_given else "अस्वीकृत / Unconfirmed"
        }

        # Symptoms formatted
        symptoms_hi = [cls.SYMPTOM_NAME_MAP.get(s, {}).get("hi", s) for s in state.symptoms_present]
        symptoms_en = [cls.SYMPTOM_NAME_MAP.get(s, {}).get("en", s) for s in state.symptoms_present]

        # Medications & Adherence
        meds_str = ", ".join(state.medications) if state.medications else "कोई दवा नहीं बताई / None mentioned"
        adherence_str = state.medication_adherence or "लागू नहीं / Not Applicable"

        # Risk Classification
        risk_tier = state.risk_tier
        if state.is_emergency:
            risk_tier = "EMERGENCY_ESCALATED"

        # Construct Hindi Clinical Summary
        summary_hi_lines = [
            f"**मरीज का नाम:** {demo['name']} | **उम्र:** {demo['age']} | **लिंग:** {demo['gender']}",
            f"**सहमति (Consent):** {demo['consent_recorded']}",
            f"**डायबिटीज स्थिति (Branch):** {state.diabetic_status_raw or 'निर्धारित नहीं'}",
            f"**रोग की अवधि (Duration):** {state.diagnosis_duration or 'लागू नहीं / ज्ञात नहीं'}",
            f"**हालिया ब्लड शुगर रीडिंग:** {state.last_sugar_reading or 'उपलब्ध नहीं'} (HbA1c: {state.hba1c or 'उपलब्ध नहीं'})",
            f"**वर्तमान दवाएं व अनुपालन:** {meds_str} | अनुपालन: {adherence_str}",
            f"**सक्रिय लक्षण (Active Symptoms):** {', '.join(symptoms_hi) if symptoms_hi else 'कोई मुख्य लक्षण नहीं बताया'}",
            f"**पारिवारिक इतिहास (Family History):** {'हाँ (पारिवारिक इतिहास मौजूद)' if state.family_history else 'नहीं / ज्ञात नहीं'}",
            f"**शारीरिक गतिविधि:** {state.physical_activity_level or 'सामान्य / अनिर्दिष्ट'}",
            f"**जोखिम स्तर (Triage Risk):** {risk_tier}"
        ]

        if state.is_emergency and state.emergency_details:
            summary_hi_lines.append(f"⚠️ **आपातकालीन अलर्ट (Emergency Triggered):** {state.emergency_details.get('alert_message')}")

        if state.additional_notes:
            summary_hi_lines.append(f"**मरीज द्वारा अतिरिक्त टिप्पणी:** \"{state.additional_notes}\"")

        summary_hi = "\n".join(summary_hi_lines)

        # Construct English Clinical Summary
        branch_en_map = {
            "BRANCH_A": "Branch A: Established / Known Diabetic",
            "BRANCH_B": "Branch B: Unsure / Suspected Diabetic (Symptom Screen)",
            "BRANCH_C": "Branch C: Non-Diabetic (Risk Factor Screening)"
        }
        branch_en = branch_en_map.get(state.branch, "Unresolved")

        summary_en_lines = [
            f"**Patient Name:** {demo['name']} | **Age:** {demo['age']} | **Sex:** {demo['gender']}",
            f"**Consent Recorded:** {'Yes (Audio recording consented)' if state.consent_given else 'Pending / Unconfirmed'}",
            f"**Clinical Branch:** {branch_en}",
            f"**Diagnosis Duration:** {state.diagnosis_duration or 'N/A'}",
            f"**Latest Blood Glucose Reading:** {state.last_sugar_reading or 'Not reported'} (HbA1c: {state.hba1c or 'Not reported'})",
            f"**Current Pharmacotherapy:** {meds_str} | **Adherence:** {adherence_str}",
            f"**Positive Symptoms Screened:** {', '.join(symptoms_en) if symptoms_en else 'None reported'}",
            f"**Family History of Diabetes:** {'Positive (Parent / Sibling)' if state.family_history else 'Negative / Unknown'}",
            f"**Physical Activity Profile:** {state.physical_activity_level or 'Not specified'}",
            f"**Triage Risk Tier:** {risk_tier}"
        ]

        if state.is_emergency and state.emergency_details:
            summary_en_lines.append(f"🚨 **EMERGENCY ESCALATION OCCURRED DURING SESSION:** Triggered by '{state.emergency_details.get('triggering_utterance')}' [{state.emergency_details.get('category')}]")

        if state.additional_notes:
            summary_en_lines.append(f"**Patient's Additional Notes:** \"{state.additional_notes}\"")

        summary_en = "\n".join(summary_en_lines)

        # Build clean Q&A transcript
        clean_transcript = []
        for item in state.qa_transcript:
            clean_transcript.append({
                "turn": item.get("turn"),
                "question_id": item.get("question_id"),
                "cura_question_hi": item.get("question_hi"),
                "patient_answer_hi": item.get("patient_response_hi")
            })

        return {
            "session_id": state.session_id,
            "timestamp": time.time(),
            "demographics": demo,
            "branch": state.branch,
            "risk_tier": risk_tier,
            "is_emergency": state.is_emergency,
            "emergency_details": state.emergency_details,
            "questions_asked_count": state.questions_asked_count,
            "is_completed": state.is_completed,
            "clinical_summary_hi": summary_hi,
            "clinical_summary_en": summary_en,
            "default_view": default_view,
            "qa_transcript": clean_transcript,
            "symptoms_identified": state.symptoms_present,
            "disclaimer": "DISCLAIMER: Diabetes Dost is an automated clinical pre-screening tool and does not formulate medical diagnoses or alter treatments. For physician review only."
        }
