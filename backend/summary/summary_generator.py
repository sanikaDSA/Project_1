import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("sehat_summary")

class SummaryGenerator:
    """
    Bilingual Structured Clinical Summary & Q&A Transcript Generator for Sehat AI.
    Never outputs a medical diagnosis; produces pre-screening triage for clinician review.
    Persists dedicated patient clinical summary reports (both JSON and human-readable text).
    """
    def __init__(self, sarvam_api_key: Optional[str] = None, summary_dir: str = "outputs/reports/summaries"):
        self.api_key = sarvam_api_key or os.getenv("SARVAM_API_KEY", "")
        self.summary_dir = summary_dir
        os.makedirs(self.summary_dir, exist_ok=True)

    def generate(self, session_state) -> Dict[str, Any]:
        """Generate both Q&A Transcript with English gloss and Structured Clinical Summary."""
        d = session_state.demographics
        answers = session_state.answers
        risk_signals = session_state.risk_signals
        symptoms = session_state.symptoms_reported
        meds = session_state.medications
        sugars = session_state.blood_sugar_readings
        urgency = session_state.recommended_urgency
        emergency = session_state.emergency_escalation

        # 1. Build Q&A Transcript with English gloss
        qa_transcript = []
        for a in answers:
            q_hi = a.get("question_asked_hi", "")
            ans_hi = a.get("patient_answer_hi", "")
            qa_transcript.append({
                "question_id": a.get("question_id", ""),
                "question_asked_hi": q_hi,
                "patient_answer_hi": ans_hi,
                "timestamp": a.get("timestamp", ""),
                "english_gloss": self._gloss_to_english(q_hi, ans_hi)
            })

        # 2. Build Clinical Narrative (Hindi & English)
        narrative_hi = self._build_narrative_hi(session_state)
        narrative_en = self._build_narrative_en(session_state)

        # 3. Sanitize Patient Name for file naming
        p_name_raw = d.get("name") or "Patient"
        safe_name = re.sub(r"[^\w\-_]", "_", p_name_raw.strip())
        if not safe_name:
            safe_name = "Patient"

        json_rel_path = f"{self.summary_dir}/Summary_{safe_name}_{session_state.session_id}.json"
        txt_rel_path = f"{self.summary_dir}/Summary_{safe_name}_{session_state.session_id}.txt"

        # 4. Structured Clinical Summary Dictionary
        structured_summary = {
            "session_id": session_state.session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient": {
                "name": d.get("name") or "अज्ञात (Not specified)",
                "age": d.get("age"),
                "sex": d.get("gender") or "अनिर्दिष्ट (Not specified)"
            },
            "branch": session_state.branch,
            "consent_status": {
                "consent_given": session_state.consent,
                "audio_recording_consent": session_state.audio_recording_consent,
                "consent_timestamp": session_state.consent_timestamp
            },
            "risk_signals": risk_signals,
            "symptoms_reported": symptoms,
            "diabetes_type": getattr(session_state, "diabetes_type", None) if session_state.branch == "known_diabetic" else None,
            "diabetes_history": session_state.diabetes_history if session_state.branch == "known_diabetic" else None,
            "medications": meds,
            "blood_sugar_readings": sugars,
            "lifestyle_factors": session_state.lifestyle_factors,
            "family_history": session_state.family_history,
            "pregnancy_history": getattr(session_state, "pregnancy_history", None),
            "gestational_history": session_state.gestational_history,
            "emergency_escalation": emergency,
            "emergency_details": session_state.emergency_details,
            "summary_narrative": narrative_en,
            "clinical_summary_hi": narrative_hi,
            "clinical_summary_en": narrative_en,
            "recommended_urgency": urgency,
            "urgency_reasons": session_state.urgency_reasons,
            "qa_transcript": qa_transcript,
            "summary_file_json": json_rel_path,
            "summary_file_txt": txt_rel_path,
            "raw_transcript_ref": f"outputs/reports/sessions/{session_state.session_id}.json"
        }

        # 5. Persist Individual Patient Summary Files to Disk
        os.makedirs(self.summary_dir, exist_ok=True)
        try:
            with open(json_rel_path, "w", encoding="utf-8") as f:
                json.dump(structured_summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving summary JSON to {json_rel_path}: {e}")

        try:
            with open(txt_rel_path, "w", encoding="utf-8") as f:
                f.write("======================================================================\n")
                f.write("  POONA HOSPITAL & RESEARCH CENTRE — CLINICAL TRIAGE REPORT\n")
                f.write("======================================================================\n\n")
                f.write(f"Session ID: {session_state.session_id}\n")
                f.write(f"Generated At: {structured_summary['generated_at']}\n")
                f.write(f"Patient Name: {d.get('name') or 'N/A'}\n")
                f.write(f"Age: {d.get('age') or 'N/A'} | Gender: {d.get('gender') or 'N/A'}\n")
                f.write(f"Triage Urgency: {urgency.upper()}\n\n")
                f.write("----------------------------------------------------------------------\n")
                f.write("ENGLISH CLINICAL SUMMARY FOR PHYSICIAN:\n")
                f.write("----------------------------------------------------------------------\n")
                f.write(narrative_en + "\n\n")
                f.write("----------------------------------------------------------------------\n")
                f.write("HINDI CLINICAL SUMMARY:\n")
                f.write("----------------------------------------------------------------------\n")
                f.write(narrative_hi + "\n\n")
                f.write("----------------------------------------------------------------------\n")
                f.write("COMPLETE CONSULTATION Q&A TRANSCRIPT:\n")
                f.write("----------------------------------------------------------------------\n")
                for idx, qa in enumerate(qa_transcript, 1):
                    f.write(f"Q{idx} [{qa.get('question_id')}]: {qa.get('question_asked_hi')}\n")
                    f.write(f"Patient: {qa.get('patient_answer_hi')}\n")
                    if qa.get('english_gloss'):
                        f.write(f"Gloss: {qa.get('english_gloss')}\n")
                    f.write("\n")
                f.write("======================================================================\n")
                f.write("DISCLAIMER: This is an AI-assisted pre-screening summary for clinician review.\n")
                f.write("======================================================================\n")
        except Exception as e:
            logger.error(f"Error saving summary TXT to {txt_rel_path}: {e}")

        session_state.generated_summary = structured_summary
        return structured_summary

    def _gloss_to_english(self, q_hi: str, ans_hi: str) -> str:
        # Clinical keyword gloss dictionary
        gloss_map = {
            "टाइप 1": "Type 1 Diabetes inquiry",
            "टाइप 2": "Type 2 Diabetes inquiry",
            "प्यास": "polydipsia (excessive thirst)",
            "पेशाब": "polyuria (frequent urination)",
            "थकान": "fatigue/weakness",
            "धुंधला": "blurred vision",
            "वजन": "unexplained weight changes",
            "घाव": "delayed wound healing",
            "सुन्न": "peripheral neuropathy/numbness",
            "दवा": "medication adherence",
            "इंसुलिन": "insulin therapy",
            "परिवार": "family history inquiry",
            "माता-पिता": "parental diabetes history",
            "family": "family history inquiry",
            "गर्भावस्था": "pregnancy history inquiry",
            "pregnancy": "pregnancy history inquiry",
            "गेस्टेशनल": "gestational diabetes inquiry",
            "gestational": "gestational diabetes inquiry",
            "इमरजेंसी": "acute emergency flag"
        }
        detected = [v for k, v in gloss_map.items() if k in q_hi or k in ans_hi]
        return f"Topic: {', '.join(detected) if detected else 'General pre-screening'}; Patient noted: {ans_hi[:60]}"

    def _build_narrative_hi(self, s) -> str:
        d = s.demographics
        dtype = getattr(s, "diabetes_type", None)
        preg = getattr(s, "pregnancy_history", None)
        gest = getattr(s, "gestational_history", None)
        
        if s.branch == "not_diabetic":
            duration_hi = "लागू नहीं (डायबिटीज नहीं है)"
        elif s.branch == "unsure":
            duration_hi = "लागू नहीं (नया मूल्यांकन / लक्षण आधारित)"
        else:
            duration_hi = s.diabetes_history or "अनिर्दिष्ट अवधि"

        lines = [
            f"मरीज का नाम: {d.get('name') or 'उपलब्ध नहीं'}, उम्र: {d.get('age') or '—'}, लिंग: {d.get('gender') or '—'}",
            f"डायबिटीज वर्ग / शाखा: {s.branch.upper()}{f' ({dtype})' if dtype else ''}",
            f"डायबिटीज अवधि / इतिहास: {duration_hi}",
            f"रिपोर्ट किए गए लक्षण: {', '.join(s.symptoms_reported) if s.symptoms_reported else 'कोई विशिष्ट लक्षण नहीं बताया गया'}",
            f"दवाइयाँ / इंसुलिन: {', '.join(s.medications) if s.medications else 'कोई दवा नहीं बताई गई'}",
            f"ब्लड शुगर रीडिंग: {', '.join([str(x.get('value')) + ' mg/dL' for x in s.blood_sugar_readings]) if s.blood_sugar_readings else 'हालिया रिकॉर्ड नहीं'}",
            f"पारिवारिक इतिहास (Family History): {'हाँ (पॉजिटिव)' if s.family_history is True else ('नहीं (नेगेटिव)' if s.family_history is False else 'अनिर्दिष्ट (Unspecified)')}",
            f"गर्भावस्था इतिहास (Pregnancy History): {'हाँ (पॉजिटिव)' if preg is True else ('नहीं (नेगेटिव)' if preg is False else 'लागू नहीं / अनिर्दिष्ट')}{', जेस्टेशनल डायबिटीज: हाँ (पॉजिटिव)' if gest is True else (', जेस्टेशनल डायबिटीज: नहीं (नेगेटिव)' if gest is False else '')}",
            f"पहचाने गए जोखिम संकेत (Risk Signals): {', '.join(s.risk_signals) if s.risk_signals else 'कोई गंभीर जोखिम संकेत नहीं मिला'}",
            f"क्लिनिकल अर्जेंसी (Poona Hospital Triage): {s.recommended_urgency.upper()} ({', '.join(s.urgency_reasons)})",
            "क्लीनिकल राय व मार्गदर्शन: नियमित फास्टिंग/PP शुगर जांच एवं त्रैमासिक HbA1c टेस्ट कराने की सलाह। Poona Hospital विशेषज्ञ डॉक्टर से परामर्श अनुशंसित।",
            "नोट: यह केवल AI प्री-स्क्रीनिंग सारांश है, यह कोई मेडिकल डायग्नोसिस नहीं है। चिकित्सक द्वारा जांच आवश्यक है।"
        ]
        return "\n".join(lines)

    def _build_narrative_en(self, s) -> str:
        d = s.demographics
        dtype = getattr(s, "diabetes_type", None)
        preg = getattr(s, "pregnancy_history", None)
        gest = getattr(s, "gestational_history", None)

        if s.branch == "not_diabetic":
            duration_en = "N/A (Non-Diabetic)"
        elif s.branch == "unsure":
            duration_en = "N/A (New Evaluation / Symptomatic Screen)"
        else:
            duration_en = s.diabetes_history or "Unspecified duration"

        lines = [
            f"Patient: {d.get('name') or 'Unknown'} | Age: {d.get('age') or 'N/A'} | Sex: {d.get('gender') or 'N/A'}",
            f"Screening Branch: {s.branch.upper()}{f' ({dtype})' if dtype else ''}",
            f"Diabetes History / Duration: {duration_en}",
            f"Reported Symptoms: {', '.join(s.symptoms_reported) if s.symptoms_reported else 'None reported'}",
            f"Medications/Insulin: {', '.join(s.medications) if s.medications else 'None reported'}",
            f"Blood Sugar Readings: {', '.join([str(x.get('value')) + ' mg/dL' for x in s.blood_sugar_readings]) if s.blood_sugar_readings else 'None reported'}",
            f"Family History: {'Positive' if s.family_history is True else ('Negative' if s.family_history is False else 'Unspecified')}",
            f"Pregnancy History: {'Positive' if preg is True else ('Negative' if preg is False else 'N/A / Unspecified')}{', Gestational Diabetes: Positive' if gest is True else (', Gestational Diabetes: Negative' if gest is False else '')}",
            f"Risk Signals Identified: {', '.join(s.risk_signals) if s.risk_signals else 'None'}",
            f"Emergency Escalation: {'YES - IMMEDIATE CARE REQUIRED' if s.emergency_escalation else 'NO'}",
            f"Recommended Clinician Urgency: {s.recommended_urgency.upper()} (Rationale: {', '.join(s.urgency_reasons)})",
            "Clinical Impression: Pre-screening completed. Recommend comprehensive metabolic panel (FBS/PPBS, HbA1c) and clinician consultation at Poona Hospital and Research Centre.",
            "Disclaimer: AI-generated pre-screening summary for human physician review. Not a diagnostic decision."
        ]
        return "\n".join(lines)


# Global instance
GLOBAL_SUMMARY_GENERATOR = SummaryGenerator()
