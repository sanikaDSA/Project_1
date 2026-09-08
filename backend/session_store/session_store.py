import os
import json
import time
import threading
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("sehat_session_store")

class SehatSessionState:
    """Dataclass-like state container for a single Sehat AI session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        
        # Demographics & Consent
        self.demographics = {"name": "", "age": None, "gender": ""}
        self.consent = False
        self.consent_timestamp: Optional[str] = None
        self.audio_recording_consent = False
        self.consent_rejected = False

        # Conversation State
        self.questions_asked_count = 0
        self.max_budget = 12
        self.branch: str = "common" # common, known_diabetic, unsure, not_diabetic
        self.current_question_id: str = "COM_CONSENT"
        self.asked_question_ids: List[str] = []
        self.completed_topics: List[str] = []
        self.answers: List[Dict[str, Any]] = []
        self.question_retry_count: Dict[str, int] = {}
        self.demographic_retry_count: int = 0

        # Clinical Findings
        self.risk_signals: List[str] = []
        self.symptoms_reported: List[str] = []
        self.diabetes_history: Dict[str, Any] = {}
        self.medications: List[str] = []
        self.blood_sugar_readings: List[Dict[str, Any]] = []
        self.lifestyle_factors: List[str] = []
        self.family_history: Optional[bool] = None
        self.pregnancy_history: Optional[bool] = None
        self.gestational_history: Optional[bool] = None
        self.diabetes_type: Optional[str] = None

        # Emergency & Urgency
        self.emergency_escalation: bool = False
        self.emergency_details: Optional[Dict[str, Any]] = None
        self.recommended_urgency: str = "routine"
        self.urgency_reasons: List[str] = []

        # Completion & Summary
        self.is_completed: bool = False
        self.generated_summary: Optional[Dict[str, Any]] = None
        self.audio_ref: Optional[str] = None

    def record_answer(self, question_id: str, question_hi: str, patient_answer_hi: str, confidence: float = 0.95):
        self.answers.append({
            "question_id": question_id,
            "question_asked_hi": question_hi,
            "patient_answer_hi": patient_answer_hi,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": confidence
        })
        if question_id not in self.asked_question_ids:
            self.asked_question_ids.append(question_id)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "patient": self.demographics,
            "consent": self.consent,
            "consent_timestamp": self.consent_timestamp,
            "audio_recording_consent": self.audio_recording_consent,
            "consent_rejected": self.consent_rejected,
            "questions_asked_count": self.questions_asked_count,
            "max_budget": self.max_budget,
            "branch": self.branch,
            "current_question_id": self.current_question_id,
            "asked_question_ids": self.asked_question_ids,
            "completed_topics": self.completed_topics,
            "answers": self.answers,
            "risk_signals": self.risk_signals,
            "symptoms_reported": self.symptoms_reported,
            "diabetes_type": self.diabetes_type,
            "diabetes_history": self.diabetes_history,
            "medications": self.medications,
            "blood_sugar_readings": self.blood_sugar_readings,
            "lifestyle_factors": self.lifestyle_factors,
            "family_history": self.family_history,
            "pregnancy_history": self.pregnancy_history,
            "gestational_history": self.gestational_history,
            "emergency_escalation": self.emergency_escalation,
            "emergency_details": self.emergency_details,
            "recommended_urgency": self.recommended_urgency,
            "urgency_reasons": self.urgency_reasons,
            "is_completed": self.is_completed,
            "generated_summary": self.generated_summary,
            "audio_ref": self.audio_ref
        }


class SessionStore:
    """Thread-safe storage manager for all Sehat AI sessions."""
    def __init__(self, storage_dir: str = "outputs/reports/sessions"):
        self.storage_dir = storage_dir
        self.sessions: Dict[str, SehatSessionState] = {}
        self.lock = threading.Lock()
        os.makedirs(self.storage_dir, exist_ok=True)

    def get_or_create(self, session_id: str) -> SehatSessionState:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SehatSessionState(session_id)
            return self.sessions[session_id]

    def get(self, session_id: str) -> Optional[SehatSessionState]:
        with self.lock:
            return self.sessions.get(session_id)

    def save(self, session: SehatSessionState):
        with self.lock:
            self.sessions[session.session_id] = session
            try:
                filepath = os.path.join(self.storage_dir, f"{session.session_id}.json")
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to persist session {session.session_id}: {e}")

    def list_all_sessions(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [s.to_dict() for s in self.sessions.values()]


# Global instance
GLOBAL_SESSION_STORE = SessionStore()
