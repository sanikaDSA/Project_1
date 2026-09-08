import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("sehat_question_selector")

class QuestionSelector:
    """
    Context-Aware Dynamic Question Selector for Sehat AI.
    Selects next questions dynamically based on clinical priority, patient branch,
    symptom signals, missing information, and question budget (6-12).
    """
    def __init__(self, question_bank_file: Optional[str] = None):
        self.question_bank_file = question_bank_file or self._find_question_bank()
        self.bank = self._load_bank()

    def _find_question_bank(self) -> str:
        candidates = [
            "data/question_bank.json",
            "../data/question_bank.json",
            "e:/Diabetes/data/question_bank.json",
            "backend/data/question_bank.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "data/question_bank.json"

    def _load_bank(self) -> Dict[str, Any]:
        if os.path.exists(self.question_bank_file):
            try:
                with open(self.question_bank_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded question bank version {data.get('version', 'unknown')} from {self.question_bank_file}")
                    return data
            except Exception as e:
                logger.error(f"Error loading question bank from {self.question_bank_file}: {e}")
        return {}

    def select_next_question(
        self,
        current_branch: str,
        asked_question_ids: List[str],
        completed_topics: List[str],
        questions_asked_count: int,
        demographics: Dict[str, Any],
        symptoms_reported: List[str],
        risk_signals: List[str],
        max_budget: int = 12
    ) -> Optional[Dict[str, Any]]:
        """
        Dynamically determine the best next question.
        Enforces:
        1. Mandatory Common Questions (Consent -> Demographics -> Diabetic Status)
        2. Clinical Branching (known_diabetic, unsure, not_diabetic)
        3. Female-only restriction on gestational diabetes
        4. Budget limits (Min 6, Target 6-10, Max 12)
        5. Closing Question before normal termination
        """
        # 1. Check Mandatory Common Flow
        common_questions = self.bank.get("common", [])
        for q in common_questions:
            qid = q.get("question_id")
            if qid not in asked_question_ids:
                return q

        # 2. If budget reached (or sufficient info collected after >= 6 turns and closing needed)
        is_closing_asked = any(q.startswith("CLS_") for q in asked_question_ids)
        if questions_asked_count >= (max_budget - 1) and not is_closing_asked:
            return self._get_closing_question(asked_question_ids)

        # 3. If patient has already completed minimum required screening and closing is ready
        branch_key = current_branch if current_branch in ["known_diabetic", "unsure", "not_diabetic"] else "unsure"
        available_branch_questions = self.bank.get(branch_key, [])

        # Priority Sorting: high -> medium -> low
        priority_map = {"mandatory": 0, "high": 1, "medium": 2, "low": 3}
        sorted_candidates = sorted(
            available_branch_questions,
            key=lambda x: priority_map.get(x.get("priority", "medium"), 2)
        )

        patient_gender = str(demographics.get("gender", "")).lower()
        is_female = any(w in patient_gender for w in ["महिला", "स्त्री", "female", "woman", "f", "girl", "lady", "फीमेल"])
        patient_age = demographics.get("age")

        # Check for contextually relevant missing topics
        for q in sorted_candidates:
            qid = q.get("question_id")
            topic = q.get("topic")
            
            # Prevent duplicate question asking
            if qid in asked_question_ids or topic in completed_topics:
                continue

            # Respect Gender and Age Restriction (Pregnancy & Gestational diabetes only for adult females >= 18)
            if q.get("gender_restriction") == "female":
                if not is_female:
                    continue
                # If patient age is known and under 18 -> never ask pregnancy/gestational questions
                if patient_age is not None and isinstance(patient_age, (int, float)) and patient_age < 18:
                    continue

            # Prerequisite topic check: Gestational diabetes should only follow completed pregnancy history
            prereq = q.get("prerequisite_topic")
            if prereq and prereq not in completed_topics:
                continue

            return q

        # 4. If all branch questions exhausted or sufficient info collected (>= 6 questions asked)
        if not is_closing_asked:
            return self._get_closing_question(asked_question_ids)

        return None

    def _get_closing_question(self, asked_question_ids: List[str]) -> Optional[Dict[str, Any]]:
        closing_list = self.bank.get("closing", [])
        for q in closing_list:
            if q.get("question_id") not in asked_question_ids:
                return q
        return {
            "question_id": "CLS_FINAL_FEEDBACK",
            "topic": "closing_remarks",
            "branch": "closing",
            "question_hi": "क्या आप Doctor को अपनी Health से जुड़ी कोई और Important बात बताना चाहते हैं?",
            "priority": "mandatory",
            "required": True
        }


# Global instance
GLOBAL_QUESTION_SELECTOR = QuestionSelector()
