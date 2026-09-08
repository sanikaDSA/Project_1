import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("sehat_urgency")

class UrgencyEngine:
    """
    Explicit, auditable Clinical Urgency Rule Engine for clinician triage guidance.
    Never relies solely on an LLM.
    """
    def __init__(self, rules_file: Optional[str] = None):
        self.rules_file = rules_file or self._find_rules_file()
        self.rules = self._load_rules()

    def _find_rules_file(self) -> str:
        candidates = [
            "data/urgency_rules.json",
            "../data/urgency_rules.json",
            "e:/Diabetes/data/urgency_rules.json",
            "backend/data/urgency_rules.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "data/urgency_rules.json"

    def _load_rules(self) -> Dict[str, Any]:
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded urgency rules version {data.get('version', 'unknown')} from {self.rules_file}")
                    return data
            except Exception as e:
                logger.error(f"Error loading urgency rules from {self.rules_file}: {e}")
        return {}

    def evaluate(
        self,
        emergency_escalation: bool,
        risk_signals: List[str],
        symptoms_reported: List[str],
        blood_sugar_readings: List[Dict[str, Any]],
        family_history: Optional[bool] = None,
        gestational_history: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Evaluate patient state against deterministic clinical triage rules.
        Returns:
        {
            "recommended_urgency": "urgent" | "soon" | "routine",
            "urgency_reasons": ["..."]
        }
        """
        reasons: List[str] = []

        # Rule 1: Emergency Triggered
        if emergency_escalation:
            reasons.append("Emergency clinical interruption triggered during voice pre-screening")
            return {
                "recommended_urgency": "urgent",
                "urgency_reasons": reasons
            }

        # Rule 2: Extreme Glucose Readings
        for r in blood_sugar_readings:
            val = r.get("value")
            if val is not None:
                try:
                    num = float(val)
                    if num > 300:
                        reasons.append(f"Severe Hyperglycemia reported ({num} mg/dL > 300)")
                    elif num < 60:
                        reasons.append(f"Severe Hypoglycemia reported ({num} mg/dL < 60)")
                except ValueError:
                    pass

        # Rule 3: Diabetic Foot & Neuropathy Co-occurrence
        symptoms_lower = [s.lower() for s in symptoms_reported]
        has_ulcer = any("wound" in s or "घाव" in s or "ulcer" in s or "छाला" in s for s in symptoms_lower)
        has_numbness = any("numbness" in s or "सुन्न" in s or "झना" in s or "tingling" in s for s in symptoms_lower)
        if has_ulcer and has_numbness:
            reasons.append("Diabetic peripheral neuropathy with non-healing foot wound")

        # Rule 4: Multiple Red-Flag Risk Signals
        if len(risk_signals) >= 3:
            reasons.append(f"Multiple concurrent high-risk signals reported ({len(risk_signals)} signals)")

        if reasons:
            return {
                "recommended_urgency": "urgent",
                "urgency_reasons": reasons
            }

        # Check for 'soon' triage tier
        soon_reasons: List[str] = []
        if len(risk_signals) >= 1:
            soon_reasons.append(f"Reported risk signals requiring clinician review: {', '.join(risk_signals[:3])}")

        for r in blood_sugar_readings:
            val = r.get("value")
            if val is not None:
                try:
                    num = float(val)
                    if 200 <= num <= 300:
                        soon_reasons.append(f"Elevated blood glucose ({num} mg/dL)")
                except ValueError:
                    pass

        if family_history and len(symptoms_reported) >= 1:
            soon_reasons.append("Strong familial diabetic history presenting with emerging symptoms")

        if gestational_history:
            soon_reasons.append("Prior history of gestational diabetes")

        if soon_reasons:
            return {
                "recommended_urgency": "soon",
                "urgency_reasons": soon_reasons
            }

        return {
            "recommended_urgency": "routine",
            "urgency_reasons": ["No acute red-flag risk signals identified; standard clinician consultation recommended."]
        }


# Global instance
GLOBAL_URGENCY_ENGINE = UrgencyEngine()
