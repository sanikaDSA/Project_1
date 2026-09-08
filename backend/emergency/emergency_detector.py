import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger("sehat_emergency")

class EmergencyDetector:
    """
    Independent Hybrid Emergency Detection Module for Sehat AI.
    Loads externalized, versioned emergency lexicon from data/emergency_phrases.json.
    """
    def __init__(self, phrases_file: Optional[str] = None):
        self.phrases_file = phrases_file or self._find_phrases_file()
        self.lexicon = self._load_lexicon()

    def _find_phrases_file(self) -> str:
        candidates = [
            "data/emergency_phrases.json",
            "../data/emergency_phrases.json",
            "e:/Diabetes/data/emergency_phrases.json",
            "backend/data/emergency_phrases.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "data/emergency_phrases.json"

    def _load_lexicon(self) -> Dict[str, Any]:
        if os.path.exists(self.phrases_file):
            try:
                with open(self.phrases_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded emergency lexicon version {data.get('version', 'unknown')} from {self.phrases_file}")
                    return data
            except Exception as e:
                logger.error(f"Error loading emergency phrases from {self.phrases_file}: {e}")
        return {}

    def scan(self, transcript: str, session_id: str = "") -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Scan a patient transcript for acute emergency signals.
        Returns (is_emergency, emergency_event_dict).
        """
        if not transcript or not transcript.strip():
            return False, None

        cleaned = transcript.lower().strip()
        detected_signals: List[str] = []
        matched_categories: List[str] = []

        categories = self.lexicon.get("categories", {})
        
        # 1. Direct Lexicon Scanning
        for cat_name, cat_info in categories.items():
            for kw in cat_info.get("keywords_devanagari", []):
                if kw in transcript:
                    detected_signals.append(kw)
                    matched_categories.append(cat_name)
            for kw in cat_info.get("keywords_latin", []):
                if kw.lower() in cleaned:
                    detected_signals.append(kw)
                    matched_categories.append(cat_name)

        # 2. Critical RegEx Pattern Matching (covering variants, breathlessness, fainting, severe hypo)
        regex_patterns = [
            (r"(chest\s*pain|seene\s*m[ea]\s*(\w+\s*){0,3}dard|cha+ti\s*m[ea]\s*(\w+\s*){0,3}dard|सीने\s*में\s*(\w+\s*){0,3}दर्द|छाती\s*में\s*(\w+\s*){0,3}दर्द|दिल\s*में\s*(\w+\s*){0,3}दर्द|हार्ट\s*अटैक|दिल\s*का\s*दौरा|दिल\s*दब\s*रहा)", "cardiac"),
            (r"(saans\s*nahi\s*aa\s*rahi|difficulty\s*breathing|दम\s*घुट\s*रहा|सांस\s*फूल\s*रही|सांस\s*लेने\s*में\s*(\w+\s*){0,3}तकलीफ|सांस\s*रुक\s*रही)", "respiratory"),
            (r"(faint(ing|ed)?|behosh(i)?|unconscious|चक्कर\s*खाकर\s*गिर|होश\s*खो|बेहोश)", "fainting_consciousness"),
            (r"(seizure|fits|daura|दौरा|झटके\s*आ\s*रहे|शरीर\s*अकड़\s*गया)", "seizure"),
            (r"(severe\s*vomit(ing)?|उल्टी\s*रुक\s*नहीं\s*रही|खून\s*की\s*उल्टी|अभी\s*बहुत\s*तकलीफ\s*हो\s*रही\s*है|असहनीय\s*दर्द)", "severe_distress_vomiting"),
            (r"(sugar\s*(level\s*)?(is\s*)?([234][0-9]|50)\b|शुगर\s*([234][0-9]|50)\b|शुगर\s*50\s*से\s*कम)", "severe_hypoglycemia_confusion"),
            (r"(suicide|mar\s*jana\s*chahta|आत्महत्या|जान\s*देना|खुद\s*को\s*नुकसान)", "self_harm")
        ]

        for pat, cat_name in regex_patterns:
            match = re.search(pat, transcript, re.IGNORECASE)
            if match:
                matched_str = match.group(0)
                if matched_str not in detected_signals:
                    detected_signals.append(matched_str)
                if cat_name not in matched_categories:
                    matched_categories.append(cat_name)

        if detected_signals:
            unique_signals = list(set(detected_signals))
            primary_cat = matched_categories[0] if matched_categories else "general_acute"
            
            verbal_response = (
                "आपके बताए Symptoms गंभीर हो सकते हैं। कृपया अभी तुरंत Poona Hospital के Emergency Department "
                "में संपर्क करें या Local Emergency Service (112 / 108) को Call करें। "
                "आपकी Safety के लिए इस Screening को अभी रोक दिया जा रहा है।"
            )

            event = {
                "event_type": "EMERGENCY",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "triggering_utterance": transcript,
                "detected_signals": unique_signals,
                "categories": list(set(matched_categories)),
                "confidence": 0.99,
                "recommended_urgency": "urgent",
                "verbal_instruction_hi": verbal_response
            }
            logger.warning(f"EMERGENCY DETECTED in session '{session_id}': {unique_signals}")
            return True, event

        return False, None


# Global instance
GLOBAL_EMERGENCY_DETECTOR = EmergencyDetector()
