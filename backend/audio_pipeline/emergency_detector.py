import re
import time
from typing import Dict, Any, List, Optional, Tuple

class EmergencyDetector:
    """
    Lightweight, auditable hybrid emergency detection module.
    Runs in parallel on every patient utterance to catch acute clinical distress,
    chest/heart pain, fainting, and diabetic emergencies in Hindi, Hinglish, and English.
    """

    EMERGENCY_RULES: List[Dict[str, Any]] = [
        {
            "id": "EMERGENCY_CHEST_CARDIAC",
            "category": "cardiac_emergency",
            "severity": "CRITICAL",
            "patterns": [
                # Hindi & Hinglish Chest/Heart Pain
                r"(?:सीने|छाती|छातीत|दिल|हार्ट|heart|chest)\s*(?:में)?\s*(?:बहुत\s*)?(?:तेज\s*|गंभीर\s*)?(?:दर्द|दबाव|जलन|घबराहट|भारी|भारीपन|pain|pressure|tightness|heaviness)",
                r"(?:heart|chest)\s*(?:pain|attack|pressure|tightness|heaviness|burn|burning)",
                r"(?:severe\s*(?:heart|chest)\s*pain|pain\s*in\s*(?:heart|chest|left\s*arm))",
                r"(?:हार्ट\s*अटैक|heart\s*attack|दिल\s*का\s*दौरा|कार्डियक|cardiac)",
                r"(?:seene\s*me|chhati\s*me|dil\s*me|heart\s*me)\s*(?:bahut\s*)?(?:tez\s*)?(?:dard|pain|heavy|pressure)",
                r"(?:बाईं|बाएं|बायां|left)\s*(?:हाथ|बांह|कंधे|arm|shoulder)\s*(?:में)?\s*(?:दर्द|भारी|pain)"
            ],
            "response_hi": "आपातकालीन चेतावनी: सीने या दिल में तेज दर्द, दबाव या हार्ट अटैक के लक्षण एक अति-गंभीर मेडिकल इमरजेंसी हैं। कृपया तुरंत 112 / एम्बुलेंस बुलाएं या नजदीकी अस्पताल के इमरजेंसी विभाग (ER) में जाएं।"
        },
        {
            "id": "EMERGENCY_FAINTING_NEURO",
            "category": "neurological_fainting_emergency",
            "severity": "CRITICAL",
            "patterns": [
                # Fainting / Unconsciousness / Collapse
                r"\b(?:faint|fainting|fainted|unconscious|unconsciousness|collapsing|collapsed)\b",
                r"(?:बेहोश|बेहोशी|मूर्छित|अचेत|सुध-बुध\s*खो|चक्कर\s*खाकर\s*गिर)",
                r"(?:behosh|behoshi|chakkar\s*aake\s*gir|gir\s*pata|faint\s*ho)",
                r"(?:चक्कर\s*आकर\s*गिर|अचानक\s*गिर\s*पड़ा|आंखों\s*के\s*आगे\s*अंधेरा\s*छाकर\s*गिर)",
                r"(?:दौरा\s*पड़ा|seizure|झटके\s*आ\s*रहे|fit\s*आ\s*गया|fits)",
                r"(?:लकवा|पक्षाघात|चेहरा\s*टेढ़ा|बोली\s*लड़खड़ा|paralysis|stroke)"
            ],
            "response_hi": "आपातकालीन चेतावनी: अचानक बेहोशी, दौरा या गिर पड़ना एक गंभीर आपातकालीन स्थिति है। मरीज को सुरक्षित लिटाएं और बिना देरी किए तुरंत आपातकालीन एम्बुलेंस (112) बुलाएं।"
        },
        {
            "id": "EMERGENCY_BREATHLESSNESS",
            "category": "respiratory_emergency",
            "severity": "CRITICAL",
            "patterns": [
                r"(?:सांस\s*लेने\s*में\s*बहुत\s*तकलीफ|सांस\s*फूल\s*रही\s*है|दम\s*घुट\s*रहा)",
                r"(?:सांस\s*नहीं\s*आ\s*रही|श्वास\s*घेता\s*येत\s*नाही|श्वास\s*त्रास)",
                r"(?:severe\s*breathlessness|cannot\s*breathe|shortness\s*of\s*breath|gasping|suffocation)",
                r"(?:saans\s*nahi\s*aa\s*rahi|saans\s*fool|dam\s*ghut)"
            ],
            "response_hi": "आपातकालीन चेतावनी: सांस लेने में अत्यधिक तकलीफ या दम घुटना एक अति-गंभीर स्थिति है। कृपया तुरंत सीधे बैठें और बिना किसी देरी के नजदीकी अस्पताल के आपातकालीन कक्ष (ER) से संपर्क करें या एम्बुलेंस बुलाएं।"
        },
        {
            "id": "EMERGENCY_SEVERE_HYPOGLYCEMIA",
            "category": "severe_hypoglycemia",
            "severity": "CRITICAL",
            "patterns": [
                r"(?:शुगर|sugar)\s*(?:30|35|40|45|50|55)\s*(?:हो\s*गई|आ\s*गई|है|mg)?",
                r"(?:बहुत\s*)?(?:कांप\s*रहा|कांपने\s*लगा|पसीना\s*छूट\s*रहा|थरथर\s*कांप)\s*(?:और)?\s*(?:बेहोशी|चक्कर|होश\s*खो)",
                r"(?:severe\s*hypoglycemia|sugar\s*drop\s*unconscious|blood\s*sugar\s*(?:30|40|50))"
            ],
            "response_hi": "आपातकालीन चेतावनी: यह खतरनाक रूप से कम ब्लड शुगर (Severe Hypoglycemia) का संकेत है। यदि मरीज होश में है तो तुरंत 3-4 चम्मच चीनी, ग्लूकोज पानी या मीठा जूस दें। यदि मरीज अचेत हो रहा है तो मुंह में कुछ न डालें और तुरंत अस्पताल ले जाएं।"
        },
        {
            "id": "EMERGENCY_DKA_RED_FLAGS",
            "category": "diabetic_ketoacidosis_dka",
            "severity": "CRITICAL",
            "patterns": [
                r"(?:सांस\s*में\s*फलों\s*जैसी\s*महक|fruity\s*breath|कीटोन्स|ketones)",
                r"(?:लगातार\s*उल्टी|severe\s*vomiting|पानी\s*भी\s*नहीं\s*पच\s*रहा|उल्टी\s*रुक\s*नहीं\s*रही)\s*(?:और\s*शुगर\s*बढ़ी)?",
                r"(?:confusion|अत्यधिक\s*भ्रम|सुध-बुध\s*खो)"
            ],
            "response_hi": "आपातकालीन चेतावनी: लगातार अनियंत्रित उल्टी, अत्यधिक भ्रम या सांस में असामान्य गंध डायबिटिक कीटोएसिडोसिस (DKA) जैसी गंभीर जटिलता का संकेत हो सकती है। मरीज को तुरंत नजदीकी अस्पताल के इमरजेंसी वार्ड में भर्ती कराएं।"
        },
        {
            "id": "EMERGENCY_SEVERE_BLEEDING",
            "category": "severe_trauma_bleeding",
            "severity": "CRITICAL",
            "patterns": [
                r"(?:बहुत\s*ज्यादा\s*खून\s*बह\s*रहा|खून\s*रुक\s*नहीं\s*रहा|severe\s*bleeding)",
                r"(?:गंभीर\s*चोट|uncontrolled\s*bleeding)"
            ],
            "response_hi": "आपातकालीन चेतावनी: अत्यधिक रक्तस्राव एक गंभीर स्थिति है। घाव पर साफ कपड़े से सीधा दबाव बनाएं और तुरंत निकटतम ट्रॉमा सेंटर या अस्पताल पहुंचें।"
        },
        {
            "id": "EMERGENCY_ACUTE_DISTRESS_SUICIDAL",
            "category": "acute_distress_mental_health",
            "severity": "CRITICAL",
            "patterns": [
                r"(?:अभी\s*बहुत\s*ज्यादा\s*तकलीफ\s*हो\s*रही\s*है|बर्दाश्त\s*नहीं\s*हो\s*रहा|मरने\s*का\s*मन|जान\s*देने)",
                r"(?:suicide|suicidal|end\s*my\s*life|cannot\s*take\s*this\s*pain)"
            ],
            "response_hi": "आपातकालीन सहायता: यदि आप अत्यधिक गंभीर शारीरिक कष्ट या मानसिक संकट में हैं, तो कृपया तुरंत किसी परिजन को बताएं और आपातकालीन हेल्पलाइन (112 या राष्ट्रीय टेली-मानस हेल्पलाइन 14416) पर संपर्क करें।"
        }
    ]

    @classmethod
    def evaluate_utterance(cls, utterance: str, session_id: str = "") -> Optional[Dict[str, Any]]:
        if not utterance or not utterance.strip():
            return None

        clean_text = utterance.strip().lower()

        for rule in cls.EMERGENCY_RULES:
            for pattern in rule["patterns"]:
                if re.search(pattern, clean_text, re.IGNORECASE):
                    emergency_payload = {
                        "is_emergency": True,
                        "session_id": session_id,
                        "timestamp": time.time(),
                        "rule_id": rule["id"],
                        "category": rule["category"],
                        "severity": rule["severity"],
                        "triggering_utterance": utterance.strip(),
                        "response_hi": rule["response_hi"],
                        "alert_message": f"🚨 EMERGENCY ALERT: [{rule['category']}] Triggered by: '{utterance.strip()}'"
                    }
                    return emergency_payload

        return None
