import re
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger("clinical_flow")

MANDATORY_FIRST_QUESTION = "नमस्ते। आपको अभी सबसे ज्यादा किस बात की परेशानी है या आप आज किस समस्या के बारे में बात करना चाहते हैं?"

# Backward Compatibility Protocol Questions
DIABETES_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "DIA_GREETING": {
        "id": "DIA_GREETING", "category": "greeting",
        "text": MANDATORY_FIRST_QUESTION
    },
    "DIA_Q01": {
        "id": "DIA_Q01", "category": "chief_complaint",
        "text": MANDATORY_FIRST_QUESTION
    }
}

GLOBAL_SESSIONS: Dict[str, 'ClinicalConversationSession'] = {}

def get_or_create_session(session_id: Optional[str] = None) -> 'ClinicalConversationSession':
    """Retrieve an existing session by ID or create a fresh one."""
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    if session_id not in GLOBAL_SESSIONS:
        GLOBAL_SESSIONS[session_id] = ClinicalConversationSession(session_id=session_id)
    return GLOBAL_SESSIONS[session_id]

# =====================================================================
# 1. EMERGENCY & RED-FLAG PROTOCOLS
# =====================================================================
EMERGENCY_TRIGGERS: List[Dict[str, Any]] = [
    {
        "id": "EMERGENCY_CHEST_PAIN",
        "category": "cardiac_emergency",
        "patterns": [r"सीने.*दर्द", r"छाती.*दर्द", r"छातीत.*दुख", r"chest.*pain", r"हार्ट.*अटैक", r"heart.*attack", r"बांह.*दर्द", r"छाती.*भारी"],
        "response": "सावधानी: सीने में तेज दर्द या दबाव एक गंभीर आपातकालीन स्थिति (Heart Emergency) हो सकती है। कृपया तुरंत Poona Hospital के इमरजेंसी विभाग में जाएं या एम्बुलेंस बुलाएं। खुद गाड़ी न चलाएं और तुरंत डॉक्टरी सहायता लें।"
    },
    {
        "id": "EMERGENCY_BREATHING",
        "category": "respiratory_emergency",
        "patterns": [r"सांस.*तकलीफ", r"दम.*घुट", r"सांस.*नहीं आ", r"breath.*difficult", r"सांस.*फूलना.*ज्यादा", r"श्वास.*त्रास"],
        "response": "महत्वपूर्ण चेतावनी: सांस लेने में अत्यधिक तकलीफ या दम घुटना गंभीर स्थिति है। कृपया तुरंत सीधे बैठें, खिड़की या खुली हवा में आएं और बिना देरी किए Poona Hospital के आपातकालीन कक्ष (Emergency Department) में संपर्क करें।"
    },
    {
        "id": "EMERGENCY_SEVERE_HYPO",
        "category": "severe_hypoglycemia",
        "patterns": [r"शुगर.*\b(?:30|40|50)\b", r"sugar.*\b(?:30|40|50)\b", r"शुगर.*कम.*बेहोश", r"गंभीर.*लो.*शुगर"],
        "response": "आपातकालीन चेतावनी: ब्लड शुगर का स्तर बहुत खतरनाक रूप से कम हो गया है। यदि मरीज होश में है तो तुरंत 3-4 चम्मच चीनी, ग्लूकोज पानी या मीठा जूस दें। यदि मरीज अचेत हो तो मुंह में कुछ न डालें और तुरंत Poona Hospital इमरजेंसी में ले जाएं।"
    },
    {
        "id": "EMERGENCY_STROKE_NEURO",
        "category": "stroke_emergency",
        "patterns": [r"बेहोश", r"अचेत", r"unconscious", r"लकवा", r"चेहरा.*टेढ़ा", r"बोली.*लड़खड़ा", r"stroke", r"paralysis", r"पक्षाघात"],
        "response": "अति आवश्यक चेतावनी: अचानक बेहोशी, चेहरे का टेढ़ापन या बोली लड़खड़ाना स्ट्रोक (Stroke) या न्यूरोलॉजिकल इमरजेंसी के लक्षण हो सकते हैं। रोगी को तुरंत Poona Hospital इमरजेंसी विभाग ले जाएं।"
    }
]

# =====================================================================
# 2. SYMPTOM KNOWLEDGE BASE (EXPANDED TO ALL MAJOR CONDITIONS)
# =====================================================================
SYMPTOM_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "fever": {
        "id": "SYM_FEVER", "category": "fever", "name_hi": "बुखार",
        "patterns": [r"बुखार", r"fever", r"ताप", r"कपकपी", r"ठंड लगकर", r"अंगात ताप"],
        "guidance": "बुखार के लिए पर्याप्त आराम करें और दिनभर में 3 से 4 लीटर गुनगुना पानी या तरल पदार्थ पिएं। यदि बुखार 100°F से अधिक है, तो आवश्यकता पड़ने पर डॉक्टर की सलाह से Paracetamol 650 mg ले सकते हैं।",
        "home_remedies": "माथे पर सामान्य पानी की ठंडी पट्टियां रखें, हल्का सुपाच्य भोजन लें और पर्याप्त पानी पिएं।",
        "diet": "गरमा-गरम वेज सूप, नारियल पानी, पतली खिचड़ी और ताजे फलों का जूस लें।",
        "investigation_chain": ["duration", "temperature", "associated_cough_bodyache", "red_flags"]
    },
    "cough": {
        "id": "SYM_COUGH", "category": "cough", "name_hi": "खांसी",
        "patterns": [r"खांसी", r"cough", r"बलगम", r"कफ", r"खोकला", r"खोखला"],
        "guidance": "खांसी के लिए दिन में दो से तीन बार गर्म पानी की भाप (Steam) लें और गुनगुने पानी में नमक डालकर गरारे करें। शहद और अदरक का रस भी गले को राहत देता है। ठंडी व खट्टी चीजों से परहेज करें।",
        "home_remedies": "एक चम्मच शहद में थोड़ा अदरक का रस मिलाकर दिन में दो बार लें और रात को हल्दी वाला गुनगुना दूध पिएं।",
        "diet": "गुनगुना पानी, हर्बल तुलसी-अदरक वाली चाय और गर्म सूप पिएं। आइसक्रीम व ठंडी चीजें न लें।",
        "investigation_chain": ["duration", "cough_type_dry_wet", "fever_breathlessness", "red_flags"]
    },
    "cold": {
        "id": "SYM_COLD", "category": "cold", "name_hi": "जुकाम / सर्दी",
        "patterns": [r"जुकाम", r"सर्दी", r"cold", r"छींक", r"नाक बहना", r"बंद नाक"],
        "guidance": "जुकाम और बंद नाक के लिए गर्म पानी की भाप लें और गुनगुना पानी पिएं। तुलसी, अदरक वाली गर्म चाय से आराम मिलता है।",
        "home_remedies": "अजवाइन या यूकेलिप्टस तेल की भाप लें और गुनगुना पानी पिएं।",
        "diet": "विटामिन सी युक्त आहार जैसे संतरा, नींबू पानी और गर्म दाल का पानी लें।",
        "investigation_chain": ["duration", "headache_fever", "throat_pain"]
    },
    "sore_throat": {
        "id": "SYM_SORE_THROAT", "category": "sore_throat", "name_hi": "गले में दर्द",
        "patterns": [r"गले.*दर्द", r"गले.*खराश", r"throat", r"निगलने.*दर्द", r"गळा.*दुख", r"खवखव"],
        "guidance": "गले में दर्द और खराश के लिए गुनगुने पानी में थोड़ा नमक डालकर दिन में 3 बार गरारे (Gargles) करें।",
        "home_remedies": "गुनगुने पानी में नमक के गरारे और मुलेठी या लौंग चूसने से गले को तुरंत राहत मिलती है।",
        "diet": "मुलायम और गुनगुना भोजन लें, बहुत गर्म या बहुत ठंडा भोजन न लें।",
        "investigation_chain": ["duration", "swallowing_difficulty", "voice_change"]
    },
    "headache": {
        "id": "SYM_HEADACHE", "category": "headache", "name_hi": "सिरदर्द",
        "patterns": [r"सिर.*दर्द", r"headache", r"माइग्रेन", r"migraine", r"माथा.*दुख", r"डोके.*दुख"],
        "guidance": "सिरदर्द के लिए शांत और कम रोशनी वाले कमरे में विश्राम करें, भरपूर पानी पिएं और स्क्रीन (मोबाइल/टीवी) से दूर रहें।",
        "home_remedies": "माथे पर हल्का बाम लगाएं या पुदीने का तेल लगाएं, और गहरी सांस लेने का अभ्यास करें।",
        "diet": "नारियल पानी और हल्का सूप लें; कैफीन, चाय और खाली पेट रहने से बचें।",
        "investigation_chain": ["duration", "severity", "associated_vomiting_vision", "bp_history"]
    },
    "stomach_pain": {
        "id": "SYM_STOMACH_PAIN", "category": "stomach_pain", "name_hi": "पेट दर्द",
        "patterns": [r"पेट.*दर्द", r"stomach.*pain", r"abdominal.*pain", r"पोट.*दुख", r"मरोड़", r"ऐंठन"],
        "guidance": "पेट दर्द के लिए हल्का और सुपाच्य भोजन (जैसे मूंग दाल की खिचड़ी) लें और गर्म पानी का सेवन करें। भारी, तला-भुना और मसालेदार भोजन बंद रखें।",
        "home_remedies": "एक चुटकी हींग को गुनगुने पानी में घोलकर पिएं या अजवाइन-सेंधा नमक चबाएं।",
        "diet": "पतली खिचड़ी, छाछ (तक्र) और ओट्स लें। तली-भुनी चीजें बिलकुल न खाएं।",
        "investigation_chain": ["duration", "location_upper_lower", "vomiting_diarrhea", "food_intake"]
    },
    "dizziness": {
        "id": "SYM_DIZZINESS", "category": "dizziness", "name_hi": "चक्कर आना",
        "patterns": [r"चक्कर", r"dizziness", r"vertigo", r"सिर घूमना", r"घिरनी", r"भोवळ"],
        "guidance": "चक्कर आने पर तुरंत बैठ जाएं या लेट जाएं ताकि गिरने से चोट न लगे। धीरे-धीरे एक गिलास ओआरएस या नींबू पानी पिएं।",
        "home_remedies": "अचानक खड़े न हों, धीरे-धीरे करवट बदलकर उठें और ग्लूकोज या नमक-चीनी का पानी लें।",
        "diet": "ताजे फलों का रस, नारियल पानी और नियमित अंतराल पर थोड़ा-थोड़ा भोजन लें।",
        "investigation_chain": ["duration", "sugar_check", "bp_check", "weakness"]
    },
    "weakness": {
        "id": "SYM_WEAKNESS", "category": "weakness", "name_hi": "कमजोरी व थकान",
        "patterns": [r"कमजोरी", r"weakness", r"थकान", r"fatigue", r"सुस्ती", r"अशक्तपणा", r"त्राण नाही"],
        "guidance": "कमजोरी के लिए पर्याप्त 8 घंटे की नींद लें, पोषक तत्वों से भरपूर आहार लें और दिनभर में हाइड्रेटेड रहें।",
        "home_remedies": "दूध में मुनक्का या खजूर उबालकर लें और ताजे फलों का सेवन करें।",
        "diet": "अंकुरित अनाज, हरी सब्जियां, दालें, सूखे मेवे और मौसमी फल लें।",
        "investigation_chain": ["duration", "diet_intake", "sugar_bp_history", "fever_history"]
    },
    "nausea_vomiting": {
        "id": "SYM_VOMITING", "category": "vomiting", "name_hi": "उल्टी व मितली",
        "patterns": [r"उल्टी", r"vomiting", r"मितली", r"nausea", r"जी मचलाना", r"उलटी", r"मळमळ"],
        "guidance": "उल्टी होने पर एक साथ ज्यादा पानी न पिएं; थोड़ी-थोड़ी देर में चम्मच से ओआरएस या नारियल पानी लें।",
        "home_remedies": "अदरक का छोटा टुकड़ा या नींबू चाटने से मितली में तुरंत आराम मिलता है।",
        "diet": "सादा उबला आलू, पतली खिचड़ी, सेब का रस लें। दूध और गरिष्ठ भोजन न लें।",
        "investigation_chain": ["frequency", "blood_in_vomit", "stomach_pain", "hydration_status"]
    },
    "diarrhea": {
        "id": "SYM_DIARRHEA", "category": "diarrhea", "name_hi": "दस्त / पेट खराब",
        "patterns": [r"दस्त", r"diarrhea", r"loose.*motion", r"पेचिश", r"पातळ संडास", r"जुलाब"],
        "guidance": "दस्त में शरीर में पानी की कमी (Dehydration) न होने दें। हर बार दस्त के बाद एक गिलास ORS का घोल जरूर पिएं।",
        "home_remedies": "ताजा दही-चावल, केला और जीरा पानी लेने से आंतों को राहत मिलती है।",
        "diet": "खिचड़ी, दही, उबला सेब, केला और छाछ लें। मसालेदार भोजन न लें।",
        "investigation_chain": ["duration", "blood_in_stool", "fever", "hydration_check"]
    },
    "excessive_thirst": {
        "id": "SYM_THIRST", "category": "diabetes_risk", "name_hi": "अत्यधिक प्यास",
        "patterns": [r"प्यास.*ज्यादा", r"ज्यादा.*प्यास", r"thirst", r"खूप तहान", r"तहान.*लाग", r"प्यास.*लगती"],
        "guidance": "बार-बार बहुत ज्यादा प्यास लगना हाई ब्लड शुगर या डिहाइड्रेशन का संकेत हो सकता है।",
        "home_remedies": "पर्याप्त सादा पानी पिएं और मीठे शरबत या कोल्ड ड्रिंक से बचें।",
        "diet": "खीरा, ककड़ी, नारियल पानी लें। मीठे पेय बंद रखें।",
        "investigation_chain": ["duration", "urination_frequency", "sugar_test_history"]
    },
    "frequent_urination": {
        "id": "SYM_URINATION", "category": "diabetes_risk", "name_hi": "बार-बार पेशाब आना",
        "patterns": [r"बार-बार.*पेशाब", r"पेशाब.*ज्यादा", r"frequent.*urination", r"रात्री.*लघवी", r"लघवी.*वारंवार", r"पेशाब.*आता"],
        "guidance": "रात में या दिन में बार-बार पेशाब आना बढ़े हुए ब्लड शुगर (डायबिटीज) अथवा यूरिनरी इन्फेक्शन का लक्षण हो सकता है।",
        "home_remedies": "शाम के समय चाय, कॉफी का सेवन कम करें और नियमित पानी पिएं।",
        "diet": "तरल पदार्थ दिन में अधिक लें, सोने से ठीक पहले अत्यधिक पानी न पिएं।",
        "investigation_chain": ["duration", "burning_urination", "thirst_association", "sugar_test_history"]
    },
    "bp_hypertension": {
        "id": "SYM_BP", "category": "bp_hypertension", "name_hi": "ब्लड प्रेशर (High BP)",
        "patterns": [r"ब्लड प्रेशर", r"blood pressure", r"high bp", r"रक्तदाब", r"बीपी.*वाढ", r"बीपी", r"bp.*high"],
        "guidance": "हाई ब्लड प्रेशर नियंत्रण के लिए भोजन में नमक की मात्रा कम करें, रोजाना 30 मिनट वॉक करें और डॉक्टर की दी हुई बीपी की गोली समय पर लें।",
        "home_remedies": "लहसुन की एक कली सुबह गुनगुने पानी के साथ लें, तनाव कम करें और प्राणायाम करें।",
        "diet": "पोटेशियम युक्त आहार जैसे केला, नारियल पानी, हरी सब्जियां लें; अचार, पापड़ और ज्यादा नमक बंद करें।",
        "investigation_chain": ["duration", "headache_chest_heaviness", "medication_history", "bp_reading"]
    },
    "acidity_gerd": {
        "id": "SYM_ACIDITY", "category": "acidity_gerd", "name_hi": "एसिडिटी व सीने में जलन",
        "patterns": [r"एसिडिटी", r"acidity", r"gerd", r"जलन.*सीने", r"खट्टी डकार", r"पित्त", r"आंबट ढेकर", r"जळजळ"],
        "guidance": "एसिडिटी के लिए खाली पेट न रहें, भोजन के तुरंत बाद न लेटें और चाय, कॉफी व तले-मसालेदार खाने से परहेज रखें।",
        "home_remedies": "सौंफ और मिश्री का पानी पिएं या ठंडा फीका दूध घूंट-घूंट लें।",
        "diet": "ओट्स, तरबूज, पपीता, छाछ और नारियल पानी लें। तली-भुनी चीजें बिलकुल न लें।",
        "investigation_chain": ["duration", "after_meals", "night_reflux", "throat_sourness"]
    },
    "urinary_uti": {
        "id": "SYM_UTI", "category": "urinary_uti", "name_hi": "मूत्र संक्रमण (UTI) व पेशाब में जलन",
        "patterns": [r"पेशाब.*जलन", r"uti", r"urinary", r"लघवी.*जळजळ", r"लघवीत त्रास", r"पेशाब.*रुक"],
        "guidance": "पेशाब में जलन के लिए दिनभर में 3 से 4 लीटर पानी पिएं, नारियल पानी और नींबू पानी लें ताकि बैक्टीरिया बाहर निकल सकें।",
        "home_remedies": "एक गिलास पानी में एक चुटकी मीठा सोडा या नींबू का रस मिलाकर पिएं।",
        "diet": "क्रैनबेरी जूस, जौ का पानी और ताजे फलों का रस अधिक लें।",
        "investigation_chain": ["duration", "fever_chills", "back_pain", "urine_color"]
    },
    "thyroid": {
        "id": "SYM_THYROID", "category": "thyroid", "name_hi": "थायरॉइड (Thyroid / TSH)",
        "patterns": [r"थायरॉइड", r"thyroid", r"tsh", r"थायरॉईड", r"गले में सूजन"],
        "guidance": "थायरॉइड के लिए अपनी थायरॉक्सिन गोली रोजाना सुबह बिल्कुल खाली पेट लें और हर 3 महीने में TSH टेस्ट कराएं।",
        "home_remedies": "नियमित योग और गर्दन के हल्के व्यायाम करें और पर्याप्त नींद लें।",
        "diet": "आयोडीन युक्त संतुलित भोजन, सूखे मेवे और हरी सब्जियां लें; फूलगोभी व पत्तागोभी कच्ची न खाएं।",
        "investigation_chain": ["duration", "weight_change", "fatigue_hairfall", "medication_adherence"]
    },
    "asthma_allergy": {
        "id": "SYM_ASTHMA", "category": "asthma_allergy", "name_hi": "अस्थमा व धूल एलर्जी",
        "patterns": [r"अस्थमा", r"asthma", r"दम", r"allergy", r"एलर्जी", r"धूळ", r"घबराहट.*सांस"],
        "guidance": "अस्थमा और एलर्जी से बचाव के लिए धूल, धुएं और ठंडी हवा से बचें। बाहर निकलते समय मास्क पहनें और इनहेलर हमेशा पास रखें।",
        "home_remedies": "अदरक-तुलसी की भाप लें और गुनगुना पानी पिएं।",
        "diet": "गर्म सूप, हल्दी दूध और ताजा भोजन लें; ठंडे पेय व आइसक्रीम से पूरी तरह बचें।",
        "investigation_chain": ["duration", "inhaler_usage", "night_worsening", "trigger_dust_cold"]
    },
    "cholesterol": {
        "id": "SYM_CHOLESTEROL", "category": "cholesterol", "name_hi": "हाई कोलेस्ट्रॉल (High Cholesterol)",
        "patterns": [r"कोलेस्ट्रॉल", r"cholesterol", r"लिपिड", r"lipid", r"चरबी"],
        "guidance": "कोलेस्ट्रॉल घटाने के लिए तेल, घी, डालडा और जंक फूड बंद करें, रोजाना 40 मिनट तेज चाल से चलें और फाइबर युक्त आहार लें।",
        "home_remedies": "सुबह खाली पेट लहसुन की 1-2 कलियां और मेथी दाने का पानी लें।",
        "diet": "ओट्स, मेथी, लहसुन, सेब, दालें और हरी पत्तेदार सब्जियां लें।",
        "investigation_chain": ["duration", "diet_pattern", "exercise_routine", "lipid_profile_test"]
    }
}

# =====================================================================
# 3. STRUCTURED MEDICAL STATE
# =====================================================================
class StructuredMedicalState:
    """
    Complete Structured Conversation State representing patient and session context.
    """
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.patient_name: Optional[str] = None
        self.patient_age: Optional[int] = None
        self.patient_gender: Optional[str] = None
        self.primary_complaint: Optional[str] = None
        self.active_symptoms: List[str] = []
        self.secondary_symptoms: List[str] = []
        self.severity: Dict[str, str] = {}
        self.duration: Dict[str, str] = {}
        self.body_locations: Dict[str, str] = {}
        self.known_diabetes: Optional[bool] = None
        self.blood_sugar_context: Dict[str, Any] = {"active": False, "sugar_values": None, "medications": [], "hba1c": None}
        self.medications: List[str] = []
        self.known_information: Dict[str, Any] = {}
        self.missing_information: List[str] = []
        self.tests_discussed: List[str] = []
        self.hospital_recommendation_given: bool = False
        self.conversation_history: List[Dict[str, str]] = []
        self.risk_level: Optional[str] = "LOW"
        self.red_flags: List[str] = []
        self.turn_count: int = 0
        self.is_emergency: bool = False
        self.asked_questions: List[str] = []

        # Compatibility Aliases
        self.current_primary_intent: str = "general"
        self.diabetes_context: Dict[str, Any] = self.blood_sugar_context
        self.body_location: Dict[str, str] = self.body_locations
        self.current_risk_level: str = "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_name": self.patient_name,
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "primary_complaint": self.primary_complaint,
            "active_symptoms": self.active_symptoms,
            "secondary_symptoms": self.secondary_symptoms,
            "severity": self.severity,
            "duration": self.duration,
            "body_locations": self.body_locations,
            "known_diabetes": self.known_diabetes,
            "blood_sugar_context": self.blood_sugar_context,
            "medications": self.medications,
            "known_information": self.known_information,
            "missing_information": self.missing_information,
            "tests_discussed": self.tests_discussed,
            "hospital_recommendation_given": self.hospital_recommendation_given,
            "conversation_history": self.conversation_history,
            "risk_level": self.risk_level
        }


# =====================================================================
# 4. CLINICAL CONVERSATION SESSION (POONA HOSPITAL ASSISTANT)
# =====================================================================
class ClinicalConversationSession:
    """
    Poona Hospital Continuous Conversational Medical Assistant.
    """
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.state = StructuredMedicalState(session_id=self.session_id)
        self.domain = "general"
        self.turn_count = 0
        self.is_emergency = False

    def get_first_greeting(self) -> Dict[str, Any]:
        """
        Mandatory First Opening Spoken Interaction.
        Must speak exactly: 'नमस्ते। आपको अभी सबसे ज्यादा किस बात की परेशानी है या आप आज किस समस्या के बारे में बात करना चाहते हैं?'
        """
        self.turn_count = 1
        self.state.turn_count = 1
        return {
            "session_id": self.session_id,
            "question_id": "MANDATORY_Q01",
            "category": "chief_complaint",
            "domain": "general",
            "doctor_text": MANDATORY_FIRST_QUESTION,
            "turn_count": self.turn_count,
            "language": "hi-IN"
        }

    def check_emergency(self, text: str) -> Optional[Dict[str, Any]]:
        """Instant emergency escalation check."""
        text_lower = text.lower()
        for em in EMERGENCY_TRIGGERS:
            for pat in em["patterns"]:
                if re.search(pat, text_lower):
                    self.is_emergency = True
                    self.state.is_emergency = True
                    self.domain = "emergency"
                    self.state.current_risk_level = "EMERGENCY"
                    self.state.risk_level = "EMERGENCY"
                    self.state.red_flags.append(em["category"])
                    return {
                        "session_id": self.session_id,
                        "question_id": em["id"],
                        "category": em["category"],
                        "domain": "emergency",
                        "doctor_text": em["response"],
                        "turn_count": self.turn_count,
                        "is_emergency": True,
                        "language": "hi-IN"
                    }
        return None

    def extract_demographics(self, user_text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Extracts patient name, age, and gender naturally from utterance."""
        name = None
        age = None
        gender = None
        text_lower = user_text.lower()

        # 1. Extract Name
        name_patterns = [
            r"(?:मेरा\s*नाम|नाम\s*है|नाव\s*आहे|नाव|नाम|my\s*name\s*is|i\s*am)\s*[:=]?\s*([A-Za-z\u0900-\u097F]+)",
            r"^([A-Za-z\u0900-\u097F]+)\s*(?:हूँ|आहे|here)$"
        ]
        for pat in name_patterns:
            m = re.search(pat, user_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if cand.lower() not in ["मुझे", "मेरी", "मेरा", "हाँ", "नहीं", "नाही", "बुखार", "खांसी", "दो", "तीन", "doctor", "डॉक्टर", "साल", "उम्र", "पुरुष", "महिला", "male", "female", "वय", "लिंग", "ling", "gender", "सेक्स"]:
                    name = cand
                    break

        # 2. Extract Age
        age_patterns = [
            r"(?:मेरी\s*उम्र|उम्र|वय|age\s*is|age)\s*[:=]?\s*(\d{1,2})",
            r"(\d{1,2})\s*(?:साल|वर्ष|years|yrs|varsh|वर्षांचा|वर्षांची)",
            r"\b(?:उम्र|वय)\s*(\d{1,2})\b"
        ]
        for pat in age_patterns:
            m = re.search(pat, user_text, re.IGNORECASE)
            if m:
                try:
                    a = int(m.group(1))
                    if 1 <= a <= 120:
                        age = a
                        break
                except ValueError:
                    pass

        # 3. Extract Gender
        if any(w in text_lower for w in ["पुरुष", "male", "लड़का", "आदमी", "man", "boy", "गृहस्थ"]):
            gender = "पुरुष"
        elif any(w in text_lower for w in ["महिला", "female", "स्त्री", "लड़की", "औरत", "woman", "girl", "माताजी", "बहन"]):
            gender = "महिला"

        return name, age, gender

    def extract_name_and_age(self, user_text: str) -> Tuple[Optional[str], Optional[int]]:
        """Legacy compatibility wrapper."""
        n, a, _ = self.extract_demographics(user_text)
        return n, a

    def extract_and_update_state(self, user_text: str) -> None:
        """Extracts medical entities, severity, durations, locations, and updates state."""
        text_lower = user_text.lower()
        self.state.turn_count = self.turn_count

        # A. Extract Name, Age & Gender
        ext_name, ext_age, ext_gender = self.extract_demographics(user_text)
        if ext_name and not self.state.patient_name:
            self.state.patient_name = ext_name
            self.state.known_information["patient_name"] = ext_name
        if ext_age and not self.state.patient_age:
            self.state.patient_age = ext_age
            self.state.known_information["patient_age"] = ext_age
        if ext_gender and not self.state.patient_gender:
            self.state.patient_gender = ext_gender
            self.state.known_information["patient_gender"] = ext_gender

        # B. Extract Severity
        if any(w in text_lower for w in ["बहुत तेज", "असहनीय", "severe", "तीव्र", "खूप जास्त", "खूप तेज", "तेज"]):
            sev = "HIGH"
        elif any(w in text_lower for w in ["हल्का", "mild", "थोड़ा", "हळू"]):
            sev = "MILD"
        else:
            sev = "MODERATE"

        # C. Extract Duration (Digits or Words - excluding age mentions)
        text_no_age = re.sub(r'(?:मेरी\s*उम्र|उम्र|वय|age\s*is|age)\s*[:=]?\s*\d{1,2}\s*(?:साल|वर्ष|years|yrs)?', '', user_text, flags=re.IGNORECASE)
        text_no_age = re.sub(r'\b\d{1,2}\s*(?:साल|वर्ष|years)\s*(?:की\s*उम्र|का|की)?', '', text_no_age, flags=re.IGNORECASE)
        dur_matches = re.findall(r'((?:\d+|एक|दो|तीन|चार|पांच|छह|सात|दोन|तीन|चार|पाच)\s*(?:दिन|महीने|हफ्ते|घंटे|दिवस|महिने|days|weeks|months))', text_no_age)
        current_dur = dur_matches[0] if dur_matches else None
        if current_dur:
            self.state.known_information["duration"] = current_dur

        # D. Extract Blood Sugar
        sugar_matches = [m for m in re.findall(r'\b(1\d\d|2\d\d|3\d\d|[5-9]\d)\b', user_text) if m.isdigit()]
        if any(w in text_lower for w in ["sugar", "शुगर", "साखर", "फास्टिंग", "fasting", "glucose", "पीपी", "pp"]) and sugar_matches:
            sugar_val = sugar_matches[0]
            self.state.blood_sugar_context["active"] = True
            self.state.blood_sugar_context["sugar_values"] = sugar_val
            self.state.known_information["sugar_values"] = sugar_val
            self.state.known_diabetes = True
            self.domain = "diabetes"

        # E. Extract HbA1c
        hba1c_matches = re.findall(r'\b([5-9]\.\d|1[0-4]\.\d)\b', user_text)
        if "hba1c" in text_lower and hba1c_matches:
            self.state.blood_sugar_context["active"] = True
            self.state.blood_sugar_context["hba1c"] = hba1c_matches[0]
            self.state.known_information["hba1c"] = hba1c_matches[0]
            self.state.known_diabetes = True
            self.domain = "diabetes"

        # F. Extract Medicines
        med_map = {
            "metformin": "Metformin", "मेटफॉर्मिन": "Metformin", "मेटफार्मिन": "Metformin",
            "glimepiride": "Glimepiride", "ग्लिमेपिराइड": "Glimepiride",
            "insulin": "Insulin", "इंसुलिन": "Insulin", "इन्सुलिन": "Insulin",
            "telmisartan": "Telmisartan", "टेल्मिसार्टन": "Telmisartan",
            "amlodipine": "Amlodipine", "एम्लोडिपिन": "Amlodipine",
            "dolo": "Dolo", "डोलो": "Dolo",
            "paracetamol": "Paracetamol", "पैरासिटामोल": "Paracetamol",
            "pan-d": "Pan-D", "पैन-डी": "Pan-D"
        }
        for med_key, med_val in med_map.items():
            if med_key in text_lower and med_val not in self.state.medications:
                self.state.medications.append(med_val)
                self.state.known_information["medications"] = self.state.medications

        # G. Extract Known Diabetes Flag
        if any(w in text_lower for w in ["शुगर है", "डायबिटीज है", "diabetes patient", "मधुमेह आहे", "insulin", "इंसुलिन", "मेटफॉर्मिन", "metformin"]):
            self.state.known_diabetes = True
            self.state.blood_sugar_context["active"] = True
            self.domain = "diabetes"

        # H. Extract Symptoms mentioned
        new_symptoms = []
        for sym_key, sym_data in SYMPTOM_KNOWLEDGE_BASE.items():
            for pat in sym_data["patterns"]:
                if re.search(pat, text_lower):
                    if sym_key not in new_symptoms:
                        new_symptoms.append(sym_key)
                    if sym_key not in self.state.active_symptoms:
                        self.state.active_symptoms.append(sym_key)
                    if current_dur:
                        self.state.duration[sym_key] = current_dur
                    self.state.severity[sym_key] = sev
                    break

        if new_symptoms:
            if not self.state.primary_complaint:
                self.state.primary_complaint = ", ".join([SYMPTOM_KNOWLEDGE_BASE[s]["name_hi"] for s in new_symptoms])
                self.state.current_primary_intent = new_symptoms[0]
            for s in self.state.active_symptoms:
                if s != self.state.current_primary_intent and s not in self.state.secondary_symptoms:
                    self.state.secondary_symptoms.append(s)

        # Recalculate multi-symptom risk level
        self.assess_symptom_combinations_and_risk()

    def assess_symptom_combinations_and_risk(self) -> Tuple[str, Optional[str]]:
        """Assesses multi-symptom combinations and calculates risk."""
        sym_set = set(self.state.active_symptoms)
        dur = self.state.known_information.get("duration", "")
        dur_txt = f"{dur} से " if dur else ""

        # 1. HEADACHE + FEVER + VOMITING (High Infection Risk)
        if "headache" in sym_set and "fever" in sym_set and "nausea_vomiting" in sym_set:
            self.state.current_risk_level = "HIGH"
            self.state.risk_level = "HIGH"
            advisory = "सिरदर्द, बुखार और उल्टी का एक साथ होना किसी गंभीर संक्रमण का संकेत हो सकता है। यह सामान्य नहीं है; कृपया तुरंत किसी चिकित्सक से प्रत्यक्ष जांच कराएं। क्या आपको गर्दन में जकड़न भी हो रही है?"
            return "HIGH", advisory

        # 2. FEVER + COUGH (Moderate Risk)
        if "fever" in sym_set and "cough" in sym_set and "cough_type" not in self.state.known_information and "fever_cough_asked" not in self.state.asked_questions:
            self.state.asked_questions.append("fever_cough_asked")
            self.state.current_risk_level = "MODERATE"
            self.state.risk_level = "MODERATE"
            advisory = f"{dur_txt}बुखार और खांसी दोनों हैं। खांसी सूखी है या बलगम के साथ आ रही है?"
            return "MODERATE", advisory

        # 3. FEVER + HEADACHE
        if "headache" in sym_set and "fever" in sym_set and len(sym_set) == 2 and "fever_headache_advised" not in self.state.asked_questions:
            self.state.asked_questions.append("fever_headache_advised")
            self.state.current_risk_level = "MODERATE"
            self.state.risk_level = "MODERATE"
            advisory = "सिरदर्द के साथ बुखार होना मौसमी संक्रमण का संकेत है। पर्याप्त पानी पिएं और आराम करें। क्या इसके साथ उल्टी या आंखों में भारीपन भी है?"
            return "MODERATE", advisory

        return self.state.current_risk_level, None

    def select_next_response(self, user_text: str) -> Dict[str, Any]:
        """
        Full Dynamic Multi-Turn Conversational Reasoning.
        Priority:
        1. Emergency / Red Flags
        2. Extract State (Symptoms, Duration, Name, Age)
        3. Name & Age Collection on Turn 1 -> Turn 2
        4. Diabetes Awareness when risk symptoms detected
        5. Direct User Questions (Diet, Remedies, Med Timing, Tests)
        6. Specialized Known Diabetes Consultation
        7. Clinical Dynamic Symptom Investigation
        8. Pooja Hospital Referral when appropriate
        """
        self.turn_count += 1
        self.state.turn_count = self.turn_count
        self.state.conversation_history.append({"user": user_text})
        text_lower = user_text.lower()

        # 1. Emergency Red-Flag Check
        em_res = self.check_emergency(user_text)
        if em_res:
            return em_res

        # 2. Update Structured State
        self.extract_and_update_state(user_text)

        # Name vocative helper (e.g. 'राहुल जी, ')
        name_prefix = f"{self.state.patient_name} जी, " if self.state.patient_name else ""

        # 3. Turn 1 (Chief Complaint received) -> Mandatory Second Question: Collect Name, Age, and Gender
        if self.turn_count == 2 and "demographics_asked" not in self.state.asked_questions:
            if not (self.state.patient_name and self.state.patient_age):
                self.state.asked_questions.append("demographics_asked")
                resp = "ठीक है, मैं आपकी समस्या समझ रहा हूँ। आपकी बेहतर मदद और सही रिकॉर्ड के लिए, क्या मैं आपका नाम और उम्र (तथा लिंग) जान सकता हूँ?"
                return self._format_response("ASK_NAME_AGE_GENDER", "demographics", resp)

        # 4. Turn 2 (Name, Age, Gender just provided) -> Acknowledge & proceed directly into clinical assessment
        if "demographics_asked" in self.state.asked_questions and "demographics_acknowledged" not in self.state.asked_questions:
            if self.state.patient_name or self.state.patient_age or self.state.patient_gender:
                self.state.asked_questions.append("demographics_acknowledged")
                primary = self.state.current_primary_intent
                dur = self.state.known_information.get("duration", "")
                dur_txt = f"{dur} से " if dur else ""
                ack_name = f"धन्यवाद {self.state.patient_name} जी। " if self.state.patient_name else "धन्यवाद। "
                
                # Check active symptoms and proceed
                if "fever" in self.state.active_symptoms and "cough" in self.state.active_symptoms:
                    resp = f"{ack_name}क्या {dur_txt}बुखार और खांसी के साथ आपको गले में दर्द, उल्टी या सांस लेने में परेशानी भी है?"
                elif primary in SYMPTOM_KNOWLEDGE_BASE:
                    sym_name = SYMPTOM_KNOWLEDGE_BASE[primary]["name_hi"]
                    resp = f"{ack_name}क्या {dur_txt}{sym_name} के साथ आपको कमजोरी, उल्टी या अन्य कोई लक्षण भी महसूस हो रहा है?"
                else:
                    resp = f"{ack_name}बताइए, आपको यह तकलीफ कब से हो रही है?"
                return self._format_response("ACK_DEMOGRAPHICS_CONTINUE", "clinical_progression", resp)

        # 5. Check Diabetes Risk Symptoms & Provide Awareness + Hospital Recommendation
        diabetes_risk_symptoms = ["excessive_thirst", "frequent_urination"]
        has_diabetes_risk = any(s in self.state.active_symptoms for s in diabetes_risk_symptoms) or any(w in text_lower for w in ["प्यास लगती", "पेशाब आता", "तहान", "लघवी", "थकान और कमजोरी", "वजन कम"])
        if has_diabetes_risk and "diabetes_awareness_given" not in self.state.asked_questions and not self.state.known_diabetes:
            self.state.asked_questions.append("diabetes_awareness_given")
            self.state.hospital_recommendation_given = True
            self.state.tests_discussed.append("Blood Sugar Screening")
            resp = f"{name_prefix}बहुत अधिक प्यास लगना और बार-बार पेशाब आना ब्लड शुगर (डायबिटीज) से संबंधित लक्षण भी हो सकते हैं। इसलिए ब्लड शुगर और आवश्यक जांचों की जरूरत हो सकती है। आप Poona Hospital में डॉक्टर से परामर्श और जरूरी टेस्ट के लिए संपर्क कर सकते हैं।"
            return self._format_response("DIABETES_AWARENESS_POONA", "diabetes_awareness", resp)

        # 6. Multi-Symptom Risk Assessment
        risk_level, combo_advisory = self.assess_symptom_combinations_and_risk()
        if combo_advisory and self.turn_count >= 3:
            return self._format_response("COMBO_RISK_ALERT", "combination_assessment", f"{name_prefix}{combo_advisory}")

        # 7. Direct User Questions (Diet, Remedies, Med Timing, Tests)
        
        # A. Home Remedies
        if any(w in text_lower for w in ["घरगुती", "घरेलू", "उपाय", "home remedies", "घरी काय करू", "घर पर क्या करें"]):
            primary = self.state.current_primary_intent
            if primary in SYMPTOM_KNOWLEDGE_BASE:
                sym_data = SYMPTOM_KNOWLEDGE_BASE[primary]
                resp = f"{name_prefix}{sym_data['name_hi']} के लिए घरेलू उपाय: {sym_data['home_remedies']} साथ ही पर्याप्त पानी पिएं और आराम करें।"
                return self._format_response(f"REMEDY_{primary.upper()}", "home_remedy", resp)

        # B. Diet & Nutrition
        if any(w in text_lower for w in ["काय खावे", "काय खाऊ", "डाइट", "diet", "क्या खाएं", "क्या खाना चाहिए", "आहार", "परहेज"]):
            if self.state.known_diabetes or self.state.blood_sugar_context.get("active"):
                resp = f"{name_prefix}डायबिटीज में डाइट: मीठा, चीनी, गुड़, सफेद चावल और आलू से परहेज करें। हरी सब्जियां, मेथी, करेला, दालें और सलाद अधिक लें।"
                return self._format_response("DIET_DIABETES", "diet", resp)
            primary = self.state.current_primary_intent
            if primary in SYMPTOM_KNOWLEDGE_BASE:
                sym_data = SYMPTOM_KNOWLEDGE_BASE[primary]
                resp = f"{name_prefix}{sym_data['name_hi']} में खान-पान: {sym_data['diet']} गरिष्ठ और तले-भुने भोजन से बचें।"
                return self._format_response(f"DIET_{primary.upper()}", "diet", resp)

        # C. Medicine Timing
        if any(w in text_lower for w in ["कधी घ्यावी", "कब खानी चाहिए", "दवा कब", "timing", "जेवणानंतर", "जेवणाआधी", "खाली पेट"]):
            if "Metformin" in self.state.medications or "metformin" in text_lower or "मेटफॉर्मिन" in text_lower:
                resp = f"{name_prefix}Metformin की गोली हमेशा भोजन (खाने) के तुरंत बाद ली जाती है ताकि पेट में गैस या जलन न हो।"
                return self._format_response("MED_TIMING_METFORMIN", "medication", resp)
            elif "Insulin" in self.state.medications or "insulin" in text_lower or "इंसुलिन" in text_lower:
                resp = f"{name_prefix}Insulin का इंजेक्शन मुख्य भोजन से 15 से 20 मिनट पहले लिया जाता है।"
                return self._format_response("MED_TIMING_INSULIN", "medication", resp)

        # D. Lab Test / Blood Test / Consultation Inquiries -> Recommend Hospital
        if any(w in text_lower for w in ["रक्त तपासणी", "ब्लड टेस्ट", "cbc", "खून की जांच", "test कब", "टेस्ट कब", "डॉक्टर को दिखाना", "जांच कहां"]):
            self.state.hospital_recommendation_given = True
            resp = f"{name_prefix}लक्षणों की सही पुष्टि के लिए CBC और ब्लड टेस्ट की सलाह दी जाती है। आप Poona Hospital में डॉक्टर से परामर्श और सभी जरूरी टेस्ट करवा सकते हैं।"
            return self._format_response("POONA_HOSPITAL_TESTS", "investigation", resp)

        # 8. Specialized Diabetes Flow (When Active)
        if self.state.known_diabetes or self.state.blood_sugar_context.get("active"):
            sugar_val = int(self.state.blood_sugar_context.get("sugar_values", 0)) if str(self.state.blood_sugar_context.get("sugar_values", "")).isdigit() else 0

            # Low Sugar Guidance (< 70)
            if (sugar_val > 0 and sugar_val < 70) or any(w in text_lower for w in ["कम", "लो शुगर", "low sugar", "low", "कमी"]):
                if "low_sugar_advised" not in self.state.asked_questions:
                    self.state.asked_questions.append("low_sugar_advised")
                    resp = f"{name_prefix}सावधानी: यह लो ब्लड शुगर है। तुरंत 3 चम्मच चीनी या आधा गिलास मीठा जूस लें और 15 मिनट आराम करें।"
                    return self._format_response("DIA_LOW_SUGAR", "hypoglycemia", resp)

            # High Sugar Guidance (>= 180)
            if (sugar_val >= 180 or any(w in text_lower for w in ["ज्यादा है", "बढ़ी", "high sugar", "240", "245", "250", "300"])) and "high_sugar_advised" not in self.state.asked_questions:
                self.state.asked_questions.append("high_sugar_advised")
                val_txt = f"{sugar_val} mg/dL " if sugar_val > 0 else ""
                resp = f"{name_prefix}आपकी ब्लड शुगर {val_txt}बढ़ी हुई है। मीठा और चावल तुरंत बंद करें। क्या आप अभी Metformin या कोई अन्य दवा ले रहे हैं?"
                return self._format_response("DIA_HIGH_SUGAR", "hyperglycemia", resp)

            # Medication acknowledgment
            if self.state.medications and "med_ack_done" not in self.state.asked_questions:
                self.state.asked_questions.append("med_ack_done")
                med_list = ", ".join(self.state.medications)
                resp = f"ठीक है {name_prefix}आप {med_list} ले रहे हैं। इसे भोजन के तुरंत बाद लें। क्या आपको बार-बार पेशाब या बहुत प्यास लग रही है?"
                return self._format_response("DIA_MED_ACK", "current_medications", resp)

            # Follow-up with Hospital Recommendation
            if "hospital_advised_dia" not in self.state.asked_questions:
                self.state.asked_questions.append("hospital_advised_dia")
                self.state.hospital_recommendation_given = True
                resp = f"{name_prefix}डायबिटीज को नियंत्रित रखने के लिए नियमित दवा लें और हर 3 महीने में HbA1c टेस्ट कराएं। आप Poona Hospital में डॉक्टर से परामर्श और चेकअप के लिए संपर्क कर सकते हैं।"
                return self._format_response("DIA_POONA_RECOMMEND", "diabetes_summary", resp)

        # 9. General Clinical Slot Investigation & Progression
        primary_sym = self.state.current_primary_intent
        if primary_sym in SYMPTOM_KNOWLEDGE_BASE:
            sym_data = SYMPTOM_KNOWLEDGE_BASE[primary_sym]
            dur = self.state.duration.get(primary_sym) or self.state.known_information.get("duration")

            # Duration question if missing
            if "duration" not in self.state.known_information and "duration" not in self.state.asked_questions:
                self.state.asked_questions.append("duration")
                follow_up = f"यह {sym_data['name_hi']} आपको कब से हो रहा है?"
                return self._format_response(f"ASK_DUR_{primary_sym.upper()}", sym_data["category"], f"{name_prefix}{follow_up}")

            # Associated symptoms check
            if "associated_checked" not in self.state.asked_questions:
                self.state.asked_questions.append("associated_checked")
                if primary_sym == "fever":
                    follow_up = "क्या इसके साथ खांसी, गले में दर्द या उल्टी भी है?"
                elif primary_sym == "headache":
                    follow_up = "क्या इसके साथ उल्टी, चक्कर या आंखों में भारीपन भी है?"
                elif primary_sym == "stomach_pain":
                    follow_up = "क्या इसके साथ उल्टी, दस्त या पेट में मरोड़ भी है?"
                else:
                    follow_up = "क्या इसके साथ चक्कर या कमजोरी भी महसूस हो रही है?"
                return self._format_response(f"ASK_ASSOC_{primary_sym.upper()}", sym_data["category"], f"{name_prefix}{follow_up}")

            # Guidance + Hospital recommendation for persistent symptoms
            dur_text = f"{dur} से " if dur else ""
            if not self.state.hospital_recommendation_given and "hospital_rec_done" not in self.state.asked_questions:
                self.state.asked_questions.append("hospital_rec_done")
                self.state.hospital_recommendation_given = True
                full_reply = f"{name_prefix}आपको {dur_text}{sym_data['name_hi']} है। {sym_data['guidance']} यदि तकलीफ 2-3 दिनों में ठीक न हो, तो आप Poona Hospital में डॉक्टर से परामर्श और जरूरी जांच के लिए संपर्क कर सकते हैं।"
            else:
                full_reply = f"{name_prefix}आपको {dur_text}{sym_data['name_hi']} है। {sym_data['guidance']}"

            return self._format_response(sym_data["id"], sym_data["category"], full_reply)

            return self._format_response(sym_data["id"], sym_data["category"], full_reply)

        # Fallback General Response
        resp = f"{name_prefix}कृपया अपने लक्षणों के बारे में विस्तार से बताएं ताकि सही मार्गदर्शन दिया जा सके।"
        return self._format_response("GENERAL_FALLBACK", "general", resp)

    def _format_response(self, q_id: str, cat: str, text: str) -> Dict[str, Any]:
        self.state.conversation_history.append({"doctor": text})
        return {
            "session_id": self.session_id,
            "question_id": q_id,
            "category": cat,
            "domain": self.domain,
            "doctor_text": text,
            "turn_count": self.turn_count,
            "is_emergency": self.is_emergency,
            "language": "hi-IN",
            "state": self.state.to_dict()
        }
