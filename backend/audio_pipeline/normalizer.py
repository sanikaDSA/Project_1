import re
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger("medical_normalizer")

# Devanagari to Arabic digits mapping
DEVANAGARI_DIGITS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Common Marathi / Hindi medical number words to standard representations
HINDI_MARATHI_NUMBER_WORDS = {
    r'\bएक\b': '1',
    r'\bदोन\b': '2',
    r'\bदो\b': '2',
    r'\bतीन\b': '3',
    r'\bचार\b': '4',
    r'\bपाच\b': '5',
    r'\bपांच\b': '5',
    r'\bसहा\b': '6',
    r'\bछह\b': '6',
    r'\bसात\b': '7',
    r'\bआठ\b': '8',
    r'\bनऊ\b': '9',
    r'\bनौ\b': '9',
    r'\bदहा\b': '10',
    r'\bदस\b': '10',
    r'\bपन्नास\b': '50',
    r'\bपचास\b': '50',
    r'\bशंभर\b': '100',
    r'\bसौ\b': '100',
    r'\bदोनशे\b': '200',
    r'\bपाचशे\b': '500',
    r'\bपाँच सौ\b': '500',
    r'\bहजार\b': '1000',
}

# Standard Clinical Abbreviations & Formats
CLINICAL_ABBREVIATIONS: Dict[str, str] = {
    r'\bb\s*\.?\s*p\s*\.?\b': 'BP',
    r'\bblood\s+pressure\b': 'BP',
    r'\bरक्तदाब\b': 'BP (रक्तदाब)',
    r'\bp\s*\.?\s*r\s*\.?\b': 'PR',
    r'\bpulse\s+rate\b': 'Pulse Rate',
    r'\bनाडी\b': 'नाडी (Pulse)',
    r'\bh\s*\.?\s*r\s*\.?\b': 'HR',
    r'\bheart\s+rate\b': 'Heart Rate',
    r'\bsp\s*o\s*2\b': 'SpO2',
    r'\bspo2\b': 'SpO2',
    r'\boxygensaturation\b': 'SpO2',
    r'\br\s*\.?\s*b\s*\.?\s*s\b': 'RBS',
    r'\brandom\s+blood\s+sugar\b': 'RBS',
    r'\bf\s*\.?\s*b\s*\.?\s*s\b': 'FBS',
    r'\bfasting\s+blood\s+sugar\b': 'FBS',
    r'\bhba1c\b': 'HbA1c',
    r'\bhb\s+a1c\b': 'HbA1c',
    r'\be\s*\.?\s*c\s*\.?\s*g\b': 'ECG',
    r'\bu\s*\.?\s*s\s*\.?\s*g\b': 'USG',
    r'\bm\s*\.?\s*r\s*\.?\s*i\b': 'MRI',
    r'\bc\s*\.?\s*t\s+scan\b': 'CT Scan',
    r'\bo\s*\.?\s*d\b': 'OD (Once Daily)',
    r'\bb\s*\.?\s*d\b': 'BD (Twice Daily)',
    r'\bt\s*\.?\s*d\s*\.?\s*s\b': 'TDS (Thrice Daily)',
    r'\bq\s*\.?\s*i\s*\.?\s*d\b': 'QID (4 Times Daily)',
    r'\bs\s*\.?\s*o\s*\.?\s*s\b': 'SOS (As Needed)',
    r'\btab\b': 'Tab',
    r'\btablet\b': 'Tab',
    r'\bगोळी\b': 'गोळी (Tab)',
    r'\bदवा\b': 'दवा (Medicine)',
    r'\bऔषध\b': 'औषध (Medicine)',
    r'\bcap\b': 'Cap',
    r'\bcapsule\b': 'Cap',
    r'\binj\b': 'Inj',
    r'\binjection\b': 'Inj',
    r'\bsyp\b': 'Syp',
    r'\bsyrup\b': 'Syrup',
}

# Common Indian Diabetes & General Clinical Medicine Names Normalization
MEDICINE_NAMES: Dict[str, str] = {
    # Diabetes Medications
    r'\bmetformin\b': 'Metformin',
    r'\bमेटफॉर्मिन\b': 'Metformin',
    r'\bglimepiride\b': 'Glimepiride',
    r'\bग्लिमेपिराइड\b': 'Glimepiride',
    r'\bvildagliptin\b': 'Vildagliptin',
    r'\bविल्डाग्लिप्टिन\b': 'Vildagliptin',
    r'\bsitagliptin\b': 'Sitagliptin',
    r'\bसिताग्लिप्टिन\b': 'Sitagliptin',
    r'\bdapagliflozin\b': 'Dapagliflozin',
    r'\bडपाग्लिफ्लोजिन\b': 'Dapagliflozin',
    r'\bempagliflozin\b': 'Empagliflozin',
    r'\bएम्पाग्लिफ्लोजिन\b': 'Empagliflozin',
    r'\binsulin\b': 'Insulin',
    r'\bइन्सुलिन\b': 'Insulin',
    r'\bइंसुलिन\b': 'Insulin',
    r'\bglibenclamide\b': 'Glibenclamide',
    r'\bpioglitazone\b': 'Pioglitazone',
    
    # Hypertension / Cardiac
    r'\btelmisartan\b': 'Telmisartan',
    r'\bटेल्मिसार्टन\b': 'Telmisartan',
    r'\bamlodipine\b': 'Amlodipine',
    r'\bअम्लोडिपाइन\b': 'Amlodipine',
    r'\batenolol\b': 'Atenolol',
    r'\batorvastatin\b': 'Atorvastatin',
    r'\bअटोर्वास्टाटिन\b': 'Atorvastatin',
    r'\brosuvastatin\b': 'Rosuvastatin',
    r'\becosprin\b': 'Ecosprin',
    r'\bइकोस्प्रिन\b': 'Ecosprin',
    r'\baspirin\b': 'Aspirin',
    
    # Antibiotics / Analgesics / Gastro
    r'\bparacetamol\b': 'Paracetamol',
    r'\bपॅरासिटामॉल\b': 'Paracetamol',
    r'\bपैरासिटामोल\b': 'Paracetamol',
    r'\bdolo\s*650\b': 'Dolo 650',
    r'\bडोलो\s*६५०\b': 'Dolo 650',
    r'\bडोलो\s*650\b': 'Dolo 650',
    r'\bpantoprazole\b': 'Pantoprazole',
    r'\bपँटोप्राझोल\b': 'Pantoprazole',
    r'\bpantocid\b': 'Pantocid',
    r'\brantac\b': 'Rantac',
    r'\bpan\s*40\b': 'Pan 40',
    r'\bpan\s*d\b': 'Pan-D',
    r'\bpan-d\b': 'Pan-D',
    r'\bamoxicillin\b': 'Amoxicillin',
    r'\baugmentin\b': 'Augmentin',
    r'\bazithromycin\b': 'Azithromycin',
    r'\bazithral\b': 'Azithral',
    r'\bcetirizine\b': 'Cetirizine',
    r'\bsetrizine\b': 'Cetirizine',
    r'\bसेट्रिझिन\b': 'Cetirizine',
    r'\bmontair\s*lc\b': 'Montair LC',
}

# Units normalization
UNITS_MAPPING = {
    r'(\d+)\s*(?:m\s*g|एम\s*जी|मिश्रण|मिलीग्राम)\b': r'\1 mg',
    r'(\d+)\s*(?:m\s*l|एम\s*एल|मिली)\b': r'\1 ml',
    r'(\d+)\s*(?:u\s*n\s*i\s*t\s*s|units|युनिट|युनिट्स)\b': r'\1 Units',
    r'(\d+)\s*/\s*(\d+)\s*(?:mm\s*hg|एमएम\s*एचजी)\b': r'\1/\2 mmHg',
}


def normalize_devanagari_digits(text: str) -> str:
    """Convert Devanagari numerals to standard digits (e.g. १ -> 1, ५ -> 5)."""
    for dev_char, arab_char in DEVANAGARI_DIGITS.items():
        text = text.replace(dev_char, arab_char)
    return text


def clean_medical_transcript(
    raw_transcript: str, 
    preserve_raw: bool = True
) -> Dict[str, str]:
    """
    Clean and normalize clinical transcript without altering spoken meaning.
    
    Returns:
      {
        "raw_transcript": str,
        "cleaned_transcript": str
      }
    """
    if not raw_transcript:
        return {"raw_transcript": "", "cleaned_transcript": ""}

    text = raw_transcript.strip()

    # 1. Normalize Devanagari digits
    text = normalize_devanagari_digits(text)

    # 2. Standardize Units (e.g., 500mg -> 500 mg)
    for pattern, replacement in UNITS_MAPPING.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Standardize Clinical Abbreviations (case-insensitive boundary match)
    for pattern, replacement in CLINICAL_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 4. Standardize Medicine Names
    for pattern, replacement in MEDICINE_NAMES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 5. Normalize common whitespace, dashes, and duplicate spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s+([.,?!:;])', r'\1', text)
    text = text.strip()

    return {
        "raw_transcript": raw_transcript,
        "cleaned_transcript": text,
    }


def validate_medical_terms(transcript: str) -> List[str]:
    """Extract recognized medical terms found in the transcript."""
    detected = []
    text_lower = transcript.lower()
    for med_key, med_val in MEDICINE_NAMES.items():
        if re.search(med_key, text_lower):
            detected.append(med_val)
    return sorted(list(set(detected)))


if __name__ == "__main__":
    sample = "डॉक्टर मला शुगर आहे. Fasting blood sugar 180 आणि HbA1c 8.5 आले आहे. Metformin 500 mg आणि Pan D सकाळी घ्या."
    res = clean_medical_transcript(sample)
    print("Raw:    ", res["raw_transcript"])
    print("Cleaned:", res["cleaned_transcript"])
    print("Detected Medical Entities:", validate_medical_terms(res["cleaned_transcript"]))
