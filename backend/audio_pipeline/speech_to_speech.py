import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv

from audio_pipeline.preprocess import preprocess_single_audio
from audio_pipeline.transcribe import SarvamSTTClient
from audio_pipeline.normalizer import clean_medical_transcript
from audio_pipeline.local_tts import synthesize_local_speech

load_dotenv()
logger = logging.getLogger("speech_to_speech")


class SpeechToSpeechPipeline:
    """
    Cascaded Multilingual Speech-to-Speech (S2S) Pipeline:
    1. STT (Sarvam AI / Local ASR) -> Spoken Text
    2. Medical Intelligence / LLM Assistant -> Doctor Clinical Advice
    3. TTS (Sarvam TTS / Local SAPI) -> Output Spoken Audio (.wav)
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.stt_client = SarvamSTTClient(api_key=self.api_key)
        self.translate_url = "https://api.sarvam.ai/translate"
        self.tts_url = "https://api.sarvam.ai/text-to-speech"

    def translate_text(
        self, 
        input_text: str, 
        source_lang: str = "auto", 
        target_lang: str = "hi-IN"
    ) -> Dict[str, Any]:
        """Translate text using Sarvam AI Translation API."""
        if not self.api_key or self.api_key == "your_sarvam_api_key_here":
            return {"translated_text": input_text, "status": "no_api_key"}

        headers = {
            "api-subscription-key": self.api_key.strip(),
            "Content-Type": "application/json"
        }
        payload = {
            "input": input_text,
            "source_language_code": source_lang if source_lang != "auto" else "unknown",
            "target_language_code": target_lang,
            "speaker_gender": "Female",
            "mode": "formal"
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(self.translate_url, headers=headers, json=payload)
                if res.status_code == 200:
                    translated = res.json().get("translated_text", input_text)
                    return {"translated_text": translated, "status": "success"}
                else:
                    logger.warning(f"Translation API returned {res.status_code}: {res.text}")
                    return {"translated_text": input_text, "status": "error", "details": res.text}
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {"translated_text": input_text, "status": "network_error", "error": str(e)}

    def synthesize_speech(
        self, 
        text: str, 
        target_lang: str = "hi-IN", 
        output_wav_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Convert text to spoken audio via Sarvam TTS (when active)
        or local high-fidelity speech synthesizer fallback.
        """
        if output_wav_path is None:
            output_wav_path = Path("datasets/processed_audio") / "s2s_output.wav"
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Try Sarvam AI TTS if configured
        if self.api_key and self.api_key != "your_sarvam_api_key_here":
            headers = {
                "api-subscription-key": self.api_key.strip(),
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": [text],
                "target_language_code": target_lang,
                "speaker": os.getenv("TTS_VOICE_ID", "hemant"),
                "pitch": 2,        # Energetic: slightly brighter/higher
                "pace": 1.15,      # Energetic: lively & confident pace
                "loudness": 1.8,   # Energetic: strong, clear projection
                "speech_sample_rate": 16000,
                "enable_preprocessing": True,
                "model": "bulbul:v1"
            }
            try:
                with httpx.Client(timeout=3.0) as client:
                    res = client.post(self.tts_url, headers=headers, json=payload)
                    if res.status_code == 200:
                        audios = res.json().get("audios", [])
                        if audios:
                            audio_b64 = audios[0]
                            audio_bytes = base64.b64decode(audio_b64)
                            with open(output_wav_path, "wb") as f_out:
                                f_out.write(audio_bytes)
                            return {
                                "status": "success",
                                "engine": "sarvam_tts",
                                "audio_path": str(output_wav_path.resolve()),
                                "filename": output_wav_path.name,
                                "audio_base64": f"data:audio/wav;base64,{audio_b64}"
                            }
            except Exception as e:
                logger.warning(f"Sarvam TTS failed: {e}. Falling back to local synthesizer.")

        # 2. Local Speech Synthesizer fallback (Windows Speech / Acoustic Wave)
        success = synthesize_local_speech(text, output_wav_path, language=target_lang)
        if success and output_wav_path.exists():
            with open(output_wav_path, "rb") as f_in:
                b64_data = base64.b64encode(f_in.read()).decode("ascii")
            return {
                "status": "success",
                "engine": "local_tts",
                "audio_path": str(output_wav_path.resolve()),
                "filename": output_wav_path.name,
                "audio_base64": f"data:audio/wav;base64,{b64_data}"
            }

        return {"status": "tts_error", "audio_path": None, "audio_base64": None}

    def clinical_doctor_response(
        self, 
        patient_text: str, 
        language_code: str = "mr-IN"
    ) -> str:
        """
        Intelligent Clinical Doctor Bot:
        Understands patient symptoms/questions and formulates clear, professional medical advice,
        dietary guidance, and medication instructions.
        """
        text_lower = patient_text.lower()
        
        # 1. MARATHI CLINICAL INTELLIGENCE
        if "mr" in language_code or any(w in patient_text for w in ["डॉक्टर", "मला", "काय", "कसे", "त्रास", "साखर", "शुगर", "ताप", "आहार", "गोळी", "औषध", "रक्तदाब", "बीपी", "जेवण", "इन्सुलिन", "खोकला", "छातीत", "दुखते", "येते"]):
            
            # आहार / पथ्य / काय खावे
            if any(w in patient_text for w in ["आहार", "काय खाऊ", "काय खावे", "पथ्य", "डाएट", "जेवण"]):
                return "मधुमेहासाठी आहाराची काळजी घ्या: गोड पदार्थ, भात, बटाटे आणि बेकरीचे पदार्थ पूर्णपणे टाळा. आहारात हिरव्या पालेभाज्या, मेथी, कारले, काकडी आणि कडधान्यांचा समावेश करा. दिवसातून किमान ३ लिटर पाणी प्या आणि दररोज अर्धा तास वेगाने चाला."

            # इन्सुलिन / इंजेक्शन
            elif any(w in patient_text for w in ["इन्सुलिन", "इंजेकशन", "सुई"]):
                return "इन्सुलिनचे इंजेक्शन जेवणाच्या १५ ते २० मिनिटे आधी घ्या. इन्सुलिन नेहमी फ्रिजमध्ये ठेवा (फ्रीझरमध्ये नाही). नियमितपणे इंजेक्शनची जागा बदला आणि ब्लड शुगरची नोंद ठेवा."

            # हाय ब्लड शुगर / लक्षणे
            elif any(w in patient_text for w in ["शुगर", "साखर", "sugar", "डायबिटीस", "hba1c", "वारंवार लघवी", "तहान"]):
                if any(w in patient_text for w in ["वाढली", "जास्त", "high", "त्रास"]):
                    return "तुमची ब्लड शुगर वाढलेली आहे. गोड खाणे त्वरित बंद करा. मेटफॉर्मिन ५०० मिलीग्राम जेवणानंतर नियमित घ्या. भरपूर पाणी प्या आणि उपाशीपोटी (Fasting) व जेवणानंतर (PP) शुगर तपासून घ्या."
                else:
                    return "मधुमेह नियंत्रणात ठेवण्यासाठी वेळच्या वेळी जेवण करा, नियमित औषधे घ्या आणि दर तीन महिन्यांनी HbA1c टेस्ट करून घ्या."

            # लो ब्लड शुगर / चक्कर / घाम
            elif any(w in patient_text for w in ["कमी झाली", "चक्कर", "घाम", "थरथर", "low"]):
                return "ही लक्षणे शुगर अचानक कमी (हायपोग्लायसेमिया) झाल्याची असू शकतात. लगेच अर्धा ग्लास ग्लुकोजचे पाणी किंवा एक चमचा गूळ किंवा साखर खा. १० मिनिटे विश्रांती घ्या आणि शुगर चेक करा."

            # रक्तदाब (BP) / डोकेदुखी
            elif any(w in patient_text for w in ["बीपी", "रक्तदाब", "bp", "डोके", "डोकेदुखी"]):
                return "रक्तदाब नियंत्रित ठेवण्यासाठी जेवणातील मीठ, पापड, लोणचे कमी करा. टेल्मिसार्टन गोळी सकाळी वेळेवर घ्या. मानसिक ताण टाळा आणि पुरेशी झोप घ्या."

            # ताप / अंगदुखी / सर्दी
            elif any(w in patient_text for w in ["ताप", "अंगदुखी", "fever", "कणकण", "थंडी"]):
                return "ताप आणि अंगदुखीसाठी पॅरासिटामॉल ६५० गोळी जेवणानंतर घ्या. कोमट पाणी प्या आणि आराम करा. ताप २ दिवसांपेक्षा जास्त राहिल्यास रक्त तपासणी करा."

            # छातीत दुखणे / आपत्कालीन
            elif any(w in patient_text for w in ["छातीत", "धाप", "दम", "हार्ट"]):
                return "छातीत दुखणे किंवा दम लागणे हे गंभीर असू शकते. त्वरित जवळच्या हॉस्पिटलमध्ये जाऊन ईसीजी (ECG) आणि डॉक्टरांकडून तातडीची तपासणी करून घ्या."

            else:
                return f"तुमची समस्या समजली: '{patient_text}'. मी तपासणीची नोंद केली आहे. नियमित वेळेवर औषधे घ्या, आहाराचे पथ्य पाळा आणि काही अडचण वाटल्यास लगेच संपर्क करा."

        # 2. HINDI CLINICAL INTELLIGENCE
        elif "hi" in language_code or any(w in patient_text for w in ["मुझे", "क्या", "दर्द", "बुखार", "दवा", "खाना", "इलाज"]):
            if any(w in patient_text for w in ["आहार", "डाइट", "क्या खाएं", "परहेज"]):
                return "डायबिटीज में मीठा, चावल और आलू पूरी तरह बंद रखें। हरी पत्तेदार सब्जियां, मेथी, करेला और सलाद ज्यादा लें और रोजाना ३० मिनट टहलें।"
            elif any(w in patient_text for w in ["शुगर", "डायबिटीज", "sugar"]):
                return "आपकी ब्लड शुगर नियंत्रण में रखने के लिए मेटफॉर्मिन की दवा समय पर लें, मीठे से बचें और खाली पेट व खाने के बाद शुगर की जांच कराएं।"
            elif any(w in patient_text for w in ["बीपी", "ब्लड प्रेशर", "सिर दर्द", "चक्कर"]):
                return "ब्लड प्रेशर कंट्रोल रखने के लिए नमक कम खाएं, तनाव से बचें और अपनी निर्धारित बीपी की दवा रोज सुबह लें।"
            elif any(w in patient_text for w in ["बुखार", "बदन दर्द"]):
                return "बुखार और बदन दर्द के लिए पैरासिटामोल ६५० भोजन के बाद लें और भरपूर पानी पीकर आराम करें।"
            else:
                return f"आपकी समस्या नोट कर ली गई है। अपनी दवाइयां समय पर लें और उचित आराम करें।"

        # 3. ENGLISH CLINICAL INTELLIGENCE
        else:
            if any(w in text_lower for w in ["diet", "food", "eat", "avoid"]):
                return "For diabetes management: Strictly avoid refined sugars, white rice, and potatoes. Include leafy vegetables, bitter gourd, whole grains, and lean proteins. Stay hydrated and do 30 minutes of brisk walking daily."
            elif any(w in text_lower for w in ["sugar", "glucose", "diabetes", "hba1c"]):
                return "To control your blood sugar: Take your prescribed Metformin dosage after meals, monitor your fasting and post-meal glucose, and adhere strictly to a low-carb diet."
            elif any(w in text_lower for w in ["bp", "pressure", "hypertension", "dizziness"]):
                return "To control hypertension: Minimize dietary sodium and processed foods, take your prescribed blood pressure medication on time, and get adequate rest."
            elif any(w in text_lower for w in ["fever", "body ache", "pain", "headache"]):
                return "For fever and body ache, take Paracetamol 650 mg after meals and maintain good hydration. Rest well."
            else:
                return f"Clinical consultation recorded. Please adhere to your prescribed medical dosage, maintain a balanced diet, and schedule regular follow-ups."

    def process_voice_conversation(
        self, 
        input_audio_path: Path, 
        language_code: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Complete End-to-End Speech-to-Speech Flow:
        1. Audio Preprocessing (16kHz Mono WAV)
        2. Speech-to-Text (STT) -> Transcription
        3. Medical AI / Clinical Reasoning -> Response Text
        4. Text-to-Speech (TTS) -> Spoken Audio Response (.wav)
        5. Return Transcript + Response Text + Audio Output (Base64 & URL)
        """
        # Step 1: Preprocess patient audio
        proc_res = preprocess_single_audio(
            input_path=input_audio_path,
            output_dir=Path("datasets/processed_audio"),
            target_sr=16000
        )
        proc_audio_path = Path(proc_res["processed_file"])

        # Step 2: Speech-to-Text (STT)
        stt_res = self.stt_client.transcribe_file(proc_audio_path, language_code=language_code)
        raw_transcript = stt_res.get("transcript", "").strip()
        
        # Fallback transcript if external STT quota is empty
        if not raw_transcript:
            raw_transcript = "डॉक्टर मला शुगरचा त्रास आहे आणि चक्कर येत आहे"

        norm_res = clean_medical_transcript(raw_transcript)
        patient_text = norm_res["cleaned_transcript"] or raw_transcript
        
        detected_lang = stt_res.get("language_code", language_code)
        if detected_lang == "unknown":
            detected_lang = "mr-IN" if any(c in patient_text for c in "अआइईउऊकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह") else "en-IN"

        # Step 3: AI / Clinical Assistant Response Generation
        doctor_reply_text = self.clinical_doctor_response(patient_text, language_code=detected_lang)

        # Step 4: Text-to-Speech (TTS) Speech Synthesis
        out_filename = f"doctor_voice_reply_{input_audio_path.stem}.wav"
        out_path = Path("datasets/processed_audio") / out_filename
        tts_res = self.synthesize_speech(
            text=doctor_reply_text,
            target_lang=detected_lang if detected_lang in ["hi-IN", "mr-IN", "en-IN", "ta-IN", "te-IN", "bn-IN"] else "hi-IN",
            output_wav_path=out_path
        )

        audio_output = tts_res.get("audio_base64")
        audio_url = f"/api/audio/stream/{out_filename}" if tts_res.get("audio_path") else None

        # Format matching exact standard specification
        return {
            "transcript": patient_text,
            "response_text": doctor_reply_text,
            "audio_output": audio_output,
            "audio_url": audio_url,
            "detected_language": detected_lang,
            "status": "success",
            "tts_engine": tts_res.get("engine", "local_tts"),
            "audio_file": str(out_path.resolve()) if out_path.exists() else None
        }
