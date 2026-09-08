import re
import os
import io
import wave
import math
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger("sehat_tts")

def clean_text_for_speech_audio(text: str) -> str:
    """
    Sanitizes Hindi+English code-mixed medical dialogue text for seamless TTS synthesis.
    Converts slashes, brackets, abbreviations, and units so TTS speaks naturally without
    pronouncing literal punctuation marks or slashes.
    """
    if not text:
        return ""
    
    t = text
    # 1. Convert slashes and alternatives to natural spoken words
    t = re.sub(r"Male\s*/\s*Female", "Male या Female", t, flags=re.IGNORECASE)
    t = re.sub(r"पुरुष\s*/\s*महिला", "पुरुष या महिला", t)
    t = re.sub(r"112\s*/\s*108", "112 या 108", t)
    t = re.sub(r"हाँ\s*/\s*नहीं", "हाँ या नहीं", t)
    t = re.sub(r"(\w+)\s*/\s*(\w+)", r"\1 या \2", t)
    
    # 2. Convert common medical phrasing in brackets
    t = re.sub(r"\(Fasting\s*व\s*PP\)", " Fasting और PP ", t, flags=re.IGNORECASE)
    t = re.sub(r"\(GDM\)", " गेस्टेशनल डायबिटीज ", t, flags=re.IGNORECASE)
    
    # 3. Medical units and common symbols
    t = re.sub(r"\bmg/dL\b", " मिलीग्राम ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmg\b", " मिलीग्राम ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bml\b", " मिलीलीटर ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bHbA1c\b", " एचबी ए वन सी ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bCBC\b", " सीबीसी ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bFBS\b", " फास्टिंग ब्लड शुगर ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bPPBS\b", " पीपी ब्लड शुगर ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bBP\b", " ब्लड प्रेशर ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bPoona Hospital\b", " पूना हॉस्पिटल ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bMetformin\b", " मेटफॉर्मिन ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bInsulin\b", " इंसुलिन ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bParacetamol\b", " पैरासिटामोल ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b650\b", " छः सौ पचास ", t)
    t = re.sub(r"°F\b", " डिग्री ", t)
    t = re.sub(r"%", " प्रतिशत ", t)
    
    # 4. Remove brackets and special symbols while preserving letters, digits, and spaces
    t = re.sub(r"[\(\)\[\]\{\}\<\>\"\'`*#~_—–|:;/\\]", " ", t)
    
    # 5. Clean up multiple spaces and whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

class TTSService(ABC):
    """Abstract Base Class for replaceable Male Hindi TTS provider adapters."""
    
    @abstractmethod
    def synthesize(self, text_hi: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Synthesize Hindi text to audio bytes/file."""
        pass

    @abstractmethod
    def stream(self, text_hi: str):
        """Stream synthesized audio chunks."""
        pass

    @abstractmethod
    def stop(self):
        """Stop current TTS playback/stream."""
        pass


class SarvamMaleHindiTTS(TTSService):
    """Sarvam AI Male Hindi TTS (aditya / rahul / amit / ratan - Indian Male Voice) with local fallback."""
    
    def __init__(self, api_key: Optional[str] = None, speaker: str = "aditya"):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        # aditya / rahul / amit / ratan / shubh / dev are valid Sarvam male voices
        self.speaker = os.getenv("TTS_VOICE_ID", speaker or "manan")
        self.endpoint = "https://api.sarvam.ai/text-to-speech"

    def synthesize(self, text_hi: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        if not text_hi or not text_hi.strip():
            return {"audio_bytes": b"", "status": "empty", "audio_url": None}

        clean_hi = clean_text_for_speech_audio(text_hi)

        if self.api_key:
            headers = {
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": [clean_hi],
                "target_language_code": "hi-IN",
                "speaker": os.getenv("TTS_VOICE_ID", "hemant"),
                "pitch": 2.0,      # Energetic: bright & confident male pitch
                "pace": 1.15,      # Energetic: lively, natural speed
                "loudness": 1.8,   # Energetic: strong voice projection
                "speech_sample_rate": 16000,
                "enable_preprocessing": True,
                "model": "bulbul:v2"
            }

            try:
                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    if resp.status == 200:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        audios = res_json.get("audios", [])
                        if audios:
                            import base64
                            audio_data = base64.b64decode(audios[0])
                            if output_path:
                                with open(output_path, "wb") as f:
                                    f.write(audio_data)
                            return {
                                "audio_bytes": audio_data,
                                "status": "success",
                                "provider": "sarvam_male_hindi",
                                "audio_url": output_path
                            }
            except Exception as e:
                logger.warning(f"Sarvam TTS request failed or timed out: {e}")

        # Local fallback using pyttsx3 or SAPI5 if available on Windows
        return self._local_speech_synthesize(text_hi, output_path)

    def _local_speech_synthesize(self, text_hi: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)   # Energetic: lively speech rate
            engine.setProperty('volume', 1.0)  # Energetic: full volume
            
            # Select male voice
            voices = engine.getProperty('voices')
            for v in voices:
                if 'male' in v.name.lower() or 'hindi' in v.name.lower() or 'david' in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break

            temp_path = output_path or "outputs/reports/temp_sehat_tts.wav"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            engine.save_to_file(text_hi, temp_path)
            engine.runAndWait()

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                with open(temp_path, "rb") as f:
                    data = f.read()
                return {"audio_bytes": data, "status": "success_pyttsx3", "provider": "pyttsx3_male", "audio_url": temp_path}
        except Exception as e:
            logger.debug(f"pyttsx3 fallback not available: {e}")

        # Clean silent WAV file fallback so client Web Speech API cleanly takes over without buzzing/beeping
        silent_data = self._generate_clean_silent_wav(duration_sec=0.2)
        if output_path:
            with open(output_path, "wb") as f:
                f.write(silent_data)
        return {"audio_bytes": silent_data, "status": "fallback_client_speech", "provider": "web_speech_client", "audio_url": output_path}

    def _generate_clean_silent_wav(self, duration_sec: float = 0.2, sample_rate: int = 16000) -> bytes:
        num_samples = int(duration_sec * sample_rate)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b'\x00\x00' * num_samples)
        return buf.getvalue()

    def stream(self, text_hi: str):
        res = self.synthesize(text_hi)
        yield res.get("audio_bytes", b"")

    def stop(self):
        pass


def get_tts_service() -> TTSService:
    return SarvamMaleHindiTTS()
