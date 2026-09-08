import os
import io
import wave
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger("sehat_asr")

class ASRService(ABC):
    """Abstract Base Class for replaceable ASR provider adapters."""
    
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language_code: str = "hi-IN") -> Dict[str, Any]:
        """Transcribe audio bytes to Hindi text transcript."""
        pass

    @abstractmethod
    def stream(self, audio_chunk: bytes) -> Dict[str, Any]:
        """Process real-time streaming audio chunk."""
        pass


class SarvamASRProvider(ASRService):
    """Sarvam AI ASR implementation with automatic retry and error resilience."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.endpoint = "https://api.sarvam.ai/speech-to-text"

    def transcribe(self, audio_bytes: bytes, language_code: str = "hi-IN") -> Dict[str, Any]:
        if not audio_bytes or len(audio_bytes) < 100:
            return {"transcript": "", "confidence": 0.0, "status": "empty"}
            
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not configured. Returning empty transcript for browser ASR.")
            return {"transcript": "", "confidence": 0.0, "status": "no_api_key"}

        # Build multipart boundary
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(audio_bytes)
        body.write(b"\r\n")
        
        for k, v in [("language_code", language_code), ("model", "saarika:v2")]:
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            body.write(f"{v}\r\n".encode("utf-8"))
        body.write(f"--{boundary}--\r\n".encode("utf-8"))
        data_bytes = body.getvalue()

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }

        for attempt in range(2):
            try:
                req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    if resp.status == 200:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        transcript = res_json.get("transcript", "").strip()
                        return {
                            "transcript": transcript,
                            "confidence": 0.95 if transcript else 0.0,
                            "status": "success",
                            "raw": res_json
                        }
            except Exception as e:
                logger.warning(f"Sarvam ASR attempt {attempt+1} failed: {e}")

        return {"transcript": "", "confidence": 0.0, "status": "failed"}

    def stream(self, audio_chunk: bytes) -> Dict[str, Any]:
        return self.transcribe(audio_chunk)


class MockLocalASRProvider(ASRService):
    """Local fallback / test ASR provider."""
    
    def transcribe(self, audio_bytes: bytes, language_code: str = "hi-IN") -> Dict[str, Any]:
        return {"transcript": "नमस्ते, मुझे कोई गंभीर समस्या नहीं है", "confidence": 0.99, "status": "mock"}

    def stream(self, audio_chunk: bytes) -> Dict[str, Any]:
        return self.transcribe(audio_chunk)


def get_asr_service() -> ASRService:
    """Factory to get active ASR provider."""
    key = os.getenv("SARVAM_API_KEY", "")
    if key:
        return SarvamASRProvider(api_key=key)
    return SarvamASRProvider()
