import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
from dotenv import load_dotenv
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio_pipeline.normalizer import clean_medical_transcript

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sarvam_transcribe")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        base_dir = Path(__file__).resolve().parent.parent
        alt_path = base_dir / "config.yaml"
        if alt_path.exists():
            config_path = str(alt_path)
        else:
            return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SarvamSTTClient:
    """
    Client for Sarvam AI Speech-To-Text API.
    Supports Marathi ('mr-IN'), Hindi ('hi-IN'), English ('en-IN'), and code-mixed speech.
    """
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str = "saaras:v3"):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.api_url = api_url or "https://api.sarvam.ai/speech-to-text"
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_sarvam_api_key_here")

    def transcribe_file(
        self, 
        audio_path: Path, 
        language_code: str = "unknown",
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send audio file to Sarvam STT API.
        Preserves multilingual code-mixed speech as spoken.
        """
        if not self.is_configured():
            logger.warning(f"SARVAM_API_KEY is not set or placeholder. Please provide a valid Sarvam AI API Key.")
            return {
                "transcript": "",
                "language_code": language_code,
                "confidence": None,
                "status": "error_missing_api_key",
                "error": "Sarvam API Key not configured. Set SARVAM_API_KEY in .env or environment.",
            }

        headers = {
            "api-subscription-key": self.api_key.strip()
        }

        with open(audio_path, "rb") as f:
            files = {
                "file": (audio_path.name, f, "audio/wav")
            }
            data: Dict[str, Any] = {
                "model": self.model,
            }
            if language_code and language_code != "unknown":
                data["language_code"] = language_code
            if prompt:
                data["prompt"] = prompt

            try:
                with httpx.Client(timeout=90.0) as client:
                    response = client.post(
                        self.api_url,
                        headers=headers,
                        data=data,
                        files=files
                    )
                    
                if response.status_code == 200:
                    res_json = response.json()
                    # Sarvam returns { "transcript": "...", "language_code": "...", ... }
                    transcript = res_json.get("transcript", "")
                    detected_lang = res_json.get("language_code", language_code)
                    confidence = res_json.get("confidence", None)
                    speaker_count = res_json.get("speaker_count", None)
                    
                    return {
                        "transcript": transcript,
                        "language_code": detected_lang,
                        "confidence": confidence,
                        "speaker_count": speaker_count,
                        "status": "success",
                        "raw_response": res_json
                    }
                else:
                    err_text = response.text
                    logger.error(f"Sarvam API Error HTTP {response.status_code}: {err_text}")
                    return {
                        "transcript": "",
                        "language_code": language_code,
                        "confidence": None,
                        "status": "api_error",
                        "error": f"HTTP {response.status_code}: {err_text}"
                    }
            except Exception as e:
                logger.error(f"Network error during Sarvam transcription: {e}")
                return {
                    "transcript": "",
                    "language_code": language_code,
                    "confidence": None,
                    "status": "network_error",
                    "error": str(e)
                }


def transcribe_processed_dataset(
    processed_audio_dir: Optional[str] = None,
    output_transcripts_dir: Optional[str] = None,
    config_path: str = "config.yaml",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Transcribes all processed WAV files in processed_audio_dir.
    Applies medical normalization.
    Stores raw and cleaned transcripts in transcripts_dir.
    """
    config = load_config(config_path)
    audio_dir = processed_audio_dir or config.get("dataset", {}).get("processed_audio_dir", "datasets/processed_audio")
    out_dir = output_transcripts_dir or config.get("dataset", {}).get("transcripts_dir", "datasets/transcripts")
    
    sarvam_cfg = config.get("sarvam", {})
    api_url = sarvam_cfg.get("api_url", "https://api.sarvam.ai/speech-to-text")
    model = sarvam_cfg.get("model", "saaras:v2")
    lang_code = sarvam_cfg.get("language_code", "unknown")

    client = SarvamSTTClient(api_key=api_key, api_url=api_url, model=model)

    p_dir = Path(audio_dir)
    o_dir = Path(out_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(list(p_dir.glob("*.wav")))
    if not audio_files:
        logger.warning(f"No WAV files found in {audio_dir}")
        return []

    master_transcripts_file = o_dir / "transcripts_master.json"
    existing_records: Dict[str, Dict[str, Any]] = {}
    if master_transcripts_file.exists():
        try:
            with open(master_transcripts_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                existing_records = {item["filename"]: item for item in loaded if "filename" in item}
        except Exception:
            existing_records = {}

    results: List[Dict[str, Any]] = []

    for file_path in audio_files:
        fn = file_path.name
        
        # Check if already transcribed
        if fn in existing_records and existing_records[fn].get("processing_status") == "success":
            logger.info(f"Skipping already transcribed file: {fn}")
            results.append(existing_records[fn])
            continue

        # Get audio duration & sample rate
        import soundfile as sf
        try:
            info = sf.info(str(file_path))
            duration = round(info.duration, 3)
            sr = info.samplerate
        except Exception:
            duration = 0.0
            sr = 16000

        logger.info(f"Transcribing {fn} with Sarvam AI ({duration}s)...")
        stt_res = client.transcribe_file(file_path, language_code=lang_code)

        raw_text = stt_res.get("transcript", "")
        norm_res = clean_medical_transcript(raw_text)

        record = {
            "id": file_path.stem,
            "filename": fn,
            "audio_filepath": str(file_path.resolve()),
            "duration": duration,
            "sample_rate": sr,
            "raw_transcript": norm_res["raw_transcript"],
            "cleaned_transcript": norm_res["cleaned_transcript"],
            "corrected_transcript": norm_res["cleaned_transcript"], # default for reviewer to edit
            "language": stt_res.get("language_code", lang_code),
            "speaker_count": stt_res.get("speaker_count", None),
            "confidence": stt_res.get("confidence", None),
            "processing_status": stt_res.get("status", "unknown"),
            "error": stt_res.get("error", None),
            "review_status": "pending",  # 'pending', 'approved', 'rejected'
            "verified_by_human": False,
            "speaker_labels": None,  # Preserved if diarization provided
        }

        results.append(record)

    # Save master file
    with open(master_transcripts_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Transcripts saved to {master_transcripts_file} ({len(results)} records).")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sarvam AI Transcription for Doctor-Patient STT")
    parser.add_argument("--processed_dir", type=str, default="datasets/processed_audio")
    parser.add_argument("--transcripts_dir", type=str, default="datasets/transcripts")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    transcribe_processed_dataset(
        processed_audio_dir=args.processed_dir,
        output_transcripts_dir=args.transcripts_dir,
        config_path=args.config
    )
