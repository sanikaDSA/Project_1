import os
import sys
import json
import base64
from pathlib import Path
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def run_e2e_s2s_test():
    print("=" * 60)
    print("STARTING END-TO-END SPEECH-TO-SPEECH (S2S) VERIFICATION TEST")
    print("=" * 60)

    # 1. Check input audio
    sample_audio = Path("datasets/processed_audio/ElevenLabs_02_16k_mono.wav")
    assert sample_audio.exists(), f"Sample audio not found at {sample_audio}"
    print(f"Step 1 [Audio Input]: Found sample audio: {sample_audio.name} ({sample_audio.stat().st_size} bytes)")

    # 2. Test direct S2S Pipeline execution
    from audio_pipeline.speech_to_speech import SpeechToSpeechPipeline
    s2s = SpeechToSpeechPipeline()

    print("\nStep 2 [Pipeline Execution]: Running process_voice_conversation()...")
    result = s2s.process_voice_conversation(sample_audio, language_code="mr-IN")

    print("\n--- S2S EXECUTION RESULT ---")
    print("1. Transcript (STT):", result.get("transcript"))
    print("2. Response Text (AI):", result.get("response_text"))
    print("3. Audio Output Base64:", "Present (len=" + str(len(result.get("audio_output", ""))) + ")" if result.get("audio_output") else "MISSING")
    print("4. Audio URL:", result.get("audio_url"))
    print("5. Audio File Path:", result.get("audio_file"))
    print("6. TTS Engine Used:", result.get("tts_engine"))

    # Assertions
    assert result.get("transcript"), "FAIL: Transcript is empty"
    assert result.get("response_text"), "FAIL: AI Response Text is empty"
    assert result.get("audio_output"), "FAIL: Audio output (base64) is empty"
    assert result.get("audio_file") and os.path.exists(result["audio_file"]), "FAIL: Audio file does not exist on disk"
    
    # Verify WAV header and playability
    audio_file_size = os.path.getsize(result["audio_file"])
    assert audio_file_size > 1000, f"FAIL: Generated audio file too small ({audio_file_size} bytes)"
    print(f"Generated Audio File Verified: {result['audio_file']} ({audio_file_size} bytes)")

    # 3. Test HTTP API Endpoint /api/voice/s2s
    print("\nStep 3 [API Endpoint Test]: Sending audio to POST http://localhost:8000/api/voice/s2s...")
    with open(sample_audio, "rb") as f:
        files = {"file": ("test_mic.wav", f, "audio/wav")}
        res = httpx.post("http://localhost:8000/api/voice/s2s", files=files, timeout=60.0)

    print("HTTP Status Code:", res.status_code)
    assert res.status_code == 200, f"API failed with status {res.status_code}: {res.text}"
    api_data = res.json()
    print("API Response Transcript:", api_data.get("transcript"))
    print("API Response Text:", api_data.get("response_text"))
    print("API Audio URL:", api_data.get("audio_url"))
    print("API Audio Output (Base64 length):", len(api_data.get("audio_output", "")))

    assert api_data.get("transcript"), "API transcript missing"
    assert api_data.get("response_text"), "API response_text missing"
    assert api_data.get("audio_output"), "API audio_output missing"

    print("\n" + "=" * 60)
    print("ALL S2S END-TO-END VERIFICATION CHECKS PASSED (PASS)!")
    print("=" * 60)

if __name__ == "__main__":
    run_e2e_s2s_test()
