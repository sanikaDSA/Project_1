# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from orchestrator.orchestrator import GLOBAL_ORCHESTRATOR, SehatOrchestrator
from session_store.session_store import GLOBAL_SESSION_STORE, SehatSessionState
from emergency.emergency_detector import GLOBAL_EMERGENCY_DETECTOR, EmergencyDetector
from doctor_delivery.doctor_service import GLOBAL_DOCTOR_DELIVERY, DoctorDeliveryService
from audit.audit_logger import GLOBAL_AUDIT_LOGGER
from audio_pipeline.cura_clinical_flow import GLOBAL_CURA_SESSIONS
from fastapi import WebSocket, WebSocketDisconnect

EMERGENCY_ALERTS_FEED = GLOBAL_DOCTOR_DELIVERY.emergency_feed

load_dotenv()

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    for candidate in [config_path, "backend/config.yaml", "../config.yaml", "e:/Diabetes/backend/config.yaml"]:
        if os.path.exists(candidate):
            try:
                import yaml
                with open(candidate, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sehat_app")

app = FastAPI(
    title="Diabetes Dost — AI Voice Pre-Screener for Diabetes",
    description="Context-aware diabetes pre-screening voice assistant in Hindi speech with dynamic question selection, independent emergency detection, and clinician triage summaries.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

training_state = {
    
    "is_training": False,
    "status": "idle",
    "current_epoch": 0,
    "total_epochs": 0,
    "best_wer": None,
    "latest_metrics": {},
    "error": None,
    "logs": []
}

class TranscriptUpdateRequest(BaseModel):
    id: str
    corrected_transcript: str
    review_status: str
    verified_by_human: bool = True

class PipelineConfigUpdate(BaseModel):
    sarvam_api_key: Optional[str] = None
    language_code: Optional[str] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    learning_rate: Optional[float] = None

for folder in [
    "datasets/raw_audio", "datasets/processed_audio", "datasets/transcripts", 
    "datasets/manifests", "datasets/reviewed", "outputs/checkpoints", 
    "outputs/logs", "outputs/reports", "outputs/reports/sessions"
]:
    Path(folder).mkdir(parents=True, exist_ok=True)


# ==========================================
# GENERAL & AUDIO API ENDPOINTS
# ==========================================

@app.get("/api/health")
async def health_check():
    try:
        import torch
        cuda_ok = torch.cuda.is_available() if torch is not None else False
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU Mode"
    except Exception:
        cuda_ok = False
        gpu_name = "CPU Mode"
    return {
        "status": "online",
        "app_name": "Diabetes Dost — AI Voice Pre-Screener for Diabetes",
        "hospital_name": "Poona Hospital and Research Centre",
        "motto": "॥ आरोग्यक्षेमं वहाम्यहम् ॥",
        "bot_name": "Diabetes Dost",
        "language": "hi-IN",
        "voice_gender": "Male",
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "active_sessions": len(GLOBAL_SESSION_STORE.sessions),
        "sarvam_configured": bool(os.getenv("SARVAM_API_KEY") and os.getenv("SARVAM_API_KEY") != "your_sarvam_api_key_here")
    }


@app.get("/api/config")
async def get_config():
    cfg = load_config()
    key = os.getenv("SARVAM_API_KEY", "")
    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else ("Configured" if key else "Not Set")
    return {
        "config": cfg,
        "sarvam_api_key_status": masked_key
    }


@app.get("/api/audio/stream/{filename}")
async def stream_audio(filename: str):
    for folder in ["outputs/reports", "datasets/processed_audio", "datasets/raw_audio"]:
        target = Path(folder) / filename
        if target.exists():
            return FileResponse(str(target), media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")


@app.post("/api/voice/s2s")
async def process_voice_speech_to_speech(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("hi-IN")
):
    """
    Direct Speech-to-Speech (S2S) Endpoint:
    Processes raw recorded patient audio (.wav), performs STT, AI Clinical reasoning, and TTS.
    Returns: transcript, doctor_reply, response_text, audio_url, audio_path, audio_output, audio_file.
    """
    from audio_pipeline.speech_to_speech import SpeechToSpeechPipeline
    s2s = SpeechToSpeechPipeline()
    
    # Save uploaded file to raw_audio
    temp_dir = Path("datasets/raw_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"direct_voice_in_{file.filename}"
    with open(temp_path, "wb") as f_out:
        f_out.write(await file.read())
        
    result = s2s.process_voice_conversation(temp_path, language_code=language_code or "hi-IN")
    return JSONResponse(content=result)


# ==========================================
# SEHAT AI VOICE PRE-SCREENER API ENDPOINTS
# ==========================================

@app.post("/api/sehat/session/start")
@app.post("/api/cura/session/start")
async def start_sehat_voice_session(session_id: Optional[str] = Form(None)):
    res = GLOBAL_ORCHESTRATOR.start_session(session_id=session_id)
    return JSONResponse(content=res)


@app.post("/api/sehat/session/turn")
@app.post("/api/cura/session/turn")
async def process_sehat_voice_turn(
    file: Optional[UploadFile] = File(None),
    session_id: str = Form(...),
    transcript: Optional[str] = Form(None)
):
    audio_bytes = None
    if file is not None:
        audio_bytes = await file.read()
    
    res = GLOBAL_ORCHESTRATOR.process_patient_turn(
        session_id=session_id,
        transcript_text=transcript,
        audio_bytes=audio_bytes
    )
    return JSONResponse(content=res)


@app.get("/api/sehat/session/{session_id}/summary")
@app.get("/api/cura/session/{session_id}/summary")
async def get_sehat_session_summary(session_id: str):
    summary = GLOBAL_ORCHESTRATOR.get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {
        "status": "success",
        "summary": summary
    }


@app.get("/api/sehat/session/{session_id}/download")
@app.get("/api/cura/session/{session_id}/download")
async def download_sehat_session_report(session_id: str):
    summary = GLOBAL_ORCHESTRATOR.get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    txt_file = summary.get("summary_file_txt")
    if txt_file and os.path.exists(txt_file):
        p_name = summary.get("patient", {}).get("name") or "Patient"
        filename = f"Summary_{p_name}_{session_id}.txt"
        return FileResponse(txt_file, media_type="text/plain; charset=utf-8", filename=filename)
    
    # Fallback to narrative text
    narrative = summary.get("clinical_summary_en", "") + "\n\n" + summary.get("clinical_summary_hi", "")
    return HTMLResponse(content=narrative, media_type="text/plain; charset=utf-8")


@app.get("/api/sehat/session/{session_id}/download/json")
async def download_sehat_session_json(session_id: str):
    summary = GLOBAL_ORCHESTRATOR.get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    json_file = summary.get("summary_file_json")
    if json_file and os.path.exists(json_file):
        p_name = summary.get("patient", {}).get("name") or "Patient"
        filename = f"Summary_{p_name}_{session_id}.json"
        return FileResponse(json_file, media_type="application/json", filename=filename)
    
    return JSONResponse(content=summary)


@app.get("/api/sehat/summaries")
async def list_all_patient_summaries():
    """Lists all saved patient summary reports."""
    summaries_dir = "outputs/reports/summaries"
    results = []
    if os.path.exists(summaries_dir):
        for f in sorted(os.listdir(summaries_dir), reverse=True):
            if f.endswith(".json"):
                fpath = os.path.join(summaries_dir, f)
                try:
                    with open(fpath, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                        results.append({
                            "session_id": data.get("session_id"),
                            "patient": data.get("patient"),
                            "branch": data.get("branch"),
                            "recommended_urgency": data.get("recommended_urgency"),
                            "generated_at": data.get("generated_at"),
                            "summary_file_txt": data.get("summary_file_txt"),
                            "summary_file_json": data.get("summary_file_json")
                        })
                except Exception:
                    pass
    return {
        "status": "success",
        "total": len(results),
        "summaries": results
    }


@app.get("/api/sehat/emergencies")
@app.get("/api/cura/emergencies")
async def get_sehat_emergency_alerts():
    emergencies = GLOBAL_DOCTOR_DELIVERY.get_emergency_feed()
    return {
        "status": "success",
        "count": len(emergencies),
        "emergencies": emergencies
    }


@app.get("/api/sehat/sessions")
@app.get("/api/cura/sessions")
async def list_all_sehat_sessions():
    sessions_list = GLOBAL_DOCTOR_DELIVERY.get_doctor_queue()
    if not sessions_list:
        sessions_list = GLOBAL_SESSION_STORE.list_all_sessions()
    return {
        "status": "success",
        "total": len(sessions_list),
        "sessions": sessions_list
    }


@app.get("/api/sehat/audit/{session_id}")
async def get_session_audit_log(session_id: str):
    events = GLOBAL_AUDIT_LOGGER.get_session_events(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "count": len(events),
        "events": events
    }


# ==========================================
# WEBSOCKET REAL-TIME STREAMING ENDPOINT
# ==========================================

@app.websocket("/ws/session")
async def websocket_session_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            event_type = msg.get("event")
            
            if event_type == "SESSION_START":
                session_id = msg.get("session_id")
                start_res = GLOBAL_ORCHESTRATOR.start_session(session_id)
                session_id = start_res["session_id"]
                await websocket.send_json({
                    "event": "SESSION_STARTED",
                    "data": start_res
                })
                
            elif event_type in ["AUDIO_CHUNK", "ASR_FINAL", "PATIENT_TURN"]:
                transcript = msg.get("transcript", "")
                sid = msg.get("session_id") or session_id
                
                await websocket.send_json({"event": "AI_THINKING", "session_id": sid})
                
                turn_res = GLOBAL_ORCHESTRATOR.process_patient_turn(
                    session_id=sid,
                    transcript_text=transcript
                )
                
                await websocket.send_json({
                    "event": "AI_RESPONSE",
                    "data": turn_res
                })
                
                if turn_res.get("is_emergency"):
                    await websocket.send_json({
                        "event": "EMERGENCY_DETECTED",
                        "data": turn_res
                    })
                elif turn_res.get("is_completed"):
                    await websocket.send_json({
                        "event": "SESSION_COMPLETED",
                        "data": turn_res
                    })
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"event": "SESSION_ERROR", "error": str(e)})
        except Exception:
            pass


# ==========================================
# MODERN POOJA HOSPITAL WEB APP
# ==========================================
TEMPLATES_DIR = Path(__file__).parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_cura_app():
    if INDEX_HTML_PATH.exists():
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Index template not found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
