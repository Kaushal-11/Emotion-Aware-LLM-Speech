"""
server.py
---------
FastAPI backend for the Emotional AI speech-to-speech pipeline.

Two input modes:
  - Audio mode  /ws/turn_audio  — mic recording -> ASR + SER + full pipeline
  - Text mode   /ws/turn_text   — typed text -> classifier only (SER skipped)

REST endpoints
--------------
GET  /api/status   health check + GPU info
GET  /api/config   speakers, backends
POST /api/reset    new conversation
POST /api/switch   switch SER / LLM / TTS backend

WebSocket: Audio mode  /ws/turn_audio
------------------------------------------
Client sends:
    {
        "audio_b64":   "<base64 encoded wav/webm>",
        "speaker_id":  "speaker_1",
        "sample_rate": 16000
    }

WebSocket: Text mode  /ws/turn_text
------------------------------------------
Client sends:
    {
        "text":       "I failed my exam today",
        "speaker_id": "speaker_1"
    }

Both WebSockets stream back:
    { "type": "progress", "step": N, "total": T, "label": "..." }
    ...
    { "type": "transcript", "text": "..." }   <- audio mode only, sent as soon as ASR done
    { "type": "result", ... final output ... }
    { "type": "error",  "message": "..." }

Run
---
    python server.py
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import base64
import io
import json
import os
import tempfile
import traceback

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    DEFAULT_SPEAKER,
    DEVICE,
    LLM_BACKEND,
    LLM_DEVICE,
    PRESET_SPEAKERS,
    SER_BACKEND,
    TTS_BACKEND,
)
from pipeline import EmotionalAIPipeline


# ============================================================================
# App
# ============================================================================

app = FastAPI(title="Emotional AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Load pipeline once at startup
# ============================================================================

print("=" * 60)
print("Initializing Emotional AI pipeline...")
print(f"  Light models  ->  {DEVICE}")
print(f"  LLM           ->  {LLM_DEVICE}")
print("=" * 60)

pipeline = EmotionalAIPipeline()
print("\nServer ready.\n")


# ============================================================================
# Helpers
# ============================================================================

def gpu_info() -> list:
    result = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            result.append({
                "index":       i,
                "name":        props.name,
                "total_gb":    round(props.total_memory / 1024**3, 1),
                "alloc_gb":    round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
            })
    return result


def audio_to_b64(audio: np.ndarray, sample_rate: int) -> str:
    """numpy float32 -> base64-encoded WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sample_rate,
             format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def b64_to_wav_file(audio_b64: str, sample_rate: int) -> str:
    """
    Decode base64 audio (WAV or WebM from browser MediaRecorder) into a
    temporary mono 16-bit WAV file. Returns the temp filepath.
    """
    raw = base64.b64decode(audio_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        # soundfile can read WAV / FLAC / OGG from bytes
        buf = io.BytesIO(raw)
        data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)          # stereo -> mono
        sf.write(tmp_path, data, sr, format="WAV", subtype="PCM_16")
    except Exception:
        # Fallback: raw PCM int16 bytes
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(tmp_path, data, sample_rate, format="WAV", subtype="PCM_16")

    return tmp_path


def result_payload(out) -> dict:
    """Serialize PipelineOutput to a JSON-safe dict."""
    audio_b64 = ""
    if out.audio is not None and out.sample_rate:
        try:
            audio_b64 = audio_to_b64(out.audio, out.sample_rate)
        except Exception as e:
            print(f"[Server] Audio encode error: {e}")

    return {
        "type":              "result",
        "transcript":        out.transcript,
        "ser_emotion":       out.ser_emotion,
        "ser_confidence":    round(out.ser_confidence, 3),
        "cl_emotion":        out.cl_emotion,
        "cl_target":         out.cl_target,
        "cl_intensity":      round(out.cl_intensity, 3),
        "fused_emotion":     out.fused_emotion,
        "emotion_agreement": out.emotion_agreement,
        "ai_emotion":        out.ai_emotion,
        "ai_intensity":      round(out.ai_intensity, 3),
        "mode":              out.mode,
        "vector":            out.vector,
        "vector_intensity":  round(out.vector_intensity, 3),
        "response_text":     out.response_text,
        "recommendations":   out.recommendations,
        "audio_b64":         audio_b64,
        "sample_rate":       out.sample_rate or 24000,
    }


# ============================================================================
# REST endpoints
# ============================================================================

@app.get("/api/status")
async def status():
    return {
        "status":     "ok",
        "device":     DEVICE,
        "llm_device": LLM_DEVICE,
        "gpu":        gpu_info(),
        "backends": {
            "ser": pipeline.ser.backend,
            "llm": pipeline.llm.backend,
            "tts": pipeline.tts.backend,
        },
    }


@app.get("/api/config")
async def config():
    return {
        "speakers": [
            {"id": k, "label": v["label"]}
            for k, v in PRESET_SPEAKERS.items()
        ],
        "default_speaker": DEFAULT_SPEAKER,
        "ser_backends":    ["sensevoice", "wavlm"],
        "llm_backends":    ["mistral", "qwen"],
        "tts_backends":    ["f5tts", "cosyvoice2"],
        "current": {
            "ser": SER_BACKEND,
            "llm": LLM_BACKEND,
            "tts": TTS_BACKEND,
        },
    }


@app.post("/api/reset")
async def reset():
    pipeline.reset()
    return {"status": "ok", "message": "Conversation reset."}


class SwitchRequest(BaseModel):
    component: str   # "ser" | "llm" | "tts"
    backend:   str


@app.post("/api/switch")
async def switch(req: SwitchRequest):
    try:
        if req.component == "ser":
            pipeline.switch_ser_backend(req.backend)
        elif req.component == "llm":
            pipeline.switch_llm_backend(req.backend)
        elif req.component == "tts":
            pipeline.switch_tts_backend(req.backend)
        else:
            raise HTTPException(400, f"Unknown component: {req.component}")
        return {"status": "ok", "component": req.component, "backend": req.backend}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================================
# WebSocket — AUDIO MODE
# ============================================================================

@app.websocket("/ws/turn_audio")
async def ws_turn_audio(websocket: WebSocket):
    """
    Audio turn:
        1. Client sends base64 audio blob
        2. Server saves to temp WAV
        3. ASR + SER run in parallel (progress streamed)
        4. As soon as ASR finishes -> send {"type":"transcript","text":"..."}
           so the UI can display it in the chat bubble immediately
        5. Remaining pipeline steps streamed as progress
        6. Final result sent (includes base64 audio response)
    """
    await websocket.accept()
    try:
        while True:
            raw        = await websocket.receive_text()
            payload    = json.loads(raw)
            audio_b64  = payload.get("audio_b64", "")
            speaker_id = payload.get("speaker_id", DEFAULT_SPEAKER)
            sample_rate = int(payload.get("sample_rate", 16000))

            if not audio_b64:
                await websocket.send_json({"type": "error", "message": "No audio received."})
                continue

            # Decode audio to temp file
            try:
                tmp_path = b64_to_wav_file(audio_b64, sample_rate)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Audio decode error: {e}"})
                continue

            # Run pipeline — stream every step
            out = None
            try:
                for status in pipeline.run_turn_stream(tmp_path, speaker_id=speaker_id):
                    if status["done"]:
                        out = status["result"]
                        break

                    # As soon as ASR is done the transcript is embedded in
                    # the next progress label — we send a dedicated transcript
                    # message when we detect step 3 (classifier step) starts,
                    # meaning ASR (step 1) has already resolved.
                    # The actual transcript comes from inside the generator
                    # via the "transcript" key when present.
                    if "transcript" in status:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": status["transcript"],
                        })

                    await websocket.send_json({
                        "type":  "progress",
                        "step":  status["step"],
                        "total": 7,
                        "label": status["label"],
                    })

            except Exception as e:
                traceback.print_exc()
                await websocket.send_json({"type": "error", "message": str(e)})
                continue
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            if out is None:
                await websocket.send_json({"type": "error", "message": "Pipeline returned no output."})
                continue

            # Send transcript explicitly (always available on result)
            await websocket.send_json({
                "type": "transcript",
                "text": out.transcript,
            })

            await websocket.send_json(result_payload(out))

    except WebSocketDisconnect:
        print("[Server] Audio WebSocket disconnected.")
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ============================================================================
# WebSocket — TEXT MODE
# ============================================================================

@app.websocket("/ws/turn_text")
async def ws_turn_text(websocket: WebSocket):
    """
    Text turn — SER is skipped entirely.
        1. Client sends plain text message
        2. Text Classifier runs (no ASR, no SER)
        3. Remaining pipeline identical to audio mode
        4. Final result sent (includes base64 audio response)
    """
    await websocket.accept()
    try:
        while True:
            raw        = await websocket.receive_text()
            payload    = json.loads(raw)
            text       = payload.get("text", "").strip()
            speaker_id = payload.get("speaker_id", DEFAULT_SPEAKER)

            if not text:
                await websocket.send_json({"type": "error", "message": "No text received."})
                continue

            out = None
            try:
                for status in pipeline.run_text_turn_stream(text, speaker_id=speaker_id):
                    if status["done"]:
                        out = status["result"]
                        break

                    await websocket.send_json({
                        "type":  "progress",
                        "step":  status["step"],
                        "total": 5,
                        "label": status["label"],
                    })

            except Exception as e:
                traceback.print_exc()
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            if out is None:
                await websocket.send_json({"type": "error", "message": "Pipeline returned no output."})
                continue

            await websocket.send_json(result_payload(out))

    except WebSocketDisconnect:
        print("[Server] Text WebSocket disconnected.")
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,       # MUST be 1 — models live in single process memory
        log_level="info",
    )