import React from "react";
import { useState, useEffect, useRef, useCallback } from "react";
import ChatPanel from "./components/ChatPanel";
import SignalsPanel from "./components/SignalsPanel";
import InputBar from "./components/InputBar";
import SettingsDrawer from "./components/SettingsDrawer";
import SessionSidebar from "./components/SessionSidebar";
import "./App.css";

const SERVER_URL = import.meta.env.VITE_SERVER_URL || "http://localhost:8000";
const WS_URL     = SERVER_URL.replace(/^https/, "wss").replace(/^http/, "ws");

// Audio mode: 7 steps. Text mode: 5 steps (no ASR/SER).
const TOTAL_STEPS_AUDIO = 7;
const TOTAL_STEPS_TEXT  = 5;

let sessionCounter = 1;
function newSession() {
  return { id: Date.now(), name: `Session ${sessionCounter++}`, turns: [] };
}

export default function App() {
  const [sessions,          setSessions]          = useState([newSession()]);
  const [activeSessionId,   setActiveSessionId]   = useState(sessions[0].id);
  const [config,            setConfig]            = useState(null);
  const [processing,        setProcessing]        = useState(false);
  const [inputMode,         setInputMode]         = useState("audio"); // "audio" | "text"
  const [progressStep,      setProgressStep]      = useState(0);
  const [progressLabel,     setProgressLabel]     = useState("");
  const [progressTotal,     setProgressTotal]     = useState(TOTAL_STEPS_AUDIO);
  const [lastSignals,       setLastSignals]       = useState(null);
  const [liveSignals,       setLiveSignals]       = useState(null);
  const [interimTranscript, setInterimTranscript] = useState(null);
  const [settingsOpen,      setSettingsOpen]      = useState(false);
  const [backends,          setBackends]          = useState({ ser: null, llm: null, tts: null });
  const [gpuInfo,           setGpuInfo]           = useState(null);
  const [serverError,       setServerError]       = useState(null);

  const wsRef = useRef(null);
  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  const updateSession = useCallback((id, updater) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, ...updater(s) } : s));
  }, []);

  // ── Fetch config on mount ──────────────────────────────────────────────
  useEffect(() => {
    fetch(`${SERVER_URL}/api/config`)
      .then(r => r.json())
      .then(cfg => {
        setConfig(cfg);
        // server returns either `current` or `current_backends` or `backends`
        setBackends(cfg.current || cfg.current_backends || cfg.backends || {});
      })
      .catch(() => setServerError("Cannot reach server. Check VITE_SERVER_URL in .env"));

    fetch(`${SERVER_URL}/api/status`)
      .then(r => r.json())
      .then(d => setGpuInfo(d.gpu_info || d.gpu || null))
      .catch(() => {});
  }, []);

  // ── Session management ─────────────────────────────────────────────────
  const handleNewSession = useCallback(() => {
    fetch(`${SERVER_URL}/api/reset`, { method: "POST" }).catch(() => {});
    const s = newSession();
    setSessions(prev => [...prev, s]);
    setActiveSessionId(s.id);
    setLastSignals(null);
    setLiveSignals(null);
    setInterimTranscript(null);
    setProgressStep(0);
    setProgressLabel("");
  }, []);

  const handleSelectSession = useCallback((id) => {
    setActiveSessionId(id);
    setLastSignals(null);
    setLiveSignals(null);
    setInterimTranscript(null);
    setProgressStep(0);
  }, []);

  // ── Backend switching ──────────────────────────────────────────────────
  const handleSwitchBackend = useCallback(async (component, backend) => {
    const res = await fetch(`${SERVER_URL}/api/switch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ component, backend }),
    });
    const data = await res.json();
    if (data.status === "ok") setBackends(prev => ({ ...prev, [component]: backend }));
    return data;
  }, []);

  // ── Shared WebSocket message handler ───────────────────────────────────
  const handleWsMessage = useCallback((msg, sessionId, isTextMode) => {
    // --- transcript message (audio mode only, sent as soon as ASR done) ---
    if (msg.type === "transcript") {
      setInterimTranscript(msg.text);
      return;
    }

    // --- progress ---
    if (msg.type === "progress") {
      setProgressStep(msg.step);
      setProgressTotal(msg.total || (isTextMode ? TOTAL_STEPS_TEXT : TOTAL_STEPS_AUDIO));
      setProgressLabel(msg.label || "");
      return;
    }

    // --- error ---
    if (msg.type === "error") {
      setProgressLabel(`❌ ${msg.message}`);
      setProcessing(false);
      setInterimTranscript(null);
      return;
    }

    // --- final result ---
    if (msg.type === "result") {
      setProgressStep(isTextMode ? TOTAL_STEPS_TEXT : TOTAL_STEPS_AUDIO);
      setProgressLabel("Done");
      setProcessing(false);
      setInterimTranscript(null);
      setLiveSignals(null);

      setLastSignals({
        serEmotion:      msg.ser_emotion,
        serConfidence:   msg.ser_confidence,
        clEmotion:       msg.cl_emotion,
        clTarget:        msg.cl_target,
        clIntensity:     msg.cl_intensity,
        fusedEmotion:    msg.fused_emotion,
        agreement:       msg.emotion_agreement,
        aiEmotion:       msg.ai_emotion,
        aiIntensity:     msg.ai_intensity,
        mode:            msg.mode,
        vector:          msg.vector,
        vectorIntensity: msg.vector_intensity,
        isTextMode,
      });

      const turn = {
        id:          Date.now(),
        inputMode:   isTextMode ? "text" : "audio",
        transcript:  msg.transcript,
        userEmotion: msg.fused_emotion,
        intensity:   msg.cl_intensity,
        clTarget:    msg.cl_target,
        response:    msg.response_text,
        aiEmotion:   msg.ai_emotion,
        aiIntensity: msg.ai_intensity,
        mode:        msg.mode,
        recommendations: msg.recommendations || [],
        audioBlobUrl: msg.audio_b64
          ? URL.createObjectURL(new Blob(
              [Uint8Array.from(atob(msg.audio_b64), c => c.charCodeAt(0))],
              { type: "audio/wav" }
            ))
          : null,
      };
      updateSession(sessionId, s => ({ turns: [...s.turns, turn] }));
    }
  }, [updateSession]);

  // ── AUDIO turn ─────────────────────────────────────────────────────────
  const handleAudio = useCallback((audioBlob, speakerId) => {
    if (processing) return;
    setProcessing(true);
    setInputMode("audio");
    setProgressStep(0);
    setProgressTotal(TOTAL_STEPS_AUDIO);
    setProgressLabel("Connecting…");
    setLiveSignals(null);
    setInterimTranscript(null);

    const sessionId = activeSessionId;
    const ws = new WebSocket(`${WS_URL}/ws/turn_audio`);
    wsRef.current = ws;

    ws.onopen = () => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const b64 = reader.result.split(",")[1];
        ws.send(JSON.stringify({ audio_b64: b64, speaker_id: speakerId, sample_rate: 16000 }));
      };
      reader.readAsDataURL(audioBlob);
    };

    ws.onmessage = (ev) => handleWsMessage(JSON.parse(ev.data), sessionId, false);
    ws.onerror   = () => { setProgressLabel("❌ WebSocket error"); setProcessing(false); setInterimTranscript(null); };
    ws.onclose   = () => { wsRef.current = null; };
  }, [processing, activeSessionId, handleWsMessage]);

  // ── TEXT turn ──────────────────────────────────────────────────────────
  const handleText = useCallback((text, speakerId) => {
    if (processing || !text.trim()) return;
    setProcessing(true);
    setInputMode("text");
    setProgressStep(0);
    setProgressTotal(TOTAL_STEPS_TEXT);
    setProgressLabel("Connecting…");
    setLiveSignals(null);
    setInterimTranscript(text); // show user text immediately — no ASR needed

    const sessionId = activeSessionId;
    const ws = new WebSocket(`${WS_URL}/ws/turn_text`);
    wsRef.current = ws;

    ws.onopen  = () => ws.send(JSON.stringify({ text, speaker_id: speakerId }));
    ws.onmessage = (ev) => handleWsMessage(JSON.parse(ev.data), sessionId, true);
    ws.onerror   = () => { setProgressLabel("❌ WebSocket error"); setProcessing(false); setInterimTranscript(null); };
    ws.onclose   = () => { wsRef.current = null; };
  }, [processing, activeSessionId, handleWsMessage]);

  // FILE turn — same pipeline as audio, file sent as base64
  const handleFile = useCallback((file, speakerId) => {
    if (processing) return;
    setProcessing(true);
    setInputMode("audio"); // file uses the full audio pipeline (ASR + SER)
    setProgressStep(0);
    setProgressTotal(TOTAL_STEPS_AUDIO);
    setProgressLabel("Reading file...");
    setLiveSignals(null);
    setInterimTranscript(null);

    const sessionId = activeSessionId;
    const reader = new FileReader();

    reader.onloadend = () => {
      const b64 = reader.result.split(",")[1];
      const ws = new WebSocket(`${WS_URL}/ws/turn_file`);
      wsRef.current = ws;

      ws.onopen = () => {
        setProgressLabel("Connecting...");
        ws.send(JSON.stringify({
          audio_b64:   b64,
          speaker_id:  speakerId,
          sample_rate: 16000,
          filename:    file.name,
        }));
      };
      ws.onmessage = (ev) => handleWsMessage(JSON.parse(ev.data), sessionId, false);
      ws.onerror   = () => { setProgressLabel("WebSocket error"); setProcessing(false); };
      ws.onclose   = () => { wsRef.current = null; };
    };

    reader.onerror = () => {
      setProgressLabel("Could not read file");
      setProcessing(false);
    };

    reader.readAsDataURL(file);
  }, [processing, activeSessionId, handleWsMessage]);

  if (serverError) {
    return (
      <div className="error-screen">
        <div className="error-card">
          <h2>⚠️ Server Unreachable</h2>
          <p>{serverError}</p>
          <code>Edit VITE_SERVER_URL in frontend/.env</code>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <SessionSidebar
        sessions={sessions}
        activeId={activeSessionId}
        onSelect={handleSelectSession}
        onNew={handleNewSession}
      />

      <div className="main-area">
        <header className="app-header">
          <div className="header-left">
            <div className="header-logo-wrap">
              <span className="header-logo">◎</span>
            </div>
            <div className="header-titles">
              <span className="header-title">Emotional AI</span>
              <span className="header-sub">Speech · Text · Emotion</span>
            </div>
          </div>
          <div className="header-right">
            {gpuInfo && gpuInfo[0] && gpuInfo[0].total_gb > 0 && (
              <span className="gpu-badge">
                {(gpuInfo[0].alloc_gb ?? gpuInfo[0].used_gb ?? 0).toFixed(1)} / {gpuInfo[0].total_gb} GB
              </span>
            )}
            <button className="settings-btn" onClick={() => setSettingsOpen(true)}>⚙ Settings</button>
          </div>
        </header>

        <ChatPanel
          turns={activeSession.turns}
          processing={processing}
          inputMode={inputMode}
          progressStep={progressStep}
          progressTotal={progressTotal}
          progressLabel={progressLabel}
          interimTranscript={interimTranscript}
        />

        <div className="bottom-bar">
          <InputBar
            onAudioReady={handleAudio}
            onTextReady={handleText}
            onFileReady={handleFile}
            disabled={processing}
            speakers={config?.speakers || []}
            defaultSpeaker={config?.default_speaker || "speaker_1"}
          />
          {(processing || progressStep > 0) && (
            <div className="progress-strip">
              <div className="progress-bar-track">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${(progressStep / progressTotal) * 100}%` }}
                />
              </div>
              <span className="progress-label">{progressLabel}</span>
            </div>
          )}
        </div>
      </div>

      <SignalsPanel
        signals={lastSignals}
        liveSignals={liveSignals}
        processing={processing}
        progressStep={progressStep}
        inputMode={inputMode}
      />

      {settingsOpen && (
        <SettingsDrawer
          backends={backends}
          gpuInfo={gpuInfo}
          onSwitch={handleSwitchBackend}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}