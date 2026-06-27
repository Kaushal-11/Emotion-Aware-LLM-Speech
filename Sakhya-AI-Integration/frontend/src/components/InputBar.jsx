import React from "react";
import { useState, useRef, useEffect } from "react";
import "./InputBar.css";

/**
 * InputBar — Claude-style unified input.
 * Layout: [ textarea ] [ 📎 file ] [ 🎙 mic ] [ ↑ send ]
 *
 * - Textarea + send  → text mode  (/ws/turn_text)
 * - Mic button       → audio mode (/ws/turn_audio)
 * - File button      → file mode  (/ws/turn_file)  same pipeline as audio
 *
 * Props:
 *   onAudioReady(blob, speakerId)
 *   onTextReady(text, speakerId)
 *   onFileReady(file, speakerId)
 *   disabled, speakers, defaultSpeaker
 */
export default function InputBar({
  onAudioReady, onTextReady, onFileReady,
  disabled, speakers, defaultSpeaker,
}) {
  const [recording,  setRecording]  = useState(false);
  const [duration,   setDuration]   = useState(0);
  const [speakerId,  setSpeakerId]  = useState(defaultSpeaker);
  const [text,       setText]       = useState("");
  const [fileName,   setFileName]   = useState(null); // name of picked file

  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);
  const timerRef         = useRef(null);
  const textareaRef      = useRef(null);
  const fileInputRef     = useRef(null);

  useEffect(() => { if (defaultSpeaker) setSpeakerId(defaultSpeaker); }, [defaultSpeaker]);

  // Recording timer
  useEffect(() => {
    if (recording) {
      timerRef.current = setInterval(() => setDuration(d => d + 1), 1000);
    } else {
      clearInterval(timerRef.current);
      setDuration(0);
    }
    return () => clearInterval(timerRef.current);
  }, [recording]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [text]);

  // ── Mic recording ──────────────────────────────────────────────────────
  const startRecording = async () => {
    if (disabled || recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        onAudioReady(blob, speakerId);
        stream.getTracks().forEach(t => t.stop());
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch {
      alert("Microphone access denied. Please allow microphone access in your browser.");
    }
  };

  const stopRecording = () => {
    if (!recording) return;
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  // ── File upload ────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file || disabled) return;
    setFileName(file.name);
    onFileReady(file, speakerId);
    // reset so same file can be re-picked
    e.target.value = "";
    // clear the name indicator after a moment
    setTimeout(() => setFileName(null), 3000);
  };

  // ── Text send ──────────────────────────────────────────────────────────
  const sendText = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onTextReady(trimmed, speakerId);
    setText("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText();
    }
  };

  const fmt = s => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  const canSend = text.trim().length > 0 && !disabled;

  return (
    <div className="input-bar">

      {/* ── Speaker selector row (compact, above input) ── */}
      {speakers.length > 0 && (
        <div className="speaker-row">
          <span className="speaker-label">AI Voice</span>
          <select
            className="speaker-select"
            value={speakerId}
            onChange={e => setSpeakerId(e.target.value)}
            disabled={recording || disabled}
          >
            {speakers.map(s => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
        </div>
      )}

      {/* ── Main input row ── */}
      <div className={`input-row ${recording ? "input-row-recording" : ""}`}>

        {/* File upload button */}
        <button
          className="action-btn file-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || recording}
          title="Upload audio file"
          aria-label="Upload audio file"
        >
          📎
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a,.webm"
          style={{ display: "none" }}
          onChange={handleFileChange}
          disabled={disabled}
        />

        {/* Textarea */}
        <div className="textarea-wrap">
          {recording ? (
            /* Recording state — replace textarea content with rec indicator */
            <div className="recording-indicator">
              <span className="rec-dot" />
              <span className="rec-timer">{fmt(duration)}</span>
              <span className="rec-hint">Release mic to send</span>
            </div>
          ) : fileName ? (
            /* File picked indicator */
            <div className="file-indicator">
              <span className="file-icon">🎵</span>
              <span className="file-name">{fileName}</span>
              <span className="file-hint">Sending…</span>
            </div>
          ) : (
            <textarea
              ref={textareaRef}
              className="text-input"
              placeholder="Type a message, or use mic / file upload…"
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={disabled}
              rows={1}
            />
          )}
        </div>

        {/* Mic button — hold to record */}
        <button
          className={`action-btn mic-btn ${recording ? "recording" : ""}`}
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          onTouchStart={e => { e.preventDefault(); startRecording(); }}
          onTouchEnd={e => { e.preventDefault(); stopRecording(); }}
          disabled={disabled && !recording}
          title={recording ? "Release to send" : "Hold to speak"}
          aria-label={recording ? "Release to send" : "Hold to speak"}
        >
          {recording && (
            <>
              <span className="mic-ring mic-ring-1" />
              <span className="mic-ring mic-ring-2" />
            </>
          )}
          <span className="btn-icon">🎙</span>
        </button>

        {/* Send button */}
        <button
          className={`action-btn send-btn ${canSend ? "ready" : ""}`}
          onClick={sendText}
          disabled={!canSend}
          title="Send message"
          aria-label="Send message"
        >
          <span className="btn-icon send-arrow">↑</span>
        </button>

      </div>

      {/* ── Subtle hint line ── */}
      {!recording && !disabled && !fileName && (
        <div className="input-hints">
          <span>Enter to send · Shift+Enter for new line</span>
          <span className="hint-sep">·</span>
          <span>Hold 🎙 to speak</span>
          <span className="hint-sep">·</span>
          <span>📎 to upload audio</span>
        </div>
      )}
      {disabled && !recording && (
        <div className="input-hints processing-hint">
          <span className="proc-dot" />
          Processing…
        </div>
      )}

    </div>
  );
}