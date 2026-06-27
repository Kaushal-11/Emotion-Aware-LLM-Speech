import React from "react";
import { useEffect, useRef, useState } from "react";
import "./ChatPanel.css";

const EMOTION_EMOJI = {
  anger: "😠", sadness: "😢", happiness: "😊",
  fear: "😨", disgust: "🤢", surprise: "😲",
};

// Audio mode: 7 steps. Text mode: 5 steps (no ASR/SER).
const AUDIO_STEPS = [
  null,
  { icon: "🎙️", label: "Transcribing speech (ASR)…" },
  { icon: "🎵", label: "Detecting speech emotion (SER)…" },
  { icon: "📝", label: "Classifying text emotion…" },
  { icon: "🧠", label: "Fusing signals & updating state…" },
  { icon: "💡", label: "Finding recommendations…" },
  { icon: "🤖", label: "Generating response…" },
  { icon: "🔊", label: "Synthesising voice…" },
];
const TEXT_STEPS = [
  null,
  { icon: "📝", label: "Classifying text emotion…" },
  { icon: "🧠", label: "Updating emotional state…" },
  { icon: "💡", label: "Finding recommendations…" },
  { icon: "🤖", label: "Generating response…" },
  { icon: "🔊", label: "Synthesising voice…" },
];

function EmotionTag({ emotion, intensity }) {
  if (!emotion || emotion === "n/a") return null;
  return (
    <span className={`emotion-tag emotion-${emotion}`}>
      {EMOTION_EMOJI[emotion] || "😐"}{" "}
      {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
      {intensity != null ? ` · ${Math.round(intensity * 100)}%` : ""}
    </span>
  );
}

function RecommendationsAccordion({ recs }) {
  const [open, setOpen] = useState(false);
  if (!recs || recs.length === 0) return null;
  return (
    <div className="recs-accordion">
      <button className="recs-toggle" onClick={() => setOpen(o => !o)}>
        💡 {recs.length} recommendation{recs.length > 1 ? "s" : ""} {open ? "▲" : "▼"}
      </button>
      {open && (
        <div className="recs-list">
          {recs.map((r, i) => (
            <div key={i} className="rec-item">
              <div className="rec-title">{r.title} <span className="rec-cat">{r.category}</span></div>
              <div className="rec-text">{r.text}</div>
              {r.options?.length > 0 && (
                <ul className="rec-options">{r.options.map((o, j) => <li key={j}>{o}</li>)}</ul>
              )}
              {r.tags?.length > 0 && (
                <div className="rec-tags">
                  {r.tags.map((t, j) => <span key={j} className="rec-tag">{t}</span>)}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Avatar({ label, variant }) {
  return <div className={`bubble-avatar avatar-${variant}`}>{label}</div>;
}

/* ── Pending audio bubble — mic released, ASR running ── */
function PendingAudioBubble() {
  return (
    <div className="bubble-row user-row">
      <div className="bubble user-bubble pending-bubble">
        <div className="pending-dots"><span /><span /><span /></div>
        <span className="pending-label">Transcribing…</span>
      </div>
      <Avatar label="You" variant="user" />
    </div>
  );
}

/* ── Interim audio bubble — ASR done, emotion analysis running ── */
function InterimAudioBubble({ transcript }) {
  return (
    <div className="bubble-row user-row">
      <div className="bubble user-bubble">
        <div className="input-mode-badge audio-badge">🎙 Voice</div>
        <p className="bubble-text">{transcript}</p>
        <div className="bubble-meta-row">
          <span className="analysing-label">Analysing emotion…</span>
        </div>
      </div>
      <Avatar label="You" variant="user" />
    </div>
  );
}

/* ── Text input bubble — shows immediately when text submitted ── */
function InterimTextBubble({ text }) {
  return (
    <div className="bubble-row user-row">
      <div className="bubble user-bubble">
        <div className="input-mode-badge text-badge">⌨ Text</div>
        <p className="bubble-text">{text}</p>
        <div className="bubble-meta-row">
          <span className="analysing-label">Analysing…</span>
        </div>
      </div>
      <Avatar label="You" variant="user" />
    </div>
  );
}

/* ── Completed user bubble ── */
function UserBubble({ turn }) {
  return (
    <div className="bubble-row user-row">
      <div className="bubble user-bubble">
        <div className={`input-mode-badge ${turn.inputMode === "text" ? "text-badge" : "audio-badge"}`}>
          {turn.inputMode === "text" ? "⌨ Text" : "🎙 Voice"}
        </div>
        <p className="bubble-text">{turn.transcript}</p>
        <div className="bubble-meta-row">
          <EmotionTag emotion={turn.userEmotion} intensity={turn.intensity} />
          {turn.clTarget && <span className="target-badge">→ {turn.clTarget}</span>}
        </div>
      </div>
      <Avatar label="You" variant="user" />
    </div>
  );
}

/* ── AI processing bubble — live step log ── */
function ProcessingAIBubble({ progressStep, inputMode }) {
  const steps = inputMode === "text" ? TEXT_STEPS : AUDIO_STEPS;
  const completedSteps = steps.slice(1, progressStep).filter(Boolean);
  const currentStep    = steps[progressStep];

  return (
    <div className="bubble-row ai-row">
      <Avatar label="AI" variant="ai" />
      <div className="bubble ai-bubble processing-ai-bubble">
        {completedSteps.map((s, i) => (
          <div key={i} className="step-done">
            <span className="step-done-check">✓</span>
            <span className="step-done-icon">{s.icon}</span>
            <span className="step-done-label">{s.label}</span>
          </div>
        ))}
        {currentStep && (
          <div className="step-current">
            <span className="step-current-icon">{currentStep.icon}</span>
            <span className="step-current-label">{currentStep.label}</span>
            <span className="step-spinner" />
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Completed AI bubble ── */
function AIBubble({ turn }) {
  const audioRef = useRef(null);
  useEffect(() => {
    if (turn.audioBlobUrl && audioRef.current) {
      audioRef.current.play().catch(() => {});
    }
  }, [turn.audioBlobUrl]);

  return (
    <div className="bubble-row ai-row">
      <Avatar label="AI" variant="ai" />
      <div className="bubble ai-bubble">
        <p className="bubble-text">{turn.response}</p>
        <div className="bubble-meta-row">
          <EmotionTag emotion={turn.aiEmotion} intensity={turn.aiIntensity} />
          {turn.mode && <span className="mode-badge">{turn.mode}</span>}
        </div>
        {turn.audioBlobUrl && (
          <audio ref={audioRef} className="audio-player" controls src={turn.audioBlobUrl} />
        )}
        <RecommendationsAccordion recs={turn.recommendations} />
      </div>
    </div>
  );
}

export default function ChatPanel({
  turns, processing, inputMode,
  progressStep, progressTotal, progressLabel,
  interimTranscript,
}) {
  const bottomRef = useRef(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, processing, progressStep, interimTranscript]);

  // Show AI processing bubble once past the initial input step
  const showAIBubble = processing && (
    inputMode === "text" ? progressStep >= 1 : progressStep >= 2
  );

  return (
    <div className="chat-panel">
      {turns.length === 0 && !processing && (
        <div className="chat-empty">
          <div className="empty-orb">◎</div>
          <p>Speak or type.<br />I'll listen — and respond with feeling.</p>
          <div className="empty-modes">
            <span className="empty-mode-chip">🎙 Hold mic to speak</span>
            <span className="empty-mode-chip">⌨ Switch to text</span>
          </div>
        </div>
      )}

      {/* Completed turns */}
      {turns.map(turn => (
        <div key={turn.id} className="turn-pair">
          <UserBubble turn={turn} />
          <AIBubble   turn={turn} />
        </div>
      ))}

      {/* Live turn */}
      {processing && (
        <div className="turn-pair live-turn">
          {/* Audio mode: pending → interim as ASR finishes */}
          {inputMode === "audio" && !interimTranscript && progressStep <= 1 && (
            <PendingAudioBubble />
          )}
          {inputMode === "audio" && interimTranscript && (
            <InterimAudioBubble transcript={interimTranscript} />
          )}

          {/* Text mode: user text shown immediately */}
          {inputMode === "text" && interimTranscript && (
            <InterimTextBubble text={interimTranscript} />
          )}

          {/* AI processing log */}
          {showAIBubble && (
            <ProcessingAIBubble progressStep={progressStep} inputMode={inputMode} />
          )}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}