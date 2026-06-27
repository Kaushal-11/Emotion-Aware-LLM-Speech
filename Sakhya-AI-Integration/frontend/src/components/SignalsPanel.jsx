import React from "react";
import "./SignalsPanel.css";

const EMOTION_EMOJI = {
  anger: "😠", sadness: "😢", happiness: "😊",
  fear: "😨", disgust: "🤢", surprise: "😲",
};

// Audio mode step mapping
const AUDIO_STEP = { ser: 2, text: 3, fusion: 4, decision: 4, ai: 4 };
// Text mode step mapping (SER skipped — no entry)
const TEXT_STEP  = { text: 1, fusion: 2, decision: 2, ai: 2 };

function BarRow({ label, value, color }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="sig-row">
      <div className="sig-row-header">
        <span className="sig-label">{label}</span>
        <span className="sig-pct" style={{ color: color || "var(--accent)" }}>{pct}%</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color || "var(--accent)" }} />
      </div>
    </div>
  );
}

function TextRow({ label, value }) {
  return (
    <div className="sig-row sig-row-inline">
      <span className="sig-label">{label}</span>
      <span className="sig-value">{value || "—"}</span>
    </div>
  );
}

function EmotionRow({ label, emotion }) {
  const emoji = EMOTION_EMOJI[emotion] || "😐";
  const cls = emotion && emotion !== "n/a"
    ? `sig-emotion-pill emotion-${emotion}`
    : "sig-emotion-pill emotion-neutral";
  return (
    <div className="sig-row sig-row-inline">
      <span className="sig-label">{label}</span>
      <span className={cls}>
        {emotion && emotion !== "n/a" ? emoji : "—"}{" "}
        {emotion && emotion !== "n/a"
          ? emotion.charAt(0).toUpperCase() + emotion.slice(1)
          : "n/a"}
      </span>
    </div>
  );
}

function Section({ title, children, visible, live }) {
  return (
    <section className={`sig-section ${visible ? "visible" : "dimmed"} ${live ? "live" : ""}`}>
      <div className="sig-section-title">
        <span>{title}</span>
        {live && <span className="live-dot" />}
      </div>
      <div className="sig-section-body">{children}</div>
    </section>
  );
}

export default function SignalsPanel({ signals, liveSignals, processing, progressStep, inputMode }) {
  const s       = processing ? (liveSignals || {}) : (signals || {});
  const isDone  = !processing && signals;
  const isText  = (inputMode === "text") || (signals?.isTextMode);
  const stepMap = isText ? TEXT_STEP : AUDIO_STEP;

  const show   = (key) => isDone || (processing && progressStep >= (stepMap[key] ?? 99));
  const isLive = (key) => processing && progressStep === (stepMap[key] ?? 99);

  return (
    <aside className="signals-panel">
      <div className="signals-header">
        <span className="signals-title">Signal Breakdown</span>
        <div className="signals-header-right">
          {isText && <span className="mode-indicator text-mode-ind">⌨ Text</span>}
          {!isText && !processing && signals && <span className="mode-indicator audio-mode-ind">🎙 Voice</span>}
          {processing && <span className="live-badge">● Live</span>}
        </div>
      </div>

      {!signals && !processing ? (
        <div className="signals-empty">
          <div className="signals-empty-icon">◎</div>
          <p>Signals appear after<br />your first turn.</p>
        </div>
      ) : (
        <div className="signals-body">

          {/* SER section — only in audio mode */}
          {!isText && (
            <Section title="🎵 Speech Emotion (SER)" visible={show("ser")} live={isLive("ser")}>
              <EmotionRow label="Detected"   emotion={s.serEmotion} />
              <BarRow     label="Confidence" value={s.serConfidence} color="var(--em-happiness)" />
            </Section>
          )}

          <Section title="📝 Text Classifier" visible={show("text")} live={isLive("text")}>
            <EmotionRow label="Emotion"   emotion={s.clEmotion} />
            <TextRow    label="Target"    value={s.clTarget} />
            <BarRow     label="Intensity" value={s.clIntensity} color="var(--em-sadness)" />
          </Section>

          <Section title="🔗 Fusion" visible={show("fusion")} live={isLive("fusion")}>
            <EmotionRow label="Final emotion" emotion={s.fusedEmotion} />
            {!isText && (
              <TextRow
                label="SER ↔ Text"
                value={s.agreement === true ? "✅ Agree" : s.agreement === false ? "⚠️ Disagree" : "—"}
              />
            )}
          </Section>

          {/* Decision ABOVE AI State */}
          <Section title="⚙️ Decision Engine" visible={show("decision")} live={isLive("decision")}>
            <TextRow label="Mode"   value={s.mode} />
            <TextRow label="Vector" value={s.vector} />
            <BarRow  label="Steering strength" value={s.vectorIntensity} color="var(--accent)" />
          </Section>

          <Section title="🧠 AI Emotional State" visible={show("ai")} live={isLive("ai")}>
            <EmotionRow label="AI feeling" emotion={s.aiEmotion} />
            <BarRow     label="Intensity"  value={s.aiIntensity} color="var(--em-fear)" />
          </Section>

        </div>
      )}
    </aside>
  );
}