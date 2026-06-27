import React from "react";
import { useState } from "react";
import "./SettingsDrawer.css";

function BackendToggle({ label, note, options, current, onSwitch }) {
  const [loading, setLoading] = useState(false);
  const [msg,     setMsg]     = useState("");

  const handle = async (val) => {
    if (val === current || loading) return;
    setLoading(true);
    setMsg("Switching…");
    const res = await onSwitch(val);
    setMsg(res.status === "ok" ? `✓ ${res.message}` : `✕ ${res.message}`);
    setLoading(false);
  };

  return (
    <div className="backend-group">
      <div className="backend-label">{label}</div>
      {note && <div className="backend-note">{note}</div>}
      <div className="backend-options">
        {options.map(o => (
          <button
            key={o}
            className={`backend-btn ${o === current ? "active" : ""}`}
            onClick={() => handle(o)}
            disabled={loading}
          >
            {o}
          </button>
        ))}
      </div>
      {msg && (
        <div className={`backend-msg ${msg.startsWith("✓") ? "ok" : msg === "Switching…" ? "wait" : "err"}`}>
          {msg}
        </div>
      )}
    </div>
  );
}

export default function SettingsDrawer({ backends, gpuInfo, onSwitch, onClose }) {
  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={e => e.stopPropagation()}>
        <div className="drawer-header">
          <span className="drawer-title">Settings</span>
          <button className="drawer-close" onClick={onClose}>✕</button>
        </div>

        <div className="drawer-body">

          <section className="drawer-section">
            <div className="drawer-section-title">Model Backends</div>
            <BackendToggle
              label="SER — Speech Emotion Recognition"
              options={["sensevoice", "wavlm"]}
              current={backends.ser}
              onSwitch={v => onSwitch("ser", v)}
            />
            <BackendToggle
              label="LLM — Language Model"
              options={["mistral", "qwen"]}
              current={backends.llm}
              onSwitch={v => onSwitch("llm", v)}
              note="⚠ Switching LLM resets conversation history (~30s reload)"
            />
            <BackendToggle
              label="TTS — Text-to-Speech"
              options={["f5tts", "cosyvoice2"]}
              current={backends.tts}
              onSwitch={v => onSwitch("tts", v)}
            />
          </section>

          {gpuInfo && gpuInfo.length > 0 && (
            <section className="drawer-section">
              <div className="drawer-section-title">GPU Memory</div>
              <div className="gpu-list">
                {gpuInfo.map((g, i) => {
                  const used  = g.alloc_gb ?? g.used_gb ?? 0;
                  const total = g.total_gb ?? 0;
                  const pct   = total > 0 ? (used / total) * 100 : 0;
                  return (
                    <div key={i} className="gpu-card">
                      <div className="gpu-card-top">
                        <span className="gpu-name">GPU {g.index ?? i} — {g.name}</span>
                        <span className="gpu-mem-text">{used.toFixed(2)} / {total} GB</span>
                      </div>
                      <div className="gpu-bar-track">
                        <div
                          className="gpu-bar-fill"
                          style={{
                            width: `${pct}%`,
                            background: pct > 85 ? "var(--em-anger)" : pct > 65 ? "var(--em-surprise)" : "var(--accent)",
                          }}
                        />
                      </div>
                      <div className="gpu-bar-label">{Math.round(pct)}% used</div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

        </div>
      </div>
    </div>
  );
}