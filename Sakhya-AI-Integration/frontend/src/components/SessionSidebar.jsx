import React from "react";
import "./SessionSidebar.css";

export default function SessionSidebar({ sessions, activeId, onSelect, onNew }) {
  return (
    <aside className="session-sidebar">
      <div className="session-header">
        <div className="session-brand">
          <span className="session-brand-mark">◎</span>
          <span className="session-brand-name">Emotional AI</span>
        </div>
        <button className="new-session-btn" onClick={onNew} title="New conversation">
          <span>+</span>
        </button>
      </div>

      <div className="session-label">Conversations</div>

      <div className="session-list">
        {sessions.map(s => (
          <button
            key={s.id}
            className={`session-item ${s.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(s.id)}
          >
            <span className="session-icon">{s.id === activeId ? "▸" : "○"}</span>
            <span className="session-name">{s.name}</span>
            {s.turns.length > 0 && (
              <span className="session-count">{s.turns.length}</span>
            )}
          </button>
        ))}
      </div>

      <div className="session-footer">
        <span className="session-footer-text">Speech · Emotion · AI</span>
      </div>
    </aside>
  );
}