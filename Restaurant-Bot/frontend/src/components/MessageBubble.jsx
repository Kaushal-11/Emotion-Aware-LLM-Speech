import React from 'react'
import EmotionBadge from './EmotionBadge'

export default function MessageBubble({ msg }) {
  const isUser = msg.role === "user"
  return (
    <div className={msg.isNew ? "msg-enter" : ""} style={{
      display: "flex", justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 10, gap: 8, alignItems: "flex-end",
    }}>
      {!isUser && (
        <div style={{
          width: 30, height: 30, borderRadius: "50%", flexShrink: 0,
          background: "#1D9E75", display: "flex", alignItems: "center",
          justifyContent: "center", fontSize: 13,
        }}>🍽️</div>
      )}
      <div style={{ maxWidth: "76%" }}>
        <div style={{
          padding: "9px 13px",
          background: isUser ? "#1D9E75" : "#fff",
          color: isUser ? "#fff" : "#2C2C2A",
          borderRadius: isUser ? "16px 16px 3px 16px" : "16px 16px 16px 3px",
          fontSize: 13.5, lineHeight: 1.55,
          border: isUser ? "none" : "1px solid #e8e6e0",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}>
          {msg.content}
        </div>
        {msg.meta && (
          <div style={{
            marginTop: 4, display: "flex", alignItems: "center", gap: 6,
            justifyContent: isUser ? "flex-end" : "flex-start",
          }}>
            <EmotionBadge emotion={msg.meta.emotion} small />
            <span style={{ fontSize: 10, color: "#bbb" }}>
              {Math.round(msg.meta.intensity * 100)}%
            </span>
          </div>
        )}
      </div>
    </div>
  )
}