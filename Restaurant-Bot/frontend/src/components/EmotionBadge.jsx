import React from 'react'

export const EMOTION_CONFIG = {
  anger:    { color: "#E24B4A", bg: "#FCEBEB", label: "Angry",     icon: "🔴" },
  sadness:  { color: "#378ADD", bg: "#E6F1FB", label: "Sad",       icon: "🔵" },
  fear:     { color: "#BA7517", bg: "#FAEEDA", label: "Fearful",   icon: "🟡" },
  happiness:{ color: "#639922", bg: "#EAF3DE", label: "Happy",     icon: "🟢" },
  disgust:  { color: "#7F77DD", bg: "#EEEDFE", label: "Disgusted", icon: "🟣" },
  surprise: { color: "#D85A30", bg: "#FAECE7", label: "Surprised", icon: "🟠" },
  "":       { color: "#888780", bg: "#F1EFE8", label: "Neutral",   icon: "⚪" },
}

export default function EmotionBadge({ emotion, small }) {
  const cfg = EMOTION_CONFIG[emotion] || EMOTION_CONFIG[""]
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 3,
      background: cfg.bg, color: cfg.color,
      padding: small ? "2px 7px" : "3px 10px",
      borderRadius: 20, fontSize: small ? 10 : 12, fontWeight: 500,
      border: `1px solid ${cfg.color}40`, whiteSpace: "nowrap",
    }}>
      {cfg.icon} {cfg.label}
    </span>
  )
}