import React from 'react'

const ICONS = {
  apology: "🙏", compensation: "💰", escalation: "📞",
  safety: "⚠️", comfort_food: "🍲", loyalty: "⭐",
  make_good: "✅", delay: "⏱️", refund: "💸",
  celebration: "🎉", dietary: "🥗", information: "ℹ️",
  complaint: "📝", special_occasion: "🎂", unavailable: "😞",
}

export default function RecCard({ rec }) {
  return (
    <div style={{
      background: "#fff", border: "1px solid #e8e6e0", borderRadius: 8,
      padding: "8px 10px", marginBottom: 6,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 3 }}>
        <span style={{ fontSize: 12 }}>{ICONS[rec.category] || "📌"}</span>
        <span style={{ fontSize: 11, fontWeight: 600, color: "#2C2C2A" }}>{rec.title}</span>
      </div>
      <p style={{ fontSize: 10.5, color: "#5F5E5A", margin: "0 0 5px", lineHeight: 1.4 }}>
        {rec.text}
      </p>
      {rec.options?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
          {rec.options.map((o, i) => (
            <span key={i} style={{
              fontSize: 9.5, background: "#EAF3DE", color: "#3B6D11",
              padding: "1px 6px", borderRadius: 10,
            }}>{o}</span>
          ))}
        </div>
      )}
    </div>
  )
}