import React from 'react'

export default function IntensityBar({ value, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ flex: 1, height: 4, borderRadius: 2, background: "#e8e6e0", overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${Math.round(value * 100)}%`,
          background: color, borderRadius: 2, transition: "width 0.5s ease",
        }} />
      </div>
      <span style={{ fontSize: 10, color: "#aaa", minWidth: 26 }}>{Math.round(value * 100)}%</span>
    </div>
  )
}