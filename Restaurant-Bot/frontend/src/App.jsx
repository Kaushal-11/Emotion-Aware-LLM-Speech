import React, { useState } from 'react'
import ChatView from './components/ChatView'
import DashboardView from './components/DashboardView'

const API = "https://protection-latina-ink-lucy.trycloudflare.com"

export default function App() {
  const [view, setView] = useState("chat")

  const tabs = [
    { id: "chat",      label: "💬 Chat" },
    { id: "dashboard", label: "📊 Dashboard" },
  ]

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Nav */}
      <div style={{
        background: "#1D9E75", padding: "0 18px", height: 50,
        display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span style={{ fontSize: 20 }}>🍽️</span>
          <span style={{ color: "#fff", fontWeight: 700, fontSize: 14 }}>XO.tikka</span>
          <span style={{
            color: "#9FE1CB", fontSize: 11, background: "rgba(0,0,0,0.15)",
            padding: "1px 7px", borderRadius: 10,
          }}>Support Bot</span>
        </div>
        <div style={{ display: "flex", gap: 3 }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setView(t.id)} style={{
              background: view === t.id ? "rgba(255,255,255,0.22)" : "transparent",
              border: "none", color: "#fff", padding: "5px 12px",
              borderRadius: 7, cursor: "pointer", fontSize: 12.5, fontWeight: 500,
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        <div style={{
          display: view === "chat" ? "flex" : "none",
          flexDirection: "column", height: "100%",
          maxWidth: 480, margin: "0 auto", background: "#fff",
          boxShadow: "0 0 0 1px #e8e6e0",
        }}>
          {/* No sessionId prop — ChatView manages its own sessions now */}
          <ChatView API={API} />
        </div>
        <div style={{
          display: view === "dashboard" ? "flex" : "none",
          flexDirection: "column", height: "100%",
        }}>
          <DashboardView API={API} />
        </div>
      </div>
    </div>
  )
} 