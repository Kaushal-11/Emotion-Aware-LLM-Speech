import React, { useState, useEffect, useRef } from 'react'

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(iso) {
  if (!iso) return "—"
  const d = new Date(iso)
  return d.toLocaleString("en-IN", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: true,
  })
}
function fmtTime(iso) {
  if (!iso) return ""
  return new Date(iso).toLocaleTimeString("en-IN", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true,
  })
}
function fmtDate(iso) {
  if (!iso) return ""
  return new Date(iso).toLocaleDateString("en-IN", {
    day: "2-digit", month: "short", year: "numeric",
  })
}

const EMOTION_COLOR = {
  anger:    "#E24B4A",
  sadness:  "#378ADD",
  fear:     "#8B5CF6",
  joy:      "#1D9E75",
  neutral:  "#888",
  disgust:  "#E67E22",
  surprise: "#BA7517",
}

function EmotionDot({ emotion, intensity }) {
  const color = EMOTION_COLOR[emotion] || "#888"
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      background: `${color}18`, color, padding: "2px 8px",
      borderRadius: 20, fontSize: 11, fontWeight: 600,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: "50%", background: color,
        display: "inline-block",
      }} />
      {emotion} {intensity != null ? `${Math.round(intensity * 100)}%` : ""}
    </span>
  )
}

// ── Chat transcript bubble ────────────────────────────────────────────────────
function ChatBubble({ msg }) {
  const isUser = msg.role === "user"
  const color  = EMOTION_COLOR[msg.emotion] || null
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      marginBottom: 10,
    }}>
      <div style={{
        maxWidth: "78%",
        background: isUser ? "#1D9E75" : "#fff",
        color:      isUser ? "#fff" : "#2C2C2A",
        border: isUser ? "none" : "1px solid #e8e6e0",
        borderRadius: isUser ? "16px 16px 3px 16px" : "16px 16px 16px 3px",
        padding: "9px 13px", fontSize: 13,
      }}>
        {msg.content}
      </div>
      <div style={{
        display: "flex", gap: 6, alignItems: "center",
        marginTop: 3, paddingLeft: isUser ? 0 : 4, paddingRight: isUser ? 4 : 0,
      }}>
        {msg.emotion && !isUser && (
          <EmotionDot emotion={msg.emotion} intensity={msg.intensity} />
        )}
        {msg.emotion && isUser && (
          <EmotionDot emotion={msg.emotion} intensity={msg.intensity} />
        )}
        <span style={{ fontSize: 10, color: "#bbb" }}>{fmtTime(msg.timestamp)}</span>
      </div>
    </div>
  )
}

// ── Session transcript modal ──────────────────────────────────────────────────
function SessionModal({ session, API, onClose }) {
  const [history, setHistory] = useState(null)
  const [loading, setLoading] = useState(true)
  const bottomRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/session/${session.session_id}/history`)
      .then(r => r.json())
      .then(d => { setHistory(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [session.session_id])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [history])

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)",
      zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center",
      padding: 16,
    }} onClick={onClose}>
      <div style={{
        background: "#fafaf9", borderRadius: 16, width: "100%", maxWidth: 520,
        maxHeight: "88vh", display: "flex", flexDirection: "column",
        boxShadow: "0 20px 60px rgba(0,0,0,0.25)",
      }} onClick={e => e.stopPropagation()}>

        {/* Modal header */}
        <div style={{
          padding: "14px 18px", borderBottom: "1px solid #e8e6e0",
          background: session.escalated ? "#FCEBEB" : "#fff",
          borderRadius: "16px 16px 0 0",
          display: "flex", justifyContent: "space-between", alignItems: "flex-start",
        }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              {session.escalated && (
                <span style={{
                  background: "#E24B4A", color: "#fff", fontSize: 10,
                  padding: "2px 8px", borderRadius: 20, fontWeight: 700,
                }}>🚨 ESCALATED</span>
              )}
              <span style={{ fontWeight: 700, fontSize: 15, color: "#2C2C2A" }}>
                {session.customer || "Guest"}
              </span>
              {session.sessionNumber && (
                <span style={{
                  fontSize: 11, fontWeight: 600, color: "#1D9E75",
                  background: "#EAF3DE", padding: "2px 8px", borderRadius: 20,
                }}>Session {session.sessionNumber}</span>
              )}
            </div>
            <div style={{ fontSize: 11, color: "#888", display: "flex", gap: 12 }}>
              {session.order_id && <span>📦 {session.order_id}</span>}
              <span>Started {fmt(session.created_at)}</span>
              {session.escalated && session.escalated_at && (
                <span style={{ color: "#E24B4A" }}>🚨 {fmt(session.escalated_at)}</span>
              )}
            </div>
          </div>
          <button onClick={onClose} style={{
            background: "none", border: "none", fontSize: 20,
            cursor: "pointer", color: "#888", lineHeight: 1,
          }}>✕</button>
        </div>

        {/* Chat transcript */}
        <div style={{ flex: 1, overflowY: "auto", padding: "14px 16px" }}>
          {loading && (
            <div style={{ textAlign: "center", color: "#bbb", padding: 40 }}>Loading…</div>
          )}
          {!loading && (!history?.chat_history?.length) && (
            <div style={{ textAlign: "center", color: "#bbb", padding: 40 }}>No messages yet</div>
          )}
          {history?.chat_history?.map((msg, i) => (
            <ChatBubble key={i} msg={msg} />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Escalation notice at bottom */}
        {session.escalated && (
          <div style={{
            padding: "10px 16px", background: "#FCEBEB",
            borderTop: "1px solid #E24B4A40",
            fontSize: 12, color: "#A32D2D", fontWeight: 500,
            borderRadius: "0 0 16px 16px", textAlign: "center",
          }}>
            🚨 This session was escalated to manager
            {session.escalated_at ? ` at ${fmtTime(session.escalated_at)}` : ""}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Session card (inside customer accordion) ──────────────────────────────────
function SessionCard({ session, sessionNumber, onClick }) {
  const esc = session.escalated
  return (
    <div onClick={onClick} style={{
      background: "#fff",
      border: `1.5px solid ${esc ? "#E24B4A60" : "#e8e6e0"}`,
      borderRadius: 10, padding: "10px 14px", marginBottom: 6,
      cursor: "pointer", transition: "box-shadow 0.15s",
    }}
      onMouseEnter={e => e.currentTarget.style.boxShadow = "0 2px 12px rgba(0,0,0,0.10)"}
      onMouseLeave={e => e.currentTarget.style.boxShadow = "none"}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{
            fontSize: 11, fontWeight: 700, color: esc ? "#E24B4A" : "#1D9E75",
            background: esc ? "#FCEBEB" : "#EAF3DE",
            padding: "2px 8px", borderRadius: 20,
          }}>Session {sessionNumber}</span>
          {esc && (
            <span style={{
              background: "#E24B4A", color: "#fff", fontSize: 9,
              padding: "1px 7px", borderRadius: 20, fontWeight: 700,
            }}>🚨 ESCALATED</span>
          )}
          <span style={{ fontSize: 11, color: "#bbb" }}>
            {fmtDate(session.created_at)} {fmtTime(session.created_at)}
          </span>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {session.order_id && (
            <span style={{
              fontSize: 10, color: "#1D9E75", background: "#EAF3DE",
              padding: "1px 7px", borderRadius: 20, fontWeight: 600,
            }}>📦 {session.order_id}</span>
          )}
          <span style={{
            fontSize: 10, color: "#888", background: "#f3f2ef",
            padding: "1px 7px", borderRadius: 20,
          }}>{session.message_count || 0} msgs</span>
        </div>
      </div>
      {session.last_message && (
        <div style={{
          fontSize: 12, color: "#5F5E5A", marginTop: 5,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          {session.last_message}
        </div>
      )}
      {session.last_emotion && (
        <div style={{ marginTop: 5 }}>
          <EmotionDot emotion={session.last_emotion} intensity={session.ai_intensity} />
        </div>
      )}
    </div>
  )
}

// ── Customer accordion ────────────────────────────────────────────────────────
function CustomerAccordion({ name, sessions, onSelectSession }) {
  const [open, setOpen] = useState(false)
  const hasEscalation   = sessions.some(s => s.escalated)
  const sessionCount    = sessions.length
  const lastSeen        = sessions.reduce((a, b) =>
    (a.created_at > b.created_at ? a : b)).created_at

  return (
    <div style={{
      border: `1.5px solid ${hasEscalation ? "#E24B4A60" : "#e8e6e0"}`,
      borderRadius: 12, marginBottom: 10, overflow: "hidden",
      background: hasEscalation ? "#FFF8F8" : "#fff",
    }}>
      {/* Header row */}
      <div onClick={() => setOpen(o => !o)} style={{
        padding: "12px 16px", cursor: "pointer",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: "50%",
            background: hasEscalation ? "#FCEBEB" : "#EAF3DE",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 700, color: hasEscalation ? "#E24B4A" : "#1D9E75",
          }}>
            {(name || "G")[0].toUpperCase()}
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <span style={{ fontWeight: 700, fontSize: 14, color: "#2C2C2A" }}>{name || "Guest"}</span>
              {hasEscalation && (
                <span style={{
                  background: "#E24B4A", color: "#fff", fontSize: 9,
                  padding: "1px 7px", borderRadius: 20, fontWeight: 700,
                }}>🚨 ESCALATED</span>
              )}
            </div>
            <div style={{ fontSize: 11, color: "#888" }}>
              {sessionCount} session{sessionCount !== 1 ? "s" : ""} · Last active {fmt(lastSeen)}
            </div>
          </div>
        </div>
        <span style={{ fontSize: 14, color: "#bbb", transform: open ? "rotate(180deg)" : "none", transition: "0.2s" }}>▼</span>
      </div>

      {/* Sessions list */}
      {open && (
        <div style={{ padding: "0 14px 12px", borderTop: "1px solid #f0eeea" }}>
          <div style={{ height: 8 }} />
          {sessions
            .slice()
            .sort((a, b) => a.created_at.localeCompare(b.created_at))
            .map((s, idx) => (
              <SessionCard
                key={s.session_id}
                session={s}
                sessionNumber={idx + 1}
                onClick={() => onSelectSession({ ...s, sessionNumber: idx + 1 })}
              />
            ))}
        </div>
      )}
    </div>
  )
}

// ── Escalations tab ───────────────────────────────────────────────────────────
function EscalationsTab({ API, onSelectSession }) {
  const [escalations, setEscalations] = useState([])
  const [loading, setLoading]         = useState(true)

  useEffect(() => {
    const load = () =>
      fetch(`${API}/dashboard/escalations`)
        .then(r => r.json())
        .then(d => { setEscalations(d); setLoading(false) })
        .catch(() => setLoading(false))
    load()
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [])

  if (loading) return (
    <div style={{ textAlign: "center", color: "#bbb", padding: 60, fontSize: 14 }}>Loading…</div>
  )

  if (!escalations.length) return (
    <div style={{ textAlign: "center", padding: 60 }}>
      <div style={{ fontSize: 40, marginBottom: 12 }}>✅</div>
      <div style={{ color: "#5F5E5A", fontWeight: 600 }}>No escalations</div>
      <div style={{ color: "#bbb", fontSize: 12, marginTop: 4 }}>All customers are satisfied</div>
    </div>
  )

  return (
    <div style={{ padding: "16px 16px" }}>
      <div style={{
        fontSize: 12, fontWeight: 600, color: "#E24B4A",
        marginBottom: 14, display: "flex", alignItems: "center", gap: 6,
      }}>
        🚨 {escalations.length} ESCALATED SESSION{escalations.length > 1 ? "S" : ""} REQUIRE ATTENTION
      </div>
      {escalations.map(e => (
        <div key={e.session_id} onClick={() => onSelectSession(e)} style={{
          background: "#fff", border: "2px solid #E24B4A60",
          borderRadius: 12, padding: "14px 16px", marginBottom: 10,
          cursor: "pointer", transition: "box-shadow 0.15s",
        }}
          onMouseEnter={el => el.currentTarget.style.boxShadow = "0 2px 12px rgba(226,75,74,0.15)"}
          onMouseLeave={el => el.currentTarget.style.boxShadow = "none"}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: 22 }}>🚨</span>
                <span style={{ fontWeight: 700, fontSize: 15, color: "#2C2C2A" }}>
                  {e.customer || "Guest"}
                </span>
              </div>
              {e.order_id && (
                <span style={{
                  fontSize: 11, color: "#1D9E75", background: "#EAF3DE",
                  padding: "2px 8px", borderRadius: 20, fontWeight: 600, marginBottom: 6, display: "inline-block",
                }}>📦 {e.order_id}</span>
              )}
              <div style={{ fontSize: 12, color: "#5F5E5A", marginTop: 6, maxWidth: 320 }}>
                "{e.last_message}"
              </div>
            </div>
            <div style={{ textAlign: "right", flexShrink: 0 }}>
              {e.last_emotion && (
                <EmotionDot emotion={e.last_emotion} intensity={e.last_intensity} />
              )}
              <div style={{ fontSize: 10, color: "#E24B4A", marginTop: 4, fontWeight: 600 }}>
                {fmt(e.escalated_at)}
              </div>
            </div>
          </div>
          <div style={{
            marginTop: 10, paddingTop: 8, borderTop: "1px solid #E24B4A20",
            fontSize: 11, color: "#888",
          }}>
            Click to view full conversation →
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Stats bar ─────────────────────────────────────────────────────────────────
function StatsBar({ API }) {
  const [stats, setStats] = useState(null)
  useEffect(() => {
    const load = () => fetch(`${API}/dashboard/stats`).then(r => r.json()).then(setStats).catch(() => {})
    load()
    const t = setInterval(load, 8000)
    return () => clearInterval(t)
  }, [])
  if (!stats) return null
  return (
    <div style={{
      display: "flex", gap: 0, borderBottom: "1px solid #e8e6e0",
      background: "#fff", flexShrink: 0,
    }}>
      {[
        { label: "Total", value: stats.total_conversations, color: "#2C2C2A" },
        { label: "Messages", value: stats.total_messages, color: "#378ADD" },
        { label: "Escalated", value: stats.escalated, color: "#E24B4A" },
      ].map((s, i) => (
        <div key={i} style={{
          flex: 1, padding: "10px 0", textAlign: "center",
          borderRight: i < 2 ? "1px solid #e8e6e0" : "none",
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: s.color }}>{s.value}</div>
          <div style={{ fontSize: 10, color: "#bbb", marginTop: 1 }}>{s.label}</div>
        </div>
      ))}
    </div>
  )
}

// ── Main DashboardView ────────────────────────────────────────────────────────
export default function DashboardView({ API }) {
  const [tab, setTab]               = useState("customers") // "customers" | "escalations"
  const [sessions, setSessions]     = useState([])
  const [loading, setLoading]       = useState(true)
  const [selectedSession, setSelectedSession] = useState(null)

  useEffect(() => {
    const load = () =>
      fetch(`${API}/dashboard/sessions`)
        .then(r => r.json())
        .then(d => { setSessions(d); setLoading(false) })
        .catch(() => setLoading(false))
    load()
    const t = setInterval(load, 8000)
    return () => clearInterval(t)
  }, [])

  // Group sessions by customer name
  const byCustomer = sessions.reduce((acc, s) => {
    const key = s.customer || "Guest"
    if (!acc[key]) acc[key] = []
    acc[key].push(s)
    return acc
  }, {})

  // Sort: customers with escalations first, then by last activity
  const sortedCustomers = Object.entries(byCustomer).sort(([, a], [, b]) => {
    const aEsc = a.some(s => s.escalated)
    const bEsc = b.some(s => s.escalated)
    if (aEsc !== bEsc) return aEsc ? -1 : 1
    const aLast = Math.max(...a.map(s => new Date(s.created_at)))
    const bLast = Math.max(...b.map(s => new Date(s.created_at)))
    return bLast - aLast
  })

  const escalationCount = sessions.filter(s => s.escalated).length

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#fafaf9" }}>

      {/* Dashboard header */}
      <div style={{
        padding: "12px 16px 0", background: "#fff",
        borderBottom: "1px solid #e8e6e0", flexShrink: 0,
      }}>
        <div style={{ fontWeight: 700, fontSize: 15, color: "#2C2C2A", marginBottom: 10 }}>
          🛡️ Manager Dashboard
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {[
            { id: "customers",   label: "👥 Customers" },
            { id: "escalations", label: `🚨 Escalations${escalationCount > 0 ? ` (${escalationCount})` : ""}` },
          ].map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: "none", border: "none", cursor: "pointer",
              padding: "7px 14px", fontSize: 13, fontWeight: 600,
              color: tab === t.id ? "#1D9E75" : "#888",
              borderBottom: tab === t.id ? "2.5px solid #1D9E75" : "2.5px solid transparent",
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      {/* Stats bar */}
      <StatsBar API={API} />

      {/* Tab content */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {tab === "customers" && (
          <div style={{ padding: "16px 14px" }}>
            {loading && (
              <div style={{ textAlign: "center", color: "#bbb", padding: 60 }}>Loading…</div>
            )}
            {!loading && sortedCustomers.length === 0 && (
              <div style={{ textAlign: "center", color: "#bbb", padding: 60 }}>
                No sessions yet
              </div>
            )}
            {sortedCustomers.map(([name, userSessions]) => (
              <CustomerAccordion
                key={name}
                name={name}
                sessions={userSessions}
                onSelectSession={setSelectedSession}
              />
            ))}
          </div>
        )}
        {tab === "escalations" && (
          <EscalationsTab API={API} onSelectSession={setSelectedSession} />
        )}
      </div>

      {/* Session transcript modal */}
      {selectedSession && (
        <SessionModal
          session={selectedSession}
          API={API}
          onClose={() => setSelectedSession(null)}
        />
      )}
    </div>
  )
}