import React, { useState, useEffect, useRef, useCallback } from 'react'
import MessageBubble from './MessageBubble'
import EmotionBadge, { EMOTION_CONFIG } from './EmotionBadge'
import IntensityBar from './IntensityBar'
import RecCard from './RecCard'

const INTENTS = [
  { id: "track_order", label: "📦 Track Order" },
  { id: "late_order",  label: "⏱️ Late Order"  },
  { id: "wrong_item",  label: "❌ Wrong Item"  },
  { id: "refund",      label: "💸 Refund"      },
]

// ── Order status card ─────────────────────────────────────────────────────────
function OrderCard({ order }) {
  if (!order) return null

  // Handle different order types based on intent
  const statusColor = {
    "Out for delivery": "#1D9E75",
    "Preparing": "#BA7517",
    "Delivered": "#378ADD",
    "Order Confirmed": "#6C5CE7",
    "Significantly Delayed": "#E24B4A",
    "Refund Requested": "#FF6B6B",
  }[order.status] || "#888"

  if (!order.found) {
    return (
      <div style={{
        background: "#FCEBEB", border: "1px solid #E24B4A40",
        borderRadius: 12, padding: "12px 14px", margin: "4px 0",
      }}>
        <div style={{ fontWeight: 600, fontSize: 13, color: "#A32D2D" }}>Order not found</div>
        <div style={{ fontSize: 12, color: "#5F5E5A", marginTop: 3 }}>{order.message}</div>
      </div>
    )
  }

  return (
    <div style={{
      background: "#fff", border: "1px solid #e8e6e0",
      borderRadius: 12, padding: "12px 14px", margin: "4px 0",
      boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
    }}>
      {/* Header with Order ID and Status */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center", 
        marginBottom: 8 
      }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: "#888", letterSpacing: 0.4 }}>
          {order.order_id || "Order"}
        </span>
        <span style={{
          fontSize: 11, fontWeight: 600, color: statusColor,
          background: `${statusColor}18`, padding: "2px 8px", borderRadius: 20,
        }}>{order.status}</span>
      </div>

      {/* Customer Name */}
      {order.customer && (
        <div style={{ fontSize: 12, color: "#5F5E5A", marginBottom: 6 }}>
          👤 Customer: <strong>{order.customer}</strong>
        </div>
      )}

      {/* ETA - only if exists */}
      {order.eta && (
        <div style={{ fontSize: 13, color: "#2C2C2A", marginBottom: 6 }}>
          ⏱️ Estimated arrival: <strong>{order.eta}</strong>
        </div>
      )}

      {/* Delay information for late orders */}
      {order.intent === "late_order" && order.promised_eta && (
        <div style={{ fontSize: 12, color: "#E24B4A", marginBottom: 6 }}>
          ⚠️ Originally promised: <strong>{order.promised_eta}</strong>
          {order.delay_reason && ` · ${order.delay_reason}`}
        </div>
      )}

      {/* Timestamp */}
      {order.timestamp && (
        <div style={{ fontSize: 11, color: "#aaa", marginBottom: 6 }}>
          🕐 {new Date(order.timestamp).toLocaleString()}
        </div>
      )}

      {/* Items - handles different formats */}
      {renderItems(order)}

      {/* Total */}
      {order.total && (
        <div style={{ fontSize: 12, color: "#888", marginTop: 6 }}>
          Total: ${order.total.toFixed(2)}
        </div>
      )}

      {/* Refund information */}
      {order.intent === "refund" && order.refund_status && (
        <div style={{ 
          fontSize: 12, 
          color: order.refund_status === "Approved" ? "#1D9E75" : "#E67E22",
          marginTop: 6,
          padding: "4px 8px",
          background: order.refund_status === "Approved" ? "#E8F5E9" : "#FFF3E0",
          borderRadius: 4,
        }}>
          💰 Refund: <strong>{order.refund_status}</strong>
          {order.refund_reason && ` · ${order.refund_reason}`}
        </div>
      )}

      {/* Wrong item information */}
      {order.intent === "wrong_item" && order.issue && (
        <div style={{ 
          fontSize: 12, 
          color: "#E24B4A",
          marginTop: 6,
          padding: "4px 8px",
          background: "#FCEBEB",
          borderRadius: 4,
        }}>
          ⚠️ {order.issue}
        </div>
      )}
    </div>
  )
}

// Helper function to render items based on order type
function renderItems(order) {
  // For wrong items - show ordered vs received
  if (order.intent === "wrong_item" && order.items_ordered && order.items_received) {
    return (
      <div style={{ marginBottom: 6 }}>
        <div style={{ fontSize: 12, color: "#5F5E5A", marginBottom: 4 }}>
          🛒 Ordered:
          <ul style={{ margin: "2px 0 4px 16px", padding: 0 }}>
            {order.items_ordered.map((item, idx) => (
              <li key={`ordered-${idx}`} style={{ fontSize: 12, color: "#2C2C2A" }}>
                {item.name} (${item.price.toFixed(2)})
              </li>
            ))}
          </ul>
        </div>
        <div style={{ fontSize: 12, color: "#E24B4A" }}>
          ❌ Received:
          <ul style={{ margin: "2px 0 0 16px", padding: 0 }}>
            {order.items_received.map((item, idx) => (
              <li key={`received-${idx}`} style={{ fontSize: 12, color: "#2C2C2A" }}>
                {item.name} (${item.price.toFixed(2)})
              </li>
            ))}
          </ul>
        </div>
      </div>
    )
  }

  // For regular orders with items array
  if (order.items && Array.isArray(order.items) && order.items.length > 0) {
    return (
      <div style={{ fontSize: 12, color: "#5F5E5A", marginBottom: 6 }}>
        <div style={{ fontWeight: 600, marginBottom: 2 }}>Items:</div>
        {order.items.map((item, idx) => (
          <div key={idx} style={{ paddingLeft: 8 }}>
            • {item.name} {item.price && `($${item.price.toFixed(2)})`}
          </div>
        ))}
      </div>
    )
  }

  return null
}

// ── Intent pill buttons ───────────────────────────────────────────────────────
function IntentButtons({ onSelect, disabled }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "6px 0 2px" }}>
      {INTENTS.map(intent => (
        <button key={intent.id} onClick={() => onSelect(intent)} disabled={disabled} style={{
          background: "#EAF3DE", color: "#3B6D11", border: "1px solid #C0DD97",
          padding: "6px 12px", borderRadius: 16, fontSize: 12, fontWeight: 600,
          cursor: disabled ? "default" : "pointer", opacity: disabled ? 0.5 : 1,
        }}>{intent.label}</button>
      ))}
    </div>
  )
}

// ── Escalation card ───────────────────────────────────────────────────────────
function EscalationCard({ orderId, onNewIssue }) {
  return (
    <div style={{
      background: "#FCEBEB", border: "1.5px solid #E24B4A60",
      borderRadius: 14, padding: "20px 16px", margin: "8px 0", textAlign: "center",
    }}>
      <div style={{ fontSize: 32, marginBottom: 8 }}>🚨</div>
      <div style={{ fontWeight: 700, fontSize: 14, color: "#A32D2D", marginBottom: 6 }}>
        Manager Notified
      </div>
      <div style={{ fontSize: 12, color: "#5F5E5A", lineHeight: 1.6, marginBottom: 14 }}>
        A manager will contact you within <strong>15 minutes</strong>
        {orderId ? ` regarding order <strong>${orderId}</strong>` : ""}.
        <br />This conversation has been escalated.
      </div>
      <button onClick={onNewIssue} style={{
        background: "#1D9E75", color: "#fff", border: "none",
        padding: "9px 20px", borderRadius: 10, fontSize: 13,
        fontWeight: 600, cursor: "pointer",
      }}>
        🔄 Start New Issue
      </button>
    </div>
  )
}

// ── Recommendations toggle ────────────────────────────────────────────────────
function RecsPanel({ recs }) {
  const [open, setOpen] = useState(false)
  if (!recs || recs.length === 0) return null
  return (
    <div style={{ marginTop: 6 }}>
      <button onClick={() => setOpen(o => !o)} style={{
        background: "transparent", border: "1px solid #C0DD97",
        color: "#3B6D11", fontSize: 11, fontWeight: 600,
        padding: "4px 10px", borderRadius: 12, cursor: "pointer",
      }}>
        {open ? "▲ Hide suggestions" : `💡 ${recs.length} suggestion${recs.length > 1 ? "s" : ""}`}
      </button>
      {open && (
        <div style={{ marginTop: 6 }}>
          {recs.map((r, i) => <RecCard key={i} rec={r} />)}
        </div>
      )}
    </div>
  )
}

// ── Main ChatView ─────────────────────────────────────────────────────────────
function newSessionId() {
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
}

export default function ChatView({ API }) {
  const [sessionId, setSessionId] = useState(() => newSessionId())
  const [phase, setPhase]         = useState("greeting")
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState("")
  const [loading, setLoading]     = useState(false)
  const [lastMeta, setLastMeta]   = useState(null)
  const [escalated, setEscalated] = useState(false)
  const [name, setName]           = useState("")
  const [nameError, setNameError] = useState(false)
  const [started, setStarted]     = useState(false)
  const [showIntents, setShowIntents] = useState(false)
  const [orderAttempts, setOrderAttempts] = useState(0)

  // Use refs for values needed inside handleResponse to avoid stale closures
  const orderFoundRef   = useRef(false)
  const currentOrderRef = useRef(null)
  const escalatedRef    = useRef(false)

  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  // Show greeting + intent buttons when chat starts
  useEffect(() => {
    if (started && messages.length === 0) {
      const greeting = name
        ? `Hi ${name}! 👋 Welcome to XO.tikka. How can I help you today?`
        : `Hi there! 👋 Welcome to XO.tikka. How can I help you today?`
      setMessages([{ role: "assistant", content: greeting }])
      setShowIntents(true)
    }
  }, [started])

  // ── Start new issue = new session ID (shows as Session 2, 3... in dashboard) ──
  const startNewIssue = useCallback(() => {
    const freshId = newSessionId()
    setSessionId(freshId)
    orderFoundRef.current   = false
    currentOrderRef.current = null
    escalatedRef.current    = false
    setEscalated(false)
    setPhase("greeting")
    setLastMeta(null)
    setOrderAttempts(0)
    setMessages([{ role: "assistant", content: `Hi ${name}! 👋 What else can I help you with?` }])
    setShowIntents(true)
  }, [name])

  // ── handleResponse ────────────────────────────────────────────────────────
  const handleResponse = useCallback((data, userMsgIndex) => {
    setPhase(data.phase)
    setOrderAttempts(data.order_attempts || 0)

    // Emotion meta for the live bar
    if (data.emotion) {
      setLastMeta({
        emotion:   data.emotion,
        intensity: data.intensity,
        mode:      data.mode,
        vector:    data.vector,
      })
    }

    const isNewEscalation = data.escalated && !escalatedRef.current
    if (isNewEscalation) {
      escalatedRef.current = true
      setEscalated(true)
    }

    // Was this the first time we got a found order card?
    const isNewOrderFound = data.order_card?.found && !orderFoundRef.current
    if (isNewOrderFound) {
      orderFoundRef.current   = true
      currentOrderRef.current = data.order_card.order_id
    }

    setMessages(prev => {
      const updated = [...prev]

      // 1. Attach emotion meta to the user message (only in emotional phase)
      if (data.emotion && data.phase === "emotional" && userMsgIndex !== undefined) {
        updated[userMsgIndex] = {
          ...updated[userMsgIndex],
          emotionMeta: {
            emotion:   data.emotion,
            intensity: data.intensity,
          },
        }
      }

      // 2. Bot response bubble
      if (isNewEscalation) {
        // Final bot message before escalation card
        if (data.response) {
          updated.push({ role: "assistant", content: data.response })
        }
        // Then the escalation card as a special message
        updated.push({
          role:         "assistant",
          isEscalation: true,
          orderId:      currentOrderRef.current,
        })
      } else if (isNewOrderFound) {
        // Order just found — show card only, skip the backend placeholder text
        updated.push({
          role:        "assistant",
          content:     null,           // no text bubble, just the card
          orderCard:   data.order_card,
          recs:        data.recommendations || [],
        })
      } else if (data.response) {
        updated.push({
          role:      "assistant",
          content:   data.response,
          orderCard: null,
          recs:      data.recommendations || [],
        })
      }

      return updated
    })
  }, [])

  // ── Generic API call ──────────────────────────────────────────────────────
  const callAPI = useCallback(async (payload, userMsgIndex) => {
    try {
      const res = await fetch(`${API}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_id:    sessionId,
          customer_name: name || undefined,
          order_id:      currentOrderRef.current || undefined,
          ...payload,
        }),
      })
      const data = await res.json()
      handleResponse(data, userMsgIndex)
    } catch {
      setMessages(prev => [...prev, {
        role:    "assistant",
        content: "Sorry, I'm having trouble connecting. Please try again.",
      }])
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [sessionId, name, API, handleResponse])

  // ── send (free text) ──────────────────────────────────────────────────────
  const send = useCallback(async (text) => {
    if (!text.trim() || loading) return
    setInput("")
    setShowIntents(false)

    const userMsgIndex = messages.length  // capture BEFORE setState
    setMessages(prev => [...prev, { role: "user", content: text }])
    setLoading(true)

    await callAPI({ message: text }, userMsgIndex)
  }, [loading, messages.length, callAPI])

  // ── selectIntent ──────────────────────────────────────────────────────────
  const selectIntent = useCallback(async (intent) => {
    if (loading) return
    setShowIntents(false)

    const userMsgIndex = messages.length
    setMessages(prev => [...prev, { role: "user", content: intent.label }])
    setLoading(true)

    await callAPI({ message: intent.label, intent: intent.id }, userMsgIndex)
  }, [loading, messages.length, callAPI])

  // ── Welcome screen ─────────────────────────────────────────────────────────
  const handleStart = () => {
    if (!name.trim()) { setNameError(true); return }
    setNameError(false)
    setStarted(true)
  }

  if (!started) return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100%",
      justifyContent: "center", alignItems: "center", padding: 24,
      background: "linear-gradient(160deg, #EAF3DE 0%, #fff 60%)",
    }}>
      <div style={{ fontSize: 60, marginBottom: 12 }}>🍽️</div>
      <h1 style={{ fontSize: 24, color: "#2C2C2A", marginBottom: 6, textAlign: "center" }}>XO.tikka</h1>
      <p style={{ color: "#5F5E5A", marginBottom: 20, textAlign: "center", fontSize: 14, maxWidth: 260 }}>
        Our support bot is here to help with orders and feedback.
      </p>
      <input
        placeholder="Your name *"
        value={name}
        onChange={e => { setName(e.target.value); if (e.target.value.trim()) setNameError(false) }}
        onKeyDown={e => e.key === "Enter" && handleStart()}
        style={{
          width: "100%", maxWidth: 280, padding: "10px 14px",
          border: `1.5px solid ${nameError ? "#E24B4A" : "#d3d1c7"}`,
          borderRadius: 10, fontSize: 14, outline: "none",
          marginBottom: 4, textAlign: "center",
          background: nameError ? "#FFF5F5" : "#fff",
        }}
      />
      {nameError && (
        <div style={{ fontSize: 12, color: "#E24B4A", marginBottom: 8 }}>
          Please enter your name to continue
        </div>
      )}
      <div style={{ height: nameError ? 2 : 10 }} />
      <button onClick={handleStart} style={{
        background: "#1D9E75", color: "#fff", border: "none",
        padding: "11px 0", borderRadius: 10, fontSize: 14,
        fontWeight: 600, cursor: "pointer", width: "100%", maxWidth: 280,
      }}>
        Start Chat
      </button>
    </div>
  )

  // ── Main chat UI ───────────────────────────────────────────────────────────
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>

      {/* Header */}
      <div style={{
        padding: "11px 16px", borderBottom: "1px solid #e8e6e0", background: "#fff",
        display: "flex", alignItems: "center", gap: 10, flexShrink: 0,
      }}>
        <div style={{
          width: 34, height: 34, borderRadius: "50%", background: "#1D9E75",
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16,
        }}>🍽️</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: "#2C2C2A" }}>XO.tikka Support</div>
          <div style={{ fontSize: 10.5, color: escalated ? "#E24B4A" : "#1D9E75", fontWeight: 500 }}>
            {escalated ? "🚨 Escalated" : "● Online"}
          </div>
        </div>
        {escalated && (
          <span style={{
            background: "#FCEBEB", color: "#A32D2D", fontSize: 10,
            padding: "3px 9px", borderRadius: 20, fontWeight: 700,
            border: "1px solid #E24B4A40",
          }}>🚨 Manager Notified</span>
        )}
        {orderAttempts > 0 && phase === "guided" && (
          <span style={{
            background: "#FFF3E0", color: "#E67E22", fontSize: 10,
            padding: "3px 9px", borderRadius: 20, fontWeight: 600,
          }}>Attempt {orderAttempts}</span>
        )}
      </div>

      {/* Emotion live bar — only in emotional phase */}
      {lastMeta && phase === "emotional" && (
        <div style={{
          padding: "6px 16px", background: "#fafaf8",
          borderBottom: "1px solid #eee",
          display: "flex", alignItems: "center", gap: 10, flexShrink: 0,
        }}>
          <EmotionBadge emotion={lastMeta.emotion} small />
          <div style={{ flex: 1 }}>
            <IntensityBar
              value={lastMeta.intensity}
              color={EMOTION_CONFIG[lastMeta.emotion]?.color || "#888"}
            />
          </div>
          <span style={{ fontSize: 9.5, color: "#ccc", whiteSpace: "nowrap" }}>
            {lastMeta.mode} · {lastMeta.vector}
          </span>
        </div>
      )}

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "14px 12px", background: "#fafaf9" }}>
        {messages.map((m, i) => (
          <div key={i}>

            {m.role === "user" ? (
              /* ── USER MESSAGE ── */
              <div>
                <MessageBubble msg={m} />
                {/* Emotion classification shown below user's message */}
                {m.emotionMeta && (
                  <div style={{
                    display: "flex", justifyContent: "flex-end",
                    gap: 6, alignItems: "center",
                    paddingRight: 4, marginTop: 2, marginBottom: 8,
                  }}>
                    <EmotionBadge emotion={m.emotionMeta.emotion} small />
                    <div style={{ width: 80 }}>
                      <IntensityBar
                        value={m.emotionMeta.intensity}
                        color={EMOTION_CONFIG[m.emotionMeta.emotion]?.color || "#888"}
                      />
                    </div>
                    <span style={{ fontSize: 10, color: "#bbb" }}>
                      {Math.round(m.emotionMeta.intensity * 100)}%
                    </span>
                  </div>
                )}
              </div>

            ) : m.isEscalation ? (
              /* ── ESCALATION CARD ── */
              <div style={{ marginBottom: 10 }}>
                <EscalationCard orderId={m.orderId} onNewIssue={startNewIssue} />
              </div>

            ) : (
              /* ── BOT MESSAGE ── */
              <div>
                {/* Only render text bubble if there's actual content */}
                {m.content && <MessageBubble msg={m} />}
                {/* Order card — only on the ONE bot message that revealed it */}
                {m.orderCard?.found && (
                  <div style={{ paddingLeft: 42, paddingRight: 8, marginBottom: 4 }}>
                    <OrderCard order={m.orderCard} />
                  </div>
                )}
                {/* Recommendations as collapsible — per bot message */}
                {m.recs?.length > 0 && (
                  <div style={{ paddingLeft: 42, paddingRight: 8, marginBottom: 6 }}>
                    <RecsPanel recs={m.recs} />
                  </div>
                )}
              </div>
            )}

            {/* Intent buttons after the very first greeting message */}
            {i === 0 && m.role === "assistant" && showIntents && phase === "greeting" && (
              <div style={{ paddingLeft: 42, marginBottom: 10 }}>
                <IntentButtons onSelect={selectIntent} disabled={loading} />
              </div>
            )}

          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div style={{ display: "flex", gap: 8, alignItems: "flex-end", marginBottom: 10 }}>
            <div style={{
              width: 30, height: 30, borderRadius: "50%", background: "#1D9E75",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13,
            }}>🍽️</div>
            <div style={{
              padding: "8px 12px", background: "#fff", border: "1px solid #e8e6e0",
              borderRadius: "16px 16px 16px 3px", display: "flex", gap: 4, alignItems: "center",
            }}>
              {[0, 1, 2].map(j => (
                <div key={j} style={{
                  width: 5, height: 5, borderRadius: "50%", background: "#1D9E75",
                  animation: `bounce 1.2s ease-in-out ${j * 0.2}s infinite`,
                }} />
              ))}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar — hidden after escalation */}
      {phase !== "greeting" && !escalated && (
        <div style={{
          padding: "10px 12px", background: "#fff",
          borderTop: "1px solid #e8e6e0", display: "flex", gap: 8, flexShrink: 0,
        }}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && send(input)}
            placeholder={
              phase === "guided"
                ? "Enter your order ID (e.g. ORD-1234)…"
                : "Type your message…"
            }
            disabled={loading}
            style={{
              flex: 1, padding: "9px 13px", border: "1.5px solid #d3d1c7",
              borderRadius: 20, fontSize: 13.5, outline: "none",
              background: loading ? "#fafaf8" : "#fff",
            }}
          />
          <button
            onClick={() => send(input)}
            disabled={loading || !input.trim()}
            style={{
              width: 40, height: 40, borderRadius: "50%",
              background: loading || !input.trim() ? "#d3d1c7" : "#1D9E75",
              color: "#fff", border: "none",
              cursor: loading || !input.trim() ? "default" : "pointer",
              fontSize: 17, display: "flex", alignItems: "center",
              justifyContent: "center", flexShrink: 0,
            }}
          >➤</button>
        </div>
      )}

      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-5px); }
        }
      `}</style>
    </div>
  )
}