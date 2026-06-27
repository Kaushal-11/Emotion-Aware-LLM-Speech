"""
backend/main.py — Restaurant Support Bot API with complete logging
"""

import sys
import json
import re
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    RESTAURANT_NAME,
    RECOMMENDER_KB_PATH,
    RECOMMENDER_MODEL_NAME,
    RECOMMENDER_TOP_K,
    ESCALATION_ANGER_THRESHOLD,
    ESCALATION_TURNS_REQUIRED,
    CLASSIFIER_DIR,
    DEVICE,
)

# ── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Import actual pipeline components ──────────────────────────────────────
from core.classifier import TextEmotionClassifier
from core.state_memory import StateMemory
from core.decision_engine import DecisionEngine
from core.llm import SteeredLLM
from core.recommender import Recommender


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Restaurant Support Bot API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load pipeline components ─────────────────────────────────────────────────

logger.info("=" * 70)
logger.info("Starting Restaurant Support Bot API")
logger.info("=" * 70)

logger.info("[1/5] Loading Text Emotion Classifier ...")
try:
    classifier = TextEmotionClassifier(checkpoint_dir=CLASSIFIER_DIR, device=DEVICE)
    logger.info("[1/5] ✅ Classifier loaded")
except Exception as e:
    logger.error(f"[1/5] ❌ Classifier failed: {e}")
    sys.exit(1)

logger.info("[2/5] Initializing State Memory ...")
state_memory = StateMemory()
logger.info("[2/5] ✅ State Memory ready")

logger.info("[3/5] Initializing Decision Engine ...")
decision_engine = DecisionEngine()
logger.info("[3/5] ✅ Decision Engine ready")

logger.info("[4/5] Loading Steered LLM ...")
try:
    llm = SteeredLLM()
    logger.info("[4/5] ✅ Steered LLM ready")
except Exception as e:
    logger.error(f"[4/5] ❌ SteeredLLM failed: {e}")
    sys.exit(1)

logger.info("[5/5] Loading Recommender ...")
try:
    recommender = Recommender(
        model_name=RECOMMENDER_MODEL_NAME,
        kb_path=RECOMMENDER_KB_PATH,
        top_k=RECOMMENDER_TOP_K,
    )
    logger.info("[5/5] ✅ Recommender ready")
except Exception as e:
    logger.error(f"[5/5] ❌ Recommender failed: {e}")
    sys.exit(1)

logger.info("=" * 70)
logger.info("🚀 All components loaded successfully!")
logger.info("=" * 70)


# ── Session store ─────────────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_conversation_logs: list[dict] = []


def get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session_id":    session_id,
            "state_memory":  StateMemory(),
            "anger_streak":  0,
            "escalated":     False,
            "created_at":    datetime.utcnow().isoformat(),
            "customer_name": None,
            "phase":         "greeting",
            "intent":        None,
            "order_id":      None,
            "last_message":  "",
            "order_attempts": 0,
            "chat_history":  [],   # [{role, content, emotion, intensity, timestamp}]
            "escalated_at":  None,
        }
        logger.info(f"[Session] Created: {session_id}")
    return _sessions[session_id]


# ── Load orders from JSON file ──────────────────────────────────────────────

ORDERS_FILE = Path(__file__).parent / "data" / "orders_db.json"

def load_orders_from_file():
    try:
        with open(ORDERS_FILE, 'r') as f:
            data = json.load(f)
            orders = data.get("orders", {})
            logger.info(f"[Orders] Loaded {len(orders)} orders")
            return orders
    except FileNotFoundError:
        logger.error(f"[Orders] File not found at {ORDERS_FILE}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"[Orders] Invalid JSON: {e}")
        return {}

MOCK_ORDERS = load_orders_from_file()

def normalize_order_data(order_data: dict) -> dict:
    """Ensure order data has all required fields for frontend"""
    normalized = order_data.copy()
    
    # Ensure items is always a list
    if "items" not in normalized:
        normalized["items"] = []
    elif not isinstance(normalized["items"], list):
        normalized["items"] = []
    
    # For wrong_item orders, ensure arrays exist
    if normalized.get("intent") == "wrong_item":
        if "items_ordered" not in normalized:
            normalized["items_ordered"] = []
        if "items_received" not in normalized:
            normalized["items_received"] = []
    
    # Ensure total is a number
    if "total" in normalized:
        try:
            normalized["total"] = float(normalized["total"])
        except:
            normalized["total"] = 0
    else:
        normalized["total"] = 0
    
    # Ensure status exists
    if "status" not in normalized:
        normalized["status"] = "Unknown"
    
    return normalized

def tool_order_status(order_id: str) -> dict:
    oid = order_id.upper().strip()
    if oid in MOCK_ORDERS:
        order_data = normalize_order_data(MOCK_ORDERS[oid])
        logger.info(f"[Order] ✅ Found: {oid}")
        return {"found": True, "order_id": oid, **order_data}
    
    logger.warning(f"[Order] ❌ Not found: {oid}")
    return {"found": False, "order_id": oid,
            "message": f"Order {oid} not found. Please check your order ID."}


def extract_order_id(text: str) -> Optional[str]:
    m = re.search(r'\bORD[-\s]?(\d{3,6})\b', text, re.IGNORECASE)
    if m:
        order_id = f"ORD-{m.group(1)}"
        logger.info(f"[Extract] Found: {order_id} in '{text}'")
        return order_id
    m2 = re.search(r'\b(\d{4,6})\b', text)
    if m2:
        order_id = f"ORD-{m2.group(1)}"
        logger.info(f"[Extract] Found: {order_id} in '{text}'")
        return order_id
    logger.info(f"[Extract] No order ID in '{text[:30]}...'")
    return None


# ── Guided phase prompts ─────────────────────────────────────────────────────

INTENT_PROMPTS = {
    "track_order":  "Sure! Please share your order ID (e.g. ORD-1234) and I'll look it up right away.",
    "late_order":   "I'm sorry your order is running late. Can you share your order ID so I can check the status?",
    "late":         "I'm sorry your order is running late. Can you share your order ID so I can check the status?",
    "wrong_item":   "I apologise for the mix-up! Please share your order ID and I'll look into this",
    "wrong":        "I apologise for the mix-up! Please share your order ID and I'll sort this out for you.",
    "refund":       "Of course, I'll help with your refund. Can you share your order ID to get started?",
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id:    Optional[str] = None
    message:       str
    customer_name: Optional[str] = None
    intent:        Optional[str] = None
    order_id:      Optional[str] = None


class OrderCard(BaseModel):
    order_id: str
    found: bool
    status: Optional[str] = None
    eta: Optional[str] = None
    items: Optional[list] = None
    total: Optional[float] = None
    message: Optional[str] = None
    customer: Optional[str] = None
    timestamp: Optional[str] = None
    intent: Optional[str] = None
    # Late order fields
    promised_eta: Optional[str] = None
    delay_reason: Optional[str] = None
    # Wrong item fields
    items_ordered: Optional[list] = None
    items_received: Optional[list] = None
    issue: Optional[str] = None
    # Refund fields
    refund_status: Optional[str] = None
    refund_reason: Optional[str] = None


class ChatResponse(BaseModel):
    session_id:      str
    response:        str
    phase:           str
    order_card:      Optional[OrderCard] = None
    emotion:         Optional[str]  = None
    target:          Optional[str]  = None
    intensity:       Optional[float] = None
    ai_emotion:      Optional[str]  = None
    ai_intensity:    Optional[float] = None
    mode:            Optional[str]  = None
    vector:          Optional[str]  = None
    vector_intensity: Optional[float] = None
    escalated:       bool           = False
    recommendations: list           = []
    turn:            int            = 0
    order_attempts:  int            = 0


# ── Helper to log messages ────────────────────────────────────────────────────

def log_message(role: str, content: str, session_id: str = None):
    """Log messages in a consistent format"""
    prefix = f"[{session_id[:8]}]" if session_id else ""
    logger.info(f"{prefix} {role}: {content[:100]}")


def append_chat(session: dict, role: str, content: str,
                emotion: str = None, intensity: float = None):
    """Append a message to session chat history."""
    session["chat_history"].append({
        "role":      role,           # "user" or "bot"
        "content":   content,
        "emotion":   emotion,
        "intensity": intensity,
        "timestamp": datetime.utcnow().isoformat(),
    })


# ── Main /chat endpoint ───────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session    = get_or_create_session(session_id)
    
    # Log the user message
    log_message("User", req.message, session_id)
    append_chat(session, "user", req.message)

    if req.customer_name:
        session["customer_name"] = req.customer_name

    session["last_message"] = req.message[:120]

    # ── PHASE 1: greeting → intent selected ──────────────────────────────────
    if req.intent and session["phase"] == "greeting":
        logger.info(f"[{session_id[:8]}] Phase: GREETING → GUIDED")
        session["phase"]  = "guided"
        session["intent"] = req.intent
        
        order_id = extract_order_id(req.message)
        if order_id:
            session["order_id"] = order_id
            order_data = tool_order_status(order_id)
            # Create OrderCard from the data
            order_card = OrderCard(**order_data)
            
            if order_card.found:
                logger.info(f"[{session_id[:8]}] Order found! Moving to EMOTIONAL")
                session["phase"] = "emotional"
                session["state_memory"].reset()
                
                # Log bot response (order card only, no text)
                log_message("Bot", f"📦 Order {order_id} details displayed", session_id)
                append_chat(session, "bot", f"📦 Order {order_id} details displayed")
                
                return ChatResponse(
                    session_id = session_id,
                    response   = "",  # Empty - frontend shows order card
                    phase      = "emotional",
                    order_card = order_card,
                    recommendations = [],
                )
            else:
                response = order_card.message or "Order not found. Please check your order ID."
                log_message("Bot", response, session_id)
                append_chat(session, "bot", response)
                return ChatResponse(
                    session_id = session_id,
                    response   = response,
                    phase      = "guided",
                    order_card = order_card,
                )
        
        prompt = INTENT_PROMPTS.get(req.intent, "Please share your order ID so I can help you.")
        log_message("Bot", prompt, session_id)
        append_chat(session, "bot", prompt)
        return ChatResponse(
            session_id = session_id,
            response   = prompt,
            phase      = "guided",
        )

    # ── PHASE 2: guided → collecting order ID ────────────────────────────────
    if session["phase"] == "guided":
        logger.info(f"[{session_id[:8]}] Phase: GUIDED - Collecting order ID")
        session["order_attempts"] += 1
        
        order_id = extract_order_id(req.message)
        
        if order_id:
            session["order_id"] = order_id
            order_data = tool_order_status(order_id)
            order_card = OrderCard(**order_data)
            
            if order_card.found:
                logger.info(f"[{session_id[:8]}] ✅ Order found! Moving to EMOTIONAL")
                session["phase"] = "emotional"
                session["order_attempts"] = 0
                session["state_memory"].reset()
                
                # Log bot response (order details)
                log_message("Bot", f"📦 Order {order_id} details displayed", session_id)
                append_chat(session, "bot", f"📦 Order {order_id} details displayed")
                
                # Return order card WITHOUT emotion pipeline
                # The frontend will show the card, and the user will react
                return ChatResponse(
                    session_id = session_id,
                    response   = "",  # Empty - frontend shows order card
                    phase      = "emotional",
                    order_card = order_card,
                    recommendations = [],
                )
            else:
                response = f"Order {order_id} not found. Please check your order ID and try again."
                log_message("Bot", response, session_id)
                append_chat(session, "bot", response)
                return ChatResponse(
                    session_id = session_id,
                    response   = response,
                    phase      = "guided",
                    order_card = order_card,
                    order_attempts = session["order_attempts"],
                )
        else:
            attempts = session["order_attempts"]
            if attempts >= 3:
                response = "I notice you're having trouble with the order ID. Please type it in the format ORD-1234"
            else:
                response = "I couldn't find an order ID in your message. Please share it in the format ORD-1234 (e.g., ORD-1234)."
            log_message("Bot", response, session_id)
            append_chat(session, "bot", response)
            return ChatResponse(
                session_id = session_id,
                response   = response,
                phase      = "guided",
                order_attempts = attempts,
            )

    # ── PHASE 3: emotional → FULL PIPELINE ──────────────────────────────────
    if session["phase"] == "emotional":
        logger.info(f"[{session_id[:8]}] Phase: EMOTIONAL - Running pipeline")
        mem = session["state_memory"]
        
        # ── 1. Text Classifier ──────────────────────────────────────────────
        logger.info(f"[{session_id[:8]}] Pipeline Step 1: Text Classifier")
        emotion, target, intensity = classifier.classify(req.message)
        logger.info(f"[{session_id[:8]}] Classifier → emotion: {emotion}, target: {target}, intensity: {intensity:.3f}")
        # Update user's chat entry with detected emotion
        if session["chat_history"] and session["chat_history"][-1]["role"] == "user":
            session["chat_history"][-1]["emotion"]   = emotion
            session["chat_history"][-1]["intensity"] = intensity
        
        # ── 2. State Memory ──────────────────────────────────────────────────
        logger.info(f"[{session_id[:8]}] Pipeline Step 2: State Memory")
        ai_state = mem.update(emotion, target, intensity)
        logger.info(f"[{session_id[:8]}] State Memory → ai_emotion: {ai_state.emotion}, ai_intensity: {ai_state.ai_intensity:.3f}, turn: {ai_state.turn}")
        
        # ── 3. Decision Engine ──────────────────────────────────────────────
        logger.info(f"[{session_id[:8]}] Pipeline Step 3: Decision Engine")
        decision = decision_engine.decide(emotion, target, intensity, ai_state)
        logger.info(f"[{session_id[:8]}] Decision → mode: {decision.mode}, vector: {decision.vector}, vector_intensity: {decision.vector_intensity:.3f}")
        
        # ── 4. Escalation check ──────────────────────────────────────────────

        # Explicit manager request — user says "call manager / talk to human" etc.
        MANAGER_KEYWORDS = [
            "manager", "supervisor", "human", "speak to someone",
            "talk to someone", "real person", "call me", "phone me",
            "escalate", "complaint", "complain",
        ]
        user_text_lower = req.message.lower()
        explicit_escalation = any(kw in user_text_lower for kw in MANAGER_KEYWORDS)

        if explicit_escalation:
            logger.info(f"[{session_id[:8]}] 🚨 Explicit manager request detected")

        # Threshold-based escalation
        if emotion == "anger" and intensity >= ESCALATION_ANGER_THRESHOLD:
            session["anger_streak"] += 1
            logger.info(f"[{session_id[:8]}] Anger streak: {session['anger_streak']}")
        else:
            if not explicit_escalation:   # don't reset streak on explicit request
                session["anger_streak"] = 0

        just_escalated = (
            not session["escalated"]
            and (
                session["anger_streak"] >= ESCALATION_TURNS_REQUIRED
                or explicit_escalation
            )
        )
        if just_escalated:
            session["escalated"] = True
            logger.warning(f"[{session_id[:8]}] 🚨 MANAGER NOTIFIED! order={session.get('order_id')} customer={session.get('customer_name')}")
        
        # ── 5. Generate response ─────────────────────────────────────────────
        logger.info(f"[{session_id[:8]}] Pipeline Step 4: Steered LLM")
        if just_escalated:
            response = (
                "I completely understand how frustrating this is, and I sincerely apologise. "
                "I'm connecting you to our manager right now — they will call you within 5 minutes "
                "regarding your order " + (session.get("order_id") or "N/A") + "."
            )
            logger.info(f"[{session_id[:8]}] LLM → ESCALATION RESPONSE")
            append_chat(session, "bot", response)
            session["escalated_at"] = datetime.utcnow().isoformat()
            # Reset session so user can start a fresh issue after escalation
            session["phase"]        = "greeting"
            session["intent"]       = None
            session["order_id"]     = None
            session["anger_streak"] = 0
            session["order_attempts"] = 0
            session["state_memory"].reset()
            logger.info(f"[{session_id[:8]}] Session reset to greeting after escalation")
        else:
            response = llm.generate(req.message, decision)
            logger.info(f"[{session_id[:8]}] LLM → response: '{response}'")
            logger.info(f"[{session_id[:8]}] LLM → length: {len(response)} chars")
            append_chat(session, "bot", response)
        
        # ── 6. Recommendations ───────────────────────────────────────────────
        logger.info(f"[{session_id[:8]}] Pipeline Step 5: Recommender")
        recommendations = recommender.recommend(req.message, emotion, target, decision.vector)
        logger.info(f"[{session_id[:8]}] Recommender → {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations):
            logger.info(f"[{session_id[:8]}]   Rec {i+1}: {rec.get('title', 'N/A')} (score: {rec.get('score', 0):.3f})")
        
        logger.info(f"[{session_id[:8]}] ✅ Pipeline complete!")
        
        # Log the bot response
        log_message("Bot", response[:100], session_id)
        
        # ── 7. Log conversation ───────────────────────────────────────────────
        _conversation_logs.append({
            "session_id": session_id,
            "turn":       ai_state.turn,
            "emotion":    emotion,
            "intensity":  intensity,
            "message":    req.message[:120],
            "escalated":  session["escalated"],
            "timestamp":  datetime.utcnow().isoformat(),
            "customer":   session.get("customer_name", "Guest"),
            "order_id":   session.get("order_id"),
            "mode":       decision.mode,
            "vector":     decision.vector,
            "response":   response[:100],
        })
        
        return ChatResponse(
            session_id      = session_id,
            response        = response,
            phase           = "emotional",
            emotion         = emotion,
            target          = target,
            intensity       = intensity,
            ai_emotion      = ai_state.emotion,
            ai_intensity    = ai_state.ai_intensity,
            mode            = decision.mode,
            vector          = decision.vector,
            vector_intensity = decision.vector_intensity,
            escalated       = session["escalated"],
            recommendations = recommendations,
            turn            = ai_state.turn,
        )
    
    # Fallback
    logger.warning(f"[{session_id[:8]}] ⚠️ Fallback reached")
    return ChatResponse(
        session_id = session_id,
        response   = "I'm here to help! Please select an option to get started.",
        phase      = "greeting",
    )


# ── Lookup endpoint ──────────────────────────────────────────────────────────

@app.get("/order/{order_id}")
async def order_lookup(order_id: str):
    data = tool_order_status(order_id)
    return OrderCard(**data)


# ── Reset session endpoint ──────────────────────────────────────────────────

@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    _sessions[session_id]["state_memory"].reset()
    _sessions[session_id]["anger_streak"] = 0
    _sessions[session_id]["escalated"] = False
    _sessions[session_id]["phase"] = "greeting"
    _sessions[session_id]["intent"] = None
    _sessions[session_id]["order_id"] = None
    logger.info(f"[Session] Reset: {session_id}")
    return {"status": "ok", "message": "Session reset"}


# ── Dashboard endpoints ──────────────────────────────────────────────────────

@app.post("/session/new")
async def new_session():
    sid = str(uuid.uuid4())
    get_or_create_session(sid)
    logger.info(f"[Session] Created: {sid}")
    return {"session_id": sid}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    s  = _sessions[session_id]
    st = s["state_memory"].state
    return {
        "session_id": session_id,
        "turn":       st.turn,
        "emotion":    st.emotion,
        "ai_intensity": st.ai_intensity,
        "escalated":  s["escalated"],
        "escalated_at": s.get("escalated_at"),
        "created_at": s["created_at"],
        "order_id":   s.get("order_id"),
        "phase":      s.get("phase"),
    }


@app.get("/session/{session_id}/history")
async def session_history(session_id: str):
    """Full chat history for a session — for manager view."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    s = _sessions[session_id]
    return {
        "session_id":    session_id,
        "customer_name": s.get("customer_name", "Guest"),
        "order_id":      s.get("order_id"),
        "escalated":     s["escalated"],
        "escalated_at":  s.get("escalated_at"),
        "created_at":    s["created_at"],
        "phase":         s.get("phase"),
        "chat_history":  s["chat_history"],
    }


@app.get("/dashboard/sessions")
async def dashboard_sessions():
    result = []
    for sid, s in _sessions.items():
        st = s["state_memory"].state
        result.append({
            "session_id":   sid,
            "turn":         st.turn,
            "last_emotion": st.emotion,
            "ai_intensity": st.ai_intensity,
            "escalated":    s["escalated"],
            "escalated_at": s.get("escalated_at"),
            "created_at":   s["created_at"],
            "last_message": s.get("last_message", ""),
            "customer":     s.get("customer_name", "Guest"),
            "order_id":     s.get("order_id"),
            "phase":        s.get("phase", "greeting"),
            "message_count": len(s.get("chat_history", [])),
        })
    result.sort(key=lambda x: (not x["escalated"], x["created_at"]), reverse=True)
    return result


@app.get("/dashboard/logs")
async def dashboard_logs(limit: int = 50):
    return list(reversed(_conversation_logs[-limit:]))


@app.get("/dashboard/escalations")
async def dashboard_escalations():
    """All escalated sessions — for manager dashboard."""
    result = []
    for sid, s in _sessions.items():
        if not s["escalated"]:
            continue
        st = s["state_memory"].state
        # Find last log entry for this session
        last_log = next(
            (l for l in reversed(_conversation_logs) if l["session_id"] == sid),
            {}
        )
        result.append({
            "session_id":     sid,
            "customer":       s.get("customer_name", "Guest"),
            "order_id":       s.get("order_id"),
            "turn":           st.turn,
            "last_emotion":   last_log.get("emotion"),
            "last_intensity": last_log.get("intensity"),
            "last_message":   s.get("last_message", ""),
            "escalated_at":   s.get("escalated_at") or last_log.get("timestamp"),
            "created_at":     s["created_at"],
        })
    result.sort(key=lambda x: x.get("escalated_at") or "", reverse=True)
    return result


@app.get("/dashboard/stats")
async def dashboard_stats():
    total    = len(_sessions)
    esc      = sum(1 for s in _sessions.values() if s["escalated"])
    emotions = {}
    for l in _conversation_logs:
        emotions[l["emotion"]] = emotions.get(l["emotion"], 0) + 1
    return {
        "total_conversations": total,
        "escalated":           esc,
        "emotion_breakdown":   emotions,
        "total_messages":      len(_conversation_logs),
    }


@app.get("/llm/status")
async def llm_status():
    return {
        "steered_llm_loaded": True,
        "backend": "steered_mistral",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "restaurant": RESTAURANT_NAME,
        "llm": "steered_mistral",
        "classifier": "loaded",
        "state_memory": "ready",
        "decision_engine": "ready",
        "recommender": "ready",
    }