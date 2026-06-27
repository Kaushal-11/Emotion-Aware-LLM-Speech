"""
state_memory.py
---------------
Layer 2 — Emotional State Memory

Tracks the AI's current emotional state across turns using a
single-step blend formula with decay. No full history stored —
each turn's intensity already contains prior history via decay chain.

Blend formula:
    new_ai_intensity = (user_intensity × DELTA_WEIGHT) + (prev_ai_intensity × DECAY)

Emotion switch threshold:
    AI only switches to a new emotion if user_intensity > SWITCH_THRESHOLD.
    Prevents rapid flipping on weak emotional signals.
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Tunable constants ─────────────────────────────────────────────────────────

DECAY         = 0.85   # how strongly previous AI intensity persists
DELTA_WEIGHT  = 0.40   # how strongly new user emotion pulls AI state
SWITCH_THRESHOLD = 0.55  # minimum user intensity required to switch AI emotion


# ── Valid vocabulary ──────────────────────────────────────────────────────────

VALID_EMOTIONS = {"anger", "sadness", "fear", "happiness", "disgust", "surprise"}
VALID_TARGETS  = {"you", "other", "self", "situation"}


# ── Normalisation helpers ─────────────────────────────────────────────────────

def normalize_emotion(emotion: str) -> str:
    """Lowercase and strip. Raises ValueError if not a known emotion."""
    e = emotion.strip().lower()
    if e not in VALID_EMOTIONS:
        raise ValueError(f"Unknown emotion: '{emotion}'. Must be one of {VALID_EMOTIONS}")
    return e


def normalize_target(target: str) -> str:
    """Lowercase and strip. Raises ValueError if not a known target."""
    t = target.strip().lower()
    if t not in VALID_TARGETS:
        raise ValueError(f"Unknown target: '{target}'. Must be one of {VALID_TARGETS}")
    return t


# ── Anchor memory ─────────────────────────────────────────────────────────────

@dataclass
class AnchorMemory:
    """
    Stores what originally triggered the current AI emotional state.
    Preserved until the AI emotion switches to a new emotion, at which
    point it is overwritten with the new trigger.
    """
    trigger_emotion: str    = ""
    trigger_target:  str    = ""
    trigger_intensity: float = 0.0
    turn_triggered:  int    = 0


# ── AI Emotional State ────────────────────────────────────────────────────────

@dataclass
class AIEmotionalState:
    """
    The live emotional state of the AI.

    Fields
    ------
    emotion         : current AI emotion label (one of VALID_EMOTIONS), or
                      empty string before the first turn.
    ai_intensity    : blended intensity float in [0.0, 1.0]
    turn            : current turn counter (incremented on each update)
    anchor          : AnchorMemory — what originally set this emotion
    """
    emotion:      str           = ""
    ai_intensity: float         = 0.0
    user_intensity: float         = 0.0  
    turn:         int           = 0
    anchor:       AnchorMemory  = field(default_factory=AnchorMemory)

    # ── internal: previous-turn snapshot (single step only) ──────────────────
    _prev_emotion:   str   = field(default="",  repr=False)
    _prev_intensity: float = field(default=0.0, repr=False)


# ── State Memory ──────────────────────────────────────────────────────────────

class StateMemory:
    """
    Manages the AI emotional state across conversation turns.

    Usage
    -----
        memory = StateMemory()

        # each turn, call update() with classifier output:
        state = memory.update(
            user_emotion="surprise",
            user_target="you",
            user_intensity=0.72,
        )

        # access current state:
        state.emotion        # "surprise"
        state.ai_intensity   # 0.61  (blended float)
        state.anchor         # AnchorMemory(...)
    """

    def __init__(self) -> None:
        self.state = AIEmotionalState()

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        user_emotion:    str,
        user_target:     str,
        user_intensity:  float,
    ) -> AIEmotionalState:
        """
        Call once per turn with classifier output.

        Parameters
        ----------
        user_emotion    : raw classifier output (case-insensitive)
        user_target     : raw classifier output (case-insensitive)
        user_intensity  : float in [0.0, 1.0]

        Returns
        -------
        Updated AIEmotionalState (also mutates self.state in place).
        """
        # ── 1. normalise ──────────────────────────────────────────────────────
        emotion   = normalize_emotion(user_emotion)
        target    = normalize_target(user_target)
        intensity = float(user_intensity)
        intensity = max(0.0, min(1.0, intensity))  # clamp

        # ── 2. advance turn counter ───────────────────────────────────────────
        self.state.turn += 1

        # ── 3. snapshot previous state ────────────────────────────────────────
        prev_emotion    = self.state.emotion
        prev_intensity  = self.state.ai_intensity

        # ── 4. decide whether AI emotion switches ─────────────────────────────
        first_turn = (prev_emotion == "")

        if first_turn:
            # always adopt the first emotion
            new_emotion = emotion
            anchor_update = True

        elif emotion == prev_emotion:
            # same emotion — just blend intensity, no switch decision needed
            new_emotion = emotion
            anchor_update = False

        elif intensity > SWITCH_THRESHOLD:
            # different emotion, strong enough signal → switch
            new_emotion = emotion
            anchor_update = True

        else:
            # different emotion but too weak → AI holds current emotion
            new_emotion = prev_emotion
            anchor_update = False

        # ── 5. blend intensity ────────────────────────────────────────────────
        #
        #   new_ai_intensity = (user_intensity × DELTA_WEIGHT)
        #                    + (prev_ai_intensity × DECAY)
        #
        # When emotion switches, prev_ai_intensity resets to 0 so the new
        # emotion starts fresh rather than inheriting the old intensity.
        #
        if anchor_update and not first_turn:
            # emotion switched — reset carry-over
            carry = 0.0
        else:
            carry = prev_intensity

        new_intensity = (intensity * DELTA_WEIGHT) + (carry * DECAY)
        new_intensity = max(0.0, min(1.0, new_intensity))  # clamp to [0,1]

        # ── 6. update anchor memory if emotion switched ───────────────────────
        if anchor_update:
            self.state.anchor = AnchorMemory(
                trigger_emotion   = emotion,
                trigger_target    = target,
                trigger_intensity = intensity,
                turn_triggered    = self.state.turn,
            )

        # ── 7. write new state ────────────────────────────────────────────────
        self.state._prev_emotion   = prev_emotion
        self.state._prev_intensity = prev_intensity
        self.state.emotion         = new_emotion
        self.state.ai_intensity    = round(new_intensity, 4)
        self.state.user_intensity  = intensity   
        
        return self.state

    # ── convenience ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset to blank state (new conversation)."""
        self.state = AIEmotionalState()

    def snapshot(self) -> dict:
        """Return a plain-dict snapshot of current state (useful for logging)."""
        s = self.state
        return {
            "turn":         s.turn,
            "emotion":      s.emotion,
            "ai_intensity": s.ai_intensity,
            "anchor": {
                "trigger_emotion":    s.anchor.trigger_emotion,
                "trigger_target":     s.anchor.trigger_target,
                "trigger_intensity":  s.anchor.trigger_intensity,
                "turn_triggered":     s.anchor.turn_triggered,
            },
        }