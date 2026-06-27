"""
decision_engine.py
------------------
Layer 3 — Decision Engine

Takes classifier output + AIEmotionalState from state memory and
produces the response mode, vector type, vector intensity, and
style_contract for LLM steering + response shaping.

Decision table is implemented from:
    target  : you | other | self | situation
    emotion : anger | sadness | fear | happiness | disgust | surprise

UPDATED: Restaurant support context with improved response strategies
"""

from dataclasses import dataclass
from typing import Optional
from core.state_memory import AIEmotionalState, normalize_emotion, normalize_target


@dataclass(frozen=True)
class StyleContract:
    max_words:          int
    max_sentences:      int
    allow_bullets:      bool
    allow_questions:    bool    
    allow_commands:     bool   
    profanity:          bool   # always False


TABLE: dict[tuple[str, str], dict] = {

    # ── you / directed at AI ──────────────────────────────────────────────────

    ("you", "anger"): {
        "mode":             "apologize + resolve",  # FIXED: was too weak
        "vector":           "inject_sadness",
        "vector_intensity": 0.45,  # FIXED: raised from 0.25 to acknowledge severity
        "style_contract":   StyleContract(
            max_words          = 35,  # FIXED: slightly longer to fully acknowledge
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("you", "sadness"): {
        "mode":             "apologize + comfort",  # KEPT: appropriate
        "vector":           "mirror_sadness",
        "vector_intensity": 0.30,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("you", "fear"): {
        "mode":             "reassure + clarify",  # FIXED: better name
        "vector":           "none",
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 40,  # FIXED: longer for allergen/delivery concerns
            max_sentences      = 3,   # FIXED: allow more sentences for clarity
            allow_bullets      = True,  # FIXED: allow bullets for allergen info
            allow_questions    = True,    
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("you", "happiness"): {
        "mode":             "mirror + gratitude",
        "vector":           "mirror_happiness",
        "vector_intensity": 0.70,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("you", "disgust"): {
        "mode":             "acknowledge + fix",  # FIXED: better for restaurant issues
        "vector":           "none",
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 40,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = True,               
            allow_commands     = True,
            profanity          = False,
        ),
    },

    ("you", "surprise"): {
        "mode":             "mirror + explain",
        "vector":           "mirror_surprise",
        "vector_intensity": 0.65,
        "style_contract":   StyleContract(
            max_words          = 40,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = True,
            allow_commands     = True,
            profanity          = False,
        ),
    },

    # ── other / third person or group ─────────────────────────────────────────

    ("other", "anger"): {
        "mode":             "acknowledge + redirect",  # FIXED: was mirror solidarity (unprofessional)
        "vector":           "inject_sadness",  # FIXED: empathy instead of mirroring anger
        "vector_intensity": 0.30,  # FIXED: lower intensity, more professional
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("other", "sadness"): {
        "mode":             "soft mirror + support",
        "vector":           "mirror_sadness",
        "vector_intensity": 0.425,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("other", "fear"): {
        "mode":             "guide + solutions",
        "vector":           "inject_happiness",
        "vector_intensity": 0.25,
        "style_contract":   StyleContract(
            max_words          = 50,
            max_sentences      = 2,
            allow_bullets      = True,   # practical steps for food concerns
            allow_questions    = False,            
            allow_commands     = True,   # action steps for resolution
            profanity          = False,
        ),
    },

    ("other", "happiness"): {
        "mode":             "celebrate with them",
        "vector":           "mirror_happiness",
        "vector_intensity": 0.65,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("other", "disgust"): {
        "mode":             "acknowledge + fix",  # FIXED: not "partial mirror" - sounds like bot disgusted
        "vector":           "none",  # FIXED: don't mirror disgust
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("other", "surprise"): {
        "mode":             "mirror + react",
        "vector":           "mirror_surprise",
        "vector_intensity": 0.55,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = True,   # prompts reflection            
            allow_commands     = True,
            profanity          = False,
        ),
    },

    # ── self / directed at themselves ─────────────────────────────────────────

    ("self", "anger"): {
        "mode":             "acknowledge + resolve",  # FIXED: was comfort + reframe (too soft)
        "vector":           "inject_sadness",  # FIXED: acknowledge frustration
        "vector_intensity": 0.30,  # FIXED: appropriate for restaurant context
        "style_contract":   StyleContract(
            max_words          = 35,  # FIXED: longer to properly address
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("self", "sadness"): {
        "mode":             "uplift + reframe",
        "vector":           "inject_happiness",
        "vector_intensity": 0.275,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("self", "fear"): {
        "mode":             "calm + grounding",
        "vector":           "none",
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("self", "happiness"): {
        "mode":             "celebrate + praise",
        "vector":           "mirror_happiness",
        "vector_intensity": 0.50,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("self", "disgust"): {
        "mode":             "acknowledge + fix",  # FIXED: was "prevent self-hate" - wrong context
        "vector":           "none",  # FIXED: don't inject emotions
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 35,  # FIXED: longer for proper acknowledgment
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("self", "surprise"): {
        "mode":             "mirror + contextualize",
        "vector":           "mirror_surprise",
        "vector_intensity": 0.50,
        "style_contract":   StyleContract(
            max_words          = 40,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    # ── situation / event or circumstance ─────────────────────────────────────

    ("situation", "anger"): {
        "mode":             "acknowledge + redirect",  # FIXED: removed allow_questions for angry users
        "vector":           "inject_sadness",  # FIXED: empathy instead of mirroring anger
        "vector_intensity": 0.30,  # FIXED: moderate empathy
        "style_contract":   StyleContract(
            max_words          = 45,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,  # FIXED: NO questions for angry users
            allow_commands     = True,   # FIXED: allow commands to resolve issues
            profanity          = False,
        ),
    },

    ("situation", "sadness"): {
        "mode":             "empathize + offer solution",  # FIXED: better than warm + encouraging
        "vector":           "inject_happiness",  # FIXED: slight positive shift
        "vector_intensity": 0.25,  # FIXED: adjusted
        "style_contract":   StyleContract(
            max_words          = 40,  # FIXED: longer for solution offering
            max_sentences      = 3,   # FIXED: more sentences for empathy + solution
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = True,  # FIXED: can suggest actions
            profanity          = False,
        ),
    },

    ("situation", "fear"): {
        "mode":             "reassure + clarify",  # FIXED: better name
        "vector":           "none",
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 40,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("situation", "happiness"): {
        "mode":             "mirror joy",
        "vector":           "mirror_happiness",
        "vector_intensity": 0.55,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("situation", "disgust"): {
        "mode":             "acknowledge + normalize",  # FIXED: better for restaurant complaints
        "vector":           "none",  # FIXED: don't mirror disgust
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 40,  # FIXED: longer for proper acknowledgment
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = True,  # FIXED: allow commands to resolve
            profanity          = False,
        ),
    },

    ("situation", "surprise"): {
        "mode":             "full mirror",
        "vector":           "mirror_surprise",
        "vector_intensity": 0.50,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = True,   # opens conversation            
            allow_commands     = False,
            profanity          = False,
        ),
    },
}


# ── Decision output ───────────────────────────────────────────────────────────

@dataclass
class DecisionOutput:
    """
    What the decision engine produces each turn.

    Fields
    ------
    mode                : human-readable response mode
    vector              : which vector to inject, e.g. "mirror_surprise",
                          "inject_happiness", or "none"
    vector_intensity    : float in [0.0, 1.0]
    style_contract      : StyleContract — shape/tone/safety rules for this response
    user_emotion        : normalised emotion from classifier
    user_target         : normalised target from classifier
    user_intensity      : classifier intensity (saved in state, passed through)
    ai_emotion          : current AI emotion from state memory
    ai_intensity        : current AI intensity from state memory
    """
    mode:             str
    vector:           str
    vector_intensity: float
    style_contract:   StyleContract
    user_emotion:     str
    user_target:      str
    user_intensity:   float
    ai_emotion:       str
    ai_intensity:     float


# ── Decision Engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Layer 3 — Decision Engine.

    Updated for restaurant support context with improved response strategies.
    """

    def decide(
        self,
        user_emotion:   str,
        user_target:    str,
        user_intensity: float,
        ai_state:       AIEmotionalState,
    ) -> DecisionOutput:

        # ── normalise inputs ───────────────────────────────────────────────
        emotion   = normalize_emotion(user_emotion)
        target    = normalize_target(user_target)
        intensity = max(0.0, min(1.0, float(user_intensity)))

        # ── look up decision table ─────────────────────────────────────────
        key = (target, emotion)
        if key not in TABLE:
            raise KeyError(f"No table entry for target='{target}', emotion='{emotion}'")

        entry            = TABLE[key].copy()
        mode             = entry["mode"]
        vector           = entry["vector"]
        vector_intensity = entry["vector_intensity"]
        style_contract   = entry["style_contract"]

        # ── special handling: if intensity is very high (>0.8) ────────────
        # increase empathy/acknowledgment automatically
        if intensity > 0.8 and emotion in ["anger", "disgust"]:
            if vector != "none":
                vector_intensity = min(1.0, vector_intensity + 0.15)
            
            # Allow more words for high intensity emotions
            style_contract = StyleContract(
                max_words          = min(50, style_contract.max_words + 5),
                max_sentences      = min(3, style_contract.max_sentences + 1),
                allow_bullets      = style_contract.allow_bullets,
                allow_questions    = False,  # Force no questions when very angry
                allow_commands     = style_contract.allow_commands,
                profanity          = style_contract.profanity,
            )

        # ── round final intensity ──────────────────────────────────────────
        if vector_intensity is not None:
            vector_intensity = round(vector_intensity, 4)

        return DecisionOutput(
            mode             = mode,
            vector           = vector,
            vector_intensity = vector_intensity if vector_intensity is not None else 0.0,
            style_contract   = style_contract,
            user_emotion     = emotion,
            user_target      = target,
            user_intensity   = intensity,
            ai_emotion       = ai_state.emotion,
            ai_intensity     = ai_state.ai_intensity,
        )