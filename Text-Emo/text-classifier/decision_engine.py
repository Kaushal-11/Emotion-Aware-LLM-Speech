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

"""

from dataclasses import dataclass
from typing import Optional
from state_memory import AIEmotionalState, normalize_emotion, normalize_target


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
        "mode":             "apologize",
        "vector":           "inject_sadness",
        "vector_intensity": 0.25,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("you", "sadness"): {
        "mode":             "apologize + comfort",
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
        "mode":             "reassure + correct",
        "vector":           "none",
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
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
        "mode":             "clarify + fix",
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
        "mode":             "mirror (solidarity)",
        "vector":           "mirror_anger",
        "vector_intensity": 0.65,
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
            allow_bullets      = True,   # only mode with bullets — practical steps needed
            allow_questions    = False,            
            allow_commands     = True,   # action steps are appropriate here
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
        "mode":             "partial mirror",
        "vector":           "mirror_disgust",
        "vector_intensity": 0.525,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,   # partial mirror — don't amplify disgust at third party
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
            allow_questions    = True,   # prompts reflection on the other person            
            allow_commands     = True,
            profanity          = False,
        ),
    },

    # ── self / directed at themselves ─────────────────────────────────────────

    ("self", "anger"): {
        "mode":             "comfort + reframe",
        "vector":           "inject_happiness",
        "vector_intensity": 0.20,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,   # never "be kinder to yourself" — pure presence
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
            allow_commands     = False,   # gentle reframe at end — don't rush past the sadness
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
            allow_commands     = False,   # no injection — facts only, calm through tone
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
        "mode":             "reassure + prevent self-hate",
        "vector":           "inject_happiness",
        "vector_intensity": 0.20,
        "style_contract":   StyleContract(
            max_words          = 30,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,   # critical — no judgment, prevent self-loathing spirals
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
            allow_commands     = False,   # mirror lightly, add context to normalise
            profanity          = False,
        ),
    },

    # ── situation / event or circumstance ─────────────────────────────────────

    ("situation", "anger"): {
        "mode":             "light mirror + guide",
        "vector":           "mirror_anger",
        "vector_intensity": 0.375,
        "style_contract":   StyleContract(
            max_words          = 45,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = True,   # one question to direct the energy productively            
            allow_commands     = False,
            profanity          = False,
        ),
    },

    ("situation", "sadness"): {
        "mode":             "warm + encouraging",
        "vector":           "inject_happiness",
        "vector_intensity": 0.20,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,   # light encouragement — not forced positivity
            profanity          = False,
        ),
    },

    ("situation", "fear"): {
        "mode":             "rational grounding",
        "vector":           "none",
        "vector_intensity": 0.0,
        "style_contract":   StyleContract(
            max_words          = 40,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,   # no injection — calm conveyed through tone alone
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
        "mode":             "light mirror + normalize",
        "vector":           "mirror_disgust",
        "vector_intensity": 0.30,
        "style_contract":   StyleContract(
            max_words          = 35,
            max_sentences      = 2,
            allow_bullets      = False,
            allow_questions    = False,            
            allow_commands     = False,   # partial mirror — normalise reaction, don't amplify
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
            allow_questions    = True,   # full mirror + question opens the conversation            
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

    Usage
    -----
        engine = DecisionEngine()

        output = engine.decide(
            user_emotion   = "surprise",    # raw classifier output
            user_target    = "You",         # raw classifier output
            user_intensity = 0.72,          # classifier float
            ai_state       = memory.state,  # AIEmotionalState from StateMemory
        )

        output.vector              # "mirror_surprise"
        output.vector_intensity    # 0.45
        output.mode                # "mirror + explain"
        output.style_contract      # StyleContract(max_words=45, ...)
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