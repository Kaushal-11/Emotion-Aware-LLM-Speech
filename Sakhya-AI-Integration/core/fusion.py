"""
core/fusion.py
---------------
Layer 1.5 — Fusion Layer

Combines:
    - SER output           : (ser_emotion, ser_confidence)        [validator]
    - Text classifier output: (cl_emotion, cl_target, cl_intensity) [primary decider]

into a single FusedPerception passed to State Memory + Decision Engine.

Fusion rule (confirmed design)
-------------------------------
- target    : ALWAYS from text classifier (SER cannot detect this)
- intensity : ALWAYS from text classifier
- emotion   :
    if SER and classifier AGREE          -> classifier emotion, high confidence
    if they DISAGREE                     -> classifier emotion wins (has context)
    SER emotion is kept only as a logging / confidence signal, never overrides
"""

from dataclasses import dataclass


@dataclass
class FusedPerception:
    emotion:        str    # final emotion used downstream (always == cl_emotion)
    target:         str    # from text classifier
    intensity:      float  # from text classifier

    # raw signals, kept for logging / UI display
    cl_emotion:     str
    cl_target:      str
    cl_intensity:   float
    ser_emotion:    str
    ser_confidence: float
    agreement:      bool   # True if ser_emotion == cl_emotion


def fuse(
    cl_emotion:     str,
    cl_target:      str,
    cl_intensity:   float,
    ser_emotion:    str,
    ser_confidence: float,
) -> FusedPerception:
    """
    Merge SER + text classifier outputs per the fusion rule above.

    Parameters
    ----------
    cl_emotion, cl_target, cl_intensity : output of core.classifier.TextEmotionClassifier
    ser_emotion, ser_confidence         : output of core.ser.SER

    Returns
    -------
    FusedPerception
    """
    cl_emotion  = cl_emotion.strip().lower()
    cl_target   = cl_target.strip().lower()
    ser_emotion = ser_emotion.strip().lower()

    agreement = (ser_emotion == cl_emotion)

    return FusedPerception(
        emotion        = cl_emotion,     # classifier always wins
        target         = cl_target,
        intensity      = float(cl_intensity),
        cl_emotion     = cl_emotion,
        cl_target      = cl_target,
        cl_intensity   = float(cl_intensity),
        ser_emotion    = ser_emotion,
        ser_confidence = float(ser_confidence),
        agreement      = agreement,
    )