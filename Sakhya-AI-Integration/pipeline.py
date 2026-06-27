"""
pipeline.py
-----------
Wires together all core modules into a single per-turn pipeline:

    audio_in
        -> ASR (Whisper)              -> transcript
        -> SER (SenseVoice/WavLM)      -> ser_emotion, ser_confidence   } parallel
        -> TextClassifier(transcript)  -> cl_emotion, cl_target, cl_intensity
        -> fusion.fuse(...)            -> FusedPerception
        -> StateMemory.update(...)      -> AIEmotionalState
        -> DecisionEngine.decide(...)   -> DecisionOutput
        -> SteeredLLM.generate(...)     -> response_text     }
        -> Recommender.recommend(...)   -> recommendations    } parallel-ish
        -> TTS.synthesize(...)          -> audio_out, sample_rate

All heavy models are loaded ONCE when EmotionalAIPipeline() is constructed.
Call `run_turn(audio_path, speaker_id)` once per conversational turn.
Call `reset()` to start a brand-new conversation (clears StateMemory + LLM history).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from config import DEFAULT_SPEAKER

from core.asr import ASR
from core.ser import SER
from core.classifier import TextEmotionClassifier
from core.fusion import fuse, FusedPerception
from core.state_memory import StateMemory, AIEmotionalState
from core.decision_engine import DecisionEngine, DecisionOutput
from core.llm import SteeredLLM
from core.tts import TTS
from core.recommender import Recommender


@dataclass
class PipelineOutput:
    """Everything the UI needs to display for one turn."""

    transcript:       str

    # perception
    ser_emotion:      str
    ser_confidence:   float
    cl_emotion:       str
    cl_target:        str
    cl_intensity:     float
    fused_emotion:    str
    emotion_agreement: bool

    # state memory
    ai_emotion:       str
    ai_intensity:     float

    # decision
    mode:             str
    vector:           str
    vector_intensity: float

    # outputs
    response_text:    str
    recommendations:  List[dict]

    # audio
    audio:            Optional[np.ndarray]
    sample_rate:      Optional[int]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("audio", None)   # don't serialize raw audio in logs/UI state
        return d


class EmotionalAIPipeline:
    """
    Usage
    -----
        pipeline = EmotionalAIPipeline()          # loads everything once (~90-120s)

        out = pipeline.run_turn("turn_001.wav", speaker_id="speaker_2")
        print(out.response_text)
        # out.audio, out.sample_rate -> play/save

        pipeline.reset()                           # new conversation
    """

    def __init__(self):
        print("=" * 70)
        print("Loading Emotional AI pipeline (this happens once)...")
        print("=" * 70)

        print("\n[1/7] Loading ASR (Whisper) ...")
        self.asr = ASR()

        print("\n[2/7] Loading SER ...")
        self.ser = SER()

        print("\n[3/7] Loading Text Classifier ...")
        self.classifier = TextEmotionClassifier()

        print("\n[4/7] Initializing State Memory + Decision Engine ...")
        self.state_memory = StateMemory()
        self.decision_engine = DecisionEngine()

        print("\n[5/7] Loading LLM (steered generation) ...")
        self.llm = SteeredLLM()

        print("\n[6/7] Loading Recommender ...")
        self.recommender = Recommender()

        print("\n[7/7] Loading TTS (CosyVoice2) ...")
        self.tts = TTS()

        self._executor = ThreadPoolExecutor(max_workers=2)

        print("\n" + "=" * 70)
        print("Pipeline ready.")
        print("=" * 70)

    # ------------------------------------------------------------------ #

    def reset(self):
        """Start a brand-new conversation: clears state memory + LLM history."""
        self.state_memory.reset()
        self.llm.reset()

    # ------------------------------------------------------------------ #

    def switch_ser_backend(self, backend: str):
        """
        Reload the SER model with a different backend ("sensevoice" / "wavlm").
        This is a heavy operation (~10s) — call only on explicit user toggle,
        not per-turn.
        """
        if backend == self.ser.backend:
            return
        print(f"[Pipeline] switching SER backend -> {backend}")
        del self.ser
        import gc
        gc.collect()
        if hasattr(__import__("torch").cuda, "empty_cache"):
            __import__("torch").cuda.empty_cache()
        self.ser = SER(backend=backend)

    def switch_llm_backend(self, backend: str):
        """
        Reload the LLM with a different backend ("mistral" / "qwen").
        This is a heavy operation (~30-40s) — call only on explicit user toggle.
        Conversation history is reset since direction vectors / model differ.
        """
        if backend == self.llm.backend:
            return
        print(f"[Pipeline] switching LLM backend -> {backend}")
        del self.llm
        import gc
        gc.collect()
        if hasattr(__import__("torch").cuda, "empty_cache"):
            __import__("torch").cuda.empty_cache()
        self.llm = SteeredLLM(backend=backend)
        self.state_memory.reset()

    def switch_tts_backend(self, backend: str):
        """
        Reload TTS with a different backend ("f5tts" / "cosyvoice2").
        Heavy operation — call only on explicit user toggle.
        """
        if backend == self.tts.backend:
            return
        print(f"[Pipeline] switching TTS backend -> {backend}")
        del self.tts
        import gc
        gc.collect()
        if hasattr(__import__("torch").cuda, "empty_cache"):
            __import__("torch").cuda.empty_cache()
        self.tts = TTS(backend=backend)

    # ------------------------------------------------------------------ #

    def run_turn(self, audio_path: str, speaker_id: str = DEFAULT_SPEAKER) -> PipelineOutput:
        """
        Run one full turn (blocking). Returns final PipelineOutput.
        """
        for out in self.run_turn_stream(audio_path, speaker_id):
            pass
        return out

    def run_turn_stream(self, audio_path: str, speaker_id: str = DEFAULT_SPEAKER):
        """
        Generator version of run_turn. Yields a dict at each pipeline step
        so the UI can show live progress, then yields the final PipelineOutput.

        Yielded dict keys:
            step      : int   (1-9)
            label     : str   human-readable step name
            done      : bool  True only on the last yield (PipelineOutput)
            result    : PipelineOutput | None
        """

        def _status(step, label, done=False, result=None, transcript=None):
            d = {"step": step, "label": label, "done": done, "result": result}
            if transcript is not None:
                d["transcript"] = transcript
            return d

        # ── Step 1: ASR ────────────────────────────────────────────────────────
        yield _status(1, "🎙️ Transcribing your speech (Whisper)...")
        asr_future = self._executor.submit(self.asr.transcribe, audio_path)

        # ── Step 2: SER (runs in parallel while ASR is working) ───────────────
        yield _status(2, "🎵 Detecting speech emotion (SER)...")
        ser_future = self._executor.submit(self.ser.predict, audio_path)

        transcript = asr_future.result()
        if not transcript.strip():
            transcript = "..."

        # Emit transcript immediately so server can forward to UI before SER finishes
        yield _status(2, "🎵 Detecting speech emotion (SER)...", transcript=transcript)

        ser_emotion, ser_confidence = ser_future.result()

        # ── Step 3: Text Classifier ────────────────────────────────────────────
        yield _status(3, "📝 Classifying emotion from text...")
        cl_emotion, cl_target, cl_intensity = self.classifier.classify(transcript)

        # ── Step 4: Fusion + State Memory + Decision Engine ────────────────────
        yield _status(4, "🧠 Updating emotional state & deciding response mode...")
        fused = fuse(
            cl_emotion=cl_emotion, cl_target=cl_target, cl_intensity=cl_intensity,
            ser_emotion=ser_emotion, ser_confidence=ser_confidence,
        )
        ai_state = self.state_memory.update(
            user_emotion=fused.emotion, user_target=fused.target, user_intensity=fused.intensity,
        )
        decision = self.decision_engine.decide(
            user_emotion=fused.emotion, user_target=fused.target,
            user_intensity=fused.intensity, ai_state=ai_state,
        )

        # ── Step 5: Recommender (fast, CPU) ───────────────────────────────────
        yield _status(5, "💡 Finding recommendations...")
        rec_future = self._executor.submit(
            self.recommender.recommend,
            text=transcript, emotion=fused.emotion,
            target=fused.target, vector=decision.vector,
        )

        # ── Step 6: LLM (heavy GPU) ───────────────────────────────────────────
        yield _status(6, f"🤖 Generating steered response ({self.llm.backend})...")
        response_text  = self.llm.generate(transcript, decision)
        recommendations = rec_future.result()

        # ── Step 7: TTS ───────────────────────────────────────────────────────
        yield _status(7, f"🔊 Synthesizing emotional speech ({self.tts.backend})...")
        audio, sr = self.tts.synthesize(
            text=response_text, speaker_id=speaker_id, decision=decision,
        )

        # ── Final: package output ─────────────────────────────────────────────
        out = PipelineOutput(
            transcript=transcript,
            ser_emotion=fused.ser_emotion, ser_confidence=fused.ser_confidence,
            cl_emotion=fused.cl_emotion, cl_target=fused.cl_target,
            cl_intensity=fused.cl_intensity, fused_emotion=fused.emotion,
            emotion_agreement=fused.agreement, ai_emotion=ai_state.emotion,
            ai_intensity=ai_state.ai_intensity, mode=decision.mode,
            vector=decision.vector, vector_intensity=decision.vector_intensity,
            response_text=response_text, recommendations=recommendations,
            audio=audio, sample_rate=sr,
        )
        yield _status(8, "✅ Done.", done=True, result=out)

    def run_text_turn_stream(self, text: str, speaker_id: str = DEFAULT_SPEAKER):
        """
        Text-only pipeline — skips ASR and SER entirely.

        Flow:
            text input
            -> Text Classifier (emotion, target, intensity)
            -> Fusion (SER disabled — classifier is sole input)
            -> State Memory + Decision Engine
            -> Recommender (parallel)
            -> LLM (steered)
            -> TTS
            -> PipelineOutput  (ser_emotion="n/a", ser_confidence=0.0)
        """

        def _status(step, label, done=False, result=None):
            return {"step": step, "label": label, "done": done, "result": result}

        # Step 1: Text Classifier
        yield _status(1, "📝 Classifying emotion from text...")
        cl_emotion, cl_target, cl_intensity = self.classifier.classify(text)

        # Step 2: Fusion (SER disabled — pass classifier output as both signals)
        yield _status(2, "🧠 Updating emotional state & deciding response mode...")
        fused = fuse(
            cl_emotion=cl_emotion, cl_target=cl_target, cl_intensity=cl_intensity,
            ser_emotion=cl_emotion,   # mirror classifier — no SER signal
            ser_confidence=0.0,       # 0.0 signals SER was skipped
        )
        ai_state = self.state_memory.update(
            user_emotion=fused.emotion, user_target=fused.target,
            user_intensity=fused.intensity,
        )
        decision = self.decision_engine.decide(
            user_emotion=fused.emotion, user_target=fused.target,
            user_intensity=fused.intensity, ai_state=ai_state,
        )

        # Step 3: Recommender (fast, CPU — parallel)
        yield _status(3, "💡 Finding recommendations...")
        rec_future = self._executor.submit(
            self.recommender.recommend,
            text=text, emotion=fused.emotion,
            target=fused.target, vector=decision.vector,
        )

        # Step 4: LLM
        yield _status(4, f"🤖 Generating steered response ({self.llm.backend})...")
        response_text   = self.llm.generate(text, decision)
        recommendations = rec_future.result()

        # Step 5: TTS
        yield _status(5, f"🔊 Synthesizing emotional speech ({self.tts.backend})...")
        audio, sr = self.tts.synthesize(
            text=response_text, speaker_id=speaker_id, decision=decision,
        )

        out = PipelineOutput(
            transcript=text,
            ser_emotion="n/a",        # SER was skipped in text mode
            ser_confidence=0.0,
            cl_emotion=fused.cl_emotion, cl_target=fused.cl_target,
            cl_intensity=fused.cl_intensity, fused_emotion=fused.emotion,
            emotion_agreement=True,   # no disagreement possible without SER
            ai_emotion=ai_state.emotion, ai_intensity=ai_state.ai_intensity,
            mode=decision.mode, vector=decision.vector,
            vector_intensity=decision.vector_intensity,
            response_text=response_text, recommendations=recommendations,
            audio=audio, sample_rate=sr,
        )
        yield _status(6, "✅ Done.", done=True, result=out)