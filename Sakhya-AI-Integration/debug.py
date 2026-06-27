"""
debug_modules.py
----------------
Single-file interactive tester for every core module in the pipeline.

Run from the project root (where config.py lives):
    python debug_modules.py

Menu lets you test each module independently — no need to load all models at once.
Pick only what you want to test.

Modules covered
---------------
  1. Config check          — verify all paths exist, print device assignment
  2. ASR (Whisper)         — transcribe a WAV file
  3. SER (SenseVoice/WavLM)— predict emotion from a WAV file
  4. Text Classifier        — classify emotion/target/intensity from text
  5. Fusion                 — merge SER + classifier outputs
  6. State Memory           — simulate multi-turn state updates
  7. Decision Engine        — get mode/vector/style for emotion+target
  8. LLM (steered)          — generate a steered response
  9. TTS                    — synthesize speech and save to WAV
 10. Recommender            — get top-3 recommendations
 11. Full pipeline (1 turn) — run everything end-to-end on a WAV file
  0. Exit
"""

import json
import os
import sys
import traceback
from pathlib import Path

# ── Make sure project root is on path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ============================================================================
# HELPERS
# ============================================================================

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"


def hdr(title: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


def ok(msg: str):
    print(f"{GREEN}  ✅ {msg}{RESET}")


def warn(msg: str):
    print(f"{YELLOW}  ⚠️  {msg}{RESET}")


def err(msg: str):
    print(f"{RED}  ❌ {msg}{RESET}")


def info(key: str, val):
    print(f"  {BLUE}{key:<22}{RESET} {val}")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"\n  {YELLOW}>{RESET} {prompt}{suffix}: ").strip()
    return val if val else default


def ask_wav(label: str = "WAV file path") -> str:
    path = ask(label)
    if not path:
        err("No path provided.")
        return ""
    if not Path(path).exists():
        err(f"File not found: {path}")
        return ""
    return path


def save_audio(audio, sr: int, out_path: str = "/tmp/tts_test_output.wav"):
    import soundfile as sf
    import numpy as np
    if audio is None:
        err("Audio is None — nothing to save.")
        return
    sf.write(out_path, audio.astype(np.float32), sr)
    ok(f"Audio saved → {out_path}  (sr={sr})")


def print_json(obj):
    print(f"  {json.dumps(obj, indent=4, default=str)}")


# ============================================================================
# MODULE 1 — CONFIG CHECK
# ============================================================================

def test_config():
    hdr("1. Config Check")
    try:
        import config as cfg
        ok("config.py imported")

        info("DEVICE",     cfg.DEVICE)
        info("LLM_DEVICE", cfg.LLM_DEVICE)
        info("SER_BACKEND", cfg.SER_BACKEND)
        info("LLM_BACKEND", cfg.LLM_BACKEND)
        info("TTS_BACKEND", cfg.TTS_BACKEND)

        print()
        paths_to_check = {
            "CLASSIFIER_DIR":     cfg.CLASSIFIER_DIR,
            "SENSEVOICE_MODEL_DIR": cfg.SENSEVOICE_MODEL_DIR,
            "WAVLM_MODEL_DIR":    cfg.WAVLM_MODEL_DIR,
            "F5TTS_CKPT_PATH":    cfg.F5TTS_CKPT_PATH,
            "COSYVOICE_MODEL_DIR": cfg.COSYVOICE_MODEL_DIR,
            "RECOMMENDER_KB_PATH": cfg.RECOMMENDER_KB_PATH,
        }
        for name, path in paths_to_check.items():
            p = Path(str(path))
            exists = p.exists()
            tag = ok if exists else warn
            symbol = "✅" if exists else "⚠️ NOT FOUND"
            print(f"  {symbol}  {name}: {path}")

        print()
        for backend, d in cfg.DIRECTIONS_DIR.items():
            for dtype_ in ["mlp", "attention"]:
                f = Path(d) / f"emo_directions_{dtype_}.pt"
                symbol = "✅" if f.exists() else "⚠️ MISSING"
                print(f"  {symbol}  directions[{backend}][{dtype_}]: {f}")

        print()
        for sid, sp in cfg.PRESET_SPEAKERS.items():
            p = Path(sp["ref_audio"])
            symbol = "✅" if p.exists() else "⚠️ MISSING"
            print(f"  {symbol}  speaker '{sid}': {sp['ref_audio']}")

        print()
        import torch
        info("torch version",  torch.__version__)
        info("CUDA available", torch.cuda.is_available())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            total = props.total_memory / 1024**3
            info(f"  GPU {i}", f"{props.name}  {alloc:.1f}/{total:.1f} GB")

    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 2 — ASR
# ============================================================================

def test_asr():
    hdr("2. ASR — Whisper")
    wav = ask_wav("Path to WAV file for transcription")
    if not wav:
        return
    try:
        print("  Loading ASR...")
        from core.asr import ASR
        asr = ASR()
        ok("ASR loaded")

        print("  Transcribing...")
        transcript = asr.transcribe(wav)
        ok(f"Transcript: \"{transcript}\"")
        return transcript
    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 3 — SER
# ============================================================================

def test_ser():
    hdr("3. SER — Speech Emotion Recognition")

    backend = ask("Backend (sensevoice / wavlm)", "sensevoice")
    wav     = ask_wav("Path to WAV file")
    if not wav:
        return

    try:
        print(f"  Loading SER ({backend})...")
        from core.ser import SER
        ser = SER(backend=backend)
        ok(f"SER ({backend}) loaded")

        print("  Predicting emotion...")
        emotion, confidence = ser.predict(wav)
        ok(f"Emotion: {emotion}  |  Confidence: {confidence:.4f}")
        return emotion, confidence
    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 4 — TEXT CLASSIFIER
# ============================================================================

def test_classifier():
    hdr("4. Text Classifier — RoBERTa multi-head")

    text = ask("Input sentence", "I can't believe you did this to me, I'm so angry")
    if not text:
        return

    try:
        print("  Loading classifier...")
        from core.classifier import TextEmotionClassifier
        clf = TextEmotionClassifier()
        ok("Classifier loaded")

        emotion, target, intensity = clf.classify(text)
        ok(f"Emotion:   {emotion}")
        ok(f"Target:    {target}")
        ok(f"Intensity: {intensity:.4f}")

        # Test multiple sentences
        more = ask("Test more sentences? (y/n)", "n")
        while more.lower() == "y":
            text2 = ask("Next sentence")
            if text2:
                e, t, i = clf.classify(text2)
                print(f"  → emotion={e}  target={t}  intensity={i:.4f}")
            more = ask("Another? (y/n)", "n")

        return emotion, target, intensity
    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 5 — FUSION
# ============================================================================

def test_fusion():
    hdr("5. Fusion Layer")

    print("  Enter SER output:")
    ser_emotion    = ask("SER emotion",    "sadness")
    ser_confidence = float(ask("SER confidence", "0.85"))

    print("\n  Enter Text Classifier output:")
    cl_emotion  = ask("CL emotion",    "sadness")
    cl_target   = ask("CL target",     "self")
    cl_intensity = float(ask("CL intensity", "0.72"))

    try:
        from core.fusion import fuse
        fused = fuse(
            cl_emotion=cl_emotion, cl_target=cl_target, cl_intensity=cl_intensity,
            ser_emotion=ser_emotion, ser_confidence=ser_confidence,
        )
        ok(f"Fused emotion:    {fused.emotion}")
        ok(f"Fused target:     {fused.target}")
        ok(f"Fused intensity:  {fused.intensity:.4f}")
        info("Agreement",       fused.agreement)
        info("SER emotion",     fused.ser_emotion)
        info("SER confidence",  fused.ser_confidence)
        return fused
    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 6 — STATE MEMORY
# ============================================================================

def test_state_memory():
    hdr("6. State Memory — Multi-turn simulation")

    try:
        from core.state_memory import StateMemory
        memory = StateMemory()
        ok("StateMemory initialized")

        print("\n  Simulating 5 turns (you can change inputs below):\n")

        turns = [
            ("sadness",   "self",      0.80),
            ("sadness",   "self",      0.70),
            ("anger",     "other",     0.40),   # weak signal — should not switch
            ("anger",     "other",     0.75),   # strong signal — should switch
            ("happiness", "situation", 0.60),
        ]

        override = ask("Use custom turns? (y/n)", "n")
        if override.lower() == "y":
            turns = []
            for i in range(1, 6):
                print(f"\n  Turn {i}:")
                e = ask("  emotion",   "sadness")
                t = ask("  target",    "self")
                iv = float(ask("  intensity", "0.5"))
                turns.append((e, t, iv))

        print()
        for i, (emotion, target, intensity) in enumerate(turns, 1):
            state = memory.update(emotion, target, intensity)
            print(f"  Turn {i:>2}  input=({emotion:<10} {target:<10} {intensity:.2f})"
                  f"  →  ai_emotion={state.emotion:<10} ai_intensity={state.ai_intensity:.4f}")

        print()
        snap = memory.snapshot()
        ok("Final state snapshot:")
        print_json(snap)

    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 7 — DECISION ENGINE
# ============================================================================

def test_decision_engine():
    hdr("7. Decision Engine")

    emotion  = ask("User emotion", "sadness")
    target   = ask("User target",  "self")
    intensity = float(ask("User intensity", "0.72"))

    try:
        from core.state_memory import StateMemory
        from core.decision_engine import DecisionEngine
        memory = StateMemory()
        engine = DecisionEngine()

        state = memory.update(emotion, target, intensity)
        decision = engine.decide(
            user_emotion=emotion, user_target=target,
            user_intensity=intensity, ai_state=state,
        )

        ok(f"Mode:             {decision.mode}")
        ok(f"Vector:           {decision.vector}")
        ok(f"Vector intensity: {decision.vector_intensity:.4f}")
        if decision.style_contract:
            sc = decision.style_contract
            ok(f"StyleContract:")
            info("  max_words",      sc.max_words)
            info("  max_sentences",  sc.max_sentences)
            info("  allow_bullets",  sc.allow_bullets)
            info("  allow_questions",sc.allow_questions)
            info("  allow_commands", sc.allow_commands)
            info("  profanity",      sc.profanity)

        return decision
    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 8 — LLM (steered generation)
# ============================================================================

def test_llm():
    hdr("8. LLM — Steered Response Generation")

    backend = ask("LLM backend (mistral / qwen)", "mistral")
    text    = ask("User message", "I just failed my exam and I feel terrible about it")
    emotion = ask("User emotion", "sadness")
    target  = ask("User target",  "self")
    intensity = float(ask("User intensity", "0.80"))

    try:
        print(f"  Loading LLM ({backend}) — this takes ~30s ...")
        from core.state_memory import StateMemory
        from core.decision_engine import DecisionEngine
        from core.llm import SteeredLLM

        memory = StateMemory()
        engine = DecisionEngine()
        llm    = SteeredLLM(backend=backend)
        ok("LLM loaded")

        state    = memory.update(emotion, target, intensity)
        decision = engine.decide(
            user_emotion=emotion, user_target=target,
            user_intensity=intensity, ai_state=state,
        )
        info("Decision mode",   decision.mode)
        info("Decision vector", decision.vector)

        print("\n  Generating response...")
        response = llm.generate(text, decision)
        ok(f"Response:\n\n  \"{response}\"\n")

        # Multi-turn test
        more = ask("Continue multi-turn? (y/n)", "n")
        while more.lower() == "y":
            text2     = ask("Next user message")
            emotion2  = ask("Emotion", emotion)
            target2   = ask("Target",  target)
            intensity2 = float(ask("Intensity", str(intensity)))
            state2    = memory.update(emotion2, target2, intensity2)
            decision2 = engine.decide(
                user_emotion=emotion2, user_target=target2,
                user_intensity=intensity2, ai_state=state2,
            )
            resp2 = llm.generate(text2, decision2)
            ok(f"Response: \"{resp2}\"")
            info("AI state emotion",    state2.emotion)
            info("AI state intensity",  state2.ai_intensity)
            more = ask("Another turn? (y/n)", "n")

    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 9 — TTS
# ============================================================================
def test_tts():
    hdr("9. TTS — Emotional Speech Synthesis")

    backend    = ask("TTS backend (f5tts / cosyvoice2)", "f5tts")
    text       = ask("Text to speak", "I'm really glad you're here. Things will get better, I promise.")
    speaker_id = ask("Speaker ID", "speaker_1")
    emotion    = ask("Emotion to steer (or 'none')", "happiness")
    alpha      = float(ask("Alpha (steering strength)", "1.0"))
    out_path   = ask("Output WAV path", "/tmp/tts_test_output.wav")

    try:
        print(f"  Loading TTS ({backend}) — may take ~20s ...")
        from core.tts import TTS
        from core.decision_engine import DecisionOutput, StyleContract

        tts = TTS(backend=backend)
        ok("TTS loaded")

        # Build a proper DecisionOutput with all required arguments
        style = StyleContract(
            max_words=50, max_sentences=3,
            allow_bullets=False, allow_questions=True,
            allow_commands=False, profanity=False,
        )
        
        vector = f"inject_{emotion}" if emotion != "none" else "none"
        
        # FIXED: Added all required positional arguments
        decision = DecisionOutput(
            mode="test", 
            vector=vector, 
            vector_intensity=alpha,
            style_contract=style,
            user_emotion=emotion,           # NEW
            user_target="self",              # NEW
            user_intensity=alpha,            # NEW
            ai_emotion=emotion,              # NEW
            ai_intensity=alpha               # NEW
        )

        print("  Synthesizing...")
        audio, sr = tts.synthesize(text=text, speaker_id=speaker_id, decision=decision)

        if audio is not None:
            save_audio(audio, sr, out_path)
        else:
            err("TTS returned None audio")

    except Exception:
        traceback.print_exc()

# ============================================================================
# MODULE 10 — RECOMMENDER
# ============================================================================

def test_recommender():
    hdr("10. Recommender — Sentence-Transformers + KB")

    text      = ask("User message", "I just failed my exam and feel awful about myself")
    emotion   = ask("Emotion",  "sadness")
    target    = ask("Target",   "self")
    vector    = ask("Vector",   "inject_happiness")

    try:
        print("  Loading Recommender...")
        from core.recommender import Recommender
        rec = Recommender()
        ok("Recommender loaded")

        results = rec.recommend(text=text, emotion=emotion, target=target, vector=vector)
        ok(f"{len(results)} recommendations:\n")

        for i, r in enumerate(results, 1):
            print(f"  {BOLD}[{i}] {r['title']}{RESET}  ({r['category']} | score={r['score']:.4f})")
            print(f"      {r['text']}")
            if r.get("options"):
                print(f"      Options: {' · '.join(r['options'])}")
            if r.get("tags"):
                print(f"      Tags: {', '.join(r['tags'])}")
            print()

        # Cascade level test
        print(f"\n  {BOLD}Cascade filter test:{RESET}")
        for level_desc, kwargs in [
            ("exact (emotion+target+vector)", {"emotion": emotion, "target": target, "vector": vector}),
            ("loose (emotion+target only)",   {"emotion": emotion, "target": target}),
            ("emotion only",                  {"emotion": emotion}),
            ("no filter (full KB)",           {}),
        ]:
            r = rec.recommend(text=text, **kwargs)
            print(f"  {level_desc:40s}  → {len(r)} results")

    except Exception:
        traceback.print_exc()


# ============================================================================
# MODULE 11 — FULL PIPELINE
# ============================================================================

def test_full_pipeline():
    hdr("11. Full Pipeline — End-to-End (1 turn)")

    wav        = ask_wav("Path to input WAV file")
    if not wav:
        return
    speaker_id = ask("Speaker ID", "speaker_1")
    out_path   = ask("Save audio output to", "/tmp/pipeline_output.wav")

    try:
        print("\n  Loading full pipeline (90-120s cold start)...")
        from pipeline import EmotionalAIPipeline
        pl = EmotionalAIPipeline()
        ok("Pipeline loaded\n")

        print("  Running pipeline (streaming steps)...\n")
        for status in pl.run_turn_stream(wav, speaker_id=speaker_id):
            if status["done"]:
                out = status["result"]
                break
            step  = status["step"]
            total = 7
            bar   = "█" * step + "░" * (total - step)
            print(f"  [{bar}] {step}/{total}  {status['label']}")

        print()
        ok(f"Transcript:       \"{out.transcript}\"")
        ok(f"SER emotion:      {out.ser_emotion}  ({out.ser_confidence:.2f})")
        ok(f"Text emotion:     {out.cl_emotion}  target={out.cl_target}  intensity={out.cl_intensity:.2f}")
        ok(f"Fused emotion:    {out.fused_emotion}  (agreement={out.emotion_agreement})")
        ok(f"AI emotion:       {out.ai_emotion}  intensity={out.ai_intensity:.2f}")
        ok(f"Decision:         mode={out.mode}  vector={out.vector}  vi={out.vector_intensity:.2f}")
        ok(f"Response text:    \"{out.response_text}\"")
        ok(f"Recommendations:  {len(out.recommendations)} items")

        if out.audio is not None:
            save_audio(out.audio, out.sample_rate, out_path)

    except Exception:
        traceback.print_exc()


# ============================================================================
# QUICK STRESS TEST — run classifier on many sentences
# ============================================================================

def test_classifier_batch():
    hdr("Bonus — Classifier Batch Test")

    sentences = [
        ("I'm so angry at you right now!",             "anger",     "you"),
        ("I feel so sad and alone today.",             "sadness",   "self"),
        ("This is amazing, I'm so happy!",             "happiness", "self"),
        ("I'm really scared about what might happen.", "fear",      "situation"),
        ("That behavior is absolutely disgusting.",    "disgust",   "other"),
        ("I can't believe you just did that!",         "surprise",  "you"),
        ("I hate myself for making that mistake.",     "anger",     "self"),
        ("She betrayed my trust completely.",          "anger",     "other"),
    ]

    try:
        print("  Loading classifier...")
        from core.classifier import TextEmotionClassifier
        clf = TextEmotionClassifier()
        ok("Classifier loaded\n")

        print(f"  {'Sentence':<48} {'Expected':<12} {'Got':<12} {'Target':<12} {'Intensity'}")
        print(f"  {'-'*100}")

        correct = 0
        for sentence, exp_emotion, exp_target in sentences:
            emotion, target, intensity = clf.classify(sentence)
            match = "✅" if emotion == exp_emotion else "❌"
            print(f"  {match} {sentence[:46]:<46}  "
                  f"{exp_emotion:<12} {emotion:<12} {target:<12} {intensity:.2f}")
            if emotion == exp_emotion:
                correct += 1

        print(f"\n  Accuracy: {correct}/{len(sentences)} = {correct/len(sentences)*100:.0f}%")

    except Exception:
        traceback.print_exc()


# ============================================================================
# MENU
# ============================================================================

MENU = """
{BOLD}{CYAN}  Emotional AI — Module Tester{RESET}
  ─────────────────────────────
   1.  Config check
   2.  ASR (Whisper)
   3.  SER (SenseVoice / WavLM)
   4.  Text Classifier (RoBERTa)
   5.  Fusion layer
   6.  State Memory (multi-turn sim)
   7.  Decision Engine
   8.  LLM (steered generation)
   9.  TTS (speech synthesis)
  10.  Recommender
  11.  Full pipeline (1 turn)
  12.  Classifier batch / stress test

   0.  Exit
"""


def main():
    handlers = {
        "1":  test_config,
        "2":  test_asr,
        "3":  test_ser,
        "4":  test_classifier,
        "5":  test_fusion,
        "6":  test_state_memory,
        "7":  test_decision_engine,
        "8":  test_llm,
        "9":  test_tts,
        "10": test_recommender,
        "11": test_full_pipeline,
        "12": test_classifier_batch,
    }

    while True:
        print(MENU.format(BOLD=BOLD, CYAN=CYAN, RESET=RESET))
        choice = input(f"  {YELLOW}Choose module (0-12):{RESET} ").strip()

        if choice == "0":
            print("\n  Bye!\n")
            break
        elif choice in handlers:
            try:
                handlers[choice]()
            except KeyboardInterrupt:
                warn("Interrupted — back to menu.")
        else:
            warn("Invalid choice.")

        input(f"\n  {YELLOW}Press Enter to return to menu...{RESET}")


if __name__ == "__main__":
    main()