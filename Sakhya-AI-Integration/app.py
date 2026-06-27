"""
app.py
------
Gradio Web UI — Emotional AI Speech-to-Speech Companion

Design
------
- LEFT  : Chat panel (message bubbles like WhatsApp)
          - User's side: transcript + detected emotion badge
          - AI's side:   response text + emotion badge + audio player
          - Below each AI turn: collapsible recommendations card
- RIGHT : Live pipeline progress + per-turn signal breakdown

- BOTTOM: Mic button (hold to record, release to send)
          Speaker selector + model toggles in a collapsible settings panel

GPU safety
----------
All models loaded on explicit single devices (config.DEVICE / config.LLM_DEVICE).
No device_map="auto" anywhere — that's what caused cross-GPU tensor errors.
"""

import os
import traceback
import time
import numpy as np

import gradio as gr

from config import (
    PRESET_SPEAKERS, DEFAULT_SPEAKER,
    SER_BACKEND, LLM_BACKEND, TTS_BACKEND,
    DEVICE, LLM_DEVICE,
)
from pipeline import EmotionalAIPipeline, PipelineOutput


# ============================================================================
# GPU info helper
# ============================================================================

def gpu_status_text():
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU only"
        lines = []
        for i in range(torch.cuda.device_count()):
            name   = torch.cuda.get_device_name(i)
            alloc  = torch.cuda.memory_allocated(i)  / 1024**3
            total  = torch.cuda.get_device_properties(i).total_memory / 1024**3
            lines.append(f"GPU {i} [{name}]  {alloc:.1f}/{total:.1f} GB used")
        lines.append(f"Light models → {DEVICE}   LLM → {LLM_DEVICE}")
        return "\n".join(lines)
    except Exception as e:
        return f"GPU info unavailable: {e}"


# ============================================================================
# Load pipeline ONCE
# ============================================================================

print("=" * 60)
print("Initializing Emotional AI pipeline …")
print(f"  Light models (ASR/SER/TTS/Classifier) → {DEVICE}")
print(f"  LLM                                   → {LLM_DEVICE}")
print("=" * 60)

pipeline = EmotionalAIPipeline()

print("Pipeline ready.\n")


# ============================================================================
# Formatting helpers
# ============================================================================

EMOTION_EMOJI = {
    "anger":     "😠", "sadness":  "😢", "happiness": "😊",
    "fear":      "😨", "disgust":  "🤢", "surprise":  "😲",
    "":          "😐",
}

STEP_TOTAL = 7


def emotion_badge(emotion: str, intensity: float = None) -> str:
    emoji = EMOTION_EMOJI.get(emotion, "🫥")
    label = f"{emoji} {emotion.capitalize()}"
    if intensity is not None:
        label += f" ({intensity:.0%})"
    return label


def progress_html(step: int, label: str) -> str:
    """Render a simple step progress bar as HTML."""
    pct = int(step / STEP_TOTAL * 100)
    bars = "".join(
        f'<div style="width:12px;height:12px;border-radius:50%;'
        f'background:{"#4CAF50" if i < step else "#e0e0e0"};'
        f'margin:0 3px;display:inline-block"></div>'
        for i in range(STEP_TOTAL)
    )
    return (
        f'<div style="font-family:sans-serif;padding:10px 0">'
        f'  <div style="margin-bottom:6px;font-size:14px">{label}</div>'
        f'  <div style="display:flex;align-items:center">'
        f'    {bars}'
        f'    <span style="margin-left:10px;color:#666;font-size:12px">'
        f'      {pct}%</span>'
        f'  </div>'
        f'</div>'
    )


def done_html() -> str:
    return (
        '<div style="font-family:sans-serif;padding:10px 0;color:#4CAF50;font-size:14px">'
        '✅ Response ready</div>'
    )


def build_chat_messages(history: list[dict]) -> list[tuple]:
    """
    Convert our history list into Gradio chatbot [(user_msg, ai_msg), ...] pairs.
    Each item in history is {"role": "user"|"ai", "html": str}.
    We pair them sequentially.
    """
    pairs = []
    i = 0
    while i < len(history):
        user_html = None
        ai_html   = None
        if history[i]["role"] == "user":
            user_html = history[i]["html"]
            i += 1
            if i < len(history) and history[i]["role"] == "ai":
                ai_html = history[i]["html"]
                i += 1
        elif history[i]["role"] == "ai":
            ai_html = history[i]["html"]
            i += 1
        pairs.append((user_html, ai_html))
    return pairs


def user_bubble(transcript: str, emotion: str, intensity: float) -> str:
    badge = emotion_badge(emotion, intensity)
    return (
        f'<div style="font-family:sans-serif">'
        f'  <div style="font-size:15px;margin-bottom:4px">{transcript}</div>'
        f'  <div style="font-size:12px;color:#888">{badge}</div>'
        f'</div>'
    )


def ai_bubble(response_text: str, emotion: str, intensity: float,
              mode: str, vector: str) -> str:
    badge = emotion_badge(emotion, intensity)
    return (
        f'<div style="font-family:sans-serif">'
        f'  <div style="font-size:15px;margin-bottom:6px">{response_text}</div>'
        f'  <div style="font-size:12px;color:#888">'
        f'    {badge} &nbsp;·&nbsp; mode: <em>{mode}</em>'
        f'    &nbsp;·&nbsp; vector: <code>{vector}</code>'
        f'  </div>'
        f'</div>'
    )


def format_signals_md(out: PipelineOutput) -> str:
    agree = "✅ agree" if out.emotion_agreement else "⚠️ disagree"
    return f"""
### 🔍 Per-turn signals

| Module | Output |
|---|---|
| 🎙️ SER (speech) | `{out.ser_emotion}` — confidence {out.ser_confidence:.2f} |
| 📝 Classifier (text) | `{out.cl_emotion}` — target `{out.cl_target}`, intensity {out.cl_intensity:.2f} |
| 🔗 Fusion | `{out.fused_emotion}` ({agree}) |
| 🧠 AI state | `{out.ai_emotion}` — intensity {out.ai_intensity:.2f} |
| ⚙️ Decision | mode `{out.mode}` · vector `{out.vector}` · strength {out.vector_intensity:.2f} |
"""


def format_recommendations_md(recs: list[dict]) -> str:
    if not recs:
        return "_No recommendations this turn._"
    lines = ["### 💡 Recommendations\n"]
    for r in recs:
        lines.append(f"**{r['title']}** _{r['category']}_\n{r['text']}")
        if r.get("options"):
            lines.append("- " + "\n- ".join(r["options"]))
        if r.get("tags"):
            lines.append(" ".join(f"`{t}`" for t in r["tags"]))
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Core callback — streaming generator
# ============================================================================

def process_audio(audio_path, speaker_id, chat_history, signals_state):
    """
    Gradio streaming generator.

    Yields tuples:
        (chatbot_pairs, progress_html, signals_md, recs_md, audio_out)

    `audio_path` is a filepath (Gradio Audio with type="filepath").
    """
    if audio_path is None:
        yield (
            build_chat_messages(chat_history),
            progress_html(0, "⚠️ No audio recorded — press the mic and speak."),
            signals_state,
            "_No recommendations yet._",
            None,
        )
        return

    # Show "Processing…" immediately
    yield (
        build_chat_messages(chat_history),
        progress_html(0, "⏳ Processing your audio…"),
        signals_state,
        "_Processing…_",
        None,
    )

    out: PipelineOutput = None

    try:
        for status in pipeline.run_turn_stream(audio_path, speaker_id=speaker_id):
            step  = status["step"]
            label = status["label"]

            if status["done"]:
                out = status["result"]
                break

            yield (
                build_chat_messages(chat_history),
                progress_html(step, label),
                signals_state,
                "_Processing…_",
                None,
            )

    except Exception as e:
        traceback.print_exc()
        yield (
            build_chat_messages(chat_history),
            f"<div style='color:red'>❌ Error: {e}</div>",
            signals_state,
            "_Error during processing._",
            None,
        )
        return

    if out is None:
        yield (
            build_chat_messages(chat_history),
            "<div style='color:red'>❌ Pipeline returned no output.</div>",
            signals_state,
            "_No output._",
            None,
        )
        return

    # ── Build chat bubbles ───────────────────────────────────────────────────
    chat_history.append({
        "role": "user",
        "html": user_bubble(out.transcript, out.fused_emotion, out.cl_intensity),
    })
    chat_history.append({
        "role": "ai",
        "html": ai_bubble(out.response_text, out.ai_emotion,
                          out.ai_intensity, out.mode, out.vector),
    })

    # ── Prepare audio output ─────────────────────────────────────────────────
    audio_out = (out.sample_rate, out.audio) if out.audio is not None else None

    signals_md   = format_signals_md(out)
    recs_md      = format_recommendations_md(out.recommendations)

    yield (
        build_chat_messages(chat_history),
        done_html(),
        signals_md,
        recs_md,
        audio_out,
    )


def reset_conversation():
    pipeline.reset()
    return (
        [],          # clear chatbot
        [],          # clear chat_history state
        "",          # clear signals
        "_Started a new conversation. Speak to begin._",
        None,        # clear audio
        progress_html(0, "Ready — press mic to speak"),
    )


def switch_ser(backend):
    try:
        pipeline.switch_ser_backend(backend)
        return f"✅ SER → **{backend}**"
    except Exception as e:
        return f"❌ {e}"


def switch_llm(backend):
    try:
        pipeline.switch_llm_backend(backend)
        return f"✅ LLM → **{backend}** (conversation reset)"
    except Exception as e:
        return f"❌ {e}"


def switch_tts(backend):
    try:
        pipeline.switch_tts_backend(backend)
        return f"✅ TTS → **{backend}**"
    except Exception as e:
        return f"❌ {e}"


# ============================================================================
# Gradio UI
# ============================================================================

speaker_choices = [(v["label"], k) for k, v in PRESET_SPEAKERS.items()]

CSS = """
.chat-col { display:flex; flex-direction:column; height:75vh; }
.chatbot  { flex:1; overflow-y:auto; }
footer    { display:none !important; }
"""

with gr.Blocks(title="Emotional AI", css=CSS, theme=gr.themes.Soft()) as demo:

    # ── State ─────────────────────────────────────────────────────────────
    chat_history  = gr.State([])   # list of {"role","html"} dicts
    signals_state = gr.State("")   # last signals markdown (persists between yields)

    # ── Header ────────────────────────────────────────────────────────────
    gr.Markdown("# 🤖 Emotional AI — Speech-to-Speech Companion")
    gr.Markdown(
        "Record your voice. The AI detects your emotion, steers its reply, "
        "and speaks back with matching emotional tone."
    )

    with gr.Row():
        # ── LEFT: Chat + Mic ──────────────────────────────────────────────
        with gr.Column(scale=3, elem_classes="chat-col"):

            chatbot = gr.Chatbot(
                label="Conversation",
                elem_classes="chatbot",
                height=480,
                bubble_full_width=False,
            )

            # AI audio output — auto-plays each turn
            audio_out = gr.Audio(
                label="🔊 AI voice response",
                autoplay=True,
                show_download_button=False,
            )

            with gr.Row():
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Hold mic button — speak — release",
                    waveform_options={"show_recording_waveform": True},
                )
                speaker_dd = gr.Dropdown(
                    choices=speaker_choices,
                    value=DEFAULT_SPEAKER,
                    label="AI Voice",
                    scale=1,
                )

            with gr.Row():
                send_btn    = gr.Button("▶ Send", variant="primary", scale=2)
                new_conv_btn = gr.Button("🔄 New Conversation", scale=1)

        # ── RIGHT: Progress + Signals + Recommendations ───────────────────
        with gr.Column(scale=2):

            progress_box = gr.HTML(
                value=progress_html(0, "Ready — record something to begin"),
                label="Pipeline progress",
            )

            signals_md = gr.Markdown(
                value="_Signals will appear after first turn._",
                label="Per-turn signals",
            )

            recs_md = gr.Markdown(
                value="_Recommendations will appear after first turn._",
                label="Recommendations",
            )

            # ── Settings (collapsible) ─────────────────────────────────
            with gr.Accordion("⚙️ Settings & Model Toggles", open=False):

                gr.Markdown("*Model reloads take ~10-40s. Conversation resets on LLM switch.*")

                with gr.Row():
                    ser_dd  = gr.Dropdown(["sensevoice","wavlm"],  value=SER_BACKEND, label="SER")
                    ser_btn = gr.Button("Apply")
                ser_status = gr.Markdown("")

                with gr.Row():
                    llm_dd  = gr.Dropdown(["mistral","qwen"],      value=LLM_BACKEND, label="LLM")
                    llm_btn = gr.Button("Apply")
                llm_status = gr.Markdown("")

                with gr.Row():
                    tts_dd  = gr.Dropdown(["f5tts","cosyvoice2"],  value=TTS_BACKEND, label="TTS")
                    tts_btn = gr.Button("Apply")
                tts_status = gr.Markdown("")

                with gr.Accordion("📊 GPU Memory", open=False):
                    gpu_md      = gr.Markdown(gpu_status_text())
                    refresh_btn = gr.Button("Refresh")
                    refresh_btn.click(fn=gpu_status_text, outputs=gpu_md)

    # ── Wiring ────────────────────────────────────────────────────────────

    # Send button (or auto-send on mic recording finish)
    send_btn.click(
        fn=process_audio,
        inputs=[mic_input, speaker_dd, chat_history, signals_state],
        outputs=[chatbot, progress_box, signals_md, recs_md, audio_out],
    )

    # Also trigger on mic recording completion (mic_input change)
    mic_input.stop_recording(
        fn=process_audio,
        inputs=[mic_input, speaker_dd, chat_history, signals_state],
        outputs=[chatbot, progress_box, signals_md, recs_md, audio_out],
    )

    new_conv_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[chatbot, chat_history, signals_md, recs_md, audio_out, progress_box],
    )

    ser_btn.click(fn=switch_ser, inputs=[ser_dd], outputs=[ser_status])
    llm_btn.click(fn=switch_llm, inputs=[llm_dd], outputs=[llm_status])
    tts_btn.click(fn=switch_tts, inputs=[tts_dd], outputs=[tts_status])


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )