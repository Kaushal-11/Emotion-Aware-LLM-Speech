"""
core/asr.py
-----------
Layer 0a — Automatic Speech Recognition (Whisper)

Takes raw audio (file path or numpy array) and returns the transcript text.
This transcript is then fed into the text classifier (core/classifier.py).

Uses faster-whisper if available (much faster on GPU), falling back to
openai-whisper otherwise. Loaded ONCE at startup and kept in memory.
"""

from typing import Union
import numpy as np

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE


class ASR:
    """
    Usage
    -----
        asr = ASR()
        transcript = asr.transcribe("turn_001.wav")
        transcript = asr.transcribe(numpy_waveform, sample_rate=16000)
    """

    def __init__(self,
                 model_size: str = WHISPER_MODEL_SIZE,
                 device: str = WHISPER_DEVICE,
                 language: str = WHISPER_LANGUAGE):

        self.language = language
        self.backend = None

        # ── Try faster-whisper first (CTranslate2, much lower latency) ────────
        try:
            from faster_whisper import WhisperModel

            compute_type = "float16" if device == "cuda" else "int8"
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self.backend = "faster_whisper"
            print(f"   [ASR] faster-whisper '{model_size}' loaded on {device} ({compute_type})")

        except ImportError:
            # ── Fallback: openai-whisper ───────────────────────────────────────
            import whisper

            # openai-whisper doesn't ship "large-v3-turbo" under that name in
            # older versions — fall back to "large-v3" if turbo isn't found.
            try:
                self.model = whisper.load_model(model_size, device=device)
            except Exception:
                fallback_size = "large-v3" if "turbo" in model_size else model_size
                print(f"   [ASR] '{model_size}' not available, falling back to '{fallback_size}'")
                self.model = whisper.load_model(fallback_size, device=device)

            self.backend = "openai_whisper"
            print(f"   [ASR] openai-whisper '{model_size}' loaded on {device}")

    def transcribe(self, audio: Union[str, np.ndarray], sample_rate: int = 16000) -> str:
        """
        Parameters
        ----------
        audio       : path to a wav/mp3/etc file, OR a 1-D float32 numpy array
        sample_rate : required if `audio` is a numpy array (ignored for file paths)

        Returns
        -------
        transcript : str (stripped)
        """
        if self.backend == "faster_whisper":
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,         # skip silence — reduces latency
                beam_size=1,              # greedy decode for lower latency
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text

        else:  # openai_whisper
            kwargs = {}
            if self.language is not None:
                kwargs["language"] = self.language

            result = self.model.transcribe(audio, **kwargs)
            return result["text"].strip()