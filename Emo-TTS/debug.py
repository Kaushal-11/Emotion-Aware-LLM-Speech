import os
from pydub import AudioSegment
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# =====================================================
# CONFIG
# =====================================================
mp3_files = [
    "/workspace/audio-em/emo-tts/data/calm_female.mp3",
    "/workspace/audio-em/emo-tts/data/calm_male.mp3"
]

output_wav_dir = "wav_files"
os.makedirs(output_wav_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# =====================================================
# LOAD MODEL
# =====================================================
model_id = "openai/whisper-large-v3-turbo"

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

model.eval()

# =====================================================
# PROCESS FILES
# =====================================================
for mp3_path in mp3_files:

    base_name = os.path.splitext(os.path.basename(mp3_path))[0]

    wav_path = os.path.join(
        output_wav_dir,
        f"{base_name}.wav"
    )

    # MP3 -> WAV
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(wav_path, format="wav")

    print(f"Saved WAV: {wav_path}")

    # =================================================
    # LOAD WAV YOURSELF
    # =================================================
    speech, sr = librosa.load(
        wav_path,
        sr=16000,
        mono=True
    )

    # =================================================
    # FEATURE EXTRACTION
    # =================================================
    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(
        device=device,
        dtype=dtype
    )

    # =================================================
    # TRANSCRIPTION
    # =================================================
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            task="transcribe"
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    print("\nTranscript:")
    print(transcription)

    with open(
        f"{base_name}_transcript.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(transcription)