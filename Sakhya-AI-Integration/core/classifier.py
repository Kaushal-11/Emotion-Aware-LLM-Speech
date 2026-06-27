"""
core/classifier.py
-------------------
Layer 1 — Text Emotion Classifier

RoBERTa-base multi-head model that predicts, from a single sentence
(typically the ASR transcript of the user's speech):

    emotion   : one of 6 classes (anger, sadness, happiness, fear, disgust, surprise)
    target    : one of 4 classes (you, other, self, situation)
    intensity : continuous regression value in [0.0, 1.0]

This is the PRIMARY decider in the fusion layer — target and intensity
always come from here, and emotion is preferred over the SER emotion
unless SER strongly disagrees (see core/fusion.py).

Architecture and checkpoint format must exactly match the training script
that produced `best_model.pt` in CLASSIFIER_DIR.
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

from config import EMOTIONS, TARGETS_CL, CLASSIFIER_DIR, CLASSIFIER_MAX_LENGTH, DEVICE


EMOTION2ID = {e: i for i, e in enumerate(EMOTIONS)}
TARGET2ID  = {t: i for i, t in enumerate(TARGETS_CL)}
ID2EMOTION = {i: e for e, i in EMOTION2ID.items()}
ID2TARGET  = {i: t for t, i in TARGET2ID.items()}


class EmotionMultiHeadModel(nn.Module):
    """
    RoBERTa-base + 3 heads: emotion (6), target (4), intensity (1).

    NOTE: this architecture must match the training script exactly,
    otherwise strict state_dict loading will fail.
    """

    def __init__(self, model_name: str = "roberta-base", dropout: float = 0.1):
        super().__init__()
        self.roberta        = RobertaModel.from_pretrained(model_name)
        hidden_size         = self.roberta.config.hidden_size
        self.dropout        = nn.Dropout(dropout)
        self.emotion_head   = nn.Linear(hidden_size, len(EMOTIONS))
        self.target_head    = nn.Linear(hidden_size, len(TARGETS_CL))
        self.intensity_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls     = self.dropout(outputs.last_hidden_state[:, 0, :])
        emotion_logits  = self.emotion_head(cls)
        target_logits   = self.target_head(cls)
        intensity_preds = self.intensity_head(cls).squeeze(-1)
        intensity_preds = torch.clamp(intensity_preds, min=0.0, max=1.0)
        return emotion_logits, target_logits, intensity_preds


class TextEmotionClassifier:
    """
    Loads EmotionMultiHeadModel once and exposes a simple inference API.

    Usage
    -----
        clf = TextEmotionClassifier()
        emotion, target, intensity = clf.classify("I can't believe you did that!")
    """

    def __init__(self, checkpoint_dir: Path = CLASSIFIER_DIR, device: str = DEVICE):
        self.device = torch.device(device)

        self.tokenizer = RobertaTokenizer.from_pretrained(str(checkpoint_dir))

        self.model = EmotionMultiHeadModel("roberta-base", dropout=0.1)
        self.model.to(self.device)

        checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"   [Classifier] loaded from {checkpoint_dir}")

    @torch.no_grad()
    def classify(self, sentence: str) -> tuple[str, str, float]:
        """
        Parameters
        ----------
        sentence : raw text (e.g. Whisper ASR transcript)

        Returns
        -------
        emotion   : str   — e.g. "surprise"
        target    : str   — e.g. "you"
        intensity : float — in [0.0, 1.0]
        """
        encoding = self.tokenizer(
            sentence,
            max_length=CLASSIFIER_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        emotion_logits, target_logits, intensity_preds = self.model(
            input_ids, attention_mask
        )

        emotion_id = emotion_logits.argmax(dim=1).item()
        target_id  = target_logits.argmax(dim=1).item()
        intensity  = float(intensity_preds.item())

        return ID2EMOTION[emotion_id], ID2TARGET[target_id], intensity