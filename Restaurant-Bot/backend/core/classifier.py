"""
core/classifier.py — Text Emotion Classifier
Loads RoBERTa multi-head model from checkpoint.
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import RobertaModel, RobertaTokenizer

EMOTIONS   = ["anger", "sadness", "fear", "happiness", "disgust", "surprise"]
TARGETS_CL = ["you", "other", "self", "situation"]

EMOTION2ID = {e: i for i, e in enumerate(EMOTIONS)}
TARGET2ID  = {t: i for i, t in enumerate(TARGETS_CL)}
ID2EMOTION = {i: e for e, i in EMOTION2ID.items()}
ID2TARGET  = {i: t for t, i in TARGET2ID.items()}


class EmotionMultiHeadModel(nn.Module):
    """Multi-head classifier for emotion, target, and intensity."""
    
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.emotion_head = nn.Linear(hidden_size, len(EMOTIONS))
        self.target_head = nn.Linear(hidden_size, len(TARGETS_CL))
        self.intensity_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return (self.emotion_head(cls),
                self.target_head(cls),
                torch.clamp(self.intensity_head(cls).squeeze(-1), 0.0, 1.0))


class TextEmotionClassifier:
    """
    Loads RoBERTa multi-head model from checkpoint.
    Raises exception if model cannot be loaded.
    """

    def __init__(self, checkpoint_dir, device="cpu"):
        self.device = device
        self._model = None
        self._tokenizer = None
        
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided")
            
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        self._load_model(checkpoint_path, device)
        
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"Failed to load model from {checkpoint_dir}")

    def _load_model(self, checkpoint_dir, device):
        """Load model and tokenizer from checkpoint directory."""
        
        print(f"[Classifier] Loading model from {checkpoint_dir}...")
        
        # Load tokenizer
        try:
            self._tokenizer = RobertaTokenizer.from_pretrained(str(checkpoint_dir))
            print("[Classifier] Tokenizer loaded from checkpoint directory")
        except Exception as e:
            try:
                self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                print("[Classifier] Tokenizer loaded from roberta-base")
            except Exception as e2:
                raise RuntimeError(f"Failed to load tokenizer: {e2}")
        
        # Find checkpoint file
        checkpoint_files = []
        for pattern in ["*.pt", "*.pth", "*.bin"]:
            checkpoint_files.extend(list(checkpoint_dir.glob(pattern)))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files (.pt, .pth, .bin) found in {checkpoint_dir}")
        
        # Use the first checkpoint file found
        checkpoint_path = checkpoint_files[0]
        print(f"[Classifier] Loading checkpoint from {checkpoint_path}")
        
        try:
            # Initialize model
            model = EmotionMultiHeadModel()
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model.to(torch.device(device))
            print(f"[Classifier] Successfully loaded model from {checkpoint_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")

    def classify(self, sentence: str) -> tuple[str, str, float]:
        """
        Classify a sentence into emotion, target, and intensity.
        
        Args:
            sentence: Input text to classify
            
        Returns:
            Tuple of (emotion, target, intensity)
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Classifier not properly initialized")
            
        # Tokenize
        enc = self._tokenizer(
            sentence, 
            max_length=128, 
            padding="max_length",
            truncation=True, 
            return_tensors="pt"
        )
        
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            emotion_logits, target_logits, intensity = self._model(ids, mask)
        
        # Get results
        emotion = ID2EMOTION[emotion_logits.argmax(dim=1).item()]
        target = ID2TARGET[target_logits.argmax(dim=1).item()]
        intensity = round(float(intensity.item()), 3)
        
        return emotion, target, intensity