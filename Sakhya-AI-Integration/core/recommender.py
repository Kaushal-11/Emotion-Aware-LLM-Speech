"""
core/recommender.py
--------------------
Recommender — Sentence-Transformers + curated JSON knowledge base (Option A)

Fully offline, loads once at startup:
    1. Embeds every entry's "title. text" in data/recommendation_kb.json
    2. On each turn, embeds the user's message
    3. Filters the KB using a CASCADING match against the same
       (emotion, target, vector) the DecisionEngine used for this turn —
       progressively loosening the filter until enough candidates remain
    4. Ranks the surviving candidates by cosine similarity to the user's
       message and returns the top-K

KB schema (data/recommendation_kb.json)
----------------------------------------
Each entry:
    {
        "id":       str,                       unique identifier
        "emotion":  str,                       one of the 6 emotions
        "target":   str,                       "you" | "other" | "self" | "situation"
        "vector":   str,                       e.g. "inject_happiness", "mirror_sadness", "none"
        "mode":     str,                       DecisionEngine mode label, e.g. "uplift + reframe"
        "category": str,                       free-form: "music" | "movie" | "activity" | "quote" | ...
        "title":    str,                       short display title
        "text":     str,                       description used for semantic search
        "options":  list[str]   (optional),    concrete suggestions (e.g. song names)
        "tags":     list[str]   (optional)     free-form tags for display/filtering
    }

`emotion`, `target`, and `vector` correspond directly to
core.decision_engine.TABLE keys/outputs, so recommendations line up with
exactly the same (target, emotion) combo that drove the LLM's response.

Cascading filter (most specific -> least specific)
-----------------------------------------------------
    1. emotion + target + vector   (exact decision match)
    2. emotion + target            (same situation, ignore vector)
    3. emotion                     (same feeling, any target)
    4. full KB                     (last resort)

The first level that yields >= top_k candidates is used. If the most
specific level already has >= top_k, it's used directly (richest match).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from config import RECOMMENDER_MODEL_NAME, RECOMMENDER_KB_PATH, RECOMMENDER_TOP_K


class Recommender:
    """
    Usage
    -----
        rec = Recommender()

        items = rec.recommend(
            text    = "I just failed my exam and feel awful",
            emotion = "sadness",
            target  = "self",
            vector  = "inject_happiness",
        )
        # -> list of dicts: title, text, category, emotion, target, vector,
        #                    mode, options, tags, score
    """

    def __init__(self,
                 model_name: str = RECOMMENDER_MODEL_NAME,
                 kb_path: Path = RECOMMENDER_KB_PATH,
                 top_k: int = RECOMMENDER_TOP_K):

        from sentence_transformers import SentenceTransformer

        self.top_k = top_k

        print(f"   [Recommender] loading '{model_name}' ...")
        self.model = SentenceTransformer(model_name)

        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb: List[Dict] = json.load(f)

        kb_texts = [f"{item['title']}. {item['text']}" for item in self.kb]
        self.kb_embeddings = self.model.encode(
            kb_texts, convert_to_numpy=True, normalize_embeddings=True
        )

        print(f"   [Recommender] loaded KB with {len(self.kb)} entries")

    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # both are L2-normalised already -> dot product == cosine similarity
        return matrix @ query_vec

    def _candidates_for_level(self, emotion: str, target: Optional[str], vector: Optional[str],
                               level: int) -> List[int]:
        """Return KB indices matching the given cascade level (1-4)."""
        idxs = range(len(self.kb))

        if level == 1 and target and vector:
            return [i for i in idxs
                    if self.kb[i]["emotion"] == emotion
                    and self.kb[i].get("target") == target
                    and self.kb[i].get("vector") == vector]

        if level == 2 and target:
            return [i for i in idxs
                    if self.kb[i]["emotion"] == emotion
                    and self.kb[i].get("target") == target]

        if level == 3:
            return [i for i in idxs if self.kb[i]["emotion"] == emotion]

        # level 4 — everything
        return list(idxs)

    # ------------------------------------------------------------------ #

    def recommend(self, text: str, emotion: str = None,
                   target: str = None, vector: str = None) -> List[Dict]:
        """
        Parameters
        ----------
        text    : user message (ASR transcript)
        emotion : fused emotion for this turn (e.g. "sadness")
        target  : fused target for this turn (e.g. "self") — from
                  core.fusion.FusedPerception.target
        vector  : steering vector chosen by DecisionEngine for this turn
                  (e.g. "inject_happiness") — from
                  core.decision_engine.DecisionOutput.vector

        `target` and `vector` are optional — if omitted, filtering falls
        back to emotion-only (cascade level 3).

        Returns
        -------
        list of up to top_k dicts:
            {id, title, text, category, emotion, target, vector, mode,
             options, tags, score}
        """
        query_vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        emotion = emotion.strip().lower() if emotion else None
        target  = target.strip().lower() if target else None
        vector  = vector.strip().lower() if vector else None

        candidates: List[int] = []
        if emotion:
            for level in (1, 2, 3):
                candidates = self._candidates_for_level(emotion, target, vector, level)
                if len(candidates) >= self.top_k:
                    break

        if len(candidates) < self.top_k:
            candidates = self._candidates_for_level(emotion or "", target, vector, 4)

        cand_embeddings = self.kb_embeddings[candidates]
        sims = self._cosine_sim(query_vec, cand_embeddings)

        top_local = np.argsort(sims)[::-1][: self.top_k]

        results = []
        for local_idx in top_local:
            kb_idx = candidates[local_idx]
            item = self.kb[kb_idx]
            results.append({
                "id":       item["id"],
                "title":    item["title"],
                "text":     item["text"],
                "category": item["category"],
                "emotion":  item.get("emotion"),
                "target":   item.get("target"),
                "vector":   item.get("vector"),
                "mode":     item.get("mode"),
                "options":  item.get("options", []),
                "tags":     item.get("tags", []),
                "score":    float(sims[local_idx]),
            })

        return results