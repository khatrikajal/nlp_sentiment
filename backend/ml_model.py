"""
ml_model.py — Loads the trained model & exposes inference helpers.

The module initialises once at startup (via lifespan) so no disk I/O
happens on every request.
"""

from __future__ import annotations

import re
import json
import logging
from pathlib  import Path
from typing   import List, Dict, Any

import joblib
import numpy  as np
import nltk
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.special import expit          # sigmoid — for LinearSVC probabilities

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR     = Path(__file__).parent / "model"
MODEL_PATH    = MODEL_DIR / "sentiment_model.pkl"
TFIDF_PATH    = MODEL_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


class SentimentEngine:
    """Singleton class that wraps the sklearn pipeline."""

    def __init__(self) -> None:
        self._model     = None
        self._tfidf     = None
        self._metadata  : Dict[str, Any] = {}
        self._lemmatizer = WordNetLemmatizer()

        # Ensure NLTK data is available
        for resource in ["stopwords", "punkt", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        self._stop_words = set(stopwords.words("english")) - {
            "not", "no", "never", "neither", "nor", "n't"
        }

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model artefacts from disk."""
        if not MODEL_PATH.exists():
            logger.warning("Model not found at %s. Run the notebook first!", MODEL_PATH)
            return

        logger.info("Loading model from %s", MODEL_PATH)
        self._model   = joblib.load(MODEL_PATH)
        self._tfidf   = joblib.load(TFIDF_PATH)

        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                self._metadata = json.load(f)

        logger.info("Model loaded — %s", self._metadata.get("best_model_name", "unknown"))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tfidf is not None

    @property
    def model_name(self) -> str:
        return self._metadata.get("best_model_name", "Unknown")

    # ── Text preprocessing ─────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Replicate the exact cleaning pipeline used during training."""
        text = text.lower()
        text = re.sub(r"<[^>]+>",          " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z'\s]",        " ", text)
        text = re.sub(r"\s+",              " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [self._lemmatizer.lemmatize(t)
                  for t in tokens
                  if t not in self._stop_words and len(t) > 1]
        return " ".join(tokens)

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a list of raw texts.

        Returns a list of dicts:
            {
                'sentiment'   : 'positive' | 'negative',
                'confidence'  : float (0-1),
                'clean_text'  : str,
                'word_count'  : int,
                'char_count'  : int,
            }
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Run the Jupyter notebook to train & save the model."
            )

        cleaned    = [self.clean_text(t) for t in texts]
        vectorized = self._tfidf.transform(cleaned)
        preds      = self._model.predict(vectorized)

        # ── Confidence scores ─────────────────────────────────────────────────
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(vectorized)  # shape (n, 2)
            # column 1 = probability of positive class
            pos_proba = proba[:, 1]
        else:
            # LinearSVC: convert decision function to [0,1] via sigmoid
            pos_proba = expit(self._model.decision_function(vectorized))

        results = []
        for original, clean, pred, pos_p in zip(texts, cleaned, preds, pos_proba):
            is_positive = int(pred) == 1
            confidence  = float(pos_p) if is_positive else float(1 - pos_p)

            results.append({
                "sentiment"  : "positive" if is_positive else "negative",
                "confidence" : round(confidence, 4),
                "clean_text" : clean,
                "word_count" : len(original.split()),
                "char_count" : len(original),
            })

        return results


# ── Module-level singleton (imported by main.py) ───────────────────────────────
engine = SentimentEngine()
