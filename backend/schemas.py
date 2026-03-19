"""
schemas.py — Pydantic request/response schemas (auto-validation & docs)
"""

from __future__ import annotations
from datetime   import datetime
from typing     import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Single-text analysis request."""
    text: str = Field(..., min_length=5, max_length=10_000,
                      example="This movie was absolutely fantastic!")

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class BatchAnalyzeRequest(BaseModel):
    """Batch analysis request — up to 50 texts."""
    texts: List[str] = Field(..., min_length=1, max_length=50,
                              example=["I loved it!", "Terrible film."])

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: List[str]) -> List[str]:
        return [t.strip() for t in texts if t.strip()]


# ── Response schemas ──────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """Sentiment result for a single text."""
    id          : Optional[int]   = None
    text        : str
    sentiment   : str             # 'positive' | 'negative'
    confidence  : float           = Field(..., ge=0.0, le=1.0)
    confidence_pct: float         # human-friendly percentage
    model_name  : str
    word_count  : int
    char_count  : int
    created_at  : Optional[datetime] = None

    model_config = {"from_attributes": True}


class BatchAnalysisResult(BaseModel):
    """Response for a batch request."""
    batch_id    : str
    total       : int
    positive    : int
    negative    : int
    results     : List[AnalysisResult]


class HistoryItem(BaseModel):
    """Row returned from /history endpoint."""
    id          : int
    text        : str
    sentiment   : str
    confidence  : float
    confidence_pct: float
    model_name  : str
    word_count  : int
    char_count  : int
    is_batch    : bool
    batch_id    : Optional[str]
    created_at  : datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    total  : int
    page   : int
    size   : int
    items  : List[HistoryItem]


class StatsResponse(BaseModel):
    """Aggregated usage statistics."""
    total_analyses  : int
    positive_count  : int
    negative_count  : int
    positive_pct    : float
    negative_pct    : float
    avg_confidence  : float
    avg_word_count  : float
    model_name      : str


class HealthResponse(BaseModel):
    status      : str
    model_loaded: bool
    db_connected: bool
    version     : str
