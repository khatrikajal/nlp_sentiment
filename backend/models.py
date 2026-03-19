"""
models.py — SQLAlchemy ORM table definitions
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean

from database import Base


class AnalysisRecord(Base):
    """Stores every sentiment-analysis request made via the API."""
    __tablename__ = "analysis_records"

    id           = Column(Integer,  primary_key=True, index=True, autoincrement=True)
    text         = Column(Text,     nullable=False)
    clean_text   = Column(Text,     nullable=True)          # preprocessed version
    sentiment    = Column(String(10), nullable=False)       # 'positive' | 'negative'
    confidence   = Column(Float,    nullable=False)         # 0.0 – 1.0
    model_name   = Column(String(50), nullable=False)
    word_count   = Column(Integer,  nullable=True)
    char_count   = Column(Integer,  nullable=True)
    is_batch     = Column(Boolean,  default=False)          # part of a batch request?
    batch_id     = Column(String(36), nullable=True)        # UUID for batch grouping
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (f"<AnalysisRecord id={self.id} sentiment={self.sentiment} "
                f"confidence={self.confidence:.2f}>")


class ModelMetric(Base):
    """Aggregated model performance stats (written from notebook results)."""
    __tablename__ = "model_metrics"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    model_name   = Column(String(60), nullable=False, unique=True)
    accuracy     = Column(Float,   nullable=False)
    cv_mean      = Column(Float,   nullable=True)
    cv_std       = Column(Float,   nullable=True)
    vocab_size   = Column(Integer, nullable=True)
    sample_size  = Column(Integer, nullable=True)
    updated_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
