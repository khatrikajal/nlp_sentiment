"""
main.py — FastAPI application for Sentiment Analysis
──────────────────────────────────────────────────────
Run:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__  import annotations

import uuid
import logging
from contextlib  import asynccontextmanager
from datetime    import datetime, timezone
from typing      import List, Optional

from fastapi             import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors     import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses   import FileResponse
from sqlalchemy          import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib             import Path

from database  import get_db, init_db
from models    import AnalysisRecord
from schemas   import (AnalyzeRequest, BatchAnalyzeRequest,
                        AnalysisResult, BatchAnalysisResult,
                        HistoryResponse, HistoryItem,
                        StatsResponse, HealthResponse)
from ml_model  import engine as ml_engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load ML model. Shutdown: nothing needed."""
    logger.info("⚡  Starting up…")
    await init_db()
    ml_engine.load()
    logger.info("✅  Ready! Model loaded: %s", ml_engine.model_name)
    yield
    logger.info("🛑  Shutting down.")

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "🧠 Sentiment Analysis API",
    description = "NLP-powered sentiment analysis using TF-IDF + Logistic Regression",
    version     = "1.0.0",
    docs_url    = "/docs",
    lifespan    = lifespan,
)

# ── CORS (allows the frontend to call the API) ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Static files (plots, frontend) ────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, tags=["Frontend"])
async def root():
    """Serve the frontend HTML."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Sentiment Analysis API is running. Visit /docs for the API."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Check API, model, and database health."""
    db_ok = True
    try:
        await db.execute(select(func.count(AnalysisRecord.id)))
    except Exception:
        db_ok = False

    return HealthResponse(
        status       = "ok" if ml_engine.is_loaded and db_ok else "degraded",
        model_loaded = ml_engine.is_loaded,
        db_connected = db_ok,
        version      = "1.0.0",
    )


# ── Core analysis endpoints ────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_text(
    body: AnalyzeRequest,
    db  : AsyncSession = Depends(get_db),
):
    """
    Analyze the sentiment of a **single** text.

    Returns the predicted sentiment ('positive' or 'negative')
    along with a confidence score and word/character counts.
    """
    if not ml_engine.is_loaded:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run the notebook first.")

    result  = ml_engine.predict([body.text])[0]
    record  = AnalysisRecord(
        text         = body.text,
        clean_text   = result["clean_text"],
        sentiment    = result["sentiment"],
        confidence   = result["confidence"],
        model_name   = ml_engine.model_name,
        word_count   = result["word_count"],
        char_count   = result["char_count"],
        is_batch     = False,
    )
    db.add(record)
    await db.flush()   # populate record.id before returning

    return AnalysisResult(
        id             = record.id,
        text           = body.text,
        sentiment      = result["sentiment"],
        confidence     = result["confidence"],
        confidence_pct = round(result["confidence"] * 100, 1),
        model_name     = ml_engine.model_name,
        word_count     = result["word_count"],
        char_count     = result["char_count"],
        created_at     = record.created_at,
    )


@app.post("/analyze/batch", response_model=BatchAnalysisResult, tags=["Analysis"])
async def analyze_batch(
    body: BatchAnalyzeRequest,
    db  : AsyncSession = Depends(get_db),
):
    """
    Analyze sentiment for a **batch** of up to 50 texts at once.

    All items in the batch share a common `batch_id` for later retrieval.
    """
    if not ml_engine.is_loaded:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run the notebook first.")

    batch_id = str(uuid.uuid4())
    raw      = ml_engine.predict(body.texts)

    results: List[AnalysisResult] = []
    for original, res in zip(body.texts, raw):
        record = AnalysisRecord(
            text         = original,
            clean_text   = res["clean_text"],
            sentiment    = res["sentiment"],
            confidence   = res["confidence"],
            model_name   = ml_engine.model_name,
            word_count   = res["word_count"],
            char_count   = res["char_count"],
            is_batch     = True,
            batch_id     = batch_id,
        )
        db.add(record)
        await db.flush()
        results.append(AnalysisResult(
            id             = record.id,
            text           = original,
            sentiment      = res["sentiment"],
            confidence     = res["confidence"],
            confidence_pct = round(res["confidence"] * 100, 1),
            model_name     = ml_engine.model_name,
            word_count     = res["word_count"],
            char_count     = res["char_count"],
            created_at     = record.created_at,
        ))

    positive = sum(1 for r in results if r.sentiment == "positive")
    return BatchAnalysisResult(
        batch_id = batch_id,
        total    = len(results),
        positive = positive,
        negative = len(results) - positive,
        results  = results,
    )


# ── History endpoints ──────────────────────────────────────────────────────────

@app.get("/history", response_model=HistoryResponse, tags=["History"])
async def get_history(
    page      : int = Query(1, ge=1),
    size      : int = Query(20, ge=1, le=100),
    sentiment : Optional[str] = Query(None, pattern="^(positive|negative)$"),
    db        : AsyncSession = Depends(get_db),
):
    """
    Retrieve paginated analysis history.

    - Filter by sentiment with `?sentiment=positive` or `?sentiment=negative`.
    - Default: newest first.
    """
    query = select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
    if sentiment:
        query = query.where(AnalysisRecord.sentiment == sentiment)

    total_result = await db.execute(
        select(func.count()).select_from(query.subquery()))
    total = total_result.scalar_one()

    paginated = query.offset((page - 1) * size).limit(size)
    records   = (await db.execute(paginated)).scalars().all()

    items = [HistoryItem(
        id             = r.id,
        text           = r.text[:200] + ("…" if len(r.text) > 200 else ""),
        sentiment      = r.sentiment,
        confidence     = r.confidence,
        confidence_pct = round(r.confidence * 100, 1),
        model_name     = r.model_name,
        word_count     = r.word_count or 0,
        char_count     = r.char_count or 0,
        is_batch       = r.is_batch,
        batch_id       = r.batch_id,
        created_at     = r.created_at,
    ) for r in records]

    return HistoryResponse(total=total, page=page, size=size, items=items)


@app.delete("/history/{record_id}", tags=["History"])
async def delete_record(record_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a single analysis record by ID."""
    record = await db.get(AnalysisRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")
    await db.delete(record)
    return {"message": f"Record {record_id} deleted."}


@app.delete("/history", tags=["History"])
async def clear_history(db: AsyncSession = Depends(get_db)):
    """⚠️  Delete ALL analysis history."""
    from sqlalchemy import delete
    await db.execute(delete(AnalysisRecord))
    return {"message": "All history cleared."}


# ── Stats endpoint ─────────────────────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Aggregated statistics for the dashboard."""
    total_res = await db.execute(select(func.count(AnalysisRecord.id)))
    total     = total_res.scalar_one() or 0

    pos_res   = await db.execute(
        select(func.count(AnalysisRecord.id))
        .where(AnalysisRecord.sentiment == "positive"))
    positive  = pos_res.scalar_one() or 0
    negative  = total - positive

    avg_conf_res  = await db.execute(select(func.avg(AnalysisRecord.confidence)))
    avg_conf      = float(avg_conf_res.scalar_one() or 0)

    avg_words_res = await db.execute(select(func.avg(AnalysisRecord.word_count)))
    avg_words     = float(avg_words_res.scalar_one() or 0)

    return StatsResponse(
        total_analyses = total,
        positive_count = positive,
        negative_count = negative,
        positive_pct   = round(positive / total * 100, 1) if total else 0,
        negative_pct   = round(negative / total * 100, 1) if total else 0,
        avg_confidence = round(avg_conf * 100, 1),
        avg_word_count = round(avg_words, 1),
        model_name     = ml_engine.model_name,
    )
