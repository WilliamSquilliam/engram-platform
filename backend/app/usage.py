"""Usage aggregation — the shared query layer both dashboards read (E10 tenant Admin,
E11 platform-admin). Rolls up the durable tables into the numbers the dashboards show:

  - corpora / documents / storage  -> Corpus + Document (TENANT-SCOPED, the isolation boundary)
  - onboarding GPU-seconds          -> Job (train_seconds lands on the corpus after a run)
  - queries + a daily series        -> Measurement (the served-query signal)

Isolation note: Measurement is a DEPLOYMENT-GLOBAL twin of the serving ring buffer — it carries no
tenant_id/corpus_id (see models.Measurement), so the `queries` count and daily series are a
deployment-level signal, not a per-tenant attribution. The load-bearing tenant-scoped facts
(corpora, documents, storage) ARE strictly filtered by tenant_id, which is what enforces
"tenant A can't see tenant B's data". Everything degrades to 0 / empty on no data — never errors.

Kept separate from the routers so the tenant view and the cross-tenant view compute identically."""
from __future__ import annotations

import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import Corpus, Document, Job, Measurement

# Default rollup window for the daily series (days). Env would be overkill; a plain constant.
USAGE_WINDOW_DAYS = 30

_BYTES_PER_GB = 1024 ** 3


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


def _day_bucket(dialect: str):
    """Portable 'YYYY-MM-DD' bucket over Measurement.created_at (SQLite strftime / Postgres to_char).
    Mirrors routers/metrics._month_bucket so the daily series works in dev/tests AND prod."""
    if dialect == "postgresql":
        return func.to_char(Measurement.created_at, "YYYY-MM-DD")
    return func.strftime("%Y-%m-%d", Measurement.created_at)


def _bytes_to_gb(n: int | None) -> float:
    return round((n or 0) / _BYTES_PER_GB, 4)


def query_series(db: Session, days: int = USAGE_WINDOW_DAYS) -> tuple[list[dict], int]:
    """Deployment-level daily served-query series + total over the window, zero-filled for every day
    so the chart has a continuous axis. `queries` counts cart-side Measurements (the served answers);
    if none exist yet, every day is 0. Returns (series, total)."""
    start = (_now() - datetime.timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    bucket = _day_bucket(db.bind.dialect.name)
    rows = (
        db.query(bucket.label("day"), func.count(Measurement.id))
        .filter(Measurement.side == "cart", Measurement.created_at >= start)
        .group_by("day")
        .all()
    )
    counts = {day: int(n or 0) for day, n in rows}
    series = []
    total = 0
    for i in range(days):
        d = (start + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        n = counts.get(d, 0)
        total += n
        series.append({"date": d, "queries": n})
    return series, total


def total_query_count(db: Session) -> int:
    """All-time cart-side served-query count (deployment-global). Used where a lifetime total,
    not a windowed series, is wanted (platform fleet rollup)."""
    return int(db.query(func.count(Measurement.id)).filter(Measurement.side == "cart").scalar() or 0)


def tenant_corpus_rollup(db: Session, tenant_id: str) -> list[dict]:
    """Per-corpus rollup for one tenant: documents, storage (summed doc sizes), gpu-seconds
    (summed job train_seconds), and the corpus query count. TENANT-SCOPED — only this tenant's
    corpora are ever returned. Empty list when the tenant has no corpora."""
    corpora = (
        db.query(Corpus)
        .filter(Corpus.tenant_id == tenant_id)
        .order_by(Corpus.created_at.desc())
        .all()
    )
    if not corpora:
        return []

    corpus_ids = [c.id for c in corpora]

    # Documents + storage per corpus (one grouped query, not N).
    doc_rows = (
        db.query(
            Document.corpus_id,
            func.count(Document.id),
            func.coalesce(func.sum(Document.size), 0),
        )
        .filter(Document.corpus_id.in_(corpus_ids))
        .group_by(Document.corpus_id)
        .all()
    )
    docs_by_corpus = {cid: (int(n or 0), int(size or 0)) for cid, n, size in doc_rows}

    out = []
    for c in corpora:
        n_docs, size_bytes = docs_by_corpus.get(c.id, (0, 0))
        out.append({
            "corpus_id": c.id,
            "name": c.name,
            "queries": 0,  # per-corpus query attribution isn't recorded (Measurement is global)
            "documents": n_docs,
            "storage_gb": _bytes_to_gb(size_bytes),
            # gpu-seconds this corpus's training consumed (last-run value on the corpus).
            "gpu_seconds": round(c.train_seconds or 0.0, 1),
        })
    return out


def tenant_usage(db: Session, tenant_id: str, days: int = USAGE_WINDOW_DAYS) -> dict:
    """Full usage rollup for ONE tenant (E10 /admin/usage). Corpora/documents/storage are strictly
    tenant-filtered; the query series is the deployment-level served-query signal. Never errors —
    a tenant with nothing yet gets zeros and empty lists."""
    by_corpus = tenant_corpus_rollup(db, tenant_id)
    documents = sum(c["documents"] for c in by_corpus)
    storage_gb = round(sum(c["storage_gb"] for c in by_corpus), 4)
    gpu_seconds = round(sum(c["gpu_seconds"] for c in by_corpus), 1)
    series, queries = query_series(db, days)
    return {
        "queries": queries,
        "documents": documents,
        "storage_gb": storage_gb,
        "gpu_seconds": gpu_seconds,
        "n_corpora": len(by_corpus),
        "by_corpus": by_corpus,
        "series": series,
    }


def tenant_gpu_seconds(db: Session, tenant_id: str) -> float:
    """Total onboarding GPU-seconds for a tenant (sum of its corpora's last-run train_seconds).
    Standalone so the platform rollup can total it per tenant without the full per-corpus payload."""
    total = (
        db.query(func.coalesce(func.sum(Corpus.train_seconds), 0.0))
        .filter(Corpus.tenant_id == tenant_id)
        .scalar()
    )
    return round(float(total or 0.0), 1)


def tenant_storage_gb(db: Session, tenant_id: str) -> float:
    """Total stored bytes -> GB for a tenant (summed document sizes across its corpora)."""
    total = (
        db.query(func.coalesce(func.sum(Document.size), 0))
        .join(Corpus, Document.corpus_id == Corpus.id)
        .filter(Corpus.tenant_id == tenant_id)
        .scalar()
    )
    return _bytes_to_gb(int(total or 0))


def tenant_document_count(db: Session, tenant_id: str) -> int:
    """Total document count for a tenant across its corpora."""
    n = (
        db.query(func.count(Document.id))
        .join(Corpus, Document.corpus_id == Corpus.id)
        .filter(Corpus.tenant_id == tenant_id)
        .scalar()
    )
    return int(n or 0)
