"""Training-job dispatch, pluggable by JOB_BACKEND:

  inline (default) — FastAPI BackgroundTask in the API process. Zero-dependency,
                     fine locally, but a run dies if the API restarts mid-train.
  rq               — enqueue to Redis (RQ); a separate `app.worker` process runs
                     the job, so it survives API restarts and scales independently.

Both paths execute the SAME `routers.jobs._run_training`, so the progress-heartbeat
and cancel-flag contract is identical. (SQS/Temporal can slot in here later behind
the same enqueue() call — see C2/C3.)"""
import logging
import os

logger = logging.getLogger(__name__)

JOB_BACKEND = os.environ.get("JOB_BACKEND", "inline").lower()
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("JOB_QUEUE_NAME", "training")
# Max wall-clock for a training run before RQ kills it. Paper-scale runs take many
# hours, so this must exceed ML_TRAIN_TIMEOUT (see ml_client.py).
TRAIN_JOB_TIMEOUT = int(os.environ.get("TRAIN_JOB_TIMEOUT", "3600"))


def _queue():
    from redis import Redis
    from rq import Queue

    return Queue(QUEUE_NAME, connection=Redis.from_url(REDIS_URL))


def enqueue_training(background, corpus_id: str, job_id: str) -> None:
    """Dispatch a training run via the configured backend."""
    # Lazy import avoids a circular import (jobs imports this module).
    from .routers.jobs import _run_training

    if JOB_BACKEND == "rq":
        _queue().enqueue(_run_training, corpus_id, job_id, job_timeout=TRAIN_JOB_TIMEOUT)
        logger.info("Enqueued training job %s on RQ queue '%s'", job_id, QUEUE_NAME)
    else:
        background.add_task(_run_training, corpus_id, job_id)
