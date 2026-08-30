"""RQ worker entrypoint (used when JOB_BACKEND=rq). Runs training jobs off the
Redis queue so they survive control-plane restarts.

    python -m app.worker

On AWS this is a separate ECS service / K8s deployment from the API."""
import logging

from .jobqueue import QUEUE_NAME, REDIS_URL
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    from redis import Redis
    from rq import Queue, Worker

    setup_logging()
    conn = Redis.from_url(REDIS_URL)
    logger.info("Starting RQ worker on queue '%s' (%s)", QUEUE_NAME, REDIS_URL)
    Worker([Queue(QUEUE_NAME, connection=conn)], connection=conn).work()


if __name__ == "__main__":
    main()
