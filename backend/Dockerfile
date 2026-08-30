# Control-plane API — light (no torch). Build context = repo root.
# Base from ECR Public (not Docker Hub) to avoid anonymous-pull rate limits in CI.
FROM public.ecr.aws/docker/library/python:3.12-slim

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code + Alembic migrations (init_db runs `alembic upgrade head` at startup).
COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini

ENV PLATFORM_DATA_DIR=/data
RUN mkdir -p /data && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
