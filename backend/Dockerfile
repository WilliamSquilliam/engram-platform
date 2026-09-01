# Control-plane API — light (no torch). Build context = repo root.
# Base from ECR Public (not Docker Hub) to avoid anonymous-pull rate limits in CI.
FROM public.ecr.aws/docker/library/python:3.12-slim

# System packages for document parsing:
#   libreoffice-writer -> `soffice` headless, used by parsing._soffice_to_docx to
#   convert genuine legacy OLE .doc files to .docx (absent, that one format fails
#   cleanly asking for a .docx). --no-install-recommends keeps the layer lean.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libreoffice-writer \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code + Alembic migrations (init_db runs `alembic upgrade head` at startup).
COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini

# /data is the single writable tree at runtime (ECS Fargate root FS is otherwise
# read-friendly but this keeps every cache in one owned dir). config.py derives
# STORAGE_DIR and FASTEMBED_CACHE_DIR from PLATFORM_DATA_DIR; pin the fastembed
# ONNX cache explicitly so the dense-retriever model downloads to a writable path.
ENV PLATFORM_DATA_DIR=/data \
    FASTEMBED_CACHE_DIR=/data/fastembed
RUN mkdir -p /data/fastembed && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
