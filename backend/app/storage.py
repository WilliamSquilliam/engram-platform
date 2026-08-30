"""Storage abstraction. Selected by PLATFORM_STORAGE_BACKEND:

  local (default) — filesystem. Zero-dependency locally; on AWS point
                    PLATFORM_STORAGE_DIR at a shared EFS mount the GPU worker also
                    sees (the worker needs real file paths for cartridge I/O).
  s3              — S3/MinIO object store for durability/portability, with a local
                    working mirror so the worker still gets filesystem paths
                    (writes go to both; reads prefer the mirror, fall back to S3).

Documents keep their *relative path* as the key (e.g. "handbook/ch1/intro.md"), so
uploading whole folders preserves structure. Keys are POSIX strings — identical
shape locally and as S3 object keys."""
import logging
import os
import shutil
from pathlib import Path

from .config import STORAGE_DIR

logger = logging.getLogger(__name__)


def safe_rel(rel_path: str) -> str:
    """Normalize an uploaded (possibly folder-nested) path to a safe relative key:
    forward slashes, no drive/leading slash, no '..' traversal."""
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts) or "document.txt"


class LocalStorage:
    """Filesystem storage (local disk or a shared EFS mount)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def corpus_dir(self, corpus_id: str) -> Path:
        d = self.root / "corpora" / corpus_id
        (d / "docs").mkdir(parents=True, exist_ok=True)
        return d

    def save_document(self, corpus_id: str, filename: str, data: bytes) -> tuple[str, int]:
        rel = safe_rel(filename)
        path = self.corpus_dir(corpus_id) / "docs" / rel
        path.parent.mkdir(parents=True, exist_ok=True)  # nested folders
        path.write_bytes(data)
        return str(path.relative_to(self.root)), len(data)

    def read_text(self, corpus_id: str, filename: str) -> str:
        path = self.corpus_dir(corpus_id) / "docs" / safe_rel(filename)
        return path.read_bytes().decode("utf-8", errors="ignore")

    def list_doc_filenames(self, corpus_id: str) -> list[str]:
        docs = self.corpus_dir(corpus_id) / "docs"
        return sorted(p.relative_to(docs).as_posix() for p in docs.rglob("*") if p.is_file())

    def delete_corpus(self, corpus_id: str) -> None:
        d = self.root / "corpora" / corpus_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


class S3Storage(LocalStorage):
    """S3/MinIO-backed object store. Inherits the local mirror behaviour from
    LocalStorage (so the GPU worker still gets file paths) and additionally writes
    every document to S3 for durability, and deletes the S3 prefix on delete."""

    def __init__(self, root: Path, bucket: str, endpoint_url: str | None = None):
        super().__init__(root)
        import boto3

        self.bucket = bucket
        self._s3 = boto3.client("s3", endpoint_url=endpoint_url)

    def _key(self, corpus_id: str, rel: str) -> str:
        return f"corpora/{corpus_id}/docs/{rel}"

    def save_document(self, corpus_id: str, filename: str, data: bytes) -> tuple[str, int]:
        key, size = super().save_document(corpus_id, filename, data)
        self._s3.put_object(Bucket=self.bucket, Key=self._key(corpus_id, safe_rel(filename)), Body=data)
        return key, size

    def read_text(self, corpus_id: str, filename: str) -> str:
        local = self.corpus_dir(corpus_id) / "docs" / safe_rel(filename)
        if not local.exists():  # mirror miss (e.g. fresh worker) -> pull from S3
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._key(corpus_id, safe_rel(filename)))
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(obj["Body"].read())
        return super().read_text(corpus_id, filename)

    def list_doc_filenames(self, corpus_id: str) -> list[str]:
        # The local mirror is empty on a fresh worker (a separate task from the API
        # that did the upload), so list authoritatively from S3 and fall back to the
        # mirror only if S3 returns nothing. read_text() then pulls each on demand.
        prefix = f"corpora/{corpus_id}/docs/"
        names: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            names.extend(o["Key"][len(prefix):] for o in page.get("Contents", []))
        names = [n for n in names if n]
        return sorted(names) if names else super().list_doc_filenames(corpus_id)

    def delete_corpus(self, corpus_id: str) -> None:
        super().delete_corpus(corpus_id)
        prefix = f"corpora/{corpus_id}/"
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self._s3.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})


def _make_storage():
    backend = os.environ.get("PLATFORM_STORAGE_BACKEND", "local").lower()
    if backend == "s3":
        bucket = os.environ["PLATFORM_S3_BUCKET"]
        endpoint = os.environ.get("S3_ENDPOINT_URL") or None  # set for MinIO
        logger.info("Storage backend: s3 (bucket=%s)", bucket)
        return S3Storage(STORAGE_DIR, bucket, endpoint)
    return LocalStorage(STORAGE_DIR)


storage = _make_storage()
