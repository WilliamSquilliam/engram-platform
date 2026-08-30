"""S3 storage backend: a fresh worker (empty local mirror) must still discover the
documents the API uploaded — by listing from S3, not the local filesystem. This is
the bug that broke training on AWS (worker saw 'no readable text documents')."""
from app.storage import S3Storage


class _FakePaginator:
    def __init__(self, keys):
        self._keys = keys

    def paginate(self, Bucket, Prefix):  # noqa: N803  (boto3 kwarg names)
        yield {"Contents": [{"Key": Prefix + k} for k in self._keys]}


class _FakeS3:
    def __init__(self, keys):
        self._keys = keys

    def get_paginator(self, _name):
        return _FakePaginator(self._keys)


def test_s3_lists_from_s3_when_mirror_empty(tmp_path):
    # Build the instance without __init__ so we don't need a real boto3 client.
    s = object.__new__(S3Storage)
    s.root = tmp_path
    s.bucket = "b"
    s._s3 = _FakeS3(["arxiv-001.txt", "sub/arxiv-002.txt"])
    # Local mirror is empty, so listing must come from S3 (sorted, prefix-stripped).
    assert s.list_doc_filenames("cid") == ["arxiv-001.txt", "sub/arxiv-002.txt"]
