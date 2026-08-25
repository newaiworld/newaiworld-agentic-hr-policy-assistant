#!/usr/bin/env bash

set -euo pipefail

python -m pip install -r requirements.txt

python -m rag.index build

python -m rag.index publish

python - <<'PY'
from pathlib import Path

from rag.chunk import (
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    TARGET_CHUNK_TOKENS,
)
from rag.embed import EMBEDDING_DIMENSION
from rag.ingest import load_manifest
from rag.store import (
    get_chroma_client,
    get_policy_collection,
    is_index_current,
    resolve_chroma_dir,
)


PROJECT_ROOT = Path.cwd()

manifest = load_manifest(
    PROJECT_ROOT / "corpus" / "version.json"
)

chroma_dir = resolve_chroma_dir()

client = get_chroma_client(
    chroma_dir
)

collection = get_policy_collection(
    client
)

count = collection.count()

print(
    f"published_index_count={count}"
)

if count != 400:
    raise SystemExit(
        "FAIL: expected 400 published policy chunks, "
        f"got {count}"
    )

current = is_index_current(
    chroma_dir,
    corpus_version=manifest.version,
    embedding_model=EMBEDDING_MODEL_NAME,
    embedding_dimension=EMBEDDING_DIMENSION,
    chunk_tokens=TARGET_CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS,
)

print(
    f"published_index_current={current}"
)

if not current:
    raise SystemExit(
        "FAIL: published policy index is stale"
    )

print(
    "deployment_index_verification=PASS"
)
PY
