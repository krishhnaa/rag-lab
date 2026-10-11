"""Index metadata: record what built the index, and catch stale indexes.

An index built with one embedding model is meaningless (or crashes on a
dimension mismatch) when queried with another, so that's a hard error.
Chunking differences are only warnings: `ingest --strategy` deliberately
overrides config.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

META_FILE = "meta.json"


def write_meta(
    index_dir: Path, conf: dict, strategy: str, n_chunks: int, dim: int
) -> None:
    meta = {
        "embed_model": conf["ollama"]["embed_model"],
        "dim": dim,
        "strategy": strategy,
        "chunk_size": conf["chunking"]["chunk_size"],
        "chunk_overlap": conf["chunking"]["chunk_overlap"],
        "n_chunks": n_chunks,
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=1))


def check_meta(index_dir: Path, conf: dict) -> tuple[list[str], list[str]]:
    """Compare the index's meta.json with config. Returns (errors, warnings)."""
    path = index_dir / META_FILE
    reingest = "Re-run: python -m rag_lab ingest"
    if not path.exists():
        return [], [f"Index has no {META_FILE}; can't verify it matches config. {reingest}"]
    meta = json.loads(path.read_text())
    if "embed_model" not in meta:
        return [], [f"Index was built by an older version without model info. {reingest}"]

    errors, warnings = [], []
    want_model = conf["ollama"]["embed_model"]
    if meta["embed_model"] != want_model:
        errors.append(
            f"Index was embedded with {meta['embed_model']!r} but config uses "
            f"{want_model!r}. {reingest}"
        )
    for key in ("chunk_size", "chunk_overlap"):
        if meta.get(key) != conf["chunking"][key]:
            warnings.append(
                f"Index {key}={meta.get(key)} differs from config "
                f"({conf['chunking'][key]}). {reingest}"
            )
    if meta.get("strategy") != conf["chunking"]["strategy"]:
        warnings.append(
            f"Index uses strategy {meta.get('strategy')!r}, config says "
            f"{conf['chunking']['strategy']!r} (fine if you ingested with --strategy)."
        )
    return errors, warnings
