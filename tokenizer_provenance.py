"""Offline integrity checks shared by training and release validation.

Hashes bind files together; they do not authenticate an untrusted manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must be a JSON object: {path.name}")
    return value


def require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Missing or invalid SHA-256: {label}")
    return value


def verify_corpus(corpus: Path, manifest_path: Path) -> str:
    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported corpus manifest schema")
    splits = manifest.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    if not isinstance(train, dict) or train.get("path") != corpus.name:
        raise ValueError("Corpus must match the manifest's train split")
    expected = require_hash(train.get("sha256"), "train split")
    benchmark = splits.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("path") == train.get("path"):
        raise ValueError("Corpus manifest requires a separate benchmark split")
    require_hash(benchmark.get("sha256"), "benchmark split")
    if benchmark["sha256"] == train.get("sha256"):
        raise ValueError("Train and benchmark corpora must differ")
    actual = sha256_file(corpus)
    if actual != expected:
        raise ValueError("Corpus SHA-256 mismatch")
    return actual


def reproducibility_inputs(manifest: dict) -> dict:
    """Exclude timestamps, duration and machine-specific file locations."""
    return {
        "corpus_sha256": manifest["corpus"]["sha256"],
        "corpus_manifest_sha256": manifest["corpus"]["manifest_sha256"],
        "max_docs": manifest["corpus"]["max_docs"],
        "requested_vocab_size": manifest["artifact"]["requested_vocab_size"],
        "tokenizer": manifest["tokenizer"],
        "seed": manifest["seed"],
        "batch_size": manifest["batch_size"],
        "environment": manifest["environment"],
        "trainer_sha256": manifest["trainer_sha256"],
        "provenance_code_sha256": manifest["provenance_code_sha256"],
    }


def verify_artifact(artifact: Path, manifest_path: Path,
                    corpus: Path, corpus_manifest: Path) -> dict:
    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError("Release requires a schema 2 training manifest; retrain legacy artifacts")
    try:
        expected = require_hash(manifest["artifact"]["sha256"], "tokenizer artifact")
        if sha256_file(artifact) != expected:
            raise ValueError("Tokenizer artifact SHA-256 mismatch")
        corpus_hash = verify_corpus(corpus, corpus_manifest)
        if require_hash(manifest["corpus"]["sha256"], "training corpus") != corpus_hash:
            raise ValueError("Training manifest refers to a different corpus")
        if require_hash(manifest["corpus"]["manifest_sha256"], "corpus manifest") != sha256_file(corpus_manifest):
            raise ValueError("Corpus manifest SHA-256 mismatch")
        inputs = reproducibility_inputs(manifest)
        if require_hash(manifest.get("reproducibility_key"), "reproducibility key") != object_hash(inputs):
            raise ValueError("Training configuration integrity mismatch")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete training manifest") from error
    return manifest
