"""Prepare pinned, deterministic Turkish train and benchmark corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", default="20231101.tr")
    parser.add_argument("--dataset-revision", help="Branch, tag, or commit to resolve; defaults to main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--train-docs", type=int, default=150000)
    parser.add_argument("--benchmark-docs", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/turkish-wikipedia"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("shuffle_buffer_size", "train_docs", "benchmark_docs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def write_split(
    documents: Iterator[dict[str, Any]], path: Path, count: int, text_column: str
) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    written = 0
    characters = 0
    utf8_bytes = 0

    with temporary.open("wb") as handle:
        while written < count:
            try:
                document = next(documents)
            except StopIteration as error:
                raise RuntimeError(
                    f"Dataset ended after {written} usable documents while preparing {path.name}."
                ) from error
            text = document.get(text_column)
            if not isinstance(text, str) or not text.strip():
                continue
            line = (json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            written += 1
            characters += len(text)
            utf8_bytes += len(text.encode("utf-8"))

    os.replace(temporary, path)
    return {
        "path": path.name,
        "documents": written,
        "characters": characters,
        "text_utf8_bytes": utf8_bytes,
        "file_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.jsonl",
        "benchmark": output_dir / "benchmark.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {names}. Pass --overwrite explicitly.")

    requested_revision = args.dataset_revision or "main"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    dataset_info = api.dataset_info(args.dataset, revision=requested_revision)
    resolved_revision = dataset_info.sha

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
        revision=resolved_revision,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    documents = iter(dataset)

    started = time.perf_counter()
    train_stats = write_split(documents, paths["train"], args.train_docs, args.text_column)
    benchmark_stats = write_split(
        documents, paths["benchmark"], args.benchmark_docs, args.text_column
    )
    elapsed = time.perf_counter() - started

    manifest = {
        "schema_version": 1,
        "dataset": {
            "repo_id": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "text_column": args.text_column,
        },
        "sampling": {
            "seed": args.seed,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "order": "deterministic streaming shuffle; train documents followed by held-out benchmark documents",
        },
        "splits": {"train": train_stats, "benchmark": benchmark_stats},
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "datasets": version("datasets"),
            "huggingface_hub": version("huggingface_hub"),
        },
        "command": [sys.executable, *sys.argv],
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

