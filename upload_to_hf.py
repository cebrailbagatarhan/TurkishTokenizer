"""Upload an explicitly selected tokenizer artifact and metadata to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from verify_tokenizer import verify_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo in owner/name form")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--repeat-tokenizer", type=Path, required=True)
    parser.add_argument("--repeat-manifest", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = [args.tokenizer, args.manifest, args.readme]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing upload files: {', '.join(missing)}")

    # Validate exactly the bytes that will be published, before any Hub mutation.
    snapshots = {path: path.read_bytes() for path in [*files, args.corpus_manifest]}
    report = verify_release(args.tokenizer, args.manifest, args.corpus,
                            args.corpus_manifest, args.repeat_tokenizer, args.repeat_manifest)
    if report["status"] != "pass":
        raise ValueError("Tokenizer release validation failed")
    if any(path.read_bytes() != content for path, content in snapshots.items()):
        raise ValueError("Upload inputs changed during verification")
    upload_plan = [
        (snapshots[args.tokenizer], "tokenizer.json"),
        (snapshots[args.manifest], "training_manifest.json"),
        (snapshots[args.corpus_manifest], "corpus_manifest.json"),
        (snapshots[args.readme], "README.md"),
        ((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), "release-evidence.json"),
    ]
    if args.dry_run:
        for _, destination in upload_plan:
            print(f"verified -> {args.repo_id}/{destination}")
        return 0

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Publish verified tokenizer and provenance bundle",
        operations=[CommitOperationAdd(path_in_repo=destination, path_or_fileobj=content)
                    for content, destination in upload_plan],
    )
    print(f"Uploaded {len(upload_plan)} explicit files to {args.repo_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
