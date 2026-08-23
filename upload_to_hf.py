"""Upload an explicitly selected tokenizer artifact and metadata to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo in owner/name form")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
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

    upload_plan = [
        (args.tokenizer, "tokenizer.json"),
        (args.manifest, "training_manifest.json"),
        (args.readme, "README.md"),
    ]
    if args.dry_run:
        for source, destination in upload_plan:
            print(f"{source} -> {args.repo_id}/{destination}")
        return 0

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    for source, destination in upload_plan:
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=destination,
            repo_id=args.repo_id,
            repo_type="model",
        )
    print(f"Uploaded {len(upload_plan)} explicit files to {args.repo_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

