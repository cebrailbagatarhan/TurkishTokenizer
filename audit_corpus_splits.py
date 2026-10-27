"""Read-only audit of prepared train/benchmark JSONL splits for exact text leakage.

Reports counts and line numbers, never corpus text. Requires only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inspect_split(path: Path) -> tuple[dict, set[str]]:
    seen: set[str] = set()
    report = {
        "lines": 0,
        "valid_documents": 0,
        "invalid_lines": 0,
        "first_invalid_lines": [],
        "duplicate_documents": 0,
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            report["lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip():
                report["invalid_lines"] += 1
                if len(report["first_invalid_lines"]) < 10:
                    report["first_invalid_lines"].append(line_number)
                continue
            report["valid_documents"] += 1
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                report["duplicate_documents"] += 1
            seen.add(digest)
    return report, seen


def audit_splits(train: Path, benchmark: Path) -> dict:
    """Check structural validity, exact duplicates and train/benchmark overlap."""
    if train.resolve() == benchmark.resolve():
        raise ValueError("train and benchmark must be different files")
    train_report, train_hashes = inspect_split(train)
    benchmark_report, benchmark_hashes = inspect_split(benchmark)
    overlap = len(train_hashes & benchmark_hashes)
    issues = []
    for name, report in (("train", train_report), ("benchmark", benchmark_report)):
        if not report["valid_documents"]:
            issues.append(f"empty_{name}")
        if report["invalid_lines"]:
            issues.append(f"invalid_{name}_records")
        if report["duplicate_documents"]:
            issues.append(f"duplicate_{name}_documents")
    if overlap:
        issues.append("train_benchmark_overlap")
    return {
        "train": train_report,
        "benchmark": benchmark_report,
        "shared_unique_documents": overlap,
        "issues": issues,
        "ok": not issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable counts")
    args = parser.parse_args(argv)
    try:
        report = audit_splits(args.train, args.benchmark)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"Audit could not run: {type(exc).__name__}\n")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Train: {report['train']['valid_documents']} documents; "
              f"benchmark: {report['benchmark']['valid_documents']} documents; "
              f"shared: {report['shared_unique_documents']}")
        print("Status: " + ("OK" if report["ok"] else ", ".join(report["issues"])))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
