"""Read-only structural checks for Hugging Face tokenizers BPE vocabularies.

This is not a tokenizer quality, security, or provenance assessment.
"""

import argparse
import json
from pathlib import Path


def audit(data):
    """Return counts and issue codes; never expose token strings."""
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise ValueError("tokenizer JSON must contain a model object")
    model = data["model"]
    if model.get("type") != "BPE":
        raise ValueError("only BPE tokenizer models are supported")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("BPE model must contain a nonempty vocab object")

    issues = []
    valid = {}
    invalid_ids = 0
    for token, token_id in vocab.items():
        if (not isinstance(token, str) or not isinstance(token_id, int)
                or isinstance(token_id, bool) or token_id < 0):
            invalid_ids += 1
        else:
            valid.setdefault(token_id, []).append(token)
    if invalid_ids:
        issues.append("invalid_vocab_ids")

    duplicate_ids = sum(len(tokens) - 1 for tokens in valid.values())
    if duplicate_ids:
        issues.append("duplicate_vocab_ids")
    ordered = sorted(valid)
    # Calculate gaps without allocating a potentially huge range of IDs.
    missing_count = ordered[-1] + 1 - len(ordered) if ordered else 0
    first_missing = []
    expected = 0
    for token_id in ordered:
        if token_id > expected and len(first_missing) < 10:
            first_missing.extend(range(expected, min(token_id, expected + 10 - len(first_missing))))
        expected = token_id + 1
    if missing_count:
        issues.append("noncontiguous_vocab_ids")

    added = data.get("added_tokens", [])
    if not isinstance(added, list):
        raise ValueError("added_tokens must be an array when present")
    seen_added = {}
    invalid_added = 0
    duplicate_added = 0
    conflicting_added = 0
    for entry in added:
        if (not isinstance(entry, dict) or not isinstance(entry.get("id"), int)
                or isinstance(entry.get("id"), bool) or entry["id"] < 0
                or not isinstance(entry.get("content"), str)):
            invalid_added += 1
            continue
        token_id, token_content = entry["id"], entry["content"]
        if token_id in seen_added:
            duplicate_added += 1
        seen_added[token_id] = token_content
        if token_id in valid and token_content not in valid[token_id]:
            conflicting_added += 1
    if invalid_added:
        issues.append("invalid_added_tokens")
    if duplicate_added:
        issues.append("duplicate_added_token_ids")
    if conflicting_added:
        issues.append("added_token_vocab_conflicts")

    return {
        "model_type": "BPE",
        "vocab_entries": len(vocab),
        "unique_valid_ids": len(valid),
        "invalid_vocab_ids": invalid_ids,
        "duplicate_vocab_ids": duplicate_ids,
        "missing_vocab_ids": missing_count,
        "first_missing_ids": first_missing,
        "added_token_entries": len(added),
        "invalid_added_tokens": invalid_added,
        "duplicate_added_token_ids": duplicate_added,
        "added_token_vocab_conflicts": conflicting_added,
        "issues": issues,
        "ok": not issues,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Local tokenizer JSON artifact")
    parser.add_argument("--json", action="store_true", help="Machine-readable summary")
    args = parser.parse_args(argv)
    try:
        with args.model.open(encoding="utf-8") as handle:
            report = audit(json.load(handle))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.exit(2, f"Audit could not run: {type(exc).__name__}\n")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"BPE vocabulary: {report['vocab_entries']} entries; "
              f"{report['missing_vocab_ids']} missing IDs; "
              f"{report['duplicate_vocab_ids']} duplicate IDs")
        print("Status: " + ("OK" if report["ok"] else ", ".join(report["issues"])))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
