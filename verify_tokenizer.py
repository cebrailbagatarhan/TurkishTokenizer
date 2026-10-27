"""Validate tokenizer provenance, repeat training and synthetic regression cases offline."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from tokenizers import Tokenizer

from tokenizer_provenance import object_hash, sha256_file, verify_artifact


# These values are deliberately synthetic. Never add production credentials.
CASES = (
    ("turkish_unicode", "İstanbul ığüşöç Iİıi e\u0301 👩🏽‍💻", "İstanbul"),
    ("malformed_json", '{"anahtar": [1, 2,], "eksik": "', '"anahtar"'),
    ("code", 'def örnek(x):\n\treturn {"değer": x != None}\n# \\n', "!="),
    ("truncated_code", 'if (ready && value != null) {\n  run("', "&&"),
    ("secret_pattern", "api_key=sk-test-NOT-A-REAL-KEY-00000000", "sk-test-"),
    ("bearer_pattern", "Authorization: Bearer SYNTHETIC_CANARY_ONLY", "Bearer"),
    ("invisible_unicode", "secret=TEST\u200b_CANARY\u202e_ONLY", "TEST\u200b_CANARY"),
    ("control_characters", "a\x00b\r\n\tson", "\x00"),
    ("path_literal", '../../sentetik/dosya.txt C:\\demo\\test.txt', "../"),
    ("special_token_literal", "<|unk|> <|pad|> <|end_of_sentence|>", "<|unk|>"),
    ("empty", "", ""),
)


def regression_report(tokenizer: Tokenizer) -> dict:
    results = []
    failures = []
    # A separate deserialization catches serialization/configuration drift.
    restored = Tokenizer.from_str(tokenizer.to_str())
    for name, text, pattern in CASES:
        first = tokenizer.encode(text, add_special_tokens=False)
        second = restored.encode(text, add_special_tokens=False)
        roundtrip = tokenizer.decode(first.ids, skip_special_tokens=False) == text
        stable = first.ids == second.ids and first.offsets == second.offsets
        covered = all(any(a <= i < b for a, b in first.offsets) for i in range(len(text)))
        start = text.find(pattern)
        end = start + len(pattern)
        count = sum(b > start and a < end for a, b in first.offsets) if pattern else 0
        results.append({
            "case": name, "roundtrip": roundtrip, "stable_encoding": stable,
            "offsets_cover_text": covered,
            "tokens": len(first.ids), "pattern_tokens": count,
            "pattern_fragmented": count > 1,
        })
        if not roundtrip or not stable or not covered:
            failures.append(name)
    return {"cases_sha256": object_hash(CASES), "cases": results, "failures": failures}


def verify_release(tokenizer_path: Path, manifest_path: Path, corpus: Path,
                   corpus_manifest: Path, repeat_tokenizer: Path | None = None,
                   repeat_manifest: Path | None = None) -> dict:
    if (repeat_tokenizer is None) != (repeat_manifest is None):
        raise ValueError("Both repeat tokenizer and repeat manifest are required together")
    manifest = verify_artifact(tokenizer_path, manifest_path, corpus, corpus_manifest)
    repeated = False
    if repeat_tokenizer is not None:
        if repeat_tokenizer.samefile(tokenizer_path) or repeat_manifest.samefile(manifest_path):
            raise ValueError("Repeat run must use separate artifact and manifest paths")
        repeat = verify_artifact(repeat_tokenizer, repeat_manifest, corpus, corpus_manifest)
        if repeat["reproducibility_key"] != manifest["reproducibility_key"]:
            raise ValueError("Repeat run uses different training inputs or environment")
        if repeat["artifact"]["sha256"] != manifest["artifact"]["sha256"]:
            raise ValueError("Repeated training produced a different tokenizer SHA-256")
        repeated = True
    if manifest["environment"]["tokenizers"] != version("tokenizers"):
        raise ValueError("Validation must use the recorded tokenizers package version")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != manifest["artifact"]["actual_vocab_size"]:
        raise ValueError("Vocabulary size differs from training manifest")
    for index, token in enumerate(manifest["tokenizer"]["special_tokens"]):
        if tokenizer.token_to_id(token) != index:
            raise ValueError("Special token IDs differ from the training configuration")
    regression = regression_report(tokenizer)
    status = "deny" if regression["failures"] else ("pass" if repeated else "incomplete")
    return {
        "schema_version": 1, "status": status,
        "artifact_sha256": sha256_file(tokenizer_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "corpus_sha256": sha256_file(corpus),
        "repeat_manifest_sha256": sha256_file(repeat_manifest) if repeated else None,
        "verifier_sha256": sha256_file(Path(__file__)),
        "tokenizers_version": version("tokenizers"),
        "reproducibility_key": manifest["reproducibility_key"],
        "repeat_training_verified": repeated, "regression": regression,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--repeat-tokenizer", type=Path)
    parser.add_argument("--repeat-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release-evidence.json"))
    args = parser.parse_args()
    inputs = [args.tokenizer, args.manifest, args.corpus, args.corpus_manifest,
              args.repeat_tokenizer, args.repeat_manifest]
    if args.output.resolve() in {p.resolve() for p in inputs if p is not None}:
        parser.error("Evidence output must not overwrite an input file")
    try:
        report = verify_release(args.tokenizer, args.manifest, args.corpus,
                                args.corpus_manifest, args.repeat_tokenizer, args.repeat_manifest)
    except Exception as error:
        # CLI boundary: even parser/native tokenizer failures produce a deny report.
        report = {"schema_version": 1, "status": "deny", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
