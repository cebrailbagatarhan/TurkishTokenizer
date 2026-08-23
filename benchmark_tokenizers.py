"""Benchmark local and Hugging Face tokenizers on the same held-out Turkish corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable


WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
DEFAULT_SUFFIXES = (
    "lar", "ler", "lık", "lik", "luk", "lük", "cı", "ci", "cu", "cü",
    "dan", "den", "tan", "ten", "dır", "dir", "dur", "dür", "mış", "miş",
    "muş", "müş", "acak", "ecek", "ımız", "imiz", "umuz", "ümüz", "ınız",
    "iniz", "unuz", "ünüz",
)


def parse_labeled_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=VALUE")
    label, target = value.split("=", 1)
    if not label.strip() or not target.strip():
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=VALUE")
    return label.strip(), target.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--local", action="append", type=parse_labeled_spec, default=[])
    parser.add_argument("--hf", action="append", type=parse_labeled_spec, default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--max-chars-per-document", type=int, default=4000)
    parser.add_argument("--max-suffix-samples", type=int, default=2000)
    parser.add_argument("--suffix", action="append", dest="suffixes")
    parser.add_argument("--output-json", type=Path, default=Path("benchmark-results/tokenizers.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("benchmark-results/tokenizers.md"))
    args = parser.parse_args()
    for name in ("batch_size", "max_docs", "max_chars_per_document", "max_suffix_samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.local and not args.hf:
        parser.error("Provide at least one --local LABEL=PATH or --hf LABEL=REPO[@REVISION]")
    args.suffixes = tuple(dict.fromkeys(args.suffixes or DEFAULT_SUFFIXES))
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_corpus(path: Path, max_docs: int, max_chars: int) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(texts) >= max_docs:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Missing string 'text' at {path}:{line_number}")
            if text:
                texts.append(text[:max_chars])
    if not texts:
        raise ValueError("Benchmark corpus contains no usable documents")
    return texts


class LocalTokenizerAdapter:
    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer

        self.path = path.resolve()
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.identity = {"type": "local", "path": str(self.path), "sha256": sha256_file(self.path)}

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [item.ids for item in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self.tokenizer.encode(text, add_special_tokens=False).offsets


class HuggingFaceTokenizerAdapter:
    def __init__(self, target: str) -> None:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        if "@" in target:
            repo_id, requested_revision = target.rsplit("@", 1)
        else:
            repo_id, requested_revision = target, "main"
        token = os.environ.get("HF_TOKEN")
        resolved_revision = HfApi(token=token).model_info(
            repo_id, revision=requested_revision
        ).sha
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=resolved_revision,
            token=token,
            use_fast=True,
            trust_remote_code=False,
        )
        self.identity = {
            "type": "huggingface",
            "repo_id": repo_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "is_fast": bool(self.tokenizer.is_fast),
        }

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return encoded["input_ids"]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if not self.tokenizer.is_fast:
            raise NotImplementedError("Suffix offsets require a fast tokenizer")
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [tuple(item) for item in encoded["offset_mapping"]]


def batched(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def suffix_metrics(
    adapter: Any, texts: list[str], suffixes: tuple[str, ...], limit: int
) -> dict[str, dict[str, float | int]]:
    samples: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        for match in WORD_PATTERN.finditer(text):
            word = match.group(0)
            for suffix in suffixes:
                if len(samples[suffix]) >= limit or len(word) <= len(suffix):
                    continue
                if word[-len(suffix) :].casefold() == suffix.casefold():
                    samples[suffix].append(word)

    metrics: dict[str, dict[str, float | int]] = {}
    for suffix in suffixes:
        exact = 0
        overlapping_counts: list[int] = []
        for word in samples[suffix]:
            suffix_start = len(word) - len(suffix)
            offsets = [pair for pair in adapter.offsets(word) if pair != (0, 0)]
            overlapping = [
                pair for pair in offsets if pair[1] > suffix_start and pair[0] < len(word)
            ]
            overlapping_counts.append(len(overlapping))
            if len(overlapping) == 1 and overlapping[0] == (suffix_start, len(word)):
                exact += 1
        count = len(overlapping_counts)
        metrics[suffix] = {
            "samples": count,
            "exact_single_token_rate": exact / count if count else 0.0,
            "mean_overlapping_tokens": sum(overlapping_counts) / count if count else 0.0,
        }
    return metrics


def benchmark_one(
    label: str, adapter: Any, texts: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    warmup = texts[: min(len(texts), args.batch_size)]
    adapter.encode_batch(warmup)
    token_count = 0
    started = time.perf_counter()
    for batch in batched(texts, args.batch_size):
        token_count += sum(len(ids) for ids in adapter.encode_batch(batch))
    elapsed = time.perf_counter() - started
    words = sum(len(WORD_PATTERN.findall(text)) for text in texts)
    characters = sum(len(text) for text in texts)
    non_whitespace_characters = sum(sum(not char.isspace() for char in text) for text in texts)
    suffixes = suffix_metrics(adapter, texts, args.suffixes, args.max_suffix_samples)
    suffix_samples = sum(item["samples"] for item in suffixes.values())
    return {
        "label": label,
        "status": "passed",
        "identity": adapter.identity,
        "vocab_size": adapter.vocab_size,
        "documents": len(texts),
        "words": words,
        "characters": characters,
        "non_whitespace_characters": non_whitespace_characters,
        "tokens": token_count,
        "tokens_per_word": token_count / words if words else None,
        "tokens_per_character": token_count / characters if characters else None,
        "tokens_per_non_whitespace_character": token_count / non_whitespace_characters if non_whitespace_characters else None,
        "encode_seconds": elapsed,
        "encode_documents_per_second": len(texts) / elapsed,
        "encode_characters_per_second": characters / elapsed,
        "encode_tokens_per_second": token_count / elapsed,
        "suffix_samples": suffix_samples,
        "suffixes": suffixes,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Tokenizer benchmark results",
        "",
        f"Effective corpus SHA-256: `{report['corpus']['effective_text_sha256']}`",
        "",
        "| Tokenizer | Status | Vocab | Token/word | Token/char | Encode char/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        if result["status"] == "passed":
            lines.append(
                f"| {result['label']} | passed | {result['vocab_size']:,} | "
                f"{result['tokens_per_word']:.4f} | {result['tokens_per_character']:.4f} | "
                f"{result['encode_characters_per_second']:,.0f} |"
            )
        else:
            lines.append(f"| {result['label']} | failed: {result['error_type']} | — | — | — | — |")

    passed = [item for item in report["results"] if item["status"] == "passed"]
    if passed:
        lines.extend([
            "",
            "## Turkish suffix spans",
            "",
            "`exact rate`, ek karakter aralığının tam olarak tek tokenizer tokenıyla örtüşme oranıdır.",
            "",
            "| Tokenizer | Suffix | Samples | Exact rate | Mean overlapping tokens |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for result in passed:
            for suffix, metric in result["suffixes"].items():
                lines.append(
                    f"| {result['label']} | {suffix} | {metric['samples']} | "
                    f"{metric['exact_single_token_rate']:.4f} | {metric['mean_overlapping_tokens']:.4f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    corpus = args.corpus.resolve()
    texts = load_corpus(corpus, args.max_docs, args.max_chars_per_document)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": {
            "path": str(corpus),
            "file_sha256": sha256_file(corpus),
            "effective_text_sha256": hash_texts(texts),
            "documents_used": len(texts),
            "max_chars_per_document": args.max_chars_per_document,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "tokenizers": package_version("tokenizers"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface-hub"),
        },
        "settings": {
            "batch_size": args.batch_size,
            "max_suffix_samples": args.max_suffix_samples,
            "suffixes": args.suffixes,
            "special_tokens": False,
        },
        "results": [],
    }

    specs: list[tuple[str, str, str]] = [
        *( ("local", label, target) for label, target in args.local ),
        *( ("hf", label, target) for label, target in args.hf ),
    ]
    for kind, label, target in specs:
        try:
            adapter = (
                LocalTokenizerAdapter(Path(target))
                if kind == "local"
                else HuggingFaceTokenizerAdapter(target)
            )
            report["results"].append(benchmark_one(label, adapter, texts, args))
        except Exception as error:
            report["results"].append(
                {
                    "label": label,
                    "status": "failed",
                    "target": target,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if any(item["status"] == "passed" for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
