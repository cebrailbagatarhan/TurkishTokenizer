"""Train a manifest-backed Turkish Byte-Level BPE tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tokenizer_provenance import object_hash, reproducibility_inputs, sha256_file, verify_corpus


SPECIAL_TOKENS = [
    "<|begin_of_sentence|>",
    "<|end_of_sentence|>",
    "<|pad|>",
    "<|unk|>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL with a string 'text' field")
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=128000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.vocab_size < len(SPECIAL_TOKENS) + len(ByteLevel.alphabet()):
        parser.error("--vocab-size is too small for the byte alphabet and special tokens")
    for name in ("min_frequency", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_docs is not None and args.max_docs < 1:
        parser.error("--max-docs must be positive")
    if args.output is None:
        args.output = Path("artifacts") / f"turkish_bpe_{args.vocab_size}.json"
    return args


class CorpusBatches:
    def __init__(self, path: Path, batch_size: int, max_docs: int | None) -> None:
        self.path = path
        self.batch_size = batch_size
        self.max_docs = max_docs
        self.documents = 0
        self.characters = 0
        self.text_utf8_bytes = 0

    def __iter__(self) -> Iterator[list[str]]:
        batch: list[str] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if self.max_docs is not None and self.documents >= self.max_docs:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from error
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"Missing string 'text' at {self.path}:{line_number}")
                batch.append(text)
                self.documents += 1
                self.characters += len(text)
                self.text_utf8_bytes += len(text.encode("utf-8"))
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


def main() -> int:
    args = parse_args()
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    manifest_output = output.with_suffix(".manifest.json")
    if output.suffix != ".json" or output == manifest_output:
        raise ValueError("Output must be a tokenizer .json file")
    if {output, manifest_output} & {corpus, args.corpus_manifest.resolve()}:
        raise ValueError("Output must not overwrite corpus inputs")
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    if (output.exists() or manifest_output.exists()) and not args.overwrite:
        raise FileExistsError("Output exists. Pass --overwrite explicitly to replace it.")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    corpus_hash = verify_corpus(corpus, args.corpus_manifest)
    corpus_manifest_hash = sha256_file(args.corpus_manifest)

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=sorted(ByteLevel.alphabet()),
        show_progress=True,
    )
    batches = CorpusBatches(corpus, args.batch_size, args.max_docs)
    started = time.perf_counter()
    tokenizer.train_from_iterator(iter(batches), trainer=trainer, length=args.max_docs)
    elapsed = time.perf_counter() - started
    if not batches.documents or not batches.characters:
        raise ValueError("Training corpus contains no text")
    if sha256_file(corpus) != corpus_hash or sha256_file(args.corpus_manifest) != corpus_manifest_hash:
        raise ValueError("Corpus or corpus manifest changed during training")

    output.parent.mkdir(parents=True, exist_ok=True)
    # Stable JSON key order removes serialization-only differences from the hash.
    output.write_text(json.dumps(json.loads(tokenizer.to_str()), ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    artifact_hash = sha256_file(output)
    training_manifest = {
        "schema_version": 2,
        "artifact": {
            "path": output.name,
            "sha256": artifact_hash,
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        },
        "tokenizer": {
            "model": "BPE",
            "pre_tokenizer": "ByteLevel(add_prefix_space=False, use_regex=True)",
            "decoder": "ByteLevel",
            "byte_fallback": True,
            "min_frequency": args.min_frequency,
            "special_tokens": SPECIAL_TOKENS,
        },
        "corpus": {
            "path": str(corpus),
            "sha256": corpus_hash,
            "manifest": str(args.corpus_manifest.resolve()),
            "manifest_sha256": corpus_manifest_hash,
            "documents": batches.documents,
            "characters": batches.characters,
            "text_utf8_bytes": batches.text_utf8_bytes,
            "max_docs": args.max_docs,
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "training_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "tokenizers": version("tokenizers"),
            "tokenizers_parallelism": os.environ["TOKENIZERS_PARALLELISM"],
        },
        "command": [sys.executable, *sys.argv],
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trainer_sha256": sha256_file(Path(__file__)),
        "provenance_code_sha256": sha256_file(Path(__file__).with_name("tokenizer_provenance.py")),
    }
    training_manifest["reproducibility_key"] = object_hash(reproducibility_inputs(training_manifest))
    manifest_output.write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(training_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
