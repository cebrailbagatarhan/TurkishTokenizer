"""Inspect a trained Turkish BPE tokenizer from the command line."""

import argparse
from pathlib import Path


def inspect_text(model_path: str, text: str) -> tuple[list[str], list[int], str]:
    """Return token pieces, token IDs and decoded text."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(model_path)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return encoded.tokens, encoded.ids, tokenizer.decode(encoded.ids, skip_special_tokens=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Turkish tokenizer output")
    parser.add_argument("text", help="Text to tokenize")
    parser.add_argument("--model", default="turkish_tokenizer.json", help="Tokenizer JSON path")
    args = parser.parse_args(argv)
    if not Path(args.model).is_file():
        parser.error(f"Tokenizer file not found: {args.model}")
    try:
        tokens, ids, decoded = inspect_text(args.model, args.text)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Unable to load or use tokenizer: {exc}\n")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {decoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
