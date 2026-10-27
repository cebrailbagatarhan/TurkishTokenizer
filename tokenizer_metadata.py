"""Summarize a local Hugging Face tokenizers JSON artifact without dependencies."""

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict[str, object]:
    """Read tokenizer configuration and report counts, not vocabulary contents."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Tokenizer JSON must be an object")
    model = data.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError("Tokenizer JSON must contain model.vocab object")
    merges = model.get("merges", [])
    added = data.get("added_tokens", [])
    if not isinstance(merges, list) or not isinstance(added, list):
        raise ValueError("Tokenizer merges and added_tokens must be arrays")
    if any(not isinstance(item, dict) for item in added):
        raise ValueError("Each added token must be an object")

    def component_type(name: str) -> str | None:
        component = data.get(name)
        return component.get("type") if isinstance(component, dict) and isinstance(component.get("type"), str) else None

    return {
        "model_type": model.get("type") if isinstance(model.get("type"), str) else None,
        "vocabulary_size": len(model["vocab"]),
        "merge_count": len(merges),
        "added_token_count": len(added),
        "special_token_count": sum(item.get("special") is True for item in added),
        "pre_tokenizer_type": component_type("pre_tokenizer"),
        "decoder_type": component_type("decoder"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show offline tokenizer metadata without loading the tokenizers library")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "turkish_tokenizer.json")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.model)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value if value is not None else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
