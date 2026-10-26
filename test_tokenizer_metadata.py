"""Tests for dependency-free tokenizer metadata inspection."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tokenizer_metadata import main, summarize


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tokenizer.json"
        self.payload = {
            "model": {"type": "BPE", "vocab": {"merhaba": 0, "dünya": 1}, "merges": ["m e"]},
            "added_tokens": [{"id": 2, "special": True}, {"id": 3, "special": False}],
            "pre_tokenizer": {"type": "ByteLevel"},
            "decoder": {"type": "ByteLevel"},
        }
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_counts_and_component_types(self):
        result = summarize(self.path)
        self.assertEqual(result["vocabulary_size"], 2)
        self.assertEqual(result["merge_count"], 1)
        self.assertEqual(result["special_token_count"], 1)
        self.assertEqual(result["pre_tokenizer_type"], "ByteLevel")
        self.assertEqual(result["model_type"], "BPE")

    def test_json_cli_does_not_leak_token_text(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--model", str(self.path), "--json"]), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data["added_token_count"], 2)
        self.assertNotIn("merhaba", output.getvalue())

    def test_missing_required_vocab_is_rejected(self):
        self.path.write_text('{"model":{"type":"BPE"}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "model.vocab"):
            summarize(self.path)

    def test_invalid_merges_are_rejected(self):
        self.payload["model"]["merges"] = "invalid"
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "arrays"):
            summarize(self.path)

    def test_malformed_json_exits_with_error(self):
        self.path.write_text("{", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                main(["--model", str(self.path)])
        self.assertEqual(exit_info.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
