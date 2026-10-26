"""Tests for the tokenizer inspection command (no model download required)."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import inspect_tokenizer


class InspectTokenizerTests(unittest.TestCase):
    def test_inspect_text_uses_model_and_round_trips(self):
        class FakeEncoding:
            tokens = ["Mer", "haba"]
            ids = [10, 20]

        class FakeTokenizer:
            def encode(self, text, add_special_tokens):
                self.assert_text = text
                assert add_special_tokens is False
                return FakeEncoding()

            def decode(self, ids, skip_special_tokens):
                assert ids == [10, 20]
                assert skip_special_tokens is False
                return "Merhaba"

        import sys
        from types import SimpleNamespace
        with patch.dict(sys.modules, {"tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: FakeTokenizer()))}):
            self.assertEqual(inspect_tokenizer.inspect_text("model.json", "Merhaba"), (["Mer", "haba"], [10, 20], "Merhaba"))

    def test_cli_prints_token_details(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tokenizer.json"
            model.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            with patch.object(inspect_tokenizer, "inspect_text", return_value=(["abc"], [42], "abc")), contextlib.redirect_stdout(output):
                self.assertEqual(inspect_tokenizer.main(["abc", "--model", str(model)]), 0)
            self.assertIn("Token IDs: [42]", output.getvalue())
            self.assertIn("Decoded: abc", output.getvalue())

    def test_missing_model_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                inspect_tokenizer.main(["hello", "--model", "/missing/model.json"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
