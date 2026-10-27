import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from vocab_audit import audit, main


class VocabularyAuditTests(unittest.TestCase):
    def setUp(self):
        self.data = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
                     "added_tokens": [{"id": 0, "content": "a", "special": True}]}

    def test_valid_vocab_and_matching_added_token(self):
        report = audit(self.data)
        self.assertTrue(report["ok"])
        self.assertEqual(report["vocab_entries"], 2)

    def test_gap_is_reported(self):
        self.data["model"]["vocab"]["b"] = 4
        report = audit(self.data)
        self.assertEqual(report["missing_vocab_ids"], 3)
        self.assertEqual(report["first_missing_ids"], [1, 2, 3])

    def test_duplicate_ids(self):
        self.data["model"]["vocab"]["b"] = 0
        self.assertIn("duplicate_vocab_ids", audit(self.data)["issues"])

    def test_bool_id_rejected(self):
        self.data["model"]["vocab"]["b"] = True
        self.assertIn("invalid_vocab_ids", audit(self.data)["issues"])

    def test_conflicting_added_token_and_duplicate(self):
        self.data["added_tokens"].append({"id": 0, "content": "not-a"})
        report = audit(self.data)
        self.assertEqual(report["duplicate_added_token_ids"], 1)
        self.assertEqual(report["added_token_vocab_conflicts"], 1)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            audit({"model": {"type": "WordPiece", "vocab": {"a": 0}}})

    def test_cli_json_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text(json.dumps(self.data), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main([str(path), "--json"]), 0)
            self.assertTrue(json.loads(stdout.getvalue())["ok"])
            self.data["model"]["vocab"]["b"] = 4
            path.write_text(json.dumps(self.data), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
