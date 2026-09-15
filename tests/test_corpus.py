import json
from pathlib import Path
import tempfile
import unittest

from prepare_corpus import write_split
from tokenizer_provenance import object_hash, sha256_file, verify_corpus


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "train.jsonl"
        self.corpus.write_text('{"text":"İstanbul ve Türkçe"}\n', encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.data = {"schema_version": 1, "splits": {
            "train": {"path": "train.jsonl", "sha256": sha256_file(self.corpus)},
            "benchmark": {"path": "benchmark.jsonl", "sha256": "a" * 64}}}
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def test_valid_corpus(self):
        self.assertEqual(verify_corpus(self.corpus, self.manifest), sha256_file(self.corpus))

    def test_missing_or_invalid_hash_rejected(self):
        for value in (None, "", "not-a-hash", 123):
            with self.subTest(value=value):
                self.data["splits"]["train"]["sha256"] = value
                self.save()
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    verify_corpus(self.corpus, self.manifest)

    def test_corpus_tamper_rejected(self):
        self.corpus.write_text('{"text":"changed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_corpus(self.corpus, self.manifest)

    def test_identical_train_and_benchmark_rejected(self):
        self.data["splits"]["benchmark"]["sha256"] = self.data["splits"]["train"]["sha256"]
        self.save()
        with self.assertRaisesRegex(ValueError, "must differ"):
            verify_corpus(self.corpus, self.manifest)

    def test_missing_benchmark_rejected(self):
        del self.data["splits"]["benchmark"]
        self.save()
        with self.assertRaises(ValueError):
            verify_corpus(self.corpus, self.manifest)

    def test_non_object_manifest_rejected(self):
        self.manifest.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            verify_corpus(self.corpus, self.manifest)

    def test_deduplication_across_splits(self):
        documents = iter({"text": t} for t in ("Türkçe", "Türkçe", "İkinci", "İkinci", "Üçüncü"))
        seen = set()
        train = write_split(documents, self.corpus, 2, "text", seen)
        bench_path = self.root / "benchmark.jsonl"
        bench = write_split(documents, bench_path, 1, "text", seen)
        self.assertEqual(train["exact_duplicates_skipped"], 1)
        self.assertEqual(bench["exact_duplicates_skipped"], 1)
        self.assertNotIn("İkinci", bench_path.read_text(encoding="utf-8"))

    def test_failed_preparation_preserves_existing_file(self):
        original = self.corpus.read_bytes()
        with self.assertRaises(RuntimeError):
            write_split(iter([]), self.corpus, 1, "text")
        self.assertEqual(self.corpus.read_bytes(), original)

    def test_canonical_hash_ignores_dictionary_order(self):
        self.assertEqual(object_hash({"a": 1, "b": 2}), object_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
