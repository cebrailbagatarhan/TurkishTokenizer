import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, normalizers

from tokenizer_provenance import object_hash, reproducibility_inputs, sha256_file, verify_corpus
from train_turkish_tokenizer import CorpusBatches
from verify_tokenizer import regression_report, verify_release
import upload_to_hf


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.corpus = cls.root / "train.jsonl"
        # Tiny data is a correctness fixture, not a quality benchmark.
        samples = ["İstanbul'da güzel bir gün. Türkçe ığüşöç Iİıi.",
                   "evlerimizden kitaplarımızı getirdik", '{"kod": true, "sayı": 42}'] * 8
        cls.corpus.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n"
                                      for t in samples), encoding="utf-8")
        cls.benchmark = cls.root / "benchmark.jsonl"
        cls.benchmark.write_text('{"text":"Ayrı değerlendirme örneği"}\n', encoding="utf-8")
        cls.corpus_manifest = cls.root / "corpus-manifest.json"
        cls.corpus_manifest.write_text(json.dumps({
            "schema_version": 1, "dataset": {"kind": "synthetic test fixture"},
            "splits": {"train": {"path": "train.jsonl", "sha256": sha256_file(cls.corpus)},
                       "benchmark": {"path": "benchmark.jsonl", "sha256": sha256_file(cls.benchmark)}}
        }), encoding="utf-8")
        cls.artifacts = []
        for index in range(2):
            artifact = cls.root / f"run-{index}.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "train_turkish_tokenizer.py"),
                "--corpus", str(cls.corpus), "--corpus-manifest", str(cls.corpus_manifest),
                "--vocab-size", "300", "--output", str(artifact),
            ], env={**os.environ, "PYTHONHASHSEED": str(index)}, capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stderr)
            cls.artifacts.append(artifact)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def manifest(self, index=0):
        return self.artifacts[index].with_suffix(".manifest.json")

    def verify(self, **overrides):
        args = dict(tokenizer_path=self.artifacts[0], manifest_path=self.manifest(),
                    corpus=self.corpus, corpus_manifest=self.corpus_manifest,
                    repeat_tokenizer=self.artifacts[1], repeat_manifest=self.manifest(1))
        args.update(overrides)
        return verify_release(**args)

    def changed_manifest(self, mutation):
        value = json.loads(self.manifest().read_text())
        mutation(value)
        path = self.root / "changed-manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_two_independent_training_processes_are_byte_identical(self):
        self.assertEqual(self.artifacts[0].read_bytes(), self.artifacts[1].read_bytes())
        self.assertEqual(self.verify()["status"], "pass")

    def test_missing_repeat_is_incomplete(self):
        self.assertEqual(self.verify(repeat_tokenizer=None, repeat_manifest=None)["status"], "incomplete")

    def test_repeat_with_different_inputs_is_rejected(self):
        def mutate(value):
            value["seed"] += 1
            value["reproducibility_key"] = object_hash(reproducibility_inputs(value))
        path = self.changed_manifest(mutate)
        with self.assertRaisesRegex(ValueError, "different training inputs"):
            self.verify(repeat_manifest=path)

    def test_repeat_with_valid_manifest_but_different_bytes_is_rejected(self):
        artifact = self.root / "different-repeat.json"
        artifact.write_bytes(self.artifacts[1].read_bytes() + b" ")
        path = self.changed_manifest(lambda m: m["artifact"].update(sha256=sha256_file(artifact)))
        with self.assertRaisesRegex(ValueError, "different tokenizer SHA-256"):
            self.verify(repeat_tokenizer=artifact, repeat_manifest=path)

    def test_same_paths_do_not_count_as_repeat(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            self.verify(repeat_tokenizer=self.artifacts[0], repeat_manifest=self.manifest())

    def test_artifact_tampering_is_rejected(self):
        path = self.root / "tampered.json"
        path.write_bytes(self.artifacts[0].read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
            self.verify(tokenizer_path=path)

    def test_missing_hash_and_changed_configuration_are_rejected(self):
        for mutate in [lambda m: m["artifact"].pop("sha256"),
                       lambda m: m["artifact"].update(sha256=""),
                       lambda m: m.update(seed=123)]:
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValueError):
                    self.verify(manifest_path=self.changed_manifest(mutate))

    def test_corpus_manifest_is_bound_to_training(self):
        path = self.root / "changed-corpus-manifest.json"
        value = json.loads(self.corpus_manifest.read_text())
        value["dataset"]["kind"] = "changed metadata"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Corpus manifest SHA-256 mismatch"):
            self.verify(corpus_manifest=path)

    def test_changed_corpus_and_benchmark_split_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text('{"text":"different"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Corpus SHA-256 mismatch"):
                verify_corpus(path, self.corpus_manifest)
            benchmark = Path(directory) / "benchmark.jsonl"
            benchmark.write_bytes(self.corpus.read_bytes())
            with self.assertRaisesRegex(ValueError, "train split"):
                verify_corpus(benchmark, self.corpus_manifest)

    def test_missing_train_hash_does_not_skip_validation(self):
        path = self.root / "missing-hash.json"
        value = json.loads(self.corpus_manifest.read_text())
        value["splits"]["train"].pop("sha256")
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_corpus(self.corpus, path)

    def test_jsonl_rejects_non_object_missing_text_and_invalid_json(self):
        path = self.root / "bad.jsonl"
        for line in ["[]", "null", '{"text":42}', "{bad json", '{"text": "\\ud800"}']:
            with self.subTest(line=line):
                path.write_text(line + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(CorpusBatches(path, 2, None))

    def test_synthetic_regressions_roundtrip_and_report_fragmentation(self):
        report = self.verify()["regression"]
        self.assertEqual(report["failures"], [])
        by_name = {r["case"]: r for r in report["cases"]}
        self.assertTrue(by_name["secret_pattern"]["pattern_fragmented"])
        self.assertTrue(by_name["malformed_json"]["roundtrip"])

    def test_destructive_normalization_is_detected(self):
        tokenizer = Tokenizer.from_file(str(self.artifacts[0]))
        tokenizer.normalizer = normalizers.Lowercase()
        self.assertIn("turkish_unicode", regression_report(tokenizer)["failures"])

    def upload_argv(self, manifest=None):
        return ["upload_to_hf.py", "--repo-id", "test/synthetic", "--tokenizer", str(self.artifacts[0]),
                "--manifest", str(manifest or self.manifest()), "--corpus", str(self.corpus),
                "--corpus-manifest", str(self.corpus_manifest), "--repeat-tokenizer", str(self.artifacts[1]),
                "--repeat-manifest", str(self.manifest(1)), "--readme", str(ROOT / "README.md")]

    def test_upload_rejects_bad_manifest_before_network(self):
        path = self.changed_manifest(lambda m: m["artifact"].update(sha256="0" * 64))
        with patch.object(sys, "argv", self.upload_argv(path)), patch.object(upload_to_hf, "HfApi") as api:
            with self.assertRaises(ValueError):
                upload_to_hf.main()
            api.assert_not_called()

    def test_dry_run_validates_without_network(self):
        with patch.object(sys, "argv", self.upload_argv() + ["--dry-run"]), \
                patch.object(upload_to_hf, "HfApi") as api, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(upload_to_hf.main(), 0)
            api.assert_not_called()

    def test_upload_uses_one_commit_for_verified_bytes(self):
        with patch.object(sys, "argv", self.upload_argv()), patch.object(upload_to_hf, "HfApi") as api, \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(upload_to_hf.main(), 0)
            api.return_value.create_commit.assert_called_once()
            operations = api.return_value.create_commit.call_args.kwargs["operations"]
            self.assertEqual(len(operations), 5)
            self.assertEqual(operations[0].path_or_fileobj, self.artifacts[0].read_bytes())

    def test_cli_writes_deny_json_and_nonzero_exit(self):
        output = self.root / "denied.json"
        result = subprocess.run([sys.executable, str(ROOT / "verify_tokenizer.py"),
            "--tokenizer", str(self.root / "missing.json"), "--manifest", str(self.manifest()),
            "--corpus", str(self.corpus), "--corpus-manifest", str(self.corpus_manifest),
            "--output", str(output)], capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(output.read_text())["status"], "deny")


if __name__ == "__main__":
    unittest.main()
