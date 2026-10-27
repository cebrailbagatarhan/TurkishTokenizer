import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from audit_corpus_splits import audit_splits, main


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.train = self.root / 'train.jsonl'
        self.benchmark = self.root / 'benchmark.jsonl'

    def write(self, train, benchmark):
        self.train.write_text(train, encoding='utf-8')
        self.benchmark.write_text(benchmark, encoding='utf-8')

    def test_disjoint_valid_splits(self):
        self.write('{"text":"İstanbul"}\n{"text":"Ankara"}\n', '{"text":"İzmir"}\n')
        report = audit_splits(self.train, self.benchmark)
        self.assertTrue(report['ok'])
        self.assertEqual(report['train']['valid_documents'], 2)
        self.assertEqual(report['shared_unique_documents'], 0)

    def test_exact_leakage_and_within_split_duplicates(self):
        self.write('{"text":"Türkçe"}\n{"text":"Türkçe"}\n', '{"text":"Türkçe"}\n')
        report = audit_splits(self.train, self.benchmark)
        self.assertFalse(report['ok'])
        self.assertEqual(report['train']['duplicate_documents'], 1)
        self.assertEqual(report['shared_unique_documents'], 1)
        self.assertIn('train_benchmark_overlap', report['issues'])

    def test_malformed_and_empty_text_are_counted_without_content(self):
        self.write('broken secret\n{"text":"  "}\n{"text":42}\n{"text":"valid"}\n', '{"text":"different"}\n')
        report = audit_splits(self.train, self.benchmark)
        self.assertEqual(report['train']['invalid_lines'], 3)
        self.assertEqual(report['train']['first_invalid_lines'], [1, 2, 3])
        self.assertNotIn('secret', json.dumps(report))

    def test_empty_split_fails(self):
        self.write('', '{"text":"valid"}\n')
        self.assertIn('empty_train', audit_splits(self.train, self.benchmark)['issues'])

    def test_same_file_rejected(self):
        self.write('{"text":"valid"}\n', '')
        with self.assertRaisesRegex(ValueError, 'different files'):
            audit_splits(self.train, self.train)

    def test_cli_exit_status_and_json(self):
        self.write('{"text":"one"}\n', '{"text":"two"}\n')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['--train', str(self.train), '--benchmark', str(self.benchmark), '--json']), 0)
        self.assertTrue(json.loads(output.getvalue())['ok'])
        self.write('{"text":"same"}\n', '{"text":"same"}\n')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--train', str(self.train), '--benchmark', str(self.benchmark)]), 1)


if __name__ == '__main__':
    unittest.main()
