import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('queue_builder', ROOT / 'build_location_layer_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueRebuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'inputs'
        self.db = self.root / 'pairs.sqlite'
        self.out = self.root / 'pairs.jsonl'
        for i in range(2):
            folder = self.inputs / f'shard-{i:03d}'
            folder.mkdir(parents=True)
            (folder / 'candidate_geocoding_input.jsonl').write_text(json.dumps({
                'predictions': [{'event_location': 'Paris;Paris;not_stated',
                                 'event_location_admin_level': 'city;city;not_stated'}]
            }) + '\n')

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            queue.build_queue(self.inputs, self.db, self.out, expected_shards=2)

    def test_rerun_replaces_counts_instead_of_adding(self):
        self.build()
        first = self.out.read_bytes()
        self.build()
        self.assertEqual(first, self.out.read_bytes())
        self.assertEqual(json.loads(first)['count'], 2)
        with sqlite3.connect(self.db) as con:
            self.assertEqual(con.execute('SELECT mention_count FROM pairs').fetchone()[0], 2)

    def test_malformed_input_preserves_previous_outputs(self):
        self.build()
        before = self.db.read_bytes(), self.out.read_bytes()
        (self.inputs / 'shard-001/candidate_geocoding_input.jsonl').write_text('invalid json\n')
        with self.assertRaises(json.JSONDecodeError):
            self.build()
        self.assertEqual(before, (self.db.read_bytes(), self.out.read_bytes()))

    def test_missing_shard_creates_no_outputs(self):
        (self.inputs / 'shard-001/candidate_geocoding_input.jsonl').unlink()
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.db.exists())
        self.assertFalse(self.out.exists())

    def test_active_database_sidecar_prevents_replacement(self):
        self.build()
        before = self.db.read_bytes(), self.out.read_bytes()
        Path(str(self.db) + '-wal').write_bytes(b'active')
        with self.assertRaises(RuntimeError):
            self.build()
        self.assertEqual(before, (self.db.read_bytes(), self.out.read_bytes()))


if __name__ == '__main__':
    unittest.main()
