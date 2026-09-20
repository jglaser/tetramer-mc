"""Standalone stdlib checks: python3 tests/viewer_coordinates.py."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_viewer', ROOT/'tools/export_viewer.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def row(sweep, shifts):
    return dict(sweep=sweep, counts={'center_shift': {'completed': shifts}})


class CoordinateHistoryTests(unittest.TestCase):
    def history(self, directory, rows):
        frames = [dict(sweep=r['sweep']) for r in rows]
        return exporter.coordinate_history(Path(directory), frames, rows, [0, 0, 0]), frames

    def test_persisted_values_are_authoritative_including_legacy_origin_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [dict(row(10, 10), coordinate_wall_center=[0, 0, 0], coordinate_origin_sweep=10),
                    dict(row(12, 12), coordinate_wall_center=[-3, 4, -5], coordinate_origin_sweep=10)]
            Path(directory, 'moves.jsonl').write_text('invalid log intentionally ignored\n')
            result, frames = self.history(directory, rows)
            self.assertTrue(result['available'])
            self.assertEqual(result['source'], 'persisted trajectory field')
            self.assertEqual(result['origin_sweep'], 10)
            self.assertEqual(frames[-1]['coordinate_wall_center'], [-3, 4, -5])

    def test_rust_shift_sum_is_negative_and_validated_at_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            moves = [dict(kind='center_shift', sweep=1, result={'displacement': [1, 2, 3]}),
                     dict(kind='gca', sweep=2),
                     dict(kind='center_shift', sweep=2, result={'displacement': [-4, 2, 1]})]
            Path(directory, 'moves.jsonl').write_text('\n'.join(map(json.dumps, moves)))
            result, frames = self.history(directory, [row(0, 0), row(1, 1), row(2, 2)])
            self.assertTrue(result['available'], result)
            self.assertEqual(frames[1]['coordinate_wall_center'], [-1, -2, -3])
            self.assertEqual(frames[2]['coordinate_wall_center'], [3, -4, -4])
            self.assertEqual(result['segment_shifts'], 2)
            failed, frames = self.history(directory, [row(0, 0), row(1, 0), row(2, 2)])
            self.assertFalse(failed['available'])
            self.assertIn('mismatch', failed['reason'])
            self.assertTrue(all('coordinate_wall_center' not in f for f in frames))

    def test_python_gzip_direction_distance_and_rejected_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            moves = [dict(kind='shift', sweep=1, direction=[0, 1, 0], distance=2.5, accepted=True),
                     dict(kind='shift', sweep=2, direction=[1, 0, 0], distance=9, accepted=False)]
            with gzip.open(Path(directory)/'moves.jsonl.gz', 'wt') as stream:
                stream.write('\n'.join(map(json.dumps, moves)))
            rows = [dict(sweep=0, counts={}), dict(sweep=2, counts={'shift': {'accepted': 1}})]
            result, frames = self.history(directory, rows)
            self.assertTrue(result['available'], result)
            self.assertEqual(frames[-1]['coordinate_wall_center'], [0, -2.5, 0])

    def test_unknown_resume_origin_is_disabled_known_checkpoint_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'previous-checkpoint.json'
            manifest = dict(initial_sweep=10, resume=str(checkpoint))
            (root/'manifest.json').write_text(json.dumps(manifest))
            checkpoint.write_text(json.dumps(dict(completed_sweeps=10)))
            (root/'moves.jsonl').write_text(json.dumps(dict(kind='center_shift', sweep=11, result={'displacement': [1, 0, 0]})))
            result, _ = self.history(directory, [row(10, 8), row(11, 9)])
            self.assertFalse(result['available'])
            self.assertIn('Legacy resume', result['reason'])
            checkpoint.write_text(json.dumps(dict(completed_sweeps=10, coordinate_wall_center=[3, 4, 5], coordinate_origin_sweep=2)))
            result, frames = self.history(directory, [row(10, 8), row(11, 9)])
            self.assertTrue(result['available'], result)
            self.assertEqual(frames[-1]['coordinate_wall_center'], [2, 4, 5])
            self.assertEqual(result['origin_sweep'], 2)

    def test_nonfinite_persisted_origin_is_not_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.history(directory, [dict(row(0, 0), coordinate_wall_center=[float('nan'), 0, 0])])
            self.assertFalse(result['available'])
            self.assertIn('finite', result['reason'])


if __name__ == '__main__':
    unittest.main()
