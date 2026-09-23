"""Synthetic in-process workflow tests; no protein sampler or classifier is run."""
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_smc_guide_workflow as workflow

RUNNER = '''import json
from pathlib import Path
def runtime(): return {'python': 'fixture'}
def process_token(pid): return 'synthetic-process-birth'
def preflight(root):
    assert not (root/'status.json').exists()
    return json.loads((root/'protocol.json').read_text())
def run(root):
    (root/'physical-called').write_text('yes')
    if (root/'fail-physical').exists(): raise RuntimeError('physical failure')
    value={'complete':True,'phase':'complete'}
    (root/'status.json').write_text(json.dumps(value))
    return value
'''
ANALYZER = '''import hashlib,json
def analyze(root,out,workers):
    assert workers==4 and json.loads((root/'status.json').read_text())['complete']
    out.mkdir()
    if (root/'fail-analysis').exists(): raise RuntimeError('classification failure')
    value={'complete':True,'protocol_sha256':hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        'status_sha256':hashlib.sha256((root/'status.json').read_bytes()).hexdigest()}
    (out/'analysis.json').write_text(json.dumps(value))
    (out/'status.json').write_text(json.dumps({'complete':True,'phase':'complete'}))
    if (root/'mutate-status').exists(): (root/'status.json').write_text('{}')
    return value
'''


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); common = self.root / 'common'; common.mkdir()
        (common / workflow.NAME).write_text(Path(workflow.__file__).read_text())
        (common / 'run_smc_guide_pilot.py').write_text(RUNNER)
        (common / 'analyze_smc_guide_pilot.py').write_text(ANALYZER)
        self.protocol = dict(runtime={'python': 'fixture'},
            python_sources={p.name: workflow.sha(p) for p in common.iterdir()})
        (self.root / 'protocol.json').write_text(json.dumps(self.protocol))
        self.digest = workflow.sha(self.root / 'protocol.json')
        source = patch.object(workflow, '__file__', str(common / workflow.NAME)); source.start(); self.addCleanup(source.stop)
        old_path = sys.path.copy(); old_bytecode = sys.dont_write_bytecode
        old_modules = {name: sys.modules.pop(name, None) for name in ('run_smc_guide_pilot', 'analyze_smc_guide_pilot')}
        def restore():
            sys.path[:] = old_path; sys.dont_write_bytecode = old_bytecode
            for name, module in old_modules.items():
                sys.modules.pop(name, None)
                if module is not None: sys.modules[name] = module
        self.addCleanup(restore)

    def test_serial_success_preserves_physical_status_and_blocks_retry(self):
        before = signal.getsignal(signal.SIGTERM)
        result = workflow.run(self.root, self.digest)
        self.assertTrue(result['complete']); self.assertEqual(result['phase'], 'complete')
        self.assertEqual(workflow.sha(self.root / 'status.json'), result['physical_status_sha256'])
        self.assertEqual(workflow.sha(self.root / 'comparison/analysis.json'), result['comparison_sha256'])
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        self.assertFalse((self.root / 'common/__pycache__').exists())
        with self.assertRaisesRegex(ValueError, 'no retries'): workflow.run(self.root, self.digest)

    def test_physical_failure_prevents_classification(self):
        (self.root / 'fail-physical').touch()
        with self.assertRaisesRegex(RuntimeError, 'physical failure'): workflow.run(self.root, self.digest)
        self.assertFalse((self.root / 'comparison').exists())
        status = workflow.read(self.root / 'workflow-status.json')
        self.assertFalse(status['complete']); self.assertEqual(status['phase'], 'physical_and_audits_failed')

    def test_classification_failure_keeps_physical_complete(self):
        (self.root / 'fail-analysis').touch()
        with self.assertRaisesRegex(RuntimeError, 'classification failure'): workflow.run(self.root, self.digest)
        self.assertTrue(workflow.read(self.root / 'status.json')['complete'])
        self.assertFalse(workflow.read(self.root / 'workflow-status.json')['complete'])

    def test_checksum_and_source_changes_fail_before_claim(self):
        with self.assertRaisesRegex(ValueError, 'checksum changed'): workflow.run(self.root, '0' * 64)
        (self.root / 'common/analyze_smc_guide_pilot.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source changed'): workflow.run(self.root, self.digest)
        self.assertFalse((self.root / 'workflow-status.json').exists())

    def test_exclusive_claim_and_existing_comparison_prevent_dependent_work(self):
        (self.root / 'workflow-status.json').write_text('{}')
        with self.assertRaises(FileExistsError): workflow.run(self.root, self.digest)
        self.assertFalse((self.root / 'physical-called').exists())
        (self.root / 'comparison').mkdir()
        with self.assertRaisesRegex(ValueError, 'no retries'): workflow.run(self.root, self.digest)

    def test_analysis_cannot_change_terminal_physical_status(self):
        (self.root / 'mutate-status').touch()
        with self.assertRaisesRegex(ValueError, 'status changed'): workflow.run(self.root, self.digest)
        self.assertFalse(workflow.read(self.root / 'workflow-status.json')['complete'])

    def test_sigterm_drains_through_underlying_stage_and_restores_handler(self):
        runner, analyzer, protocol = workflow.load_archived(self.root, self.digest)
        before = signal.getsignal(signal.SIGTERM)
        def interrupt(root): signal.raise_signal(signal.SIGTERM)
        with patch.object(runner, 'run', interrupt):
            with self.assertRaises(InterruptedError): workflow.run(self.root, self.digest)
        self.assertFalse((self.root / 'comparison').exists())
        self.assertFalse(workflow.read(self.root / 'workflow-status.json')['complete'])
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


if __name__ == '__main__': unittest.main()
