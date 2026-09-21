#!/usr/bin/env python3
"""Execute one frozen outer-shell protocol exactly once; drain children on failure."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    protocol = json.loads((root/'protocol.json').read_text())
    assert protocol['schema'] == 'mobile-competing-outer-regions-campaign-v1'
    assert len(protocol['commands']) == 2 and protocol['maximum_physical_workers'] == 8
    for name, digest in protocol['source_sha256'].items():
        assert sha(root/'provenance'/name) == digest, name
    for name, digest in protocol['input_sha256'].items():
        assert sha(root/name) == digest, name
    for spec in protocol['commands']:
        assert not (root/spec['region']).exists(), 'Refuse existing population output'
    status = dict(running=True, complete=False, started=time.time(), jobs=[],
                  protocol_sha256=sha(root/'protocol.json'), driver_sha256=sha(Path(__file__)))
    with (root/'status.json').open('x') as stream:
        json.dump(status, stream, indent=2)

    def save():
        temp = root/'status.json.tmp'
        temp.write_text(json.dumps(status, indent=2)+'\n')
        temp.replace(root/'status.json')

    children = []
    try:
        try:
            for spec in protocol['commands']:
                stream = (root/(spec['region']+'.log')).open('x')
                try:
                    process = subprocess.Popen(spec['argv'], stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                record = dict(region=spec['region'], pid=process.pid, exit_code=None)
                status['jobs'].append(record)
                children.append((process, stream, record))
                save()
        finally:
            for process, stream, record in children:
                record['exit_code'] = process.wait()
                stream.close()
                save()
        assert len(children) == 2 and all(r['exit_code'] == 0 for _, _, r in children), 'Population failure; no retry'
        for spec, record in zip(protocol['commands'], status['jobs']):
            campaign = root/spec['region']
            manifest = json.loads((campaign/'manifest.json').read_text())
            assert manifest['region_sha256'] == spec['region_sha256']
            assert len(manifest['jobs']) == 4
            assert manifest['workers'] == 4 and manifest['cloud_replicates'] == 2
            assert manifest['lambda_ratio'] == 64.
            for actual, expected in zip(manifest['jobs'], spec['expected_jobs']):
                for field in ('id', 'seed', 'samples'):
                    assert actual[field] == expected[field]
                terminal = json.loads((campaign/(actual['id']+'-status.json')).read_text())
                assert terminal['returncode'] == 0
                assert sha(Path(actual['directory'])/'provenance/source-bundle.json') == protocol['source_sha256']['source-bundle.json']
            assert not (campaign/'assessment').exists(), 'Refuse repeated audit'
            argv = [sys.executable, '-B', str(campaign/'provenance/analyze_latent_region.py'), '--root', str(campaign)]
            with (root/(spec['region']+'-assessment.log')).open('x') as stream:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
            record['audit_exit_code'] = result.returncode
            save()
            assert result.returncode == 0, 'Audit failure; preserve output'
            record['assessment_sha256'] = sha(campaign/'assessment/analysis.json')
        status['complete'] = True
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        status['traceback'] = traceback.format_exc()
        raise
    finally:
        status.update(running=False, finished=time.time())
        save()
        print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
