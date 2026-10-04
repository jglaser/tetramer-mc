"""Freeze a passive probe of evenly spaced slots in the completed two-root control.

No selection by outcome, contact, or native registry. This is a diagnostic of
clouds already used by those chains, not independent quadrature validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    bindings = {}

    def load(path, expected=None):
        path = Path(path).resolve()
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError('metadata exceeds 64 MiB')
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'changed input: {path}')
        bindings[str(path)] = digest
        return json.loads(path.read_text())

    # Frozen allocation before any trajectory is inspected.
    blocks = [513 + 256*j for j in range(16)]
    prefixes = [4096, 16384]
    plan = load(root/'execution-plan.json', '7c2cff9234f0620037f3afe42269d2f6343c29cccddd998131819b59335a0168')
    status = load(root/'execution/status.json')
    if not (status['complete'] and status['passed'] and status['active'] is None
            and status['failure'] is None and len(status['completed']) == 8):
        raise ValueError('all eight completed chains required')
    for row in status['completed']:
        if not (row['success'] and row['child_drained'] and row['returncode'] == 0):
            raise ValueError('failed or undrained chain')
        load(row['terminal']['path'], row['terminal']['sha256'])
    config = load(root/'config.json', 'b9b74b568b20573c04a12902429fd2a29b7ff35d0546f1864c77f139eaf71641')
    source = load(config['source_frame']['path'], config['source_frame']['sha256'])
    manifest_path = Path(config['preparation_output'])/'manifest.json'
    manifest = load(manifest_path, plan['files'][str(manifest_path)])
    shape = config['shape']
    load(shape['path'], shape['sha256'])
    groups, references = [], []
    output.mkdir(parents=True, exist_ok=False)
    write(output/'allocation.json', dict(blocks=blocks, prefixes=prefixes, chains=8,
        slots=128, maximum_score_evaluations=512, maximum_unpruned_evaluations=32,
        new_pose_draws=0, new_clouds=0, native_label_queries=0, retries=0))
    for job in sorted(config['jobs'], key=lambda x: x['id']):
        if job['context_index'] != 0 or job['arm'] != 'two_root_m4':
            raise ValueError('unexpected context or arm')
        directory = root/f"chains/job-{job['id']:03d}"
        terminal = load(directory/'terminal.json')
        if not terminal['complete'] or terminal['job'] != job:
            raise ValueError('terminal identity differs')
        trajectory = Path(terminal['trajectory']['path'])
        bank = next(b for b in manifest['cloud_banks'] if b['context_index'] == 0
                    and b['initialization'] == job['initialization'] and b['stream'] == job['stream'])
        metadata = load(bank['metadata']['path'], bank['metadata']['sha256'])
        if sha(bank['raw']['path']) != bank['raw']['sha256']:
            raise ValueError('cloud changed')
        bindings[bank['raw']['path']] = bank['raw']['sha256']
        cases = {}; h = hashlib.sha256()
        with trajectory.open('rb') as f:
            while line := f.readline(8*1024**2+1):
                if len(line) > 8*1024**2 or not line.endswith(b'\n'):
                    raise ValueError('oversize or partial journal row')
                h.update(line)
                match = re.search(rb'"block":\s*(\d+)[,}]', line)
                if not match or int(match[1]) not in blocks:
                    continue
                row = json.loads(line)
                if row.get('kind') != 'factorized_dimer':
                    continue
                block = row['block']
                if block in cases or row['canonical_members'] != [27, 132]:
                    raise ValueError('duplicate slot or changed labels')
                case_id = f"job-{job['id']:03d}-block-{block}"
                cases[block] = dict(id=case_id, old=row['canonical_old'],
                                    proposed=row.get('canonical_proposed'))
                reference = dict(id=case_id, job=job, block=block, status=row['status'],
                                 accepted=row['accepted'], root_slot=row['root_slot'])
                if row.get('bath') is not None:
                    counts = row['bath']['aggregate']
                    z = config['physical']['activity']; lam = z * config['physical']['lambda_ratio']
                    gained, lost = counts['gained'], counts['lost']
                    reference.update(saved_bath_log_weight=counts['log_weight'],
                        noisy_delta_physical_log_weight=z*(gained/lam-lost/(lam+z)),
                        estimated_count_variance=z*z*(gained/lam**2+lost/(lam+z)**2),
                        full_proposal_correction=row['complete_log_correction'])
                references.append(reference)
        if h.hexdigest() != terminal['trajectory']['sha256'] or sorted(cases) != blocks:
            raise ValueError('trajectory binding or fixed-slot coverage differs')
        bindings[str(trajectory)] = h.hexdigest()
        groups.append(dict(id=f"job-{job['id']:03d}", cloud=dict(raw=bank['raw'],
            low=metadata['low'], high=metadata['high'], raw_count=metadata['raw_count']),
            cases=[cases[b] for b in blocks]))
    scientific = dict(schema='dimer-surrogate-saved-endpoints-v1', shape=shape,
        radius=config['physical']['depletant_radius'], activity=config['physical']['activity'],
        spectators=[p for i,p in enumerate(source['poses']) if i not in [27,132]],
        prefixes=prefixes, groups=groups, unpruned_first_case=True)
    write(output/'plan.json', scientific)
    write(output/'references.json', references)
    write(output/'preparation.json', dict(complete=True, source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        input_sha256=bindings, plan_sha256=sha(output/'plan.json'),
        references_sha256=sha(output/'references.json'), all_slots_retained=True,
        scope='Selected by fixed block index. Reused guided clouds and nested prefixes are dependent; no independent error certification.'))
    return sha(output/'plan.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(prepare(args.root,args.output))
