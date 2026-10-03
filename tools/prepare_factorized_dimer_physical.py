#!/usr/bin/env python3
"""Freeze all audited passive rows for a later approved physical reset replay.

This program creates inputs/protocol only. It never invokes the executable or
draws a proposal, Poisson cloud or MH variate. A passed complete passive audit
is mandatory; no rows or atlases are selected by geometric/native outcomes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA = '0bde358181834b1848b2b6dbebe84d1b7747bb9faeb6d7a0ac5ce79b3b018237'
PROPOSAL_FILES = ['src/docking.rs', 'src/proposal.rs', 'src/basin_involution.rs', 'src/math.rs',
                  'src/defensive_dimer_proposal.rs', 'src/dimer_tree_proposal.rs',
                  'src/capped_dimer.rs', 'src/factorized_dimer.rs', 'src/geometry.rs', 'src/spherical.rs']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return record(target)


def cache_rows(lines):
    """Preserve one cache row per original line, including every exhausted call."""
    cached = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        if row['status'] != 'completed':
            raise ValueError('Fatal/incomplete passive row cannot become a physical rejection')
        candidate = row['outcome']['candidate']
        status = row['outcome']['status']
        if (candidate is None) != (status != 'candidate'):
            raise ValueError('Passive candidate/status mismatch')
        if candidate is None and status not in ('cap_exhausted', 'source_outside_domain', 'source_outside_contact'):
            raise ValueError('Unclassified passive null')
        if row['method'] not in ('whole_joint', 'factorized'):
            raise ValueError('Unexpected proposal method')
        for name in ('proposal_cpu_seconds', 'contact_diagnostic_cpu_seconds'):
            if not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError('Invalid cached timing')
        cached.append(dict(index=index, passive_row_sha256=hashlib.sha256(line.encode()).hexdigest(),
            **{k: row[k] for k in ('atlas', 'atlas_index', 'case', 'case_index', 'attempt', 'method',
                                 'old', 'anchor_pose', 'proposal_cpu_seconds', 'contact_diagnostic_cpu_seconds', 'raw_edge_draws')},
            proposal_status=status, candidate=candidate))
    return cached


def prepare(base, passive_base, executable, passive_review, audit_files):
    if sys.flags.optimize:
        raise RuntimeError('Preparation must run with assertions enabled; Python -O is forbidden')
    audit_path = passive_base / 'analysis.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['passed'] and not audit['failures'], 'Passive independent audit must pass first'
    review = read(passive_review)
    assert review['complete'] and review['passed'], 'Root passive completion review must pass first'
    assert review['output_hashes']['analysis.json'] == sha(audit_path), 'Root review binds a different analysis'
    assert audit_files and all(Path(p).is_file() for p in audit_files), 'Decision audit source closure required before preparation'
    run = passive_base / 'execution'
    for name, expected in audit['input_hashes'].items():
        assert sha(run / name) == expected, 'Changed passive audit input: ' + name
    terminal = read(run / 'terminal.json')
    assert terminal['summary']['complete'] and terminal['summary']['result']['outer_attempts'] == 1536
    passive_binding = read(run / 'binding.json')
    for name, key in [('config.json', 'config_sha256'), ('protocol.json', 'protocol_sha256'),
                      ('source-bundle.json', 'compiled_source_bundle_sha256'), ('example.rs', 'example_source_sha256')]:
        assert sha(run / name) == passive_binding[key], 'Changed passive binding: ' + name
    assert sha(run / 'attempts.jsonl') == terminal['attempts_sha256']
    passive_config = read(run / 'config.json')
    assert passive_config['density_law'] == 'map-factor-full-mixture-v1'
    assert [passive_config[k] for k in ('joint_cap', 'root_cap', 'internal_cap', 'factorized_joint_cap', 'attempts_per_context')] == [32, 32, 32, 1, 32]
    assert passive_config['factorized_order'] == 'root_first'
    reference_record = passive_config['reference_config']
    assert reference_record['sha256'] == REFERENCE_SHA == sha(reference_record['path'])
    reference = read(reference_record['path'])
    assert reference['depletant_radius'] == 1.4 and reference['activity'] == .0275
    assert len(reference['cases']) == 8 and len(reference['atlases']) == 3
    source_inputs = [reference[k] for k in ('panel', 'shape', 'source_config', 'source_frame', 'source_freeze_manifest')]
    source_inputs += [atlas['model'] for atlas in reference['atlases']]
    assert all(sha(r['path']) == r['sha256'] for r in source_inputs)
    passive_bundle = read(run / 'source-bundle.json')
    for name in PROPOSAL_FILES:
        assert sha(ROOT / name) == passive_bundle['files'][name]['sha256'], 'Cached proposal does not use current factor/source: ' + name
    source_files = [ROOT / name for name in ['Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md']]
    source_files += sorted((ROOT / 'src').rglob('*.rs'))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    bundles = []
    for path in (ROOT / 'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'):
        bundle = read(path)
        if {k: v['sha256'] for k, v in bundle['files'].items()} == hashes:
            bundles.append(path)
    assert bundles and len({sha(p) for p in bundles}) == 1, 'No exact compiled source bundle'
    rows = cache_rows((run / 'attempts.jsonl').read_text().splitlines())
    assert len(rows) == len(audit['rows']) == 1536
    keys = {(r['atlas_index'], r['case_index'], r['attempt'], r['method']) for r in rows}
    assert keys == {(a, c, t, m) for a in range(3) for c in range(8) for t in range(32) for m in ('whole_joint', 'factorized')}
    candidate_count = sum(r['candidate'] is not None for r in rows)
    assert candidate_count == audit['summary']['candidate_count'] == terminal['summary']['result']['candidates']
    base.mkdir()
    common = base / 'common'
    common.mkdir()
    passive_files = {}
    for name in audit['input_hashes']:
        passive_files[name] = copy(run / name, common / 'passive' / name)
    passive_files['analysis.json'] = copy(audit_path, common / 'passive/analysis.json')
    passive_files['root-review.json'] = copy(passive_review, common / 'passive/root-review.json')
    audit_records = {}
    for path in audit_files:
        path = Path(path)
        assert path.name not in audit_records, 'Duplicate decision audit basename'
        audit_records[path.name] = copy(path, common / 'audit' / path.name)
    archive_inputs = [copy(Path(r['path']), common / 'inputs' / f'{i:02d}-{Path(r["path"]).name}') for i, r in enumerate(source_inputs)]
    copy(Path(reference_record['path']), common / 'reference-config.json')
    for path in source_files + [Path(__file__), ROOT / 'examples/factorized_dimer_physical.rs', ROOT / 'tools/test_prepare_factorized_dimer_physical.py']:
        copy(path, common / 'source' / path.relative_to(ROOT))
    for path in sorted(p for p in (ROOT / 'vendor').rglob('*') if p.is_file()):
        copy(path, common / 'source' / path.relative_to(ROOT))
    binary = common / 'factorized_dimer_physical'
    copy(executable, binary)
    binary.chmod(0o755)
    copy(bundles[0], common / 'source-bundle.json')
    ledger = common / 'cached-outers.jsonl'
    with ledger.open('x') as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
    limits = dict(raw_per_leg=20_000_000, raw_per_outer=40_000_000, raw_campaign=2_000_000_000,
                  retained_per_leg=20_000_000, retained_per_outer=40_000_000, retained_campaign=2_000_000_000,
                  cpu_seconds=1200.)
    protocol = dict(schema='factorized-dimer-physical-reset-protocol-v1', prepared_utc=datetime.now(timezone.utc).isoformat(),
        execution_authorized=False, allocation=dict(total_outer=1536, candidates=candidate_count,
            failed_proposals=1536-candidate_count, atlases=3, contexts=8, attempts_per_context_and_method=32,
            methods=['whole_joint', 'factorized'], selection='Every saved outer row exactly once in immutable ledger order', extension=False),
        master_seed=6100203201, candidate_ledger=record(ledger), passive=passive_files,
        audit_files=audit_records,
        reference_config=reference_record, source_inputs=source_inputs, archived_source_inputs=archive_inputs,
        density='Cached independently audited full F0(old)F1(old)/F0(new)F1(new), current actual prepared map factors',
        proposal_source_sha256={name:hashes[name] for name in PROPOSAL_FILES},
        source='Reset all264 poses to archived sweep7400 source for EVERY outer; no sequential trajectory',
        physical_conditions=dict(depletant_radius_A=1.4, activity_A_inverse3=.0275, concentration_uM=500, auxiliary_intensity=1.76),
        original_decision_conditions='Separate: 1.5 A, activity0.035, approximately106.8uM. This run is only a growth-state reset diagnostic.',
        gate='One fair-order two-singleton path per saved candidate; independent leg clouds, copied intermediate, no intermediate hard/wall filter; wall-permeable bath',
        mh='One independently seeded Open01 variate per outer, including proposal nulls; candidate accepts iff logU < min(0,cached fullF correction + summed count factor). No retry after rejection.',
        rng='SHA256(factorized-dimer-physical-v1/{master}/{cached-ledger-sha}/{atlas-index}/{case-index}/{attempt}/{method}/{role}); first16hex u64; roles gate and mh',
        envelope=dict(max_cells=255,max_depth=8,min_width=0.), limits=limits,
        budgets='Fatal, never an ordinary rejection: planned raw count retained before point loop; processed/retained partial counters retained on abort. CPU checks before envelopes/rows and every1024points; small checkpoint granularity overrun possible.',
        timing='GateCPU includes source/endpoint geometry, path preparation, envelope and cloud work. All-row replayCPU adds binding/state bookkeeping; whole-processCPU includes setup and output serialization. Report each plus saved passive proposalCPU; whole-process plus saved proposalCPU is the complete observed cost before terminal-receipt serialization. Audit separate; no ESS interpretation.',
        no_new_proposal_draws=True, native_classifier=False, outcome_filtering=False, scientific_threads=1,
        limitations=['No trajectory, equilibrium, thermodynamic stability, native assembly or contactESS inference.',
            'A fatal budget/arithmetic/frame/input error stops the fixed allocation with partial ledger; no extension or replacement samples.',
            'Finite budget failure is a validation failure, not a complete physical rejection sample.',
            'The local bounded wrapper is validated against unchanged SingletonPath on small controls; full floating-point measure/RNG correctness remains outside that test.'])
    write(base / 'protocol.json', protocol)
    config = dict(schema='factorized-dimer-physical-reset-v1', protocol=record(base / 'protocol.json'),
        reference_config=reference_record, passive=passive_files, candidate_ledger=record(ledger),
        total_outer=1536,total_candidates=candidate_count,master_seed=protocol['master_seed'],
        depletant_radius=1.4,activity=.0275,**{'lambda':1.76},envelope=protocol['envelope'],limits=limits,
        compiled_source_sha256=hashes,output=str((base / 'execution').resolve()))
    write(base / 'config.json', config)
    binding = dict(schema='factorized-dimer-physical-binding-v1',config_sha256=sha(base / 'config.json'),
        protocol_sha256=sha(base / 'protocol.json'),example_source_sha256=sha(ROOT / 'examples/factorized_dimer_physical.rs'),
        compiled_source_bundle_sha256=sha(common / 'source-bundle.json'),executable_sha256=sha(binary))
    write(base / 'binding.json', binding)
    files = sorted(p for p in common.rglob('*') if p.is_file()) + [base / n for n in ('config.json','protocol.json','binding.json')]
    closure = dict(schema='factorized-dimer-physical-prelaunch-v1',complete=True,launched=False,execution_authorized=False,
        files={str(p.resolve()):sha(p) for p in files},original_source_inputs=source_inputs,
        environment=dict(python=sys.version,rustc=subprocess.check_output(['rustc','--version'],text=True).strip()),
        command=[str(binary.resolve()),'--config',str((base / 'config.json').resolve()),'--binding',str((base / 'binding.json').resolve())])
    write(base / 'prelaunch.json', closure)
    return dict(base=str(base),total_outer=1536,candidates=candidate_count,prelaunch_sha256=sha(base / 'prelaunch.json'),**binding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/factorized-dimer-physical-20261002')
    parser.add_argument('--passive',type=Path,default=ROOT/'results/factorized-dimer-probe-20261002')
    parser.add_argument('--executable',type=Path,default=ROOT/'target-validation-line-guide/release/examples/factorized_dimer_physical')
    parser.add_argument('--passive-review',type=Path,required=True)
    parser.add_argument('--audit-file',type=Path,action='append',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.passive,args.executable,args.passive_review,args.audit_file),indent=2))
