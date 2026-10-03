#!/usr/bin/env python3
"""Freeze the bounded sphere reference and audit closure without sampling."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def prepare(base, executable):
    sources = [ROOT/p for p in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md')]
    sources += sorted((ROOT/'src').rglob('*.rs'))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    bundles = []
    for path in (ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'):
        bundle = json.loads(path.read_text())
        if {k: v['sha256'] for k, v in bundle['files'].items()} == hashes:
            bundles.append(path)
    if not bundles or len({sha(p) for p in bundles}) != 1:
        raise ValueError('No unique matching compiled source bundle')
    base = Path(base).resolve()
    base.mkdir()
    common = base/'common'
    extra = ['examples/auxiliary_overlap_sphere_control.rs',
             'tools/prepare_auxiliary_overlap_sphere_control.py',
             'tests/auxiliary_overlap_threshold.rs', 'tests/factorized_dimer.rs',
             'docs/auxiliary-overlap-threshold.md']
    for path in dict.fromkeys(sources + [ROOT/p for p in extra] + sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())):
        copy(path, common/'source'/path.relative_to(ROOT))
    for name in ('analyze_auxiliary_overlap_sphere_control.py', 'test_analyze_auxiliary_overlap_sphere_control.py'):
        copy(ROOT/'tools'/name, common/'audit'/name)
    copy(bundles[0], common/'source-bundle.json')
    frozen_exe = common/'auxiliary_overlap_sphere_control'
    copy(executable, frozen_exe)
    frozen_exe.chmod(0o755)
    config = dict(schema='auxiliary-overlap-sphere-control-v1', master_seed=6100300201,
        populations=4, draws_per_population=4096, m=[1, 4], core_radius=.05,
        exclusion_radius=.45, uniform_half_width=1., activity=0.,
        caps=dict(root=8, internal=8, joint=1), order='root_first')
    write(base/'config.json', config)
    protocol = dict(schema='auxiliary-overlap-sphere-protocol-v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        allocation=dict(independent_sources=16384, corrected_decisions=32768,
            populations=4, draws_per_population=4096, m=[1, 4], extension=False),
        target='Hard-only uniform root cube x internal contact shell, independent Haar rotations, conditioned on no selected/anchor core overlaps; fixed anchor at identity; unit tree-coordinate Jacobian.',
        source='Independent uniform root cube, normalized four-normal Haar quaternion, normalized three-normal radial direction, uniform r^3 in [.1^3,.9^3], independent relative Haar quaternion; reject entire source on final hard failure. Retain all trials, fatal after1000; no replacement outer.',
        guide='Fixed point-mass cloud of .1*(i,j,k), i,j,k=-4..4 inside closed radius.45 root sphere. Root-body coordinates fixed across retry; threshold maximum of m unbiased integers0..Kold.',
        correction='Full F exactly constant (uniform_probability1,halfwidth1); complete accessor supplies m*log((Kold+1)/(Knew+1)); no bath at activity0.',
        matching='Shared source, proposal RNG prefix, auxiliary RNG prefix, and MH uniform across m1/m4. Independent source panels and slots. Wrong m4 reuses same candidate and MH uniform, omits only auxiliary correction.',
        seeds='SHA256(auxiliary-overlap-sphere-v1/6100300201/{population}/{slot}/{role}); first16hex u64; roles source,proposal,auxiliary,mh.',
        primary_tests=dict(observables=['separation', 'I(separation<.6)', 'overlap_count', 'root_quaternion_w_squared'],
            arms=[1,4], total_tests=8, estimator='Retained minus source, pooled paired delta across16384 independent sources per arm; normal approximation using paired sample variance.',
            family_alpha=.01, adjustment='Bonferroni min(1,8*p_two_sided)', population_summaries='Descriptive only; no secondary gate.'),
        negative_control=dict(observable='I(separation<.6)', arm='wrong m4 without auxiliary correction',
            criterion='Positive mean retained-minus-source and one-sided paired normal p<.01.'),
        failure_policy='No replacement draws or allocation extensions. A flagged test is reported; arithmetic repairs retain original output and are separately documented.',
        obligations=['Independent reconstruction of source variates, all retries, hard predicates, guidance counts in every frame, fullF and auxiliary correction, MH decisions and rejected states.',
            'Finite statistical diagnostics and simple-sphere geometry are not a general equilibrium or floating-point proof.',
            'No evolving trajectory, contact ESS, physical bath clouds, protein sampling or thermodynamic conclusion.'])
    write(base/'protocol.json', protocol)
    files = sorted(p for p in common.rglob('*') if p.is_file()) + [base/'config.json', base/'protocol.json']
    frozen = dict(schema='auxiliary-overlap-sphere-prelaunch-v1', complete=True, launched=False,
        files={str(p): sha(p) for p in files},
        environment=dict(python=sys.version, python_executable=sys.executable,
            rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
            cargo=subprocess.check_output(['cargo','--version'],text=True).strip()),
        command=[str(frozen_exe),'--config',str(base/'config.json'),'--out',str(base/'execution')],
        analysis_command=[sys.executable,str(common/'audit/analyze_auxiliary_overlap_sphere_control.py'),
            '--run',str(base/'execution'),'--prelaunch',str(base/'prelaunch.json'),'--output',str(base/'analysis.json')])
    write(base/'prelaunch.json', frozen)
    return dict(base=str(base), prelaunch_sha256=sha(base/'prelaunch.json'), files=len(files), command=frozen['command'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/auxiliary-overlap-sphere-control-20261003')
    parser.add_argument('--executable',type=Path,default=ROOT/'target-validation-line-guide/release/examples/auxiliary_overlap_sphere_control')
    args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.executable),indent=2))
