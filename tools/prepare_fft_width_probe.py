#!/usr/bin/env python3
"""Freeze FFT covariance-width passive controls; never draw clouds or poses."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy as copy_module
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import scipy
from prepare_capped_dimer_probe import ROOT, copy, read, record, sha, write
from analyze_capped_dimer_probe import MAP_FILES, REFERENCE_CONFIG_SHA256, PANEL_SHA256

SCALES = [.125, .25, .5]
MASTER = 6100300401
FFT_SHA = 'c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08'
BASELINE = ROOT/'results/factorized-dimer-probe-20261002'
BASELINE_REVIEW = '738c91a6d4a588626ce62fe567238db1ebf46683a474ad3684738795d1a98a1a'
GUIDED = ROOT/'results/auxiliary-overlap-probe-20261003'
GUIDED_REVIEW = 'd8555ed4a062784f1842cc1712b8673ac006592a336a255951e5bc3dd2cc563e'
AUDIT_FILES = ['analyze_fft_width_probe.py', 'test_analyze_fft_width_probe.py',
    'prepare_fft_width_probe.py', 'test_prepare_fft_width_probe.py',
    'analyze_auxiliary_overlap_probe.py', 'test_analyze_auxiliary_overlap_probe.py',
    'analyze_factorized_dimer_probe.py', 'test_analyze_factorized_dimer_probe.py',
    'analyze_capped_dimer_probe.py', 'test_analyze_capped_dimer_probe.py',
    'analyze_dimer_destination_probe.py', 'dimer_destination_density.py', 'dimer_destination_geometry.py',
    'normalizer_proposal_density.py', 'prepare_smc_normalizer_atlas.py', 'prepare_dimer_destination_panel.py',
    'test_dimer_destination_density.py', 'prepare_capped_dimer_probe.py']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def allocation():
    return dict(schema='fft-width-allocation-v1', contexts=8, slots_per_context=32,
        covariance_scales=SCALES, methods=['unguided', 'm4'], new_outer_attempts=1536,
        clouds=256, raw_points_per_cloud=16384, raw_cloud_points=4194304,
        raw_cloud_bytes=100663296, maximum_raw_edge_draws=98304,
        historical_tau1_attempts=512, historical_new_draws=0, extension=False,
        master_seed=MASTER, caps=dict(root=32, internal=32, joint=1), order='root_first',
        uniform_probability=.5, uniform_half_width_A=160.,
        physical_bath_draws=0, state_updates=0, native_classification=False)


def scale_atlas(original, tau):
    """Scale every 6x6 base covariance, preserving the whole reciprocal envelope."""
    require(type(tau) in (int, float) and math.isfinite(tau) and tau in SCALES,
            'Unallocated covariance scale')
    require(original.get('schema') == 'reciprocal-pose-mixture-v1', 'Reciprocal schema required')
    base = original.get('base_model', {})
    require(base.get('schema') == 'weighted-pose-mixture-v1', 'Weighted Gaussian schema required')
    require('scales' not in base, 'Ambiguous precomputed diagonal scales')
    cov = np.asarray(base.get('covariances'), dtype=float)
    require(cov.ndim == 3 and cov.shape[1:] == (6, 6) and len(cov) > 0 and np.isfinite(cov).all(),
            'Finite 6x6 covariances required')
    require(len(base.get('means', [])) == len(cov) and len(base.get('anchors', [])) == len(cov)
            and len(base.get('weights', [])) == len(cov)
            and len(original.get('reciprocal_components', [])) == len(cov), 'Component lengths differ')
    require(np.allclose(cov, np.swapaxes(cov, 1, 2), atol=0., rtol=1e-13), 'Nonsymmetric covariance')
    # Validation only. Rust construction recomputes its own factors, without jitter/floors.
    np.linalg.cholesky(cov)
    result = copy_module.deepcopy(original)
    result['base_model']['covariances'] = (cov * (tau * tau)).tolist()
    np.linalg.cholesky(np.asarray(result['base_model']['covariances']))
    return result


def history_rows(root, expected_review, method):
    """Authenticate and preserve bytes, including every cap/null, from tau=1."""
    root = Path(root)
    require(sha(root/'completed-review.json') == expected_review, 'Historical review changed')
    receipt = read(root/'completed-review.json')
    require(receipt['complete'] is True and receipt['passed'] is True
            and receipt['audit_exit_code'] == 0, 'Historical audit incomplete')
    for name, digest in receipt['output_hashes'].items():
        require(sha(root/name) == digest, 'Changed historical output: '+name)
    analysis = read(root/'analysis.json')
    require(analysis['complete'] is True and analysis['passed'] is True and not analysis['failures'],
            'Historical analysis incomplete')
    rows = []
    keys = []
    with (root/'execution/attempts.jsonl').open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            if row['atlas_index'] != 1 or row['method'] != method:
                continue
            require(row['status'] == 'completed', 'Historical failed row')
            rows.append(line)
            keys.append((row['case_index'], row['attempt']))
    require(keys == [(c, j) for c in range(8) for j in range(32)], 'Historical allocation differs')
    return rows


def prepare(base, executable):
    reference = ROOT/'results/dimer-destination-probe-20261002/config.json'
    require(sha(reference) == REFERENCE_CONFIG_SHA256, 'Reference config changed')
    ref = read(reference)
    require(ref['panel']['sha256'] == PANEL_SHA256, 'Panel changed')
    inputs = [record(reference), ref['panel'], ref['shape'], ref['source_config'],
              ref['source_frame'], ref['source_freeze_manifest']] + [a['model'] for a in ref['atlases']]
    require(all(sha(v['path']) == v['sha256'] for v in inputs), 'Changed frozen source input')
    require(len(ref['cases']) == 8 and len(ref['atlases']) == 3
            and ref['uniform_probability'] == .5 and ref['uniform_half_width'] == 160.
            and ref['depletant_radius'] == 1.4 and ref['activity'] == .0275, 'Reference contract changed')
    require(ref['atlases'][1]['name'] == 'blind_fft512slots'
            and ref['atlases'][1]['model']['sha256'] == FFT_SHA, 'FFT source differs')
    original = read(ref['atlases'][1]['model']['path'])
    transformed = [scale_atlas(original, tau) for tau in SCALES]
    require(len(original['base_model']['covariances']) == 1024, 'FFT base count changed')
    histories = [history_rows(BASELINE, BASELINE_REVIEW, 'factorized'),
                 history_rows(GUIDED, GUIDED_REVIEW, 'm4')]
    source_files = [ROOT/p for p in ['Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md']]
    source_files += sorted((ROOT/'src').rglob('*.rs'))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    bundles = []
    for path in (ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'):
        if {k: v['sha256'] for k, v in read(path)['files'].items()} == hashes:
            bundles.append(path)
    require(bundles and len({sha(p) for p in bundles}) == 1, 'No matching compiled source bundle')
    # Fail before creating a partial freeze if a requested independent auditor is absent.
    require(all((ROOT/'tools'/name).is_file() for name in AUDIT_FILES), 'Audit closure incomplete')
    shape = read(ref['shape']['path'])
    radii = np.asarray([a['radius']+ref['depletant_radius'] for a in shape['atoms']])
    centers = np.asarray([a['center'] for a in shape['atoms']])
    low = (centers-radii[:, None]).min(axis=0)
    high = (centers+radii[:, None]).max(axis=0)
    volume = float(np.prod(high-low))
    base = Path(base).resolve()
    base.mkdir()
    common = base/'common'
    common.mkdir()
    all_sources = source_files + [ROOT/p for p in ['examples/fft_width_probe.rs',
        'tools/prepare_fft_width_probe.py', 'tools/test_prepare_fft_width_probe.py',
        'tests/auxiliary_overlap_threshold.rs', 'tests/factorized_dimer.rs']]
    all_sources += sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())
    for path in dict.fromkeys(all_sources):
        copy(path, common/'source'/path.relative_to(ROOT))
    audit_files = {}
    for name in AUDIT_FILES:
        path = common/'audit'/name
        copy(ROOT/'tools'/name, path)
        audit_files[name] = record(path)
    scaled_atlases = []
    for tau, transformed_model in zip(SCALES, transformed):
        name = 'blind_fft512slots_tau'+str(tau).replace('.', 'p')
        path = common/(name+'.json')
        write(path, transformed_model)
        scaled_atlases.append(dict(name=name, tau=tau, model=record(path)))
    history = {}
    for method, rows, origin in zip(['unguided', 'm4'], histories, [BASELINE, GUIDED]):
        path = common/('historical_tau1_'+method+'.jsonl')
        with path.open('xb') as stream:
            stream.writelines(rows)
        history[method] = dict(cache=record(path), source_method='factorized' if method == 'unguided' else 'm4',
            sources={name: record(origin/name) for name in ['completed-review.json', 'analysis.json',
                'prelaunch.json', 'config.json', 'protocol.json', 'binding.json',
                'execution/attempts.jsonl', 'execution/terminal.json']})
    frozen_exe = common/'fft_width_probe'
    copy(executable, frozen_exe)
    frozen_exe.chmod(0o755)
    copy(bundles[0], common/'source-bundle.json')
    write(common/'scientific-allocation.json', allocation())
    protocol = dict(schema='fft-width-passive-protocol-v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scientific_allocation=record(common/'scientific-allocation.json'), allocation=allocation(),
        master_seed=MASTER, caps=dict(root=32, internal=32, joint=1), order='root_first',
        reference_config=record(reference), source_inputs=inputs, source_atlas_index=1,
        source_atlas_sha256=FFT_SHA, scaled_atlases=scaled_atlases,
        transformation='Every entry of each stored 6x6 base covariance multiplies by tau^2. All non-covariance JSON is identical; no floors, jitter or refit. Rust rebuilds Cholesky/inverses/determinants through a fresh immutable proposal per scale.',
        covariance_scales=SCALES, methods=['unguided', 'm4'],
        seed_rule='SHA256(fft-width-probe-v1/{master}/{case}/{attempt}/{role}); first16hex u64',
        seed_roles=['cloud', 'proposal', 'threshold', 'execution_order/{scale_index}/{method}'],
        arm_order='Sort six (scale_index,method,key) triples by (key,scale_index,method); key uses execution_order/{scale_index}/{method}.',
        matching='One cloud and separate threshold/proposal seeds per case-slot, shared six arms; each arm resets its proposal RNG. Raw prefixes can diverge after different capped stopping. Arms are dependent paired controls.',
        cloud_law=dict(kind='fixed-size-uniform-AABB-root-thinning', raw_count=16384, low=low.tolist(),
            high=high.tolist(), volume_A3=volume, effective_raw_intensity_Aminus3=16384/volume,
            membership='Closed atomic membership d_squared <= inflated_radius_squared; body overlap remains strict.',
            thinning='Keep root exclusion members only, independent of old internal pose. Never redraw for old_count0 or choose overlap points.',
            storage='All raw uniforms are LEf64 triples in cloud-uniforms.bin; transformed and retained coordinates carry hashes; kept indices saved.',
            is_poisson=False, point_frame='root body; co-transports with root during pose block'),
        threshold_law='m4: maximum of four exact uniform integers0..=Kold; fixed across retries. Unguided: no threshold or overlap queries.',
        auxiliary_correction='m4 adds4*(log(Kold+1)-log(Knew+1)) exactlyonce to completefullF; unguided zero. Fixed labels give zero selection correction.',
        density_law='map-factor-full-mixture-v1', uniform_probability=.5, uniform_half_width_A=160.,
        old_density='Record root/internal logfull,loglearned,loguniform in old_edges on everyouter, includingnulls. This diagnostic rescoring occurs before sampling; production already scores completedcandidates. Its CPU is separate from production-equivalent proposalCPU.',
        timing='Standalone proposalCPU includes proposal call and, for m4, entire sharedcloudconstruction plusguidancesetup. Source-density diagnostic and contact diagnostic CPU separate. Actual campaignCPU includes all overhead. FullF scoring within candidateproposalcall is charged.',
        history=history, history_interpretation='Tau1 savedoutcomes only; distinct historical streams/clouds andtiming; descriptive controls, not pairedreplicates.',
        density_tolerance=dict(absolute=2e-7, relative=2e-10),
        map_density_source_sha256={k: hashes[k] for k in MAP_FILES},
        guidance_source_sha256={k: hashes[k] for k in ['src/factorized_dimer.rs', 'src/auxiliary_overlap_threshold.rs']},
        audit_files=audit_files, physical_bath_draws=0, state_updates=0, native_classification=False,
        physical_conditions=dict(depletant_radius_A=1.4, activity_Aminus3=.0275, concentration_uM=500),
        decision_conditions='Original1.5A/.035A^-3/~106.8uM remain separate.',
        limitations=['Passive candidate/density comparison only; no acceptance,ESS,registry orassemblyconclusion.',
            'Fixed clouds are proposal auxiliaries, not physicaldepletants or exactvolumes.',
            'No nativefilter orwidthadaptation. Frozenfixedallocation; fatalrecords drain andstop withoutreplacement.',
            'Uniformdefense preserves measure support but does not guarantee effectiveimportancecoverage.'])
    write(base/'protocol.json', protocol)
    config = dict(schema='fft-width-screen-v1', scientific_allocation=record(common/'scientific-allocation.json'),
        protocol=record(base/'protocol.json'), reference_config=record(reference), master_seed=MASTER,
        scaled_atlases=scaled_atlases, cloud_raw_count=16384, attempts_per_context=32,
        root_cap=32, internal_cap=32, factorized_joint_cap=1, factorized_order='root_first',
        density_law=protocol['density_law'], compiled_source_sha256=hashes, output=str(base/'execution'))
    write(base/'config.json', config)
    binding = dict(schema='fft-width-probe-binding-v1', config_sha256=sha(base/'config.json'),
        protocol_sha256=sha(base/'protocol.json'), example_source_sha256=sha(ROOT/'examples/fft_width_probe.rs'),
        compiled_source_bundle_sha256=sha(common/'source-bundle.json'), executable_sha256=sha(frozen_exe))
    write(base/'binding.json', binding)
    files = sorted(p for p in common.rglob('*') if p.is_file()) + [base/'protocol.json', base/'config.json', base/'binding.json']
    externals = inputs + [v for arm in history.values() for v in arm['sources'].values()]
    prelaunch = dict(schema='fft-width-prelaunch-v1', complete=True, launched=False,
        files={str(p.resolve()): sha(p) for p in files}, external_inputs={v['path']: v['sha256'] for v in externals},
        environment=dict(python=sys.version, python_executable=sys.executable, numpy=np.__version__,
            scipy=scipy.__version__, rustc=subprocess.check_output(['rustc', '--version'], text=True).strip(),
            cargo=subprocess.check_output(['cargo', '--version'], text=True).strip()),
        command=[str(frozen_exe), '--config', str(base/'config.json'), '--binding', str(base/'binding.json')],
        analysis_command=[sys.executable, str(common/'audit/analyze_fft_width_probe.py'), '--run',
            str(base/'execution'), '--output', str(base/'analysis.json')])
    write(base/'prelaunch.json', prelaunch)
    return dict(base=str(base), prelaunch_sha256=sha(base/'prelaunch.json'), **binding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/fft-width-probe-20261003')
    parser.add_argument('--executable', type=Path, default=ROOT/'target-validation-line-guide/release/examples/fft_width_probe')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.executable), indent=2))
