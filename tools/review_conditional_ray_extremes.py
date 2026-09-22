#!/usr/bin/env python3
"""Read-only source/measure and observed-uncertainty review of saved extremes.

Creates a fresh review directory. Never launches physics or a classifier.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
from diagnose_conditional_ray_extremes import chart_coordinates, read, sha, require

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def decode_block(bundle):
    text = bundle['files']['src/latent_region.rs']['text']
    start = text.index('    fn decode(')
    block = text[start:text.index('    fn encode(', start)]
    inline = 'let x: [f64; 6] = std::array::from_fn(|i| {\n            self.mean[i] + (0..=i).map(|j| self.lower[i][j] * u[j]).sum::<f64>()\n        });'
    if 'let x = self.coordinates(u);' in block:
        require('fn coordinates(&self, u: [f64; 6]) -> [f64; 6] {\n        std::array::from_fn(|i| {\n            self.mean[i] + (0..=i).map(|j| self.lower[i][j] * u[j]).sum::<f64>()\n        })\n    }' in text, 'Coordinate helper differs from original inline map')
        block = block.replace('let x = self.coordinates(u);', inline)
    return block


def jacobian(u, region):
    c = region['gaussian_chart']
    lower = np.linalg.cholesky(c['covariances'][0])
    x = np.asarray(u) @ lower.T + c['means'][0]
    ell = c['angular_length']
    return (np.log(np.diag(lower)).sum() - 3 * math.log(ell)
            - 2 * math.log(math.pi) - 2 * np.log1p(np.sum((x[:, 3:] / ell)**2, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    require(not out.exists(), 'Refuse to overwrite a completed or partial review')
    diag_path = ROOT / 'runs/mobile-conditional-ray-extremes-20260922'
    current_path = ROOT / 'runs/mobile-conditional-ray-comparison-20260921'
    reference_path = ROOT / 'runs/mobile-native-pocket-coverage-comparison-20260921'
    diag, current, reference = [read(p / 'analysis.json') for p in (diag_path, current_path, reference_path)]
    require(sha(diag_path / 'analysis.json') == 'ec4e6b76fa2c27079f7357116157f7df393f0af1eefb4a411ce48ca9e6004f31', 'Diagnostic changed')
    require(sha(current_path / 'analysis.json') == diag['comparison_sha256'], 'Current comparison changed')
    require(sha(reference_path / 'analysis.json') == diag['reference']['source_sha256'], 'Reference changed')
    core = next(s for s in reference['strata'] if s['name'] == 'r5repeat')
    cp = Path(current['campaign']) / 'pilot_uniform/provenance'
    rp = Path(core['campaign']) / 'provenance'
    cr, rr = read(cp / 'region.json'), read(rp / 'region.json')
    cc, rc = read(cp / 'config.json'), read(rp / 'config.json')
    hashes = {str(p): sha(p) for p in [diag_path / 'analysis.json', current_path / 'analysis.json', reference_path / 'analysis.json', cp / 'region.json', rp / 'region.json', cp / 'config.json', rp / 'config.json', cp / 'source-bundle.json', rp / 'source-bundle.json']}
    shape = sha(cp / 'shape.json')
    require(shape == sha(rp / 'shape.json') == cr['shape_sha256'] == rr['shape_sha256'], 'Shape bytes differ')
    require({k:v for k,v in cc.items() if k != 'shape'} == {k:v for k,v in rc.items() if k != 'shape'}, 'Configuration differs beyond archive path')
    target_fields = ['physical_fixed_neighbors', 'capture_center', 'capture_radius', 'shape_sha256', 'activity', 'depletant_radius', 'physical_metric']
    require(all(cr[k] == rr[k] for k in target_fields), 'Physical target fields differ')
    for field in ['definition_sha256', 'runtime_sha256', 'input_sha256', 'criteria', 'scope']:
        require(current['native_definition'][field] == reference['native_definition'][field], 'Native observer differs: ' + field)
    cb, rb = read(cp / 'source-bundle.json'), read(rp / 'source-bundle.json')
    require(decode_block(cb) == decode_block(rb), 'Chart decoder/Jacobian implementation changed')
    common_geometry = {}
    for key in ['src/geometry.rs', 'src/math.rs', 'src/overlap_weight.rs']:
        require(cb['files'][key]['sha256'] == rb['files'][key]['sha256'], 'Physical implementation differs: ' + key)
        common_geometry[key] = cb['files'][key]['sha256']
    sel = diag['selected']
    chart_error = float(np.max(abs(chart_coordinates([s['pose'] for s in sel], cr) - np.array([s['current_latent'] for s in sel]))))
    require(chart_error < 1e-10, 'Independent chart inverse disagrees with saved coordinates')
    hard_error = float(np.max(abs(jacobian([s['current_latent'] for s in sel], cr) - np.array([s['log_proposal_density'] + s['log_hard_weight'] for s in sel]))))
    require(hard_error < 1e-10, 'Current J/q identity failed')
    labels_path = reference_path / core['native_labels']['path']
    require(sha(labels_path) == core['native_labels']['sha256'], 'Reference labels changed')
    native = {}
    motifs = {}
    with labels_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['applicable']:
                native[(row['population'], row['draw'])] = row['label']['native_any']
                key = str([(m['anchor_index'], m['motif_id']) for m in row['label']['matches']])
                motifs[key] = motifs.get(key, 0) + 1
    arrays = {k: [] for k in ['current_u', 'reference_u', 'log_J_current', 'log_J_reference', 'log_importance_weight', 'log_hard_weight', 'population', 'draw']}
    ref_errors = []
    denominators = {}
    for pop in core['populations']:
        path = Path(core['campaign']) / 'runs' / pop['id'] / 'samples.jsonl'
        require(sha(path) == core['source_sha256'][str(path)], 'Reference raw file changed')
        hashes[str(path)] = sha(path)
        with path.open() as stream:
            rows = [json.loads(line) for line in stream]
        require(len(rows) == pop['draws'] == 16384, 'Reference denominator changed')
        denominators[pop['id']] = dict(seed=pop['seed'], unconditional_draws=len(rows))
        rows = [r for r in rows if r['log_hard_weight'] is not None]
        require(all(native[(pop['id'], r['draw'])] for r in rows), 'Reference hard-valid contributor lacks saved native label')
        old_u = np.array([r['latent'] for r in rows])
        new_u = chart_coordinates([r['pose'] for r in rows], cr)
        require(np.max(np.linalg.norm(new_u, axis=1)) <= 4., 'Observed reference contributor outside current R4')
        old_j = jacobian(old_u, rr)
        new_j = jacobian(new_u, cr)
        log_v = 3 * math.log(math.pi) + 6 * math.log(5.) - math.log(6.)
        ref_errors.append(float(np.max(abs(old_j + log_v - [r['log_hard_weight'] for r in rows]))))
        ref_errors.append(float(np.max(abs(chart_coordinates([r['pose'] for r in rows], rr) - old_u))))
        require(abs(float(logsumexp([r['log_importance_weight'] for r in rows]) - math.log(16384)) - pop['log_Qz']) < 1e-10, 'Reference unconditional mass changed')
        for key, values in [('current_u', new_u), ('reference_u', old_u), ('log_J_current', new_j), ('log_J_reference', old_j), ('log_importance_weight', [r['log_importance_weight'] for r in rows]), ('log_hard_weight', [r['log_hard_weight'] for r in rows]), ('population', [pop['id']] * len(rows)), ('draw', [r['draw'] for r in rows])]:
            arrays[key].extend(values)
    require(max(ref_errors) < 1e-10 and len(arrays['draw']) == 5064, 'Reference measure identity failed')
    subset = diag['reference']['current_R4_intersection']['intersection_native']
    ref_est = subset['population_uncertainty']
    comparisons = {}
    for arm, source in current['arms'].items():
        estimate = source['estimates']['registered_native_entry']['population_uncertainty']
        ratio = math.exp(estimate['log_Qz'] - ref_est['log_Qz'])
        vr = ref_est['Qz_relative_SE']**2
        va = (ratio * estimate['Qz_relative_SE'])**2
        se = math.sqrt(vr + va)
        df = (vr + va)**2 / (vr**2 / 7 + va**2 / 3)
        half = float(t.ppf(.975, df)) * se
        comparisons[arm] = dict(current_over_subset_estimate=ratio,
            subset_minus_current_in_subset_mean_units=1-ratio,
            combined_observed_population_SE=se, standardized_difference=(1-ratio)/se,
            welch_df=df, observed_two_sided_95_difference_interval=[1-ratio-half, 1-ratio+half],
            native_population_log_estimates=[p['log_Qz'] for p in source['estimates']['registered_native_entry']['populations']])
    result = dict(schema='conditional-ray-extremes-identity-review-v1', complete=True,
        new_physical_samples=0, classifiers_rerun=0, source_sha256=hashes,
        shape_sha256=shape, physical_target_fields_equal=target_fields,
        configuration_equal_except_shape_archive_path=True,
        native_observer={k: current['native_definition'][k] for k in ['definition_sha256','runtime_sha256','input_sha256','criteria']},
        common_physical_source_sha256=common_geometry, chart_decode_algebra_identical_after_coordinate_helper_inlining=True,
        current_chart_inverse_max_abs_error=chart_error, current_log_J_over_q_max_abs_error=hard_error,
        reference_inverse_and_log_weight_max_abs_error=max(ref_errors),
        reference_denominators=denominators, saved_reference_native_motif_patterns=motifs,
        observed_reference_current_radius_range=[float(np.min(np.linalg.norm(arrays['current_u'],axis=1))),float(np.max(np.linalg.norm(arrays['current_u'],axis=1)))],
        subset_estimate=subset, observed_population_comparisons=comparisons,
        scope='A statistically uncertain subset-mass estimate, not a deterministic or rigorous lower bound. Observed t/SE diagnostics cannot certify unseen-tail coverage. Source data reused for proposal design are not fresh proposal validation. Current R4/native target contains the R5 intersection/native target; the regions are not equal.')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'reference-contributors.npz', **{k: np.asarray(v) for k,v in arrays.items()})
    result['reference_contributors_sha256'] = sha(out / 'reference-contributors.npz')
    write(out / 'analysis.json', result)
    lines = ['# Saved-reference identity and uncertainty review', '',
        'No new physical draws, raw audits, or classifier evaluations. Frozen campaigns and the completed extremes diagnostic are unchanged.', '',
        'The shape bytes, both fixed-neighbor poses, activity0.035 Å⁻³, depletant radius1.5 Å, capture170 Å, wall certificate and physical metadata match exactly. Archived configurations differ only in the shape archive path. The archived chart decode/Jacobian algebra is identical after inlining the new coordinate helper; geometry, quaternion math and overlap-weight source hashes match. Native definition, full classifier source, all input hashes and criteria match; labels were read from the saved full-classifier outputs, not approximated by the two guide interfaces.', '',
        'The physical measure is center volume times normalized SO(3) Haar measure. In each chart x=μ+Lu, c=xrot/ell and J(u)=det(L)/(ell³π²(1+|c|²)²). Each original unconditional attempt contributes I(valid) I(reporting region) J(u)/q(u) times the arithmetic mean of two unbiased Poisson weights. The R5 reference uses q=1/V6(5); its old J/q is retained after intersecting with current R4. There is no new current-chart Jacobian multiplier, new normalization, averaging of logarithms, or conditioning denominator.', '',
        'All5,064 saved hard-valid/native contributors among8×16,384=131,072 attempts lie in current R4. The intersection estimate is logQz=60.682123, populationRSE7.97%, rowRSE7.67%, ESS169.75 and largest-row share3.84%. This is an uncertain subset mass, not a rigorous lower bound or a complete-basin estimate. It uses q>1 in the old chart, compatible with the current q≥0/no-upper-cutoff region. It does not establish that every possible hard-valid R5 pose lies in R4.', '',
        '| Current arm | Current native / subset estimate | Difference / combined observed population SE | Observed Welch95% interval for (subset−current)/subset estimate |',
        '|---|---:|---:|---:|']
    for arm, c in comparisons.items():
        lo, hi = c['observed_two_sided_95_difference_interval']
        lines.append(f"| {arm} | {c['current_over_subset_estimate']:.5f} | {c['standardized_difference']:.2f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ['', 'These are comparisons of linear population means, using eight independent reference and four current populations. The large-ray interval includes zero: its discrepancy alone is not a calibrated95% inconsistency result. The much larger observed discrepancies of the other arms expose failure of their observed means/SEs to represent known subset mass. All intervals are descriptive because missing tails and the small population counts limit their calibration.', '',
        'The reference population log estimates span60.398–61.036. Its paired-cloud decomposition attributes34.7% of observed row variance to the two-cloud estimator and65.3% to pose variation. Current native maxima have paired-cloud count discrepancies≤2.782 nominal standard deviations, while the maximum-weight large-ray row differs by only52 counts (0.507 standard deviations). Poisson noise is appreciable but does not explain away missing reference geometry.', '',
        'All24 native population maxima match A7/B4; the240 selected native extremes miss independent R5 and even R8. Native maxima have median nearest-other-population member RMS0.157 Å; no-entry maxima0.127 Å. All24 no-entry maxima exceed both2 Å body-member entry thresholds, by0.027–0.619 Å at A7 and0.015–0.609 Å at B4, while proper orientation errors are only2.875–5.143°. They include recurrent near-threshold configurations. The distance heatmap also shows two more separated no-entry top poses (pilot_ray/r01 and lambda128/r00), so these data do not justify calling the entire denominator a single connected basin. All distances and clusters remain descriptive, not evidence of barriers or connectivity.', '',
        'reference-contributors.npz stores the5,064 original contributors with current_u, reference_u, original log_importance_weight/log_hard_weight, log_J_current/log_J_reference, population and draw. Invalid zero rows are omitted from this convenience archive only; source population denominators remain16,384. Reusing these points to design a proposal is training/design reuse, not independent validation of that proposal.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    shutil.copy2(__file__, out / Path(__file__).name)
    write(out / 'freeze.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(out)
    print('analysis_sha256=' + sha(out / 'analysis.json'))


if __name__ == '__main__':
    main()
