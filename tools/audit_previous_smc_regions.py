#!/usr/bin/env python3
"""Audit existing SMC region coverage without changing source datasets.

This reuses archived populations; it is not an independent normalizer estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp


BINS = ['q<=0.8', '0.8<q<=1', '1<q<2', '2<=q<5', 'q>=5']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose['orientation'])[[1, 2, 3, 0]])


def recompute_q(pose, environment, config):
    members = np.asarray([p['position'] for p in environment['rigid_members']])
    r = rotation(pose)
    transformed = r.apply(members) + pose['position']
    refs = environment.get('native_poses') or [environment['native_pose']]
    return min(max(
        np.linalg.norm(transformed - (rotation(ref).apply(members) + ref['position']), axis=1).max()
        / config.get('member_error_scale', 2.),
        (r * rotation(ref).inv()).magnitude() / math.radians(config.get('angle_error_scale_deg', 15.))
    ) for ref in refs)


def masks(q):
    return np.asarray([q <= .8, (q > .8) & (q <= 1), (q > 1) & (q < 2),
                       (q >= 2) & (q < 5), q >= 5])


def run(source, out):
    out.mkdir(parents=True, exist_ok=True)
    used = {Path(__file__).resolve()}
    for name in ['analysis.json', 'diagnostics.json', 'independent-audit/audit.json',
                 'implementation/kernel-validation.json',
                 'implementation/src/bin/coordination_smc.rs',
                 'implementation/src/single_body_depletion.rs']:
        used.add(source / name)
    archived_audit = read(source / 'independent-audit/audit.json')
    assert archived_audit['passed'] and archived_audit['all_campaign_jobs_checked']
    answer = dict(source=str(source), bins=BINS, sites={}, total_endpoints=0, shoulder_endpoints=0,
        endpoint_weighting='Final endpoints have equal weights after final systematic resampling and invariant rejuvenation. Across independent populations, use their linear normalizer estimates. Descendants are not independent replication units.',
        target='Q_b = integral_capture H(x) I_b(x) exp(z C(x)) dx, with Lebesgue center volume in A^3 and normalized SO(3) Haar measure.',
        scope='Retrospective endpoint coverage and normalization audit, not an independent free-energy calculation. Zero observed bin mass does not establish zero physical mass.',
        source_audit_passed=True, recomputed_q_max_error=0.)
    for site in [0, 1]:
        record = {}
        for basin in ['native', 'other_adsorbed']:
            paths = sorted((source / 'runs').glob(f'site{site}-m1-r1.5-z0.035-{basin}-*/summary.json'))
            assert len(paths) == 8
            populations = []
            for path in paths:
                cfg_path = source / 'configs' / (path.parent.name + '.json')
                counts_path = path.parent / 'final_overlap_counts.json'
                summary, config, counts = read(path), read(cfg_path), read(counts_path)
                env_path = Path(config['environment'])
                environment = read(env_path)
                used.update([path, cfg_path, counts_path, env_path, Path(config['shape'])])
                assert summary['complete'] and summary['completed_stage'] == 128
                assert config['reservoir_density'] == .035 and config['depletant_radius'] == 1.5
                assert config['basin'] == basin and config['population'] == 512
                assert environment['capture_radius'] == 18.
                endpoints = summary['final_particles']
                q = np.array([p['q'] for p in endpoints])
                answer['total_endpoints'] += len(q)
                answer['shoulder_endpoints'] += int(((q > 1) & (q < 5)).sum())
                computed = np.array([recompute_q(p['pose'], environment, config) for p in endpoints])
                error = float(np.max(np.abs(q - computed)))
                assert error < 1e-10
                answer['recomputed_q_max_error'] = max(answer['recomputed_q_max_error'], error)
                assert np.array_equal(masks(q), masks(computed))
                assert (q <= 1).all() if basin == 'native' else (q > 1).all()
                samples = counts['samples']
                assert len(samples) == len(q) == 512
                assert [p['particle'] for p in samples] == list(range(512))
                contact = np.array([p['contact_neighbor_count'] > 0 for p in samples])
                histogram = masks(q).mean(axis=1)
                assert histogram.sum() == 1.
                overlap = np.array([p['count'] / p['intensity'] for p in samples])
                for value, p in zip(overlap, samples):
                    assert abs(value - p['overlap_volume_estimate']) < 1e-10
                populations.append(dict(population=path.parent.name, logQ=summary['logZ'],
                    q_mass=histogram.tolist(), contact_q_mass=(masks(q) & contact).mean(axis=1).tolist(),
                    unbound_fraction=float((~contact).mean()), q_range=[float(q.min()), float(q.max())],
                    mean_overlap_A3=float(overlap.mean()), family_ess=summary['family_ess'],
                    distinct_families=summary['distinct_families'], initial_hits=summary['initialization']['hits'],
                    initial_draws=summary['initialization']['draws'], seed=config['seed']))
            assert len({p['seed'] for p in populations}) == len(populations)
            logs = np.array([p['logQ'] for p in populations])
            weights = np.exp(logs - logsumexp(logs))
            logq = float(logsumexp(logs) - np.log(len(logs)))
            h = np.array([p['q_mass'] for p in populations])
            mean_overlap = float(weights @ np.array([p['mean_overlap_A3'] for p in populations]))
            record[basin] = dict(logQ=logq, population_normalizer_weights=weights.tolist(),
                normalizer_ess=float(1. / (weights @ weights)),
                largest_normalizer_fraction=float(weights.max()),
                relative_standard_error=float(np.sqrt((len(weights) * (weights @ weights) - 1) / (len(weights) - 1))),
                normalizer_weighted_q_mass=(weights @ h).tolist(),
                normalizer_weighted_contact_q_mass=(weights @ np.array([p['contact_q_mass'] for p in populations])).tolist(),
                normalizer_weighted_unbound_fraction=float(weights @ np.array([p['unbound_fraction'] for p in populations])),
                unweighted_population_mean_q_mass=h.mean(axis=0).tolist(),
                population_q_mass_min=h.min(axis=0).tolist(), population_q_mass_max=h.max(axis=0).tolist(),
                mean_overlap_A3=mean_overlap,
                differential_entropy_estimate=logq - .035 * mean_overlap,
                populations=populations)
        logs = np.array([record[b]['logQ'] for b in ['native', 'other_adsorbed']])
        basin_weight = np.exp(logs - logsumexp(logs))
        record['combined'] = dict(basin_mass=dict(zip(['native', 'other_adsorbed'], basin_weight.tolist())),
            q_mass=(basin_weight @ np.array([record[b]['normalizer_weighted_q_mass'] for b in ['native', 'other_adsorbed']])).tolist(),
            contact_q_mass=(basin_weight @ np.array([record[b]['normalizer_weighted_contact_q_mass'] for b in ['native', 'other_adsorbed']])).tolist())
        record['energy_entropy'] = dict(
            native_depletion_advantage_kBT=.035 * (record['native']['mean_overlap_A3'] - record['other_adsorbed']['mean_overlap_A3']),
            other_minus_native_differential_entropy=record['other_adsorbed']['differential_entropy_estimate'] - record['native']['differential_entropy_estimate'],
            F_native_minus_other_kBT=record['other_adsorbed']['logQ'] - record['native']['logQ'],
            limitation='Derived from the same unvalidated endpoint coverage, not an independent thermodynamic check. Entropy differences use the same reference volume and normalized Haar measure.')
        answer['sites'][f'site{site}'] = record
    answer['provenance'] = {str(p): sha(p) for p in sorted(used)}
    (out / 'audit.json').write_text(json.dumps(answer, indent=2) + '\n')
    figure(answer, out)
    (out / 'artifacts.json').write_text(json.dumps({p.name: sha(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name != 'artifacts.json'}, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), q_max_error=answer['recomputed_q_max_error'],
        total_endpoints=answer['total_endpoints'], shoulder_endpoints=answer['shoulder_endpoints']), indent=2))


def figure(audit, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    palette = ['#087f8c', '#83c5be', '#f4a261', '#e76f51', '#4c5a89']
    x = np.arange(2)
    bottom = np.zeros(2)
    for k, label in enumerate(BINS):
        height = np.array([audit['sites'][f'site{s}']['combined']['q_mass'][k] for s in [0, 1]])
        axes[0].bar(x, height, bottom=bottom, color=palette[k], label=label)
        bottom += height
    axes[0].set(xticks=x, xticklabels=['Site 0', 'Site 1'], ylim=(0, 1),
        ylabel='Archived normalizer-weighted endpoint mass', title='No final endpoints in 1 < q < 5')
    axes[0].legend(fontsize=8, loc='center left', bbox_to_anchor=(1., .5))
    for site in [0, 1]:
        for index, basin in enumerate(['native', 'other_adsorbed']):
            r = audit['sites'][f'site{site}'][basin]
            xx = site * 2.5 + index
            axes[1].scatter(np.full(8, xx) + np.linspace(-.14, .14, 8),
                [p['logQ'] for p in r['populations']], color=palette[0 if index == 0 else 4], s=24)
            axes[1].plot([xx-.25, xx+.25], [r['logQ']]*2, color='black', lw=1.5)
    axes[1].set(xticks=[0, 1, 2.5, 3.5], xticklabels=['0 native', '0 other', '1 native', '1 other'],
        ylabel='log Q per independent SMC population', title='Competing normalizers vary substantially')
    axes[1].tick_params(axis='x', labelrotation=20)
    fig.suptitle('Archived SMC coverage at r_d = 1.5 Å, z = 0.035 Å⁻³\nObserved zero shoulder mass is not a physical zero; black lines are log(mean Q)', fontsize=11)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'previous-smc-region-audit.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'runs/normalizer-reference-audit')
    args = parser.parse_args()
    run(args.source.resolve(), args.out.resolve())
