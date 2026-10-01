#!/usr/bin/env python3
"""Freeze existing contact probes and densities for a score-only arc comparison.

This preparation reads saved rows only. It performs no geometric query, fitting,
pose sampling, Poisson sampling, or physical reclassification.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

ARMS = ['uniform_phi92', 'localized_phi92']
OLD_ARMS = ['baseline92', *ARMS]
EXPECTED_PROTOCOL = '7faa707bfcd96a1c10b269810488788335b4d2d9106bbc030d34b5a13308a501'
GUIDE_SHA = 'a00a4470d898e06f66b78dcb7f1c1b42d7c4dc76f9a111dff9ce6268e09a96c0'
REGION_SHA = '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
MINIMUM_ARC_MASS = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_probe_allocation(critical, breadth, critical_metadata, breadth_metadata):
    require(len(critical) == 78 and len(breadth) == 128, 'Probe allocation changed')
    all_rows = critical + breadth
    require(len({p['id'] for p in all_rows}) == 206, 'Duplicate probe identity')
    require(len({tuple(p['latent']) for p in all_rows}) == 206, 'Repeated latent geometry')
    require(all(len(p['latent']) == 6 and all(math.isfinite(v) for v in p['latent']) for p in all_rows),
            'Invalid latent coordinates')
    require([p['id'] for p in critical_metadata] == [p['id'] for p in critical],
            'Critical metadata order changed')
    require([p['id'] for p in breadth_metadata] == [p['id'] for p in breadth],
            'Breadth metadata order changed')
    require(Counter(p['source']['arm'] for p in critical_metadata) == {'baseline': 4, 'expanded': 74},
            'Original critical source partition changed')
    classes = Counter(p['coverage_class'] for p in breadth_metadata)
    require(classes == {'native_R5': 32, 'native_complement': 32, 'competing': 32, 'invalid': 32},
            'Breadth classes changed')
    for p, m in zip(critical, critical_metadata):
        require(p['latent'] == m['source']['u'] and len(m['source']['paired_log_weights']) == 2 and
                m['source']['source_attempted_draws'] > 0, 'Critical weights/pose provenance differs')
    for p, m in zip(breadth, breadth_metadata):
        require(p['latent'] == m['latent'], 'Breadth pose provenance differs')
    return classes


def validate_guides(original, guides):
    require(len(original['gaussian_components']) == 92 and
            original['defensive_uniform_shell_probability'] == .5, 'Original guide changed')
    for name, b in zip(OLD_ARMS, (0., 0., .9)):
        g = guides[name]
        require(g['schema'] == 'defensive-contact-distance-guide-v1' and
                g['gaussian_components'] == original['gaussian_components'] and
                g['defensive_uniform_shell_probability'] == .5 and
                g['conditional_probability'] == (0. if name == 'baseline92' else .5) and
                g['contact_widths_A'] == [.02, .1, .5] and g['contact_neighbor_indices'] == [0, 1] and
                len(g['component_contact_pairs']) == 92 and g['minimum_center_distance'] == 1e-8 and
                g['minimum_polygon_area'] == 1e-16 and
                g['azimuth'] == dict(localized_probability=b, radius_floor=1e-8,
                    projection_floor=1e-10, gamma_min=.01, gamma_max=math.pi), 'Distance law changed: '+name)
    require(all(guides[a]['component_contact_pairs'] == guides['baseline92']['component_contact_pairs']
                for a in ARMS), 'Contact labels differ across arms')


def combine_old_scores(probes, by_arm, metadata):
    for arm in OLD_ARMS:
        require(len(by_arm[arm]) == len(probes), 'Missing saved density rows: '+arm)
        require([r['id'] for r in by_arm[arm]] == [p['id'] for p in probes],
                'Saved density order differs: '+arm)
    result = []
    for i, probe in enumerate(probes):
        old = by_arm['baseline92'][i]
        for arm in OLD_ARMS:
            r = by_arm[arm][i]
            require(r['latent'] == probe['latent'], 'Saved density pose mismatch')
            for key in ('hard_valid', 'shell_valid', 'capture_valid', 'pose', 'raw_coordinates'):
                require(r[key] == old[key], 'Old arm geometry differs: '+key)
            require(abs(r['baseline_log_density'] - old['log_proposal_density']) < 1e-10,
                    'Saved old92 density mismatch')
            require(abs(r['log_physical_jacobian'] - old['log_physical_jacobian']) < 1e-12,
                    'Saved Jacobian mismatch')
        result.append(dict(id=probe['id'], latent=probe['latent'],
            group='critical' if i < 78 else 'breadth', metadata=metadata[i],
            old_log_q={arm: by_arm[arm][i]['log_proposal_density'] for arm in OLD_ARMS},
            old_density_CPU_seconds={arm: by_arm[arm][i]['density_cpu_seconds'] for arm in OLD_ARMS},
            saved_geometry={key:old[key] for key in ('hard_valid','shell_valid','capture_valid','pose',
                'raw_coordinates','log_physical_jacobian','width_contacts')}))
    return result


def prepare(out, preparation, passive):
    out, preparation, passive = map(lambda p:Path(p).resolve(), (out, preparation, passive))
    require(not out.exists(), 'Fresh immutable preparation required')
    bindings = {}
    def bind(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, 'Saved source changed: '+str(path))
        bindings[str(path)] = digest
        return path
    frozen = read(bind(preparation/'freeze.json'))
    for name, digest in frozen['files'].items(): bind(preparation/name, digest)
    state = read(bind(passive/'status.json'))
    require(state['complete'] and state['protocol_sha256'] == EXPECTED_PROTOCOL,
            'Original passive execution is incomplete or changed')
    bind(passive/'protocol.json', EXPECTED_PROTOCOL)
    bind(passive/'analysis.json', state['analysis_sha256'])
    for name, digest in [('original92', GUIDE_SHA), ('region', REGION_SHA), ('shape', SHAPE_SHA)]:
        bind(preparation/'common'/f'{name}.json', digest)
    original = read(preparation/'common/original92.json')
    guides = {arm:read(preparation/'guides'/f'{arm}.json') for arm in OLD_ARMS}
    validate_guides(original, guides)
    critical, breadth = rows(preparation/'probes.jsonl'), rows(preparation/'coverage-probes.jsonl')
    critical_metadata = read(preparation/'axis-diagnostics.json')['rows']
    breadth_metadata = read(preparation/'coverage-selection.json')['rows']
    classes = validate_probe_allocation(critical, breadth, critical_metadata, breadth_metadata)
    by_arm, receipts = {}, []
    for arm in OLD_ARMS:
        by_arm[arm] = []
        for population in ('r00', 'coverage'):
            job = next(j for j in [*state['jobs'], *state['coverage_jobs']]
                       if j['arm'] == arm and j['id'] == population)
            require(job['status'] == 'complete', 'Saved scoring job incomplete')
            directory = passive/'runs'/arm/population
            for name in ('probes.jsonl','manifest.json','summary.json','independent-audit.json'):
                bind(directory/name, job['output_sha256'][name])
            manifest = read(directory/'manifest.json')
            require(manifest['guide_sha256'] == sha(preparation/'guides'/f'{arm}.json') and
                    manifest['region_sha256'] == REGION_SHA and manifest['shape_sha256'] == SHAPE_SHA,
                    'Saved densities use different geometry or guide')
            require(read(directory/'independent-audit.json')['complete'], 'Saved independent audit failed')
            by_arm[arm].extend(rows(directory/'probes.jsonl'))
            receipts.append((arm, population, directory))
    combined = combine_old_scores(critical+breadth, by_arm, critical_metadata+breadth_metadata)
    out.mkdir(parents=True)
    for directory in ('common','guides','prior-scores'): (out/directory).mkdir()
    for name in ('original92.json','region.json','shape.json'):
        shutil.copy2(preparation/'common'/name, out/'common'/name)
    config = read(preparation/'common/config.json')
    require(config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035,
            'Original physical parameters changed')
    config['shape'] = str(out/'common/shape.json')
    write(out/'common/config.json', config)
    for arm in OLD_ARMS: shutil.copy2(preparation/'guides'/f'{arm}.json', out/'guides'/f'{arm}.json')
    for name in ('probes.jsonl','coverage-probes.jsonl','coverage-selection.json','axis-diagnostics.json'):
        shutil.copy2(preparation/name, out/name)
    for arm, population, directory in receipts:
        target = out/'prior-scores'/arm/population
        target.mkdir(parents=True)
        for name in ('probes.jsonl','manifest.json','summary.json','independent-audit.json'):
            shutil.copy2(directory/name, target/name)
    with (out/'score-probes.jsonl').open('x') as stream:
        for probe in critical+breadth: stream.write(json.dumps(probe, allow_nan=False)+'\n')
    write(out/'saved-scores.json', dict(schema='contact-arc-score-saved-inputs-v1', rows=combined))
    write(out/'plan.json', dict(schema='contact-arc-score-preparation-v1', source_sha256=bindings,
        original_passive_protocol_sha256=EXPECTED_PROTOCOL, arms=ARMS, baseline_arm='baseline92',
        unique_saved_queries=206, candidate_density_evaluations=412, critical_queries=78,
        critical_original_sources={'baseline':4,'expanded':74}, breadth_queries=128,
        breadth_classes=dict(classes), minimum_arc_mass=MINIMUM_ARC_MASS,
        defensive_uniform_probability=.5, conditional_probability=.5, widths_A=[.02,.1,.5],
        maximum_CPU_workers=1, new_pose_draws=0, new_Poisson_clouds=0, new_physical_mass_estimates=0,
        geometry_deduplicated_across_arms=True, fixed_probe_order='78 original critical, then128 original breadth',
        guides_unchanged=True, labels_unchanged=True, all_components_and_widths_in_density=True,
        selection='All previously frozen78critical and128breadth poses. No new selection/filtering, including all32invalid breadth rows.',
        fallback='Arc mass at or below1e-12 retains the original azimuth law at fixed orientation and radii; no rejection/retry or redraw of outer coordinates.',
        density='Complete component/width/fallback law, retaining original defensive mixture. Geometry shared between arms; azimuth masses differ by arm.',
        retrospective_moments='Separately for source84baseline4rows and source92expanded74rows, sum exp(sum(paired_log_weights)+source_log_q-candidate_log_q-log(source_attempted_draws)); compare ratios to saved old92 and old distance laws. Retain all paired-cloud terms and attempted denominators.',
        diagnostics=['full-mixture q versus saved old92 and distance q at every probe',
            'critical paired second-moment ratios separately by original source, contribution ESS and largest fraction',
            'breadth log-density ratios by all four fixed classes, including invalid',
            'CPU for full mixture and shared geometry, candidate-circle counts and fallback counts',
            'independent geometry/azimuth-mass/density reconstruction before interpretation'],
        no_optional_stopping=True, no_retries=True, no_autoextension=True,
        physical_gates_unchanged=True, physical_campaign_authorized=False,
        interpretation='Retrospective score-only proposal diagnostics, not held-out validation, physical weights, sampling speedup, or finite-system stability evidence. No assembly conclusion follows.'))
    shutil.copy2(__file__, out/'source.py')
    write(out/'freeze.json', dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(prepared=True, plan_sha256=sha(out/'plan.json'), probes_sha256=sha(out/'score-probes.jsonl'),
                source_files_bound=len(bindings), unique_saved_queries=206, candidate_densities=412,
                new_pose_draws=0, new_Poisson_clouds=0, out=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out','preparation','passive'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.preparation, args.passive), indent=2))
