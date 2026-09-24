#!/usr/bin/env python3
"""Independent audit of the explicitly native-excluded R4 SMC target.

``audit_population(context, job, out)`` never launches a sampler.  The context
contains config/current/reference dictionaries, protocol binary/source and
physical-input hashes, the original frozen ``definition`` path, and a
``compiled_definition`` path.  ``job`` has id/seed/output/argv.  An optional
``stages_to_classify`` list controls bridge profiles, never native auditing.

Analytic tests may instead supply ``independent_native`` with independently
constructed expected_compiled, source_sha256, and classify_pair(anchor, pose).
This explicit adapter is identified in the result and is not a protein audit.
The command-line interface accepts a JSON context and a JSON job (no adapter).
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import gzip
import json
import math
from pathlib import Path

from analyze_r4_smc_control import (
    Chart, Proposal, Ledger, STRATA, audit_clouds, audit_pose, close,
    jsonlines, line, logvalue, nullable_log, parse_job_options, pose_key,
    read, require, sha, systematic_parents, verify_ancestry, write,
)
from analyze_mobile_native_pocket import load_classifier
from analyze_mobile_threshold_reference import ExclusionContact
import numpy as np
from scipy.special import logsumexp

SCHEMA = 'native-excluded-smc-population-audit-v1'
CLASSES = ('total', 'contact_no_native_entry', 'unbound_no_native_entry')
LIMITATIONS = (
    'This is an explicitly native-excluded conditional target, not unrestricted '
    'protein or finite-system equilibrium. Every initialization attempt and '
    'operative native decision is independently classified. Atomic hard/capture/'
    'R4 validity is independently checked for all initialization and recorded '
    'operative poses, including initially rejected zeros. Geometrically rejected '
    'mutation proposals have counters but no saved poses; their predicate '
    'execution is not replayed. Poisson count factors and envelope accounting '
    'are reconstructed, but individual unsaved cloud coordinates, geometric '
    'thinning/envelope correctness, proposal RNG law, independent random streams, '
    'and floating-point execution remain implementation obligations. Descendant '
    'counts and genealogy are not IID ESS or convergence guarantees.'
)


def compiled_from_model(model):
    """Materialize all fields from the independently verified frozen sources."""
    return dict(schema='native-entry-compiled-v1',
        source_definition_sha256=model.definition_sha256,
        source_input_sha256=model.definition['input_sha256'],
        criteria=copy.deepcopy(model.definition['criteria']), fixed_poses=model.fixed_poses,
        members=[dict(position=p.tolist(), rotation=r.tolist())
                 for p, r in zip(model.member_positions, model.member_rotations)],
        monomer_atoms=[dict(center=p.tolist(), radius=float(r), residue=int(s))
                       for p, r, s in zip(model.atoms, model.radii, model.residues)],
        residue_count=model.residue_count,
        references=[dict(label=label, family=ref['family'],
                         position=ref['position'].tolist(), rotation=ref['rotation'].tolist(),
                         native_residue_pairs=sorted(ref['native_residue_pairs']))
                    for label, ref in model.references.items()],
        motifs=[dict(id=m['id'], position=model.motif_positions[i].tolist(),
                     rotation=model.motif_rotations[i].tolist(),
                     member_contacts=[{k: c[k] for k in
                         ('member_i', 'member_j', 'directed_class')}
                         for c in m['member_contacts']]) for i, m in enumerate(model.motifs)])


def compare_decision(expected, saved, errors, path=''):
    """Labels and IDs are exact; only geometric diagnostics have roundoff slack."""
    if isinstance(expected, dict):
        require(isinstance(saved, dict) and set(saved) == set(expected),
                'Native decision keys differ: '+path)
        for key in expected:
            compare_decision(expected[key], saved[key], errors, path+'/'+key)
    elif isinstance(expected, list):
        require(isinstance(saved, list) and len(saved) == len(expected),
                'Native decision length differs: '+path)
        for i, (a, b) in enumerate(zip(expected, saved)):
            compare_decision(a, b, errors, path+'/'+str(i))
    elif isinstance(expected, float):
        require(type(saved) in (int, float) and math.isfinite(saved),
                'Nonfinite native diagnostic: '+path)
        angular = 'orientation' in path
        delta = abs(expected-saved)
        name = 'angle_deg' if angular else 'position_or_gap_A'
        errors[name] = max(errors[name], delta)
        require(delta <= (3e-6 if angular else 2e-9),
                'Native geometric diagnostic differs: '+path)
    else:
        require(type(expected) is type(saved) and expected == saved,
                'Native discrete decision differs: '+path)


class NativeAudit:
    def __init__(self, context, ledger, stream):
        self.config = context['config']
        self.chart = Chart(context['current'])
        self.stream = stream
        self.cache = OrderedDict()
        self.calls = self.certificate_checks = self.hard_checks = 0
        self.near_zero_geometry = 0
        self.errors = dict(angle_deg=0., position_or_gap_A=0.)
        compiled_path = ledger.bind(context['compiled_definition'])
        compiled = read(compiled_path)
        self.test_adapter = 'independent_native' in context
        if self.test_adapter:
            self.model = context['independent_native']
            expected = self.model.expected_compiled
            require(self.model.source_sha256, 'Analytic adapter needs frozen source bindings')
            for path, digest in self.model.source_sha256.items():
                ledger.bind(path, digest)
        else:
            definition = ledger.bind(context['definition'], context.get('definition_sha256'))
            self.model, self.source_binding = load_classifier(definition)
            for name, digest in self.model.definition['input_sha256'].items():
                ledger.bind(self.model.root/name, digest)
            expected = compiled_from_model(self.model)
        require(compiled == expected, 'Compiled predicate differs from independent source reconstruction')
        require(compiled['schema'] == 'native-entry-compiled-v1', 'Unknown compiled predicate schema')
        require(compiled['fixed_poses'] == self.config['fixed_poses'], 'Native scaffold differs')
        shape = ledger.bind(self.config['shape'])
        require(compiled['source_input_sha256']['tetramer-shape.json'] == sha(shape),
                'Native shape binding differs')
        self.binding = dict(definition_sha256=compiled['source_definition_sha256'],
            compiled_sha256=sha(compiled_path), shape_sha256=sha(shape),
            fixed_poses=self.config['fixed_poses'])
        self.geometry = ExclusionContact(read(shape), self.config['fixed_poses'],
                                         self.config['depletant_radius'])

    def geometry_at(self, pose):
        key = pose_key(pose)
        if key not in self.cache:
            # Chart validates finite translation and normalized quaternion too.
            chart = self.chart.evaluate(pose)
            contact = self.geometry.classify(pose, require_hard_valid=False)
            gaps = [a['minimum_surface_gap_A'] for a in contact['anchors']]
            require(all(math.isfinite(g) for g in gaps), 'Nonfinite atomic surface gap')
            atomic = all(g >= 0. for g in gaps)  # Rust overlap is strictly d^2 < (ra+rb)^2.
            capture = bool(np.linalg.norm(np.asarray(pose['position'])-
                                          self.config['capture_center']) <= self.config['capture_radius'])
            near = any(abs(g) <= 1e-8 for g in gaps)
            self.near_zero_geometry += int(near)
            self.hard_checks += 1
            self.cache[key] = dict(chart=chart, contact=contact, atomic_hard_valid=atomic,
                capture_valid=capture, hard_valid=atomic and capture and chart['inside'],
                near_zero_atomic_gap=near, native=None)
            if len(self.cache) > 32768:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def native_at(self, pose):
        data = self.geometry_at(pose)
        require(data['hard_valid'], 'Operative native pose violates atomic hard/capture/R4 geometry')
        if data['native'] is None:
            groups = []
            for i, anchor in enumerate(self.config['fixed_poses']):
                matches = self.model.classify_pair(anchor, pose)
                groups.append(dict(anchor_index=i, matched_motif_ids=[m['motif_id'] for m in matches],
                                   matches=matches))
            anchors = [g['anchor_index'] for g in groups if g['matches']]
            data['native'] = dict(native_any=bool(anchors), matched_anchor_indices=anchors, per_anchor=groups)
            self.calls += 1
        return data['native']

    def certificate(self, pose, saved, location, retained=False):
        decision = self.native_at(pose)
        expected = dict(decision, pose=pose, predicate_binding=self.binding)
        # Pose and hash bindings are bitwise/discrete identities, not parity-tolerant diagnostics.
        require(isinstance(saved, dict) and saved.get('pose') == pose and
                saved.get('predicate_binding') == self.binding, 'Native certificate pose/provenance differs')
        compare_decision(decision, {k: v for k, v in saved.items()
                                   if k not in ('pose', 'predicate_binding')}, self.errors)
        require(not retained or not decision['native_any'], 'Native pose retained in excluded target')
        self.certificate_checks += 1
        line(self.stream, dict(location=location, pose=pose, **decision,
                               near_zero_atomic_gap=self.geometry_at(pose)['near_zero_atomic_gap']))
        return expected

    def classify(self, pose, target_valid):
        data = self.geometry_at(pose)
        native = self.native_at(pose) if data['hard_valid'] else None
        actual = data['hard_valid'] and not native['native_any']
        require(type(target_valid) is bool and target_valid == actual, 'Target membership differs')
        bound = data['contact']['exclusion_contact']
        return dict(classes=dict(total=actual, contact_no_native_entry=actual and bound,
                                 unbound_no_native_entry=actual and not bound),
            strata=self.chart.bins(data['chart']['latent']) if data['chart']['latent'] is not None else None,
            contact=data['contact'], geometric_hard_valid=data['hard_valid'],
            capture_valid=data['capture_valid'], atomic_hard_valid=data['atomic_hard_valid'],
            near_zero_atomic_gap=data['near_zero_atomic_gap'], classification=native)


def parse_options(job):
    options = parse_job_options(job)
    args = dict(zip(job['argv'][1::2], job['argv'][2::2]))
    require('--exclude-native-entry' in args, 'Native-excluded command flag is required')
    allowed = {'--config', '--region', '--exclude-native-entry', '--initial-reference-region',
        '--initial-current-probability', '--out', '--seed', '--initial-draws', '--population',
        '--stages', '--sweeps-per-stage', '--cloud-replicates', '--lambda-ratio', '--bridge'}
    require(set(args) <= allowed, 'Unknown restricted SMC command argument')
    options['exclude_native_entry'] = args['--exclude-native-entry']
    require(options['bridge'] in ('proposal_density', 'physical_activity'), 'Unknown SMC bridge')
    require(options['initial_draws'] > 0 and options['population'] > 0 and
            options['sweeps_per_stage'] >= 0 and options['cloud_replicates'] > 0 and
            math.isfinite(options['lambda_ratio']) and options['lambda_ratio'] > 0,
            'Invalid fixed SMC allocation')
    require(options['out'] == job['output'] and options['seed'] == job['seed'], 'Job path/seed differs')
    return options


def optional_close(saved, expected, message):
    if expected == -math.inf:
        require(saved is None, message+'; zero must remain null')
    else:
        close(saved, expected, message)


def audit_gate(event, old_pose, proposed_pose, proposal, config, options, beta):
    require(event['old_pose'] == old_pose and event['proposed_pose'] == proposed_pose,
            'Bath gate pose differs from native mutation history')
    close(event['beta'], beta, 'Gate beta differs')
    old_g = proposal.evaluate(old_pose)[2]
    c, _, new_g = proposal.evaluate(proposed_pose)
    require(c['inside'] and math.isfinite(new_g), 'Gate proposal lost current support')
    close(event['log_g_old'], old_g, 'Old gate density differs')
    close(event['log_g_new'], new_g, 'New gate density differs')
    correction = (1-beta)*(new_g-old_g) if options['bridge'] == 'proposal_density' else 0.
    close(event['deterministic_log_correction'], correction, 'Gate bridge correction differs')
    gate = event['gate']
    for name in ('gained', 'lost', 'raw_points', 'retained_points'):
        require(type(gate[name]) is int and gate[name] >= 0, 'Invalid mutation count')
    require(gate['gained']+gate['lost'] == gate['retained_points'] <= gate['raw_points'],
            'Mutation thinning accounting differs')
    activity = beta*config['reservoir_density']
    log_gate = ((gate['gained']-gate['lost'])*math.log1p(1/config['poisson_lambda_ratio'])
                if activity > 0 else 0.)
    if activity == 0:
        require(all(gate[name] == 0 for name in ('gained', 'lost', 'raw_points', 'retained_points')),
                'Nonzero gate counts at zero activity')
    close(gate['log_weight'], log_gate, 'Gained/lost gate factor differs')
    loga = min(0., log_gate+correction)
    close(event['log_acceptance'], loga, 'Combined acceptance differs')
    logu = nullable_log(event['log_uniform'])
    require(logu <= 0 and type(event['accepted']) is bool and event['accepted'] == (logu < loga),
            'Acceptance uniform/decision differs')
    return event['accepted']


def audit_stage(stage, previous, proposal, config, options, index, log_z, observer):
    n = options['population']; beta = options['schedule'][index]
    db = beta-options['schedule'][index-1]
    activity = beta*config['reservoir_density']; delta = db*config['reservoir_density']
    lam = options['lambda_ratio']*delta if delta > 0 else 1.
    require(stage['stage'] == index and len(stage['potentials']) == len(stage['particles']) ==
            len(stage['parents']) == n, 'Stage allocation/index differs')
    for field, value in [('beta', beta), ('delta_beta', db), ('activity', activity),
        ('previous_activity', activity-delta), ('delta_activity', delta), ('incremental_lambda', lam)]:
        close(stage[field], value, 'Stage path differs')
    logs = []
    for i, row in enumerate(stage['potentials']):
        require(row['particle'] == i and row['pose'] == previous[i]['pose'] and
                row['initial_ancestor'] == previous[i]['initial_ancestor'], 'Potential history differs')
        require(row['native_membership'] == previous[i]['native_membership'], 'Potential lost retained certificate')
        observer.certificate(row['pose'], row['native_membership'],
                             dict(stage=index, kind='potential', particle=i), retained=True)
        g = audit_pose(row, proposal, config)
        close(row['log_g'], g, 'Potential proposal density differs')
        correction = -db*g if options['bridge'] == 'proposal_density' else 0.
        close(row['deterministic_log_correction'], correction, 'Potential bridge correction differs')
        weight = audit_clouds(row['clouds'], delta, lam, options['cloud_replicates'])+correction
        close(row['log_incremental_weight'], weight, 'Incremental potential differs')
        logs.append(weight)
    increment = float(logsumexp(logs)-math.log(n))
    close(stage['log_Z_increment'], increment, 'Normalizer increment differs')
    log_z += increment; close(stage['log_Z'], log_z, 'Normalizer recursion differs')
    weights = np.exp(np.asarray(logs)-max(logs))
    close(stage['pre_resampling_weight_ESS'], float(weights.sum()**2/(weights@weights)), 'Resampling ESS differs')
    parents = systematic_parents(logs, n, stage['resampling_offset'])
    require(parents == stage['parents'], 'Systematic resampling parents differ')
    current = [copy.deepcopy(previous[i]) for i in parents]
    gates = {}; last = (-1, -1)
    for event in stage['mutation_density_records']:
        key = event['particle'], event['sweep']
        require(key > last and key not in gates, 'Duplicate/out-of-order bath gate')
        last = key; gates[key] = event
    accepted = native_rejected = gate_rejected = translations = rotations = raw_points = 0
    used = set(); last = (-1, -1)
    for event in stage['native_mutation_records']:
        i, sweep = event['particle'], event['sweep']; key = i, sweep
        require(type(i) is int and type(sweep) is int and 0 <= i < n and
                0 <= sweep < options['sweeps_per_stage'] and key > last,
                'Invalid/duplicate/out-of-order native mutation identity')
        last = key
        require(event['old_pose'] == current[i]['pose'] and
                event['old_membership'] == current[i]['native_membership'], 'Mutation lost retained state')
        location = dict(stage=index, kind='mutation_old', particle=i, sweep=sweep)
        observer.certificate(event['old_pose'], event['old_membership'], location, retained=True)
        candidate = observer.certificate(event['proposed_pose'], event['proposed_membership'],
            dict(stage=index, kind='mutation_candidate', particle=i, sweep=sweep))
        if candidate['native_any']:
            require(event['outcome'] == 'native_rejected' and event['accepted'] is False and
                    event['bath_gate_evaluated'] is False and key not in gates,
                    'Native rejection invoked bath gate or changed state')
            native_rejected += 1
            continue
        require(event['bath_gate_evaluated'] is True and key in gates,
                'Non-native hard-valid candidate lost bath gate')
        gate = gates[key]; used.add(key)
        take = audit_gate(gate, event['old_pose'], event['proposed_pose'], proposal, config, options, beta)
        require(event['accepted'] is take and event['outcome'] == ('accepted' if take else 'gate_rejected'),
                'Native/gate decision history differs')
        raw_points += gate['gate']['raw_points']
        if take:
            accepted += 1
            translations += int(event['old_pose']['position'] != event['proposed_pose']['position'])
            rotations += int(event['old_pose']['orientation'] != event['proposed_pose']['orientation'])
            current[i]['pose'] = event['proposed_pose']
            current[i]['native_membership'] = event['proposed_membership']
        else:
            gate_rejected += 1
    require(used == set(gates), 'Bath gate lacks a matching operative native decision')
    counts = stage['mutation_counts']
    require(all(type(v) is int and v >= 0 for v in counts.values()), 'Invalid mutation counters')
    require(counts['attempted'] == n*options['sweeps_per_stage'] and counts['attempted'] ==
            sum(counts[k] for k in ('accepted', 'capture_rejected', 'region_rejected',
                                   'hard_rejected', 'native_rejected', 'gate_rejected')),
            'Mutation attempts were lost')
    for name, value in [('accepted', accepted), ('native_rejected', native_rejected),
        ('gate_rejected', gate_rejected), ('raw_points', raw_points),
        ('accepted_translation_changes', translations), ('accepted_rotation_changes', rotations)]:
        require(counts[name] == value, 'Mutation counter differs: '+name)
    for i, particle in enumerate(stage['particles']):
        require(particle['initial_ancestor'] == current[i]['initial_ancestor'] and
                particle['pose'] == current[i]['pose'] and
                particle['native_membership'] == current[i]['native_membership'],
                'Retained endpoint differs from accepted/rejected history')
        audit_pose(particle, proposal, config)
        observer.certificate(particle['pose'], particle['native_membership'],
            dict(stage=index, kind='endpoint', particle=i), retained=True)
    return log_z, dict(stage=index, beta=beta, log_Z=log_z,
        ancestry=verify_ancestry(stage['particles'], stage['ancestry']),
        mutation_counts=counts, pre_resampling_weight_ESS=stage['pre_resampling_weight_ESS'])


def profile(particles, log_z, observer, stage, stream):
    counts = {k: 0 for k in CLASSES}
    strata = {f: {k: [0]*size for k in CLASSES} for f, size in STRATA.items()}
    families = {k: Counter() for k in CLASSES}
    for i, particle in enumerate(particles):
        label = observer.classify(particle['pose'], True)
        line(stream, dict(stage=stage, particle=i, initial_ancestor=particle['initial_ancestor'], **label))
        for name, yes in label['classes'].items():
            if yes:
                counts[name] += 1; families[name][particle['initial_ancestor']] += 1
                for family, index in label['strata'].items():
                    strata[family][name][index] += 1
    n = len(particles)
    require(counts['total'] == n == counts['contact_no_native_entry']+counts['unbound_no_native_entry'],
            'Restricted terminal partition has gap/overlap')
    def mass(c): return None if not c or log_z is None else log_z+math.log(c/n)
    return dict(stage=stage, particles=n, counts=counts,
        log_masses={k: mass(c) for k, c in counts.items()},
        log_strata={f: {k: [mass(c) for c in a] for k, a in names.items()} for f, names in strata.items()},
        class_distinct_initial_families={k: len(v) for k, v in families.items()},
        estimator='Zhat times terminal indicator fraction; whole-population uncertainty only',
        measure='restricted final physical target' if stage == 'terminal' else 'restricted beta bridge')


def validate_inputs(context, job, ledger):
    options = parse_options(job); base = Path(job['output']); protocol = context['protocol']
    state = read(ledger.bind(base/'status.json'))
    require(state['complete'] is True and state['phase'] == 'complete', 'Population is incomplete')
    summary = read(ledger.bind(base/'summary.json', state['summary_sha256']))
    manifest = read(ledger.bind(base/'manifest.json', summary['manifest_sha256']))
    require(summary['complete'] is True and summary['schema'] == 'latent-region-smc-native-excluded-summary-v1'
            and manifest['schema'] == 'latent-region-smc-native-excluded-v1', 'Wrong restricted SMC schema')
    require(manifest['options'] == options, 'Executed options differ from exact command')
    require(manifest['executable_sha256'] == protocol['binary_sha256'] and
            manifest['source_bundle_sha256'] == protocol['source_bundle_sha256'], 'Binary/source binding differs')
    binary = ledger.bind(job['argv'][0], protocol['binary_sha256'])
    provenance = base/'provenance'
    for name, key in [('config.json', 'config_sha256'), ('region.json', 'region_sha256'),
                      ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        ledger.bind(provenance/name, manifest[key])
    require((provenance/'source-bundle.json').read_bytes() in binary.read_bytes(),
            'Audited source bundle is not embedded in binary')
    ledger.bind(options['config'], manifest['config_sha256'])
    ledger.bind(options['region'], manifest['region_sha256'])
    ledger.bind(context['config']['shape'], manifest['shape_sha256'])
    require(manifest['region_sha256'] == protocol['physical_target']['region_sha256'] and
            manifest['shape_sha256'] == protocol['physical_target']['shape_sha256'], 'Physical input identity differs')
    require(read(options['config']) == context['config'] and read(options['region']) == context['current'],
            'Context config/chart does not match executed input')
    current = context['current']; config = context['config']
    require(current['mahalanobis_radius'] == 4 and current.get('minimum_mahalanobis_radius', 0) == 0
            and current['minimum_original_q'] == 0 and 'maximum_original_q' not in current,
            'Target is not the full original R4')
    require(config.get('target_region') is None, 'Unexpected additional docking target filter')
    require(current['physical_fixed_neighbors'] == config['fixed_poses'] and
            current['capture_center'] == config['capture_center'] and current['capture_radius'] == config['capture_radius']
            and current['activity'] == config['reservoir_density'] and
            current['depletant_radius'] == config['depletant_radius'] and
            current['shape_sha256'] == manifest['shape_sha256'], 'R4 target/config differs')
    if options['initial_reference_region']:
        ledger.bind(options['initial_reference_region'], manifest['initial_reference_sha256'])
        ledger.bind(provenance/'initial-reference-region.json', manifest['initial_reference_sha256'])
        require(read(options['initial_reference_region']) == context['reference'], 'Reference chart differs')
    require(manifest['final_target_unchanged'] is False and 'H_minus' in manifest['bridge_density'],
            'Restricted target metadata is ambiguous')
    compiled = ledger.bind(options['exclude_native_entry'])
    ledger.bind(provenance/'native-entry-compiled.json', sha(compiled))
    require(sha(compiled) == sha(context['compiled_definition']), 'Compiled predicate input differs')
    for path, digest in context.get('bindings', {}).items(): ledger.bind(path, digest)
    initial = ledger.bind(base/'initialization.jsonl', summary['initialization_sha256'])
    stages = ledger.bind(base/'stages.jsonl', summary['stages_sha256'])
    return options, state, summary, manifest, initial, stages


def audit_population(context, job, out):
    """Audit a NEW completed restricted population; write an immutable fresh result."""
    ledger = Ledger()
    for module_path in (__file__,): ledger.bind(module_path)
    # Include imported primitive implementations, not merely this wrapper.
    import analyze_r4_smc_control, analyze_mobile_native_pocket, analyze_mobile_threshold_reference
    for module in (analyze_r4_smc_control, analyze_mobile_native_pocket, analyze_mobile_threshold_reference):
        ledger.bind(module.__file__)
    options, state, summary, manifest, raw_initial, raw_stages = validate_inputs(context, job, ledger)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', dict(complete=False, phase='auditing'))
    try:
        with gzip.open(out/'native-decisions.jsonl.gz', 'wt') as decisions:
            observer = NativeAudit(context, ledger, decisions)
            require(manifest['native_exclusion'] == observer.binding, 'Manifest predicate binding differs')
            proposal = Proposal(context['current'], context.get('reference'), options['initial_current_probability'])
            initial = []; valid_indices = []; logs = []; branches = Counter()
            unrestricted_hard = []; restricted_hard = []; geometric_hits = native_rejected = exterior = 0
            class_logs = {k: [] for k in CLASSES}; hard_class_logs = {k: [] for k in CLASSES}
            class_counts = Counter(); zeros = Counter()
            with gzip.open(out/'initial-labels.jsonl.gz', 'wt') as labels:
                for i, row in enumerate(jsonlines(raw_initial)):
                    require(row['draw'] == i and i < options['initial_draws'], 'Initialization draw identity/count differs')
                    c, r, g = proposal.evaluate(row['pose'])
                    require(math.isfinite(g), 'Initialization lost proposal support')
                    close(row['log_physical_proposal_density'], g, 'Initialization proposal density differs')
                    for data, latent_key, jac_key in [(c, 'latent', 'log_physical_jacobian'),
                        (r, 'reference_latent', 'reference_log_physical_jacobian')]:
                        if data is None:
                            require(row[latent_key] is None and row[jac_key] is None, 'Absent reference chart gained coordinates')
                        elif data['latent'] is not None:
                            require(np.max(abs(np.asarray(row[latent_key])-data['latent'])) < 2e-8,
                                    'Initial inverse chart coordinates differ')
                            close(row[jac_key], data['log_jacobian'], 'Initial physical Jacobian differs')
                    close(row['latent_radius'], c['radius'], 'Initial chart radius differs')
                    require(row['current_ball_valid'] == c['inside'] and
                            row['reference_ball_valid'] == (r['inside'] if r else False), 'Initial chart support differs')
                    selected = row['selected_initial_chart']
                    require(selected in ('current', 'reference') and (selected != 'reference' or options['initial_current_probability'] < 1),
                            'Invalid initialization branch')
                    source = c if selected == 'current' else r
                    require(source is not None and source['inside'], 'Selected initial chart has no support')
                    close(row['selected_log_physical_jacobian'], source['log_jacobian'], 'Selected Jacobian differs')
                    require(np.max(abs(np.asarray(row['selected_latent'])-source['latent'])) < 2e-8, 'Selected chart coordinates differ')
                    optional_close(row['log_proposal_density'], -proposal.current.log_volume
                        if options['initial_current_probability'] == 1 else -math.inf, 'Legacy latent proposal density differs')
                    branches[selected] += 1; exterior += int(not c['inside'])
                    geometry = observer.geometry_at(row['pose']); hard = geometry['hard_valid']
                    require(type(row['hard_valid']) is bool and row['hard_valid'] == hard and
                            type(row['capture_valid']) is bool and row['capture_valid'] == geometry['capture_valid'],
                            'Independent initial hard/capture flag differs (including false-negative zeros)')
                    require(type(row['native_evaluated']) is bool and row['native_evaluated'] == hard,
                            'Native evaluation eligibility differs')
                    if hard:
                        decision = observer.certificate(row['pose'], row['native_membership'], dict(kind='initial', draw=i))
                        target = not decision['native_any']; geometric_hits += 1
                        native_rejected += int(not target); unrestricted_hard.append(-g)
                    else:
                        require(row['native_membership'] is None, 'Geometrically invalid initial row has native decision')
                        target = False
                    require(type(row['target_valid']) is bool and row['target_valid'] == target, 'Initial restricted target flag differs')
                    weight = (0. if options['bridge'] == 'proposal_density' else -g) if target else -math.inf
                    optional_close(row['log_hard_weight'], -g if hard else -math.inf, 'Unrestricted H/g diagnostic differs')
                    optional_close(row['log_target_hard_weight'], -g if target else -math.inf, 'Restricted Hminus/g diagnostic differs')
                    optional_close(row['log_initial_weight'], weight, 'Operative all-M initial bridge weight differs')
                    label = observer.classify(row['pose'], target)
                    line(labels, dict(draw=i, selected_initial_chart=selected, target_valid=target,
                        log_initial_weight=logvalue(weight), log_hard_weight=row['log_hard_weight'],
                        log_target_hard_weight=row['log_target_hard_weight'], **label))
                    initial.append(dict(pose=row['pose'], latent=row['latent'], initial_ancestor=i,
                                        native_membership=row['native_membership']))
                    for name, yes in label['classes'].items():
                        if yes:
                            class_counts[name] += 1; class_logs[name].append(weight); hard_class_logs[name].append(-g)
                    if target:
                        valid_indices.append(i); logs.append(weight); restricted_hard.append(-g)
                    else: zeros['native' if hard else 'geometric'] += 1
            m = options['initial_draws']
            require(len(initial) == m == summary['initial_draws'] and len(logs) == summary['initial_hits'] and
                    geometric_hits == summary['initial_geometric_hits'] and native_rejected == summary['initial_native_rejected'],
                    'Original denominator or restricted hit counters differ')
            iterator = iter(jsonlines(raw_stages)); first = next(iterator, None)
            require(first is not None and first['stage'] == 0 and first['initial_draws'] == m and
                    first['initial_hits'] == len(logs), 'Missing/allocation-mismatched initial stage')
            close(first['activity'], 0., 'Initialization activity differs')
            selected_stages = context.get('stages_to_classify', [0, len(options['schedule'])-1])
            require(all(type(s) is int and 0 <= s < len(options['schedule']) for s in selected_stages), 'Invalid profile stages')
            diagnostics = []; profiles = {}
            with gzip.open(out/'profile-labels.jsonl.gz', 'wt') as labels:
                if not logs:
                    require(summary['zero_estimate'] is True and summary['log_Z'] is None and summary['Z'] == 0 and
                        not summary['terminal_particles'] and not first['particles'] and not first['parents'] and
                        first['zero_estimate'] is True and first['log_Z'] is None and first['Z'] == 0 and
                        summary['completed_stage'] == state['completed_stage'] == 0 and next(iterator, None) is None,
                        'Zero population was replaced, dropped or extended')
                    verify_ancestry([], summary['ancestry'])
                    diagnostics.append(dict(stage=0, beta=0., log_Z=None, zero_estimate=True))
                    for stage in selected_stages: profiles[str(stage)] = profile([], None, observer, stage, labels)
                    terminal = profile([], None, observer, 'terminal', labels)
                else:
                    require(summary['zero_estimate'] is False, 'Positive population labeled zero')
                    log_z = float(logsumexp(logs)-math.log(m))
                    close(first['log_Z'], log_z, 'Initial normalizer lost all-M denominator')
                    close(first['log_Z_increment'], log_z, 'Initial normalizer increment differs')
                    w = np.exp(np.asarray(logs)-max(logs))
                    close(first['initial_weight_ESS'], float(w.sum()**2/(w@w)), 'Initial weight ESS differs')
                    parents = systematic_parents(logs, options['population'], first['resampling_offset'])
                    draws = [valid_indices[j] for j in parents]
                    require(first['parent_initial_draws'] == draws, 'Initial resampling ancestry differs')
                    previous = first['particles']
                    require(len(previous) == options['population'] and previous == [initial[j] for j in draws],
                            'Initial resampled pose/certificate differs')
                    for i, particle in enumerate(previous):
                        observer.certificate(particle['pose'], particle['native_membership'],
                            dict(stage=0, kind='endpoint', particle=i), retained=True)
                    diagnostics.append(dict(stage=0, beta=0., log_Z=log_z,
                        ancestry=verify_ancestry(previous, first['ancestry'])))
                    if 0 in selected_stages: profiles['0'] = profile(previous, log_z, observer, 0, labels)
                    seen = 0
                    for index, stage in enumerate(iterator, 1):
                        require(index < len(options['schedule']), 'Unexpected additional stage')
                        log_z, diagnostic = audit_stage(stage, previous, proposal, context['config'], options,
                                                        index, log_z, observer)
                        previous = stage['particles']; seen = index; diagnostics.append(diagnostic)
                        if index in selected_stages: profiles[str(index)] = profile(previous, log_z, observer, index, labels)
                    require(seen == len(options['schedule'])-1 == summary['completed_stage'] == state['completed_stage'],
                            'Missing fixed annealing stages')
                    close(summary['log_Z'], log_z, 'Terminal normalizer differs')
                    require(summary['terminal_particles'] == previous, 'Terminal poses differ from last stage')
                    verify_ancestry(previous, summary['ancestry'])
                    if summary['Z'] is not None:
                        require(summary['Z'] > 0 and math.isfinite(summary['Z']), 'Invalid linear normalizer')
                        close(math.log(summary['Z']), log_z, 'Linear/log normalizer differs')
                    terminal = profile(previous, log_z, observer, 'terminal', labels)
        def logmean(values): return float(logsumexp(values)-math.log(m)) if values else None
        ledger.recheck()
        result = dict(schema=SCHEMA, complete=True, id=job['id'], seed=job['seed'],
            zero_estimate=summary['zero_estimate'], initial_draws=m, initial_hits=len(logs),
            initial_geometric_hits=geometric_hits, initial_native_rejected=native_rejected,
            zero_initial_attempts=dict(zeros), initial_branch_counts=dict(branches),
            exterior_initial_attempts=exterior, initial_class_counts={k: class_counts[k] for k in CLASSES},
            initial_bridge_log_masses={k: logmean(v) for k, v in class_logs.items()},
            initial_physical_hard_log_masses={k: logmean(v) for k, v in hard_class_logs.items()},
            initial_unrestricted_H_over_g_log_mass=logmean(unrestricted_hard),
            initial_restricted_H_over_g_log_mass=logmean(restricted_hard),
            stages=diagnostics, profiles=profiles, terminal=terminal,
            native_definition=observer.binding, independent_analytic_test_adapter=observer.test_adapter,
            native_classifier_calls=observer.calls, native_certificate_checks=observer.certificate_checks,
            independent_atomic_geometry_calls=observer.hard_checks,
            near_zero_atomic_geometry_calls=observer.near_zero_geometry,
            maximum_native_diagnostic_error=observer.errors, source_sha256=ledger.files,
            sampler_cpu_seconds=summary.get('sampler_cpu_seconds'), limitations=LIMITATIONS)
        result['label_sha256'] = {name: sha(out/name) for name in
            ('initial-labels.jsonl.gz', 'profile-labels.jsonl.gz', 'native-decisions.jsonl.gz')}
        write(out/'population.json', result)
        write(out/'status.json', dict(complete=True, phase='complete', population_sha256=sha(out/'population.json')))
        write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
        return result
    except BaseException as error:
        write(out/'status.json', dict(complete=False, phase='failed', error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    context = read(args.context)
    context.setdefault('bindings', {})[str(args.context.resolve())] = sha(args.context)
    context['bindings'][str(args.job.resolve())] = sha(args.job)
    result = audit_population(context, read(args.job), args.out)
    print(json.dumps(dict(complete=result['complete'], id=result['id'], terminal=result['terminal']), indent=2))


if __name__ == '__main__':
    main()
