"""Adversarial saved-record tests; no protein sampling or historical audit replay."""
import copy
import gzip
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp

from analyze_native_excluded_smc import (
    NativeAudit, audit_population, audit_stage, compiled_from_model, parse_options,
)
from analyze_r4_smc_control import Ledger, Proposal, read, sha, systematic_parents, write
from validate_native_excluded_smc_reference import fixture as input_fixture, AnalyticSphereNative


def pose(x):
    return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


def ancestry(particles):
    counts = {}
    for p in particles:
        k = str(p['initial_ancestor']); counts[k] = counts.get(k, 0)+1
    n = len(particles)
    return dict(initial_ancestor_counts=counts, distinct_initial_ancestors=len(counts),
        initial_family_ESS=n*n/sum(c*c for c in counts.values()) if n else 0.,
        largest_initial_family_fraction=max(counts.values())/n if n else None)


def save_records(base, initial, stages, summary):
    for name, rows in [('initialization.jsonl', initial), ('stages.jsonl', stages)]:
        (base/name).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    summary['initialization_sha256'] = sha(base/'initialization.jsonl')
    summary['stages_sha256'] = sha(base/'stages.jsonl')
    summary['manifest_sha256'] = sha(base/'manifest.json')
    write(base/'summary.json', summary)
    write(base/'status.json', dict(complete=True, phase='complete',
        completed_stage=summary['completed_stage'], summary_sha256=sha(base/'summary.json')))


def saved_fixture(root, bridge='proposal_density', zero=False, reference=False):
    """Construct small nonphysical records with real one-atom predicate geometry."""
    root = Path(root); inputs = root/'inputs'; input_fixture(inputs, 4.)
    config = read(inputs/'config.json'); current = read(inputs/'region.json')
    config['rotation_steps_deg'] = [2.]
    write(inputs/'config.json', config)
    ref = copy.deepcopy(current) if reference else None
    if ref:
        ref['mahalanobis_radius'] = 5.
        ref['gaussian_chart']['means'][0][0] = -1.
        write(inputs/'reference.json', ref)
    model = AnalyticSphereNative(inputs/'shape.json')
    compiled = inputs/'native-compiled.json'; binary = root/'synthetic-inert-binary'
    binary.write_bytes(b'NOT AN EXECUTABLE: synthetic audit unit test\n{}\n')
    base = root/'raw'; provenance = base/'provenance'; provenance.mkdir(parents=True)
    for name in ('config.json', 'region.json', 'shape.json'):
        (provenance/name).write_bytes((inputs/name).read_bytes())
    (provenance/'native-entry-compiled.json').write_bytes(compiled.read_bytes())
    (provenance/'source-bundle.json').write_bytes(b'{}\n')
    if ref:
        (provenance/'initial-reference-region.json').write_bytes((inputs/'reference.json').read_bytes())
    argv = [str(binary), '--config', str(inputs/'config.json'), '--region', str(inputs/'region.json'),
        '--exclude-native-entry', str(compiled), '--out', str(base), '--seed', '153001010',
        '--bridge', bridge.replace('_', '-'), '--initial-draws', '5', '--population', '2',
        '--stages', '2', '--sweeps-per-stage', '3', '--cloud-replicates', '2', '--lambda-ratio', '8']
    if ref:
        argv += ['--initial-reference-region', str(inputs/'reference.json'), '--initial-current-probability', '.5']
    job = dict(id='synthetic', seed=153001010, output=str(base), argv=argv)
    options = parse_options(job); alpha = options['initial_current_probability']
    proposal = Proposal(current, ref, alpha)
    binding = dict(definition_sha256=model.expected_compiled['source_definition_sha256'],
        compiled_sha256=sha(compiled), shape_sha256=sha(inputs/'shape.json'), fixed_poses=config['fixed_poses'])
    def certificate(p):
        matches = model.classify_pair(config['fixed_poses'][0], p)
        return dict(native_any=bool(matches), matched_anchor_indices=[0] if matches else [],
            per_anchor=[dict(anchor_index=0, matched_motif_ids=[m['motif_id'] for m in matches], matches=matches)],
            pose=copy.deepcopy(p), predicate_binding=copy.deepcopy(binding))
    initial = []; valid = []; logs = []
    for i, x in enumerate([.8, 1., .9, 0., 2.3] if zero else [-1., -1.8, .8, 0., 2.3]):
        p = pose(x); c, r, g = proposal.evaluate(p)
        capture = abs(x) <= 2.2; hard = capture and c['inside'] and abs(x) >= .6
        cert = certificate(p) if hard else None; target = hard and not cert['native_any']
        w = (0. if bridge == 'proposal_density' else -g) if target else None
        row = dict(draw=i, pose=p, latent=c['latent'].tolist(), latent_radius=c['radius'],
            log_physical_jacobian=c['log_jacobian'], log_physical_proposal_density=g,
            reference_latent=r['latent'].tolist() if r else None,
            reference_log_physical_jacobian=r['log_jacobian'] if r else None,
            reference_ball_valid=r['inside'] if r else False, current_ball_valid=c['inside'],
            selected_initial_chart='current', selected_latent=c['latent'].tolist(),
            selected_log_physical_jacobian=c['log_jacobian'],
            log_proposal_density=-proposal.current.log_volume if alpha == 1 else None,
            capture_valid=capture, hard_valid=hard, target_valid=target, native_evaluated=hard,
            native_membership=cert, log_hard_weight=-g if hard else None,
            log_target_hard_weight=-g if target else None, log_initial_weight=w)
        initial.append(row)
        if target: valid.append(i); logs.append(w)
    if zero:
        stages = [dict(stage=0, activity=0., log_Z=None, Z=0., zero_estimate=True,
            initial_draws=5, initial_hits=0, parents=[], particles=[])]
        terminal = []; logz = None
    else:
        parents = systematic_parents(logs, 2, 0.)
        initial_draws = [valid[i] for i in parents]
        particles = [dict(pose=copy.deepcopy(initial[j]['pose']), latent=initial[j]['latent'],
            initial_ancestor=j, native_membership=copy.deepcopy(initial[j]['native_membership'])) for j in initial_draws]
        logz = float(logsumexp(logs)-math.log(5)); weights = np.exp(np.asarray(logs)-max(logs))
        first = dict(stage=0, activity=0., initial_draws=5, initial_hits=len(logs), log_Z=logz,
            log_Z_increment=logz, initial_weight_ESS=float(weights.sum()**2/(weights@weights)),
            resampling_offset=0., parent_initial_draws=initial_draws, particles=copy.deepcopy(particles),
            ancestry=ancestry(particles))
        stages = [first]
        for index in (1, 2):
            beta = index/2; delta = 2.; lam = 16.; potentials = []; increments = []
            for i, p in enumerate(particles):
                g = proposal.evaluate(p['pose'])[2]
                cloud = dict(raw_points=4, overlap_points=2+i, retained_cells=1, created_cells=1,
                    certified_cells=0, lower_volume=.2, upper_volume=.5, uncertain_volume=.3,
                    log_weight=delta*.2+(2+i)*math.log1p(delta/lam))
                correction = -.5*g if bridge == 'proposal_density' else 0.
                w = cloud['log_weight']+correction; increments.append(w)
                potentials.append(dict(particle=i, **copy.deepcopy(p), clouds=[copy.deepcopy(cloud), copy.deepcopy(cloud)],
                    log_g=g, deterministic_log_correction=correction, log_incremental_weight=w))
            parents = systematic_parents(increments, 2, 0.)
            terminal = [copy.deepcopy(particles[i]) for i in parents]
            native_events = []; gates = []
            for i, p in enumerate(terminal):
                for sweep in range(3):
                    old = copy.deepcopy(p['pose']); old_cert = copy.deepcopy(p['native_membership'])
                    new = pose(.8) if sweep == 0 else pose(old['position'][0]-.02)
                    cert = certificate(new); take = sweep == 1
                    native_events.append(dict(particle=i, sweep=sweep, old_pose=old, proposed_pose=new,
                        old_membership=old_cert, proposed_membership=cert, accepted=take,
                        outcome='native_rejected' if sweep == 0 else 'accepted' if take else 'gate_rejected',
                        bath_gate_evaluated=sweep > 0))
                    if sweep:
                        old_g = proposal.evaluate(old)[2]; new_g = proposal.evaluate(new)[2]
                        correction = (1-beta)*(new_g-old_g) if bridge == 'proposal_density' else 0.
                        gain, lost = (3, 1) if take else (0, 5)
                        gate = dict(gained=gain, lost=lost, raw_points=gain+lost,
                            retained_points=gain+lost, log_weight=(gain-lost)*math.log1p(1/8))
                        loga = min(0., gate['log_weight']+correction)
                        gates.append(dict(particle=i, sweep=sweep, old_pose=old, proposed_pose=new,
                            log_g_old=old_g, log_g_new=new_g, beta=beta, deterministic_log_correction=correction,
                            gate=gate, log_acceptance=loga, log_uniform=-1. if take else -.001, accepted=take))
                    if take:
                        p['pose'] = new; p['latent'] = proposal.evaluate(new)[0]['latent'].tolist(); p['native_membership'] = cert
            inc = float(logsumexp(increments)-math.log(2)); logz += inc
            w = np.exp(np.asarray(increments)-max(increments))
            stage = dict(stage=index, beta=beta, delta_beta=.5, activity=4*beta,
                previous_activity=4*beta-2., delta_activity=2., incremental_lambda=16.,
                log_Z=logz, log_Z_increment=inc, pre_resampling_weight_ESS=float(w.sum()**2/(w@w)),
                potentials=potentials, parents=parents, resampling_offset=0., particles=copy.deepcopy(terminal),
                native_mutation_records=native_events, mutation_density_records=gates,
                mutation_counts=dict(attempted=6, accepted=2, native_rejected=2, gate_rejected=2,
                    hard_rejected=0, region_rejected=0, capture_rejected=0, raw_points=18,
                    accepted_translation_changes=2, accepted_rotation_changes=0), ancestry=ancestry(terminal))
            stages.append(stage); particles = copy.deepcopy(terminal)
    protocol = dict(binary_sha256=sha(binary), source_bundle_sha256=sha(provenance/'source-bundle.json'),
        physical_target=dict(shape_sha256=sha(inputs/'shape.json'), region_sha256=sha(inputs/'region.json')))
    manifest = dict(schema='latent-region-smc-native-excluded-v1', options=options,
        executable_sha256=protocol['binary_sha256'], source_bundle_sha256=protocol['source_bundle_sha256'],
        config_sha256=sha(inputs/'config.json'), region_sha256=sha(inputs/'region.json'),
        shape_sha256=sha(inputs/'shape.json'), initial_reference_sha256=sha(inputs/'reference.json') if ref else None,
        final_target_unchanged=False, bridge_density='H_minus', native_exclusion=binding)
    write(base/'manifest.json', manifest)
    summary = dict(schema='latent-region-smc-native-excluded-summary-v1', complete=True, zero_estimate=zero,
        log_Z=logz, Z=0. if zero else math.exp(logz), initial_draws=5, initial_hits=len(valid),
        initial_geometric_hits=3, initial_native_rejected=3 if zero else 1, completed_stage=0 if zero else 2,
        terminal_particles=terminal, ancestry=ancestry(terminal))
    save_records(base, initial, stages, summary)
    context = dict(protocol=protocol, config=config, current=current, reference=ref,
        compiled_definition=str(compiled), independent_native=model, stages_to_classify=[0, 1, 2])
    return context, job, initial, stages, summary


class RestrictedAuditTests(unittest.TestCase):
    def test_both_bridges_all_attempts_and_terminal_indicator_masses(self):
        for bridge in ('proposal_density', 'physical_activity'):
            for reference in (False, True):
                with self.subTest(bridge=bridge, reference=reference), tempfile.TemporaryDirectory() as d:
                    context, job, initial, stages, summary = saved_fixture(d, bridge, reference=reference)
                    result = audit_population(context, job, Path(d)/'audit')
                    self.assertTrue(result['complete']); self.assertEqual(result['initial_draws'], 5)
                    self.assertEqual(result['initial_hits'], 2); self.assertEqual(result['initial_geometric_hits'], 3)
                    self.assertEqual(result['initial_native_rejected'], 1)
                    self.assertEqual(result['zero_initial_attempts'], {'native': 1, 'geometric': 2})
                    self.assertEqual(result['terminal']['counts']['total'], 2)
                    self.assertAlmostEqual(result['terminal']['log_masses']['total'], summary['log_Z'])
                    self.assertGreater(result['initial_unrestricted_H_over_g_log_mass'], result['initial_restricted_H_over_g_log_mass'])
                    self.assertEqual(set(result['profiles']), {'0', '1', '2'})
                    with gzip.open(Path(d)/'audit/initial-labels.jsonl.gz', 'rt') as stream:
                        rows = [json.loads(line) for line in stream]
                    self.assertEqual(len(rows), 5)
                    self.assertTrue(all(row['strata'] is not None for row in rows))
                    self.assertIsNone(rows[2]['log_initial_weight'])
                    self.assertGreater(result['native_certificate_checks'], 30)

    def test_zero_population_remains_zero_without_refill(self):
        with tempfile.TemporaryDirectory() as d:
            context, job, _, _, _ = saved_fixture(d, zero=True)
            result = audit_population(context, job, Path(d)/'audit')
            self.assertTrue(result['zero_estimate']); self.assertEqual(result['initial_native_rejected'], 3)
            self.assertTrue(all(v is None for v in result['terminal']['log_masses'].values()))
            self.assertEqual(result['terminal']['particles'], 0)
            self.assertEqual(set(result['profiles']), {'0', '1', '2'})
            self.assertTrue(all(all(v is None for v in a) for a in result['terminal']['log_strata']['radial'].values()))

    def test_rejects_resealed_corrupt_initial_zeros_and_densities(self):
        changes = {
            'false_negative_hard': lambda rows: rows[0].update(hard_valid=False),
            'false_positive_hard': lambda rows: rows[3].update(hard_valid=True),
            'native_becomes_target': lambda rows: rows[2].update(target_valid=True),
            'native_zero_weight': lambda rows: rows[2].update(log_initial_weight=0.),
            'dropped_native_decision': lambda rows: rows[2].update(native_membership=None),
            'wrong_motif': lambda rows: rows[2]['native_membership']['per_anchor'][0]['matched_motif_ids'].__setitem__(0, 9),
            'wrong_J': lambda rows: rows[0].update(log_physical_jacobian=0.),
            'wrong_proposal': lambda rows: rows[0].update(log_physical_proposal_density=0.),
            'drop_attempt': lambda rows: rows.pop(),
        }
        for label, change in changes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as d:
                context, job, initial, stages, summary = saved_fixture(d)
                change(initial); save_records(Path(job['output']), initial, stages, summary)
                with self.assertRaises(ValueError): audit_population(context, job, Path(d)/'audit')

    def test_reconstructs_every_native_and_gate_rejection_history(self):
        changes = {
            'native_accepted': lambda s: s['native_mutation_records'][0].update(accepted=True),
            'native_bath': lambda s: s['native_mutation_records'][0].update(bath_gate_evaluated=True),
            'missing_native': lambda s: s['native_mutation_records'].pop(0),
            'stale_old': lambda s: s['native_mutation_records'][2]['old_pose']['position'].__setitem__(0, -.8),
            'gate_count': lambda s: s['mutation_density_records'][0]['gate'].update(lost=0),
            'gate_correction': lambda s: s['mutation_density_records'][0].update(deterministic_log_correction=1.),
            'acceptance': lambda s: s['mutation_density_records'][1].update(log_uniform=-100.),
            'cloud': lambda s: s['potentials'][0]['clouds'][0].update(overlap_points=99),
            'potential_certificate': lambda s: s['potentials'][0]['native_membership'].update(native_any=True),
            'normalizer': lambda s: s.update(log_Z=s['log_Z']+.3),
            'resampling': lambda s: s['parents'].__setitem__(0, 1),
            'endpoint': lambda s: s['particles'][0]['pose']['position'].__setitem__(0, -1.5),
            'counts': lambda s: s['mutation_counts'].update(native_rejected=1),
        }
        with tempfile.TemporaryDirectory() as d:
            context, job, _, stages, _ = saved_fixture(d, reference=True)
            observer = NativeAudit(context, Ledger(), io.StringIO())
            options = parse_options(job); proposal = Proposal(context['current'], context['reference'], .5)
            args = (stages[0]['particles'], proposal, context['config'], options, 1, stages[0]['log_Z'], observer)
            audit_stage(stages[1], *args)
            for label, change in changes.items():
                with self.subTest(label=label):
                    stage = copy.deepcopy(stages[1]); change(stage)
                    with self.assertRaises(ValueError): audit_stage(stage, *args)

    def test_compiled_source_hash_schema_and_certificate_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            context, job, _, _, _ = saved_fixture(d)
            ledger = Ledger(); observer = NativeAudit(context, ledger, io.StringIO())
            p = pose(-1.); decision = dict(observer.native_at(p), pose=p, predicate_binding=observer.binding)
            observer.certificate(p, decision, dict(test=True), retained=True)
            bad = copy.deepcopy(decision); bad['predicate_binding']['compiled_sha256'] = '0'*64
            with self.assertRaises(ValueError): observer.certificate(p, bad, dict(test=True))
            compiled = Path(context['compiled_definition']); changed = read(compiled)
            changed['motifs'][0]['position'][0] += .1; write(compiled, changed)
            with self.assertRaisesRegex(ValueError, 'source reconstruction'):
                NativeAudit(context, Ledger(), io.StringIO())
            with self.assertRaisesRegex(ValueError, 'Source changed'): ledger.recheck()
        with tempfile.TemporaryDirectory() as d:
            context, job, initial, stages, summary = saved_fixture(d)
            manifest_path = Path(job['output'])/'manifest.json'; manifest = read(manifest_path)
            manifest['schema'] = 'latent-region-smc-v1'; write(manifest_path, manifest)
            save_records(Path(job['output']), initial, stages, summary)
            with self.assertRaisesRegex(ValueError, 'schema'): audit_population(context, job, Path(d)/'audit')

    def test_near_zero_gap_is_recorded_and_never_waived(self):
        with tempfile.TemporaryDirectory() as d:
            context, _, _, _, _ = saved_fixture(d)
            observer = NativeAudit(context, Ledger(), io.StringIO())
            touching = observer.geometry_at(pose(-.6)); overlapping = observer.geometry_at(pose(-.6+1e-10))
            self.assertTrue(touching['hard_valid']); self.assertTrue(touching['near_zero_atomic_gap'])
            self.assertFalse(overlapping['hard_valid']); self.assertTrue(overlapping['near_zero_atomic_gap'])
            with self.assertRaisesRegex(ValueError, 'geometry'): observer.native_at(pose(-.6+1e-10))

    def test_actual_protein_compiled_payload_reconstruction_without_sampling(self):
        from analyze_mobile_native_pocket import load_classifier
        root = Path(__file__).resolve().parents[1]
        definition = root/'runs/protected-guide-validation-20260923/common/reference-package/native-region/definition.json'
        compiled = root/'runs/native-entry-compiled-20260924/compiled.json'
        if not definition.is_file() or not compiled.is_file():
            self.skipTest('Pinned protein source fixture unavailable; no download/replay')
        model, _ = load_classifier(definition)
        self.assertEqual(compiled_from_model(model), read(compiled))
        self.assertEqual(len(compiled_from_model(model)['motifs']), 14)


if __name__ == '__main__':
    unittest.main()
