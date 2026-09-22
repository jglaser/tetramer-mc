#!/usr/bin/env python3
"""Audit frozen-posterior assembly, retaining mobile neighbors and every frame.

Native labels are the existing external-patch plus rigid-tetramer motif test.
They are observations, never a target constraint. This bounded preparation
test does not infer equilibrium, independent samples or physical kinetics.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_smc_normalizer_atlas import Density, read, sha, write
from analyze_involution_docking_campaign import ChartAudit
from mobile_posterior_metrics import summarize_graph_history


RECOVERY_MARKER = 'reference-recovery.json'
RECOVERY_SCHEMA = 'mobile-posterior-reference-recovery-v1'
RECOVERY_COORDINATES = 'results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json'
FOUR_BODY_SCHEMA = 'mobile-four-body-growth-campaign-v1'
GEOMETRY_FOUR_BODY_SCHEMA = 'mobile-geometry-four-body-growth-campaign-v1'


def campaign_body_count(manifest):
    """An explicit new experiment contract, never an inferred relaxation of N=3."""
    if manifest['schema'] in (FOUR_BODY_SCHEMA, GEOMETRY_FOUR_BODY_SCHEMA):
        assert manifest['body_count'] == 4 and type(manifest['body_count']) is int
        assert manifest['tracked_body_index'] == 3 and type(manifest['tracked_body_index']) is int
        assert manifest['scaffold_body_indices'] == [0, 1, 2]
        assert all(type(i) is int for i in manifest['scaffold_body_indices'])
        return 4
    assert manifest['schema'] in ('mobile-frozen-posterior-pilot-campaign-v1',
        'mobile-competing-atlas-benchmark-v1', 'mobile-reciprocal-atlas-benchmark-v1')
    assert manifest.get('body_count', 3) == 3
    return 3


def archived_residue_labels(reference, shape_path):
    """Run the unchanged frozen label helper without unrelated audit imports.

    These two function definitions are compiled directly from the hash-checked
    archived script, with the same globals used by that script. Its atom-order
    and coordinate-hash checks remain authoritative; no labels are synthesized.
    """
    script = reference/'scripts/analyze_depletion_mirror_benchmark.py'
    names = {'read_json', 'repaired_residue_labels'}
    definitions = [node for node in ast.parse(script.read_text(), filename=str(script)).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names and len(definitions) == 2
    namespace = dict(np=np, json=json, hashlib=hashlib, Path=Path, PROJECT=reference)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(script), 'exec'), namespace)
    return namespace['repaired_residue_labels'](shape_path)


def validate_reference(campaign, manifest, reference):
    """Accept the original reference or one explicitly bound data-only recovery."""
    campaign, reference = Path(campaign).resolve(), Path(reference).resolve()
    frozen = (campaign/'provenance/reference').resolve()
    assert Path(manifest['reference']).resolve() == frozen
    originals = {name[len('reference/'):]: digest for name, digest in manifest['input_sha256'].items()
                 if name.startswith('reference/')}
    assert originals, 'No frozen observer reference files'
    for name, digest in originals.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(frozen/relative) == digest, ('Original frozen reference changed', name)
    bindings = dict(source_campaign=str(campaign), source_campaign_manifest_sha256=sha(campaign/'manifest.json'),
        source_frozen_reference=str(frozen), original_reference_sha256=originals)
    if reference == frozen:
        return dict(mode='original-frozen-reference', reference=str(reference), **bindings)
    assert not reference.is_relative_to(campaign), 'Recovery must be a separate package outside the frozen campaign'
    marker_path = reference/RECOVERY_MARKER
    marker = read(marker_path)
    monomer = campaign/'provenance/monomer-shape.json'
    monomer_hash = manifest['input_sha256']['monomer-shape.json']
    assert sha(monomer) == monomer_hash, 'Frozen monomer shape changed'
    coordinate_hash = read(monomer)['source_coordinates_sha256']
    assert isinstance(coordinate_hash, str) and len(coordinate_hash) == 64
    addition = dict(path=RECOVERY_COORDINATES, sha256=coordinate_hash)
    expected_marker = dict(schema=RECOVERY_SCHEMA, **bindings, monomer_shape=str(monomer),
        monomer_shape_sha256=monomer_hash, added_coordinates=addition)
    assert marker == expected_marker, 'Reference recovery marker does not match frozen campaign/shape bindings'
    assert RECOVERY_COORDINATES not in originals, 'Coordinate data was not missing from the frozen reference'
    expected_files = set(originals) | {RECOVERY_MARKER, RECOVERY_COORDINATES}
    observed_files = set()
    for path in reference.rglob('*'):
        assert not path.is_symlink(), ('Recovery may not reference mutable external files', str(path))
        if path.is_file():
            observed_files.add(path.relative_to(reference).as_posix())
    assert observed_files == expected_files, ('Unexpected or missing recovery files', sorted(observed_files ^ expected_files))
    for name, digest in originals.items():
        assert sha(reference/name) == digest, ('Recovery changed a frozen reference file', name)
    coordinate_path = reference/RECOVERY_COORDINATES
    assert sha(coordinate_path) == coordinate_hash, 'Recovery coordinates differ from the monomer source hash'
    labels, residue = archived_residue_labels(reference, monomer)
    assert labels is not None and residue['available'], 'Frozen helper could not validate authoritative atom labels'
    assert Path(residue['source']).resolve() == coordinate_path
    assert residue['source_sha256'] == coordinate_hash and residue['shape_sha256'] == monomer_hash
    return dict(mode='separate-reference-data-recovery', reference=str(reference), **bindings,
        recovery_manifest_sha256=sha(marker_path), monomer_shape=str(monomer), monomer_shape_sha256=monomer_hash,
        added_coordinates=addition, label_validation=dict(available=True, atoms=len(labels),
            residues=len(residue['residues']), source=str(coordinate_path),
            maximum_coordinate_discrepancy_angstrom=residue['maximum_coordinate_discrepancy_angstrom'],
            atom_order_audit=residue['atom_order_audit']),
        scope='Separate observer recovery: every original reference script/data file is unchanged; only the missing hash-bound coordinate records are added. Original campaign, automatic observer failure and physical trajectories are preserved.')


def validate_campaign_jobs(manifest, status):
    """Require the declared experiment grid before reusing the physical audit."""
    schema = manifest['schema']
    if schema == 'mobile-frozen-posterior-pilot-campaign-v1':
        expected = {(start, mode, replicate) for start in ('dispersed', 'preassociated')
                    for mode in ('capture_only', 'c0', 'c09') for replicate in (0, 1)}
    elif schema == 'mobile-competing-atlas-benchmark-v1':
        assert manifest['atlas_variant'] in ('legacy', 'augmented')
        expected = {('competing', mode, replicate) for mode in ('c0', 'c09') for replicate in (0, 1)}
    elif schema == 'mobile-reciprocal-atlas-benchmark-v1':
        assert manifest['atlas_variant'] in ('legacy', 'reciprocal')
        expected = {('competing', mode, replicate) for mode in ('c0', 'c09') for replicate in (0, 1)}
    elif schema == FOUR_BODY_SCHEMA:
        assert manifest['atlas_variant'] in ('original', 'coverage')
        campaign_body_count(manifest)
        expected = {(start, 'c09', replicate) for start in ('triangle_free', 'retained_motif8') for replicate in (0, 1)}
    elif schema == GEOMETRY_FOUR_BODY_SCHEMA:
        assert manifest['atlas_variant'] == 'geometry'
        assert manifest['native_informed_proposal'] is False
        assert manifest['native_initial_scaffold'] is True
        campaign_body_count(manifest)
        expected = {(start, 'c09', replicate) for start in ('triangle_free', 'retained_motif8') for replicate in (0, 1)}
    else:
        raise AssertionError('Unrecognized mobile campaign schema')
    jobs = manifest['jobs']
    assert len(jobs) == len(status['jobs']) == len(expected) == len({j['id'] for j in jobs})
    assert {(j['start'], j['mode'], j['replicate']) for j in jobs} == expected
    assert {j['id'] for j in jobs} == {j['id'] for j in status['jobs']}
    assert status['complete'] and not status['running']
    assert all(j['status'] == 'complete' and j['exit_code'] == 0 for j in status['jobs'])
    return jobs


def affected_body_pairs(body_count, indices, common_isometry=False):
    """One-body changes touch all incident pairs; common rigid moves only cross pairs."""
    indices = list(indices)
    assert type(body_count) is int and body_count >= 2
    assert all(type(i) is int and 0 <= i < body_count for i in indices) and len(set(indices)) == len(indices)
    changed = set(indices)
    pairs = list(itertools.combinations(range(body_count), 2))
    return ([pair for pair in pairs if (pair[0] in changed) != (pair[1] in changed)] if common_isometry else
            [pair for pair in pairs if any(i in changed for i in pair)])


def validate_move_indices(move, body_count, selected):
    assert move['kind'] in ('local', 'global')
    i = move['moving_index']
    assert type(i) is int and 0 <= i < body_count and i not in selected
    assert move['update_in_sweep'] == len(selected)
    j = None
    if move['kind'] == 'global' and move['proposed_pose'] is not None:
        j = move['proposal']['anchor_index']
        assert type(j) is int and 0 <= j < body_count and i != j
    return i, j


def supported_registered_bonds(classification, keys, motifs, entry):
    """Deduplicate actual prescribed monomer/member/class support across motifs.

    Entry uses strict entry monomer bonds; active hysteretic body motifs use
    current relaxed stay bonds. This is support, not a separate bond-memory law.
    The archived motif record's scalar support count uses stay even for an
    entry body, so it is deliberately not reused as a strict entry count.
    """
    required = {}
    for i, j, motif in keys:
        assert motif in motifs
        required.setdefault((i, j), set()).update((c['member_i'], c['member_j'], c['directed_class'])
                                                 for c in motifs[motif]['member_contacts'])
    bonds = set()
    for bond in classification['native_entry_monomer_edges' if entry else 'native_stay_monomer_edges']:
        pair = tuple(bond['bodies']);members = tuple(bond['members'])
        if (*members, bond['class_label']) in required.get(pair, set()):
            bonds.add((*pair, *members, bond['class_label']))
    return sorted(bonds)


def four_body_graph_flags(edges, registered_keys, entry_keys, initial_scaffold_keys):
    pairs = {tuple(p) for p in edges}
    assert pairs <= set(itertools.combinations(range(4), 2))
    scaffold = {(0, 1), (0, 2), (1, 2)}
    reached = {0}
    while True:
        new = reached | {v for p in pairs if any(i in reached for i in p) for v in p}
        if new == reached:break
        reached = new
    active = {tuple(k) for k in registered_keys};entry = {tuple(k) for k in entry_keys}
    return dict(tracked_partners=sorted(i if j == 3 else j for i, j in pairs if 3 in (i, j)),
        tracked_attached=any(3 in p for p in pairs), scaffold_triangle=scaffold <= pairs,
        all_four_connected=len(reached) == 4, complete_K4=len(pairs) == 6,
        initial_scaffold_registry_retained=set(initial_scaffold_keys) <= active,
        initial_scaffold_registry_instantaneous_entry=set(initial_scaffold_keys) <= entry)


def summarize_four_body_growth(history, frames, graph_metrics, burn, total_cpu, post_cpu):
    """Descriptive all-update four-body outcomes, preserving initial censoring."""
    initial = {tuple(k) for k in history[0]['registered_keys'] if k[0] < 3 and k[1] < 3}
    assert {k[:2] for k in initial} == {(0, 1), (0, 2), (1, 2)}
    observations = []
    for row in history:
        values = dict(serial=row['serial'], sweep=row['sweep'], source=row['source'],
            moving_index=row.get('moving_index'), anchor_index=row.get('anchor_index'),
            tracked_registered_keys=[k for k in row['registered_keys'] if 3 in k[:2]],
            tracked_entry_keys=[k for k in row['instantaneous_entry_keys'] if 3 in k[:2]])
        for name, edges in (('native', row['native_edges']), ('nonspecific', row['nonspecific_edges']),
                            ('instantaneous_entry', [k[:2] for k in row['instantaneous_entry_keys']])):
            values[name] = four_body_graph_flags(edges, row['registered_keys'], row['instantaneous_entry_keys'], initial)
        observations.append(values)
    def event(row):
        return {k:row[k] for k in ('serial', 'sweep', 'source', 'moving_index', 'anchor_index')}
    def first(kind, field):
        row = next((r for r in observations if r[kind][field]), None)
        return None if row is None else dict(event(row), initial_state=row['serial'] == -1)
    outcomes = {}
    for kind in ('native', 'nonspecific', 'instantaneous_entry'):
        fields = ('tracked_attached', 'scaffold_triangle', 'all_four_connected', 'complete_K4',
                  'initial_scaffold_registry_retained', 'initial_scaffold_registry_instantaneous_entry')
        transitions = []
        for previous, row in zip(observations, observations[1:]):
            for field in fields:
                if previous[kind][field] != row[kind][field]:
                    transitions.append(dict(event(row), field=field, value=row[kind][field]))
        endpoints = {r['sweep']:r for r in observations}
        kept = [r for sweep,r in sorted(endpoints.items()) if sweep > burn]
        outcomes[kind] = dict(first_observed={field:first(kind, field) for field in fields},
            transitions=transitions,postburn_endpoint_fractions={field:sum(r[kind][field] for r in kept)/len(kept) for field in fields},
            scaffold_first_loss=next((t for t in transitions if t['field']=='scaffold_triangle' and not t['value']),None))
        if kind in ('native', 'nonspecific'):
            g = graph_metrics['graphs'][kind]
            tracked_events = [dict(e, formed_edges=[p for p in e['formed_edges'] if 3 in p],
                detached_edges=[p for p in e['detached_edges'] if 3 in p]) for e in g['events']
                if any(3 in p for p in e['formed_edges']+e['detached_edges'])]
            rates = {}
            for label, events, cpu, attempts in (('full',tracked_events,total_cpu,len(history)-1),
                ('postburn',[e for e in tracked_events if e['sweep']>burn],post_cpu,sum(r['sweep']>burn for r in history[1:]))):
                counts={key:sum(len(e[key])for e in events)for key in ('formed_edges','detached_edges')}
                rates[label]=dict(counts=counts,attempted_updates=attempts,sampling_CPU_seconds=cpu,
                    per_attempted_update={k:v/attempts for k,v in counts.items()},per_sampling_CPU_second={k:v/cpu for k,v in counts.items()})
            outcomes[kind].update(tracked_events=tracked_events,tracked_event_rates=rates,
                tracked_body=copy.deepcopy(g['bodies'][3]),
                tracked_partner_exchanges=[e for e in g['partner_exchanges']if e['body']==3],
                tracked_same_attempt_replacements=[e for e in g['same_attempt_partner_replacements']if e['body']==3],
                tracked_pending_partner_losses=[e for e in g['pending_partner_losses_at_end']if e['body']==3])
    return dict(schema='mobile-four-body-growth-observables-v1',body_count=4,tracked_body_index=3,
        scaffold_body_indices=[0,1,2],pair_labels=list(itertools.combinations(range(4),2)),
        initial_scaffold_keys=sorted(initial),observations=observations,outcomes=outcomes,
        supported_monomer_bonds_by_saved_sweep=[dict(sweep=r['sweep'],**r['registered_monomer_support'])for r in frames],
        scope='All attempted updates including rejections retained; initial native attachment is left-censored occupancy, not a formation. Native connected4, ABC triangle, and six-edge K4 are separate geometric descriptors. Entry is stateless; retained registry is hysteretic. Monomer support counts actual prescribed member/class bonds, not pair count or energy. No equilibrium, template-free assembly, or physical rate inference.')


def update_registry(active, entry, stay, affected):
    affected = set(affected)
    return {key for key in active if key[:2] not in affected or key in stay} | entry


def pair_changes(before, after):
    old, new = {key[:2] for key in before}, {key[:2] for key in after}
    return dict(formed_body_pairs=sorted(new-old), broken_body_pairs=sorted(old-new),
                changed_registry_body_pairs=sorted(
                    {key[:2] for key in before-after} & {key[:2] for key in after-before} & old & new))


def full_capture_log_density(log_g, pose, lengths, eta):
    assert len(lengths) == len(pose['position']) == 3
    assert all(math.isfinite(x) and x > 0 for x in lengths) and 0 <= eta <= 1
    assert math.isfinite(log_g) or log_g == -math.inf
    inside = all(-length/2 <= value < length/2 for value, length in zip(pose['position'], lengths))
    uniform = math.log(eta)-sum(math.log(x) for x in lengths) if inside and eta else -math.inf
    gaussian = math.log1p(-eta)+log_g if eta < 1 else -math.inf
    return float(np.logaddexp(uniform, gaussian))


def density_record(serial, kernel, move, state, audit):
    """Snapshot before later accepted moves/common shifts mutate the state."""
    info = move['proposal']
    i, j = move['moving_index'], info['anchor_index']
    old, new, anchor = copy.deepcopy((state[i], move['proposed_pose'], state[j]))
    return dict(serial=serial, kernel=kernel, accepted=move['accepted'],
        proposal=copy.deepcopy(info), old=old, new=new, anchor=anchor,
        old_relative=audit.relative(old, anchor), new_relative=audit.relative(new, anchor))


def proposal_kernel(move, cfg):
    """Reject inconsistent labels instead of auditing them as another kernel."""
    kind, info = move['kind'], move.get('proposal') or {}
    kernel = info.get('kernel', 'full-mixture-capture' if kind == 'global' else kind)
    if kind == 'local':
        assert kernel == 'local' and info['branch'] == 'local'
    elif kind == 'global':
        assert kernel in ('full-mixture-capture', 'frozen-posterior')
        if kernel == 'frozen-posterior':
            assert cfg.get('frozen_posterior') is not None
            assert info['branch'] in ('uniform', 'involution')
            assert info['correlation'] == cfg['frozen_posterior']['correlation']
            if move['proposed_pose'] is not None and info['branch'] == 'involution':
                assert info['source_law'] == 'posterior' and info.get('trace') is not None
        else:
            assert info['branch'] in ('uniform', 'learned')
    else:
        assert kind in ('gca', 'center_shift') and kernel == kind
    return kernel


def audit_single_body_gate(move, cfg, audit):
    """Check recorded arithmetic/schema, without claiming bath/RNG regeneration."""
    assert move['kind'] in ('local', 'global')
    assert type(move['hard_valid']) is bool and type(move['accepted']) is bool
    candidate, gate = move['proposed_pose'], move['gate']
    if candidate is not None:
        correction = move['proposal']['log_reverse_forward']
        assert isinstance(correction, (int, float)) and math.isfinite(correction)
        if move['kind'] == 'local':
            audit.close('local_correction', 0., correction, atol=1e-12)
    else:
        correction = 0.
    if gate is None:
        assert not move['hard_valid'] and not move['accepted'] and move['log_acceptance'] is None
        return 0
    assert move['hard_valid'] and candidate is not None
    for key in ('raw_points', 'retained_points', 'gained', 'lost', 'retained_cells', 'created_cells'):
        assert type(gate[key]) is int and gate[key] >= 0
    assert math.isfinite(gate['envelope_volume']) and gate['envelope_volume'] >= 0
    assert math.isfinite(gate['log_weight'])
    assert gate['retained_points'] == gate['gained']+gate['lost'] <= gate['raw_points']
    z, ratio = cfg['reservoir_density'], cfg['poisson_lambda_ratio']
    assert math.isfinite(z) and z >= 0 and math.isfinite(ratio) and ratio > 0
    lam = ratio*z if z > 0 else 1.
    coefficient = math.log1p(z/lam)
    if z == 0 or gate['envelope_volume'] == 0:
        assert gate['raw_points'] == 0
    audit.close('poisson_count_factor', coefficient*(gate['gained']-gate['lost']), gate['log_weight'], atol=1e-12)
    alpha = min(0., correction+gate['log_weight'])
    audit.close('acceptance_arithmetic', alpha, move['log_acceptance'], atol=1e-12)
    if alpha == 0.:
        assert move['accepted'], 'A unit-acceptance proposal cannot be rejected'
    return 1


def update_nonspecific_graph(poses, affected, active, atomic, rd, tolerance=2e-8):
    """Recompute changed pairs in open space; unchanged pairs retain their state."""
    result = set(active)
    for i, j in affected:
        a, b = poses[i], poses[j]
        displacement = np.asarray(b['position'])-a['position']
        if np.linalg.norm(displacement) > 2*atomic.bound+2*rd+tolerance:
            result.discard((i, j))
            continue
        rotations = Rotation.from_quat(np.asarray([a['orientation'], b['orientation']])[:, [1, 2, 3, 0]]).as_matrix()
        gap = atomic.exact_gap(rotations[0].T@displacement, rotations[0].T@rotations[1])
        assert gap >= -tolerance, ('accepted state has a hard overlap', i, j, gap)
        if gap <= 2*rd+tolerance:
            result.add((i, j))
        else:
            result.discard((i, j))
    return result


def environment_episodes(rows, kind, body_count):
    """Initialize from the actual first state, including a preassociated start."""
    episodes, returns = [], 0
    for body in range(body_count):
        def neighbors(row):
            return tuple(sorted(b if a == body else a for a, b in row[kind]['edges'] if body in (a, b)))
        previous, begin, seen = neighbors(rows[0]), rows[0]['sweep'], set()
        for earlier, row in zip(rows, rows[1:]):
            value = neighbors(row)
            if value != previous:
                episodes.append(dict(body=body, neighbors=list(previous), first_observed_sweep=begin,
                    last_observed_sweep=earlier['sweep'], next_observation_sweep=row['sweep'], right_censored=False))
                returns += int(value in seen)
                seen.add(previous)
                previous, begin = value, row['sweep']
        episodes.append(dict(body=body, neighbors=list(previous), first_observed_sweep=begin,
            last_observed_sweep=rows[-1]['sweep'], right_censored=True))
    return dict(episodes=episodes, observed_returns=returns,
                scope='Changes at saved sweep endpoints; returns to previously observed neighbor sets are not independent samples.')


def audit_densities(model, records, cfg, audit):
    if not records:
        return dict(global_density_records=0, full_mixture_checks=0, posterior_checks=0,
                    posterior_uniform_checks=0, map_checks=0)
    density = Density(model)
    assert np.array_equal(density.base_indices, audit.base_indices)
    assert np.array_equal(density.inverted, audit.inverted)
    old_g, _, old_logs = density.evaluate([r['old_relative'] for r in records])
    new_g, _, new_logs = density.evaluate([r['new_relative'] for r in records])
    log_weights = np.log(density.weights)
    posterior, captures, uniforms = [], 0, 0
    for k, record in enumerate(records):
        info = record['proposal']
        if record['kernel'] == 'frozen-posterior':
            if info.get('trace') is None:
                assert info['branch'] == 'uniform'
                uniforms += 1
                audit.close('posterior_uniform_correction', 0., info['log_reverse_forward'])
                continue
            posterior.append(k)
            a, b = info['trace']['source'], info['trace']['target']
            audit.check_branch_metadata(info, a, b)
            audit.close('full_old_gaussian', old_g[k], info['full_old_gaussian_log_density'])
            audit.close('full_new_gaussian', new_g[k], info['full_new_gaussian_log_density'])
            audit.close('source_posterior', old_logs[k, a]-old_g[k], info['source_log_probability'])
            audit.close('inverse_source_posterior', new_logs[k, b]-new_g[k], info['inverse_source_log_probability'])
            label = new_logs[k, b]-new_g[k]+log_weights[a]-(old_logs[k, a]-old_g[k])-log_weights[b]
            audit.close('posterior_label_ratio', label, info['label_log_reverse_forward'])
            audit.close('expanded_ratio', info['step']['log_correction']+label, info['expanded_log_reverse_forward'])
            audit.close('collapsed_ratio', old_g[k]-new_g[k], info['log_reverse_forward'])
            audit.close('expanded_collapsed', info['expanded_log_reverse_forward'], info['log_reverse_forward'])
        else:
            assert record['kernel'] == 'full-mixture-capture'
            captures += 1
            if density.inverted.any() and info.get('branch') == 'learned':
                base, inverted = info.get('component_index'), info.get('component_inverted')
                assert type(base) is int and type(inverted) is bool, 'Missing reciprocal capture branch metadata'
                matches = np.flatnonzero((density.base_indices == base) & (density.inverted == inverted))
                assert len(matches) == 1, 'Invalid reciprocal capture base/branch combination'
            lengths = cfg['uniform_proposal_cube_lengths']
            before = full_capture_log_density(old_g[k], record['old'], lengths, cfg['learned_uniform_weight'])
            after = full_capture_log_density(new_g[k], record['new'], lengths, cfg['learned_uniform_weight'])
            audit.close('capture_old_full_mixture', before, info['old_log_density'])
            audit.close('capture_new_full_mixture', after, info['new_log_density'])
            audit.close('capture_full_mixture_ratio', before-after, info['log_reverse_forward'])
    selected = set(posterior[i] for i in np.linspace(0, len(posterior)-1, min(128, len(posterior)), dtype=int)) if posterior else set()
    accepted = [k for k in posterior if records[k]['accepted']]
    if accepted:
        selected.update(accepted[i] for i in np.linspace(0, len(accepted)-1, min(128, len(accepted)), dtype=int))
    for k in sorted(selected):
        record = records[k]
        info = copy.deepcopy(record['proposal'])
        info.pop('full_old_gaussian_log_density', None)
        info.pop('full_new_gaussian_log_density', None)
        pose = audit.involution(record['old_relative'], info, cfg['frozen_posterior']['correlation'])
        audit.compare_pose(audit.absolute(pose, record['anchor']), record['new'], 'posterior_global_anchor_map')
        correlation = cfg['frozen_posterior']['correlation']
        sine = math.sqrt((1-correlation)*(1+correlation))
        target = np.asarray(info['step']['target_latent'])
        inverse = info['step']['inverse_trace']
        inverse_noise = np.asarray(inverse['noise'])
        restored, _ = audit.decode(inverse['target'], correlation*target+sine*inverse_noise)
        audit.compare_pose(restored, record['old_relative'], 'posterior_inverse_map')
        audit.close('posterior_inverse_noise', sine*target-correlation*inverse_noise, info['trace']['noise'])
    return dict(global_density_records=len(records), full_mixture_checks=captures,
                posterior_checks=len(posterior), posterior_uniform_checks=uniforms, map_checks=len(selected))


def one(task):
    campaign, output, job, reference = task
    campaign, output, reference = map(Path, (campaign, output, reference))
    started = time.monotonic()
    manifest = read(campaign/'manifest.json')
    expected_bodies = campaign_body_count(manifest)
    reference = reference.resolve()
    reference_validation = validate_reference(campaign, manifest, reference)
    # Importing verified observers must not add bytecode files to either the
    # original frozen reference or the exact-file-list recovery package.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference/'scripts'))
    from audit_tetramer_assembly import AtomicAssembly, pose_arrays, rotations
    from tetramer_order import TetramerOrder, graph_summary
    from analyze_precursor_exchange import prepare_templates
    for name in ('audit_tetramer_assembly', 'tetramer_order', 'analyze_precursor_exchange'):
        assert Path(sys.modules[name].__file__).resolve() == (reference/'scripts'/f'{name}.py').resolve()
    directory = Path(job['directory'])
    cfg, summary, provenance = [read(directory/name) for name in ('config.json', 'summary.json', 'manifest.json')]
    assert cfg.get('assembly_bias') is None, 'Physical-only observer does not audit assembly-biased trajectories'
    assert summary['complete'] and summary['completed_sweeps'] == manifest['sweeps']
    assert provenance['executable_sha256'] == manifest['binary_sha256']
    assert provenance['config_sha256'] == job['config_sha256'] == sha(job['config'])
    assert provenance['model_sha256'] == manifest['model_sha256'] == sha(directory/'provenance/frozen-relative-model.json')
    assert provenance['shape_sha256'] == manifest['shape_sha256'] == sha(directory/'provenance/shape.json')
    source = read(directory/'provenance/source-bundle.json')
    assert sha(directory/'provenance/source-bundle.json') == provenance['source_bundle_sha256']
    assert set(source['files']) == set(manifest['rust_sources'])
    for name, value in source['files'].items():
        import hashlib
        assert value['sha256'] == hashlib.sha256(value['text'].encode()).hexdigest() == manifest['rust_sources'][name]
    assert cfg['initial_poses'] == read(job['config'])['initial_poses']
    assert not cfg['fixed_body_indices'] and not cfg['seed_labels']
    assert all(cfg.get(name) is None for name in ('auxiliary_transport', 'reversible_jump', 'contact_memory', 'conditional_closure', 'atlas_transport', 'atlas_mask'))
    assert cfg['gca_probability'] == cfg['center_shift_probability'] == 1.
    assert cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and cfg['poisson_lambda_ratio'] == 64.
    if expected_bodies == 4:
        assert cfg['boundary'] == dict(kind='spherical', radius=223.32617672378387)
        assert cfg['local_translation_std_A'] == .2 and cfg['local_small_angle_std_degrees'] == 1.
        assert cfg['global_probability'] == .5 and cfg['learned_uniform_weight'] == .1
        assert cfg['frozen_posterior'] == dict(probability=.5, correlation=.9)
        assert job['mode'] == 'c09' and job['start'] in ('triangle_free', 'retained_motif8')
    if manifest['schema'] == GEOMETRY_FOUR_BODY_SCHEMA:
        assert cfg['metadata']['native_informed_proposal'] is False
        assert cfg['metadata']['native_initial_scaffold'] is True
        assert cfg['metadata']['atlas_variant'] == 'geometry'
    model, shape = read(directory/'provenance/frozen-relative-model.json'), read(directory/'provenance/shape.json')
    audit = ChartAudit(model)
    atoms = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    bound = float(np.max(np.linalg.norm(atoms, axis=1)+radii))
    radius = cfg['boundary']['radius']
    audit.close('uniform_proposal_cube', [2*(radius+bound)]*3, cfg['uniform_proposal_cube_lengths'], atol=1e-12)
    analysis_cfg = copy.deepcopy(cfg)
    analysis_cfg.update(shape=str(directory/'provenance/shape.json'), rigid_members=shape['rigid_members'], box_lengths=[8*(radius+bound)]*3)
    templates = prepare_templates(cfg['monomer_shape'], reference/'results/c1c3-scaffold/motifs.json', reference/'results/native-neighbor-classes/classification.json')
    atomic, order = AtomicAssembly(analysis_cfg, directory), TetramerOrder(analysis_cfg, directory, templates=templates)
    frames = [json.loads(line) for line in (directory/'trajectory.jsonl').open()]
    assert [f['sweep'] for f in frames] == list(range(manifest['sweeps']+1))
    state, n = copy.deepcopy(frames[0]['poses']), len(frames[0]['poses'])
    assert n == expected_bodies
    pairs = list(itertools.combinations(range(n), 2))
    motif_lookup = {m['id']:m for m in order.body_motifs}
    def classify(poses, subset=None, details=False):
        result = order.classify({'poses': poses}, pair_filter=subset)
        keys = ({(*r['bodies'], r['motif_id']) for r in result['registered_tetramer_motifs'] if r['entry']},
                {(*r['bodies'], r['motif_id']) for r in result['registered_tetramer_motifs']})
        return (*keys, result) if details else keys
    active, _ = classify(state)
    instantaneous = set(active)
    initial_native = sorted(active)
    changes, rows, density_records, graph_history = [], [], [], []
    near = set()
    counters, counts, accepted = defaultdict(Counter), Counter(), Counter()
    native_candidates = []
    gate_checks = 0
    def observe(frame):
        nonlocal state, near
        assert len(frame['poses']) == n
        for pose, expected in zip(state, frame['poses']):
            audit.compare_pose(pose, expected, 'frame_replay')
        entry, stay, classification = classify(state, details=True)
        assert active == (active & stay) | entry
        if n == 4:assert instantaneous == entry, 'Incremental instantaneous entry graph differs from frame classification'
        p, q = pose_arrays(frame)
        world = np.einsum('bij,aj->bai', rotations(q), atoms)+p[:, None, :]
        wall = float(np.min(radius-np.linalg.norm(world, axis=2)-radii[None, :]))
        assert wall >= -2e-8
        result = atomic.frame(p, q)
        assert result['hard_valid']
        observed_near, native = {tuple(e) for e in result['depletion_edges']}, {key[:2] for key in active}
        if frame['sweep'] == 0:
            near = observed_near
        else:
            assert near == observed_near, 'Incremental all-update contact graph differs from independent frame audit'
        assert native <= near
        if frame['sweep'] == 0 and job['start'] == 'dispersed':
            assert not near and not native
        rows.append(dict(sweep=frame['sweep'], sampler_cpu_seconds=frame['sampler_cpu_seconds'],
            native=graph_summary(n, native), nonspecific=graph_summary(n, near), registered_keys=sorted(active),
            native_instantaneous_entry=graph_summary(n, {key[:2] for key in entry}),
            minimum_atomic_wall_clearance_A=wall, minimum_interbody_gap_A=result['minimum_interbody_gap_A'], hard_valid=True))
        if n == 4:
            entry_bonds = supported_registered_bonds(classification, entry, motif_lookup, True)
            stay_bonds = supported_registered_bonds(classification, active, motif_lookup, False)
            rows[-1].update(instantaneous_entry_keys=sorted(entry),
                registered_monomer_support=dict(entry_bonds=entry_bonds,retained_bonds=stay_bonds,
                    entry_count=len(entry_bonds),retained_count=len(stay_bonds),
                    tracked_entry_count=sum(3 in b[:2] for b in entry_bonds),
                    tracked_retained_count=sum(3 in b[:2] for b in stay_bonds)))
        state = copy.deepcopy(frame['poses'])
    observe(frames[0])
    graph_history.append(dict(serial=-1, sweep=0, source='initial',
        native_edges=sorted({key[:2] for key in active}), nonspecific_edges=sorted(near)))
    if n == 4:
        graph_history[-1].update(registered_keys=sorted(active), instantaneous_entry_keys=sorted(instantaneous),
            moving_index=None,anchor_index=None)
        assert {(0,1),(0,2),(1,2)} <= {k[:2]for k in active}
        if job['start'] == 'triangle_free':
            assert not any(3 in p for p in near), 'Fourth body must initially be exclusion-free'
        else:
            assert (2,3,8) in active, 'Retained start must include the prescribed C-to-D motif8'
    for pose, expected in zip(state, cfg['initial_poses']):
        audit.compare_pose(pose, expected, 'initial')
    current, selected, collective = 0, set(), []
    records = 0
    with (directory/'moves.jsonl').open() as stream:
        for serial, line in enumerate(stream):
            move = json.loads(line)
            records += 1
            sweep, kind = move['sweep'], move['kind']
            if sweep != current:
                if current:
                    assert selected == set(range(n)) and collective == ['gca', 'center_shift']
                    observe(frames[current])
                assert sweep == current+1
                current, selected, collective = sweep, set(), []
            previous, affected = set(active), []
            counts[kind] += 1
            info = move.get('proposal') or {}
            kernel = proposal_kernel(move, cfg)
            source = kernel+':'+info.get('branch', kind)
            counter = counters[source]
            counter['attempted'] += 1
            if kind in ('local', 'global'):
                i, j = validate_move_indices(move, n, selected)
                assert not collective
                selected.add(i)
                audit.compare_pose(state[i], move['old_pose'], 'old_pose')
                candidate = move['proposed_pose']
                if kind == 'global' and candidate is not None:
                    if kernel == 'frozen-posterior':
                        assert info['moving_index'] == i
                        assert info['correlation'] == cfg['frozen_posterior']['correlation']
                    density_records.append(density_record(serial, kernel, move, state, audit))
                counter['hard_valid'] += int(move['hard_valid'])
                counter['accepted'] += int(move['accepted'])
                correction = info.get('log_reverse_forward', 0.)
                gate_checks += audit_single_body_gate(move, cfg, audit)
                if move['hard_valid'] and (kind == 'global' or n == 4):
                    temporary = list(state)
                    temporary[i] = candidate
                    entry, _, candidate_classification = classify(temporary, affected_body_pairs(n,[i]),details=True)
                    novel = entry-active
                    counter['new_native_candidate'] += int(bool(novel))
                    counter['accepted_new_native_candidate'] += int(bool(novel) and move['accepted'])
                    if novel:
                        native_candidates.append(dict(serial=serial, sweep=sweep, source=source, moving=i,
                            accepted=move['accepted'], new_registered_keys=sorted(novel),
                            proposal_correction=correction, gate_log_weight=move['gate']['log_weight'], log_acceptance=move['log_acceptance']))
                        if n == 4:
                            support = supported_registered_bonds(candidate_classification, novel, motif_lookup, True)
                            native_candidates[-1].update(anchor_index=j,tracked_body_in_new_edges=any(3 in k[:2]for k in novel),
                                new_native_supported_monomer_bonds=support,
                                proposed_new_supported_monomer_bond_count=len(support),
                                proposal_trace=copy.deepcopy(info.get('trace')),
                                proposal_source_component_index=info.get('source_component_index'),
                                proposal_target_component_index=info.get('target_component_index'),
                                proposal_source_inverted=info.get('source_inverted'),
                                proposal_target_inverted=info.get('target_inverted'),
                                capture_component_index=info.get('component_index'),capture_component_inverted=info.get('component_inverted'))
                if move['accepted']:
                    assert move['hard_valid'] and candidate is not None
                    assert move['retained_pose'] == candidate
                    accepted[kind] += 1
                    state[i] = copy.deepcopy(candidate)
                    affected = affected_body_pairs(n,[i])
                else:
                    audit.compare_pose(state[i], move['retained_pose'], 'self_loop')
            elif kind == 'gca':
                assert selected == set(range(n)) and not collective and move['accepted']
                collective.append(kind)
                accepted[kind] += 1
                indices = move['result']['flipped_indices']
                affected = affected_body_pairs(n,indices,common_isometry=True)
                flipped = set(indices)
                axis = np.asarray(move['axis'])
                axis /= np.linalg.norm(axis)
                matrix = 2*np.outer(axis, axis)-np.eye(3)
                for i in flipped:
                    p, r = audit.arrays(state[i])
                    state[i] = audit.pose(matrix@p, matrix@r)
                counter['transformed_bodies'] += len(flipped)
                counter['partial_components'] += int(0 < len(flipped) < n)
            elif kind == 'center_shift':
                assert collective == ['gca'] and move['accepted']
                collective.append(kind)
                accepted[kind] += 1
                for pose in state:
                    pose['position'] = (np.asarray(pose['position'])+move['result']['displacement']).tolist()
            else:
                raise AssertionError('Unexpected adaptive/physical kernel '+kind)
            if affected:
                entry, stay = classify(state, affected)
                active = update_registry(active, entry, stay, affected)
                if n == 4:
                    instantaneous = {key for key in instantaneous if key[:2] not in affected} | entry
                near = update_nonspecific_graph(state, affected, near, atomic, cfg['depletant_radius'])
            assert {key[:2] for key in active} <= near
            if previous != active:
                changes.append(dict(serial=serial, sweep=sweep, source=source, kind=kind,
                    formed=sorted(active-previous), broken=sorted(previous-active), **pair_changes(previous, active)))
            graph_history.append(dict(serial=serial, sweep=sweep, source=source, kind=kind,
                accepted=move['accepted'], native_edges=sorted({key[:2] for key in active}),
                nonspecific_edges=sorted(near)))
            if n == 4:
                graph_history[-1].update(registered_keys=sorted(active),instantaneous_entry_keys=sorted(instantaneous),
                    moving_index=move.get('moving_index'),anchor_index=info.get('anchor_index'),affected_pairs=affected)
    assert current == manifest['sweeps'] and selected == set(range(n)) and collective == ['gca', 'center_shift']
    assert records == (n+2)*manifest['sweeps']
    observe(frames[current])
    checkpoint = read(directory/'checkpoint.json')
    assert len(checkpoint['poses']) == n
    for pose, expected in zip(state, checkpoint['poses']):
        audit.compare_pose(pose, expected, 'checkpoint')
    for kind in ('local', 'global'):
        assert counts[kind] == summary['counts'][kind]['attempted']
        assert accepted[kind] == summary['counts'][kind]['accepted']
    for kind in ('gca', 'center_shift'):
        assert counts[kind] == accepted[kind] == manifest['sweeps'] == summary['counts'][kind]['completed']
    density_checks = audit_densities(model, density_records, cfg, audit)
    burn = manifest['burn_sweeps']
    cpu = frames[-1]['sampler_cpu_seconds']-frames[burn]['sampler_cpu_seconds']
    assert cpu > 0 and len(rows) == len(frames)
    graph_metrics = summarize_graph_history(graph_history, n, burn, manifest['sweeps'], cpu)
    event_counts = {}
    for scope, events, denominator in (('full', changes, summary['sampler_cpu_seconds']),
                                      ('postburn', [e for e in changes if e['sweep'] > burn], cpu)):
        totals = {key: sum(len(e[key]) for e in events) for key in ('formed_body_pairs', 'broken_body_pairs', 'changed_registry_body_pairs')}
        by_kernel = {name: {key: sum(len(e[key]) for e in events if e['source'] == name) for key in totals} for name in counters}
        event_counts[scope] = dict(counts=totals, by_kernel=by_kernel, sampling_CPU_seconds=denominator,
            events_per_sampling_CPU_second={key: count/denominator for key, count in totals.items()})
    kept = rows[burn+1:]
    occupancy = {kind: {str(size): sum(row[kind]['largest_component_size'] == size for row in kept)/len(kept)
                        for size in range(1, n+1)} for kind in ('native', 'nonspecific')}
    result = dict(id=job['id'], mode=job['mode'], start=job['start'], replicate=job['replicate'], seed=job['seed'], passed=True,
        initial_registered_keys=initial_native, rows=rows, changes=changes, native_candidates=native_candidates,
        event_counts=event_counts, branch_counts={key: dict(value) for key, value in counters.items()},
        graph_history=graph_history, graph_metrics=graph_metrics,
        postburn_largest_component_occupancy=occupancy,
        observed_environments={kind: environment_episodes(rows, kind, n) for kind in ('native', 'nonspecific')},
        protocol=order.protocol, audit=dict(all_move_records=records, atomic_frames=len(rows), gate_arithmetic_checks=gate_checks,
            **density_checks, checks=dict(audit.checks), maximum_errors=dict(audit.max_errors)),
        observer_reference=reference_validation, analyzer_sha256=sha(__file__),
        sampler_cpu_seconds=summary['sampler_cpu_seconds'], postburn_sampler_cpu_seconds=cpu,
        observer_wall_seconds=time.monotonic()-started,
        source_sha256={name: sha(directory/name) for name in ('manifest.json', 'config.json', 'summary.json', 'checkpoint.json', 'moves.jsonl', 'trajectory.jsonl')},
        scope='Every saved sweep, attempted-update graph state and repeat retained; all-mobile native-informed preparation pilot. Native graph occupancy/ESS use path-dependent entry/retention hysteresis, not instantaneous q<=1 thermodynamic regions or AB shoulder weights; instantaneous entry graphs are separately recorded at saved frames. No equilibrium, template-free assembly, speedup or physical kinetics claim. Gate arithmetic is checked; random bath realizations and unrecorded accept uniforms are not independently regenerated.')
    if n == 4:
        result.update(body_count=4,atlas_variant=manifest['atlas_variant'],
            four_body_growth=summarize_four_body_growth(graph_history,rows,graph_metrics,burn,summary['sampler_cpu_seconds'],cpu))
    if manifest['schema'] == GEOMETRY_FOUR_BODY_SCHEMA:
        result.update(native_informed_proposal=False,native_initial_scaffold=True)
        result['scope'] = ('Geometry-only proposal; supplied native scaffold and retained-start registry remain initial-condition information. '
                           + result['scope'])
    target = output/'runs'/job['id']
    target.mkdir(parents=True)
    write(target/'analysis.json', result)
    print(job['id'], event_counts['full']['counts'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Frozen reference or a separately hash-bound data-recovery package')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    assert 1 <= args.workers <= 32
    campaign, output = args.campaign.resolve(), args.out.resolve()
    assert not output.exists()
    manifest, status = read(campaign/'manifest.json'), read(campaign/'status.json')
    reference = (args.reference or Path(manifest['reference'])).resolve()
    reference_validation = validate_reference(campaign, manifest, reference)
    jobs = validate_campaign_jobs(manifest, status)
    for name, digest in manifest['input_sha256'].items():
        assert sha(campaign/'provenance'/name) == digest, name
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(one, [(campaign, output, job, reference) for job in jobs]))
    write(output/'analysis.json', dict(complete=True, runs=results, manifest_sha256=sha(campaign/'manifest.json'),
        terminal_status_sha256=sha(campaign/'status.json'), analyzer_sha256=sha(__file__),
        observer_reference=reference_validation, original_frozen_observer_sha256=manifest['observer_sha256'],
        observer_execution='separate-recovery-assessment' if reference_validation['mode'] == 'separate-reference-data-recovery' else 'frozen-reference-assessment',
        scope=manifest['scope']))


if __name__ == '__main__':
    main()
