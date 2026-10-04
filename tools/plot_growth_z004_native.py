#!/usr/bin/env python3
"""Present two completed, authenticated z=.04 native-growth observations.

Reads controller metadata and all 1001 saved native rows per arm. The existing
driver hashes frozen input bytes; no classifier, trajectory decoding or moves.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import run_native_class_physical_campaign as driver

ARMS = ('coverage', 'discovery')
LABELS = {'coverage': 'Native-informed coverage', 'discovery': 'Native-blind FFT 512'}
MODELS = {'coverage': 'feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e',
          'discovery': 'c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08'}
SHARED = dict(input_config_sha256='977dd78a90639ae16b5f7d81d5c66f0810863f092d39113305c95f085f945a82',
    shape_sha256='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9',
    sampler_executable_sha256='94c987694af7cc347fdf076ff121425464329b35ae76c80b9e0415592410010a',
    source_bundle_sha256='76950d1bc656ec99fba953c444e3b60f4c6276041a28242f63f3a6f13e0a8e30', initial_poses_equal=True)
PHYSICAL = dict(bodies=264, seed=20260926, seed_bodies=8, depletant_radius_A=1.4,
                activity_A_minus3=.04, concentration_uM=500)
SCOPE = ('Two historical trajectories with shared RNG seed and identical initial poses, '
         'not independent replicates or equilibrium samples. All saved frames, including '
         'unchanged states, remain present. Changes are instantaneous native-entry '
         'observations; excursions between saved frames are unresolved. Sampler CPU and '
         'sweeps are algorithmic coordinates, not physical time. No physical rates, '
         'IID uncertainty, p-values, or ESS are estimated.')
require, sha, write = driver.require, driver.sha, driver.write


class Metadata:
    """Bounded JSON metadata reader, separate from driver byte authentication."""
    def __init__(self, root):
        self.root = Path(root).resolve(); self.refs = {}; self.bytes = 0

    def read(self, relative):
        path = (self.root/relative).resolve()
        require(path.is_relative_to(self.root), 'Escaping metadata path')
        size = path.stat().st_size; assessment = path.name == 'assessment.json'
        if not assessment: self.bytes += size
        require(size <= (256 if assessment else 16)*1024**2 and self.bytes <= 64*1024**2,
                'Metadata byte cap exceeded')
        self.refs[str(path)] = sha(path)
        return driver.read(path)


def authenticate(root, manifest_sha256):
    """Reuse the frozen driver, including its complete input-byte validation."""
    root = Path(root).resolve(); bound = Metadata(root)
    require(set(manifest_sha256) == set(ARMS), 'Two explicit expected manifest hashes required')
    for digest in manifest_sha256.values(): driver._digest(digest)
    plan = bound.read('execution-plan.json'); claim = bound.read('execution/claim.json')
    summary = bound.read('execution/summary.json'); status = bound.read('execution/status.json')
    digest = bound.refs[str(root/'execution-plan.json')]
    require(plan['schema'] == driver.SCHEMA and plan['root'] == str(root)
            and plan['maximum_workers'] == plan['threads'] == 1 and len(plan['jobs']) == 2,
            'Wrong two-arm execution allocation')
    keys = ('schema', 'complete', 'passed', 'failure', 'active', 'unstarted', 'completed',
            'plan_sha256', 'maximum_workers', 'threads', 'retries', 'replacements')
    require(all(summary[k] == status[k] for k in keys)
            and summary['schema'] == driver.SCHEMA
            and summary['complete'] is True and summary['passed'] is True
            and summary['failure'] is None and summary['active'] is None and summary['unstarted'] == []
            and len(summary['completed']) == 2 and summary['plan_sha256'] == digest
            and summary['files'] == plan['files'] and summary['retries'] == summary['replacements'] == 0
            and summary['maximum_workers'] == summary['threads'] == 1
            and not (root/'execution/failure.json').exists(), 'Execution incomplete, failed or undrained')
    protocol = bound.read('protocol.json'); prepared = bound.read('preparation.json')
    require(protocol['schema'] == 'growth-z004-native-protocol-v1'
            and prepared['schema'] == 'growth-z004-native-preparation-v1'
            and prepared['complete'] is True and prepared['launched'] is False
            and prepared['execution_plan'] == dict(path=str(root/'execution-plan.json'), sha256=digest)
            and prepared['protocol'] == dict(path=str(root/'protocol.json'), sha256=bound.refs[str(root/'protocol.json')])
            and plan['files'].get(str(root/'protocol.json')) == bound.refs[str(root/'protocol.json')]
            and claim['preparation_receipt'] == dict(path=str(root/'preparation.json'), sha256=bound.refs[str(root/'preparation.json')])
            and not (root/'preparation-failure.json').exists(), 'Preparation/protocol binding differs')
    require(protocol['root'] == str(root) and set(protocol['arms']) == set(ARMS)
            and protocol['physical_parameters'] == PHYSICAL and protocol['shared_provenance'] == SHARED
            and protocol['geometry_jobs'] == 2 and protocol['new_saved_frames'] == 2002
            and protocol['cached_classifications'] == protocol['new_physical_draws'] == 0
            and protocol['move_log_scans'] == protocol['exclusion_graph_jobs'] == 0,
            'Protocol context, build or comparison allocation differs')
    source = root/'code/run_native_class_physical_campaign.py'
    require(source.resolve() == source and sha(source) == sha(driver.__file__) == plan['files'].get(str(source)),
            'Frozen driver differs from reviewed stable source')
    bound.refs[str(source)] = sha(source)
    spec = importlib.util.spec_from_file_location('_growth_z004_frozen_driver', source)
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    for index, arm in enumerate(ARMS):
        job = plan['jobs'][index]; terminal_path = root/arm/'strict-native/assessment.json'
        require(job['id'] == arm+'-native' and job['population'] == arm and job['phase'] == 'geometry'
                and job['terminal'] == dict(path=str(terminal_path), success_contract='complete'),
                'Declared native arm/terminal differs')
        require(terminal_path.stat().st_size <= 256*1024**2, 'Assessment byte cap exceeded')
        terminal = frozen.completed_terminal(root, plan, job['id'])
        directory = Path('execution/jobs')/f'{index:03d}-{job["id"]}'
        records = {name: bound.read(directory/(name+'.json')) for name in ('attempt', 'process', 'success', 'exit')}
        require(records['success'] == records['exit'] == summary['completed'][index]
                and summary['completed'][index]['terminal'] == terminal, 'Completed inventory differs')
    assessments = {}; all_rows = {}; native_bytes = 0
    for arm in ARMS:
        directory = Path(arm)/'strict-native'
        manifest_path = root/directory/'manifest.json'
        require(sha(manifest_path) == manifest_sha256[arm], 'Expected native manifest hash differs')
        declaration = protocol['arms'][arm]
        snapshot_manifest = Path(arm)/'snapshot-manifest.json'; snapshot = bound.read(snapshot_manifest)
        require(declaration['snapshot_manifest'] == dict(path=str(root/snapshot_manifest), sha256=bound.refs[str(root/snapshot_manifest)])
                and plan['files'].get(str(root/snapshot_manifest)) == bound.refs[str(root/snapshot_manifest)]
                and declaration['model_sha256'] == snapshot['actual_model_sha256'] == MODELS[arm]
                and declaration['native_informed_proposal'] is (arm == 'coverage')
                and declaration['effective_config_sha256'] == snapshot['sha256']['config.json']
                and declaration['frames'] == snapshot['frames'] == 1001
                and snapshot['first_sweep'] == 0 and snapshot['last_sweep'] == 100000
                and snapshot['cadence_sweeps'] == 100 and declaration['final_exhaustive_pairs'] == 34716,
                'Frozen arm/model/snapshot allocation differs')
        require(not (root/arm/'failure.json').exists() and not (root/directory/'failure.json').exists(),
                'Native arm failure is present')
        assessment = bound.read(directory/'assessment.json'); manifest = bound.read(directory/'manifest.json')
        require(manifest['complete'] is True and manifest['physical_jobs_launched'] == 0
                and manifest['source_sha256'] == assessment['source_sha256']
                and assessment['snapshot'] == str(root/arm/'snapshot')
                and assessment['reuse'] is None and assessment['newly_classified_frames'] == 1001,
                'Unexpected native observation provenance/reuse')
        for path, value in assessment['source_sha256'].items():
            require(plan['files'].get(path) == value, 'Native input is outside frozen plan')
        require({'assessment.json', 'frames.jsonl', 'report.md'} <= manifest['files'].keys(), 'Native output inventory missing')
        for name, value in manifest['files'].items():
            path = (root/directory/name).resolve()
            require(path.is_relative_to(root/directory), 'Escaping native manifest path')
            native_bytes += path.stat().st_size
            require(native_bytes <= 1024**3, 'Two-arm native artifact byte cap exceeded')
            if name != 'frames.jsonl':
                require(sha(path) == value, 'Native manifest hash mismatch: '+name)
                bound.refs[str(path)] = value
        path = root/directory/'frames.jsonl'; rows = []; hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for line in iter(lambda: stream.readline(4*1024**2+1), b''):
                require(len(line) <= 4*1024**2 and len(rows) < 1001, 'Native row/count cap exceeded')
                hasher.update(line)
                rows.append(json.loads(line, object_pairs_hook=driver._pairs,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite native JSON '+value))))
        require(hasher.hexdigest() == manifest['files']['frames.jsonl'], 'Native frame hash mismatch')
        bound.refs[str(path)] = hasher.hexdigest(); assessments[arm] = assessment; all_rows[arm] = rows
    return assessments, all_rows, bound.refs


def saved_series(assessment, rows):
    """Reduce authenticated labels only; keep membership and component sizes distinct."""
    require(assessment['complete'] is True and assessment['frames'] == len(rows) == 1001
            and assessment['first_sweep'] == 0 and assessment['last_sweep'] == 100000
            and assessment['cadence_sweeps'] == 100, 'Wrong native saved-frame allocation')
    require(assessment['seed_labels'] == list(range(8)), 'Wrong original seed labels')
    require(assessment['physical'] == dict(radius_A=1.4, activity_A_minus3=.04,
            bodies=264, concentration_uM=500.0), 'Wrong physical comparison context')
    require(assessment['initial'] == rows[0] and assessment['final'] == rows[-1],
            'Assessment endpoints differ from saved native rows')
    check = assessment['final_full_pair_check']
    require(check['passed'] is True and check['pairs'] == 34716
            and check['keys'] == len(rows[-1]['native_keys']), 'Missing exhaustive endpoint agreement')
    seeds = set(assessment['seed_labels']); result = []; previous_cpu = -1.
    previous_members = set(); previous_edges = set()
    for index, row in enumerate(rows):
        require(type(row['sweep']) is int and row['sweep'] == index*100, 'Missing or repeated saved frame')
        cpu = row['sampler_cpu_seconds']
        require(type(cpu) in (int, float) and math.isfinite(cpu) and cpu >= max(0, previous_cpu),
                'Invalid recorded sampler CPU')
        previous_cpu = cpu
        members = row['certified_seed_bodies']
        require(members == sorted(set(members)) and all(type(b) is int and 0 <= b < 264 for b in members),
                'Invalid certified seed membership')
        member_set = set(members)
        keys = row['native_keys']
        require(all(len(k) == 3 and all(type(v) is int for v in k)
                    and 0 <= k[0] < k[1] < 264 and 0 <= k[2] < 14 for k in keys)
                and len(set(map(tuple, keys))) == len(keys), 'Invalid native motif keys')
        edges = {(i, j) for i, j, _ in keys}
        require(row['registered_edges'] == len(edges)
                and row['registered_edges_outside_original_seed'] == sum(not {i, j} <= seeds for i, j in edges),
                'Saved edge counts differ from labels')
        require(row['seed_entered_bodies'] == (sorted(member_set-previous_members) if index else [])
                and row['seed_left_bodies'] == (sorted(previous_members-member_set) if index else [])
                and sorted(map(tuple, row['gained_edges'])) == (sorted(edges-previous_edges) if index else [])
                and sorted(map(tuple, row['lost_edges'])) == (sorted(previous_edges-edges) if index else []),
                'Saved membership/edge changes differ')
        previous_members, previous_edges = member_set, edges
        separate = []; certified = []; seed_components = []
        for component in row['registered_components']:
            bodies = component['bodies']
            require(type(component['certified']) is bool and len(bodies) > 1
                    and bodies == sorted(set(bodies))
                    and all(type(b) is int and 0 <= b < 264 for b in bodies), 'Invalid registered component')
            if seeds.intersection(bodies):
                if component['certified']: seed_components.append(len(bodies))
            else:
                separate.append(len(bodies))
                if component['certified']: certified.append(len(bodies))
        require(sorted(separate, reverse=True) == row['nonseed_registered_component_sizes'],
                'Nonseed registered-component sizes differ')
        largest_seed = row['largest_certified_seed_component']
        # Singleton seed components are omitted from registered_components.
        require(type(largest_seed) is int and max([0]+seed_components) <= largest_seed <= len(members)
                and (largest_seed <= 1 or largest_seed == max([0]+seed_components)),
                'Largest certified seed component differs')
        result.append(dict(sweep=row['sweep'], sampler_cpu_seconds=cpu,
            certified_seed_members=len(members), largest_certified_seed_component=largest_seed,
            registered_edges_outside_original_seed=row['registered_edges_outside_original_seed'],
            largest_nonseed_registered_component=max([0]+separate),
            largest_nonseed_certified_component=max([0]+certified)))
    return result


def numeric_summary(assessment, series):
    """Copy completed observer reductions; no stationarity or uncertainty fitting."""
    require(len(assessment['quarters']) == 4, 'Four saved quarters required')
    return dict(initial=assessment['initial'], final=assessment['final'],
        saved_windows=assessment['windows'], saved_quarters=assessment['quarters'],
        seed_entries=assessment['windows']['full']['seed_member_entries'],
        seed_exits=assessment['windows']['full']['seed_member_exits'],
        motif_families=assessment['families'],
        missing_outside_seed_families=assessment['missing_outside_seed_families'],
        seed_membership_episodes=assessment['seed_newcomers'],
        motif_change_windows=assessment['motif_change_windows'],
        observed_edge_reentry_episodes=assessment['observed_edge_reentry_episodes'],
        cycle_frustrated_frames=assessment['cycle_frustrated_frames'],
        time_series=series,
        window_convention='Copied observer windows: (start,end] transitions/residence, with start endpoint retained as baseline. Any inherited per-1000-sweep values are algorithmic descriptors.',
        component_convention='Seed members are the union of certified components touching original seed labels. Largest certified seed component is separate. Nonseed components contain no original seed labels; size zero means none with at least two bodies. Registered connectivity alone does not certify cycle consistency.')


def render(summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['path.simplify'] = False
    colors = {'coverage': '#0072B2', 'discovery': '#D55E00'}
    panels = (('certified_seed_members', 'Cycle-certified seed membership',
               'largest_certified_seed_component', 'largest certified seed component'),
              ('registered_edges_outside_original_seed', 'Registered edges outside original seed', None, None),
              ('largest_nonseed_registered_component', 'Largest separate nonseed component',
               'largest_nonseed_certified_component', 'cycle-certified'))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey='col')
    for row_index, coordinate in enumerate(('sweep', 'sampler_cpu_seconds')):
        for ax, (key, title, secondary, _) in zip(axes[row_index], panels):
            for arm in ARMS:
                rows = summaries[arm]['time_series']; x = [r[coordinate] for r in rows]
                ax.plot(x, [r[key] for r in rows], color=colors[arm], lw=1, label=LABELS[arm])
                if secondary:
                    ax.plot(x, [r[secondary] for r in rows], color=colors[arm], lw=.8, linestyle='--')
            ax.set_title(title, fontsize=11); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
            ax.set_xlabel('Algorithmic sweep' if row_index == 0 else 'Recorded sampler CPU (s)')
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Bodies'); axes[1, 0].set_ylabel('Bodies')
    axes[0, 1].set_ylabel('Edges'); axes[1, 1].set_ylabel('Edges')
    axes[0, 2].set_ylabel('Bodies'); axes[1, 2].set_ylabel('Bodies')
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.suptitle('Saved native registry and reorganization · two historical proposal arms', fontsize=15, y=.98)
    fig.subplots_adjust(left=.06, right=.98, top=.84, bottom=.20, hspace=.38, wspace=.29)
    fig.text(.06, .04, 'rd 1.4 Å · z 0.04 Å⁻³ · 500 μM · N=264 · eight initial seed bodies · every saved frame retained.\n'
             'Left: solid = union of certified seed memberships; dashed = largest certified seed component.\n'
             'Right: solid = registered connectivity; dashed = cycle-certified component; isolated bodies excluded.\n'
             'Shared RNG seed and identical initial poses. Descriptive histories, not independent replicates or equilibrium evidence.\n'
             'Lines connect sampled observations; intervening events are unresolved. CPU and sweeps are not physical time.', fontsize=9)
    for suffix in ('png', 'svg'): fig.savefig(output/('native-growth.'+suffix), dpi=180, facecolor='white')
    plt.close(fig)
    return matplotlib.__version__


def plot(root, output, manifest_sha256):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Fresh output directory required')
    source = {str(Path(__file__).resolve()): sha(__file__), str(Path(driver.__file__).resolve()): sha(driver.__file__)}
    assessments, all_rows, refs = authenticate(root, manifest_sha256)
    summaries = {arm: numeric_summary(assessments[arm], saved_series(assessments[arm], all_rows[arm])) for arm in ARMS}
    output.mkdir(parents=True)
    try:
        version = render(summaries, output)
        write(output/'numeric-summary.json', dict(schema='growth-z004-native-presentation-v1',
            complete=True, scope=SCOPE, arms=summaries, labels=LABELS, physical=assessments['coverage']['physical'],
            expected_manifest_sha256=manifest_sha256, input_sha256=refs))
        lines = [SCOPE, '', 'rd=1.4 Å, z=.04 Å⁻³, 500 μM, 264 bodies, eight original seed bodies. Each arm retains all 1001 observations.', '',
                 '| Arm | Certified seed members, start → end | Largest certified seed component, start → end | Outside-seed edges, start → end | Saved member entries / exits |',
                 '| --- | --- | --- | --- | --- |']
        for arm in ARMS:
            data = summaries[arm]; first, last = data['time_series'][0], data['time_series'][-1]
            changes = [f"{first[k]} → {last[k]}" for k in ('certified_seed_members', 'largest_certified_seed_component', 'registered_edges_outside_original_seed')]
            lines.append(f"| {LABELS[arm]} | {' | '.join(changes)} | {data['seed_entries']} / {data['seed_exits']} |")
        lines += ['', 'The numeric summary preserves observer endpoints, full/late/quarter windows, inverse motif-family coverage, censored membership episodes and motif changes. A registered nonseed component is not necessarily cycle-certified. Seed membership totals can span multiple disconnected components.', '']
        (output/'report.md').write_text('\n'.join(lines))
        for path, digest in {**refs, **source}.items(): require(sha(path) == digest, 'Presentation input changed')
        write(output/'receipt.json', dict(schema='growth-z004-native-plot-v1', complete=True, passed=True,
            source_sha256=source, input_sha256=refs, outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
            arms=list(ARMS), saved_native_rows=2002, new_geometry_queries=0, move_log_reads=0,
            raw_trajectory_decodes=0, frozen_input_byte_hashing=True,
            driver_completed_terminal_calls=2, matplotlib_version=version, scope=SCOPE))
    except BaseException as error:
        write(output/'failure.json', dict(complete=False, error=repr(error))); raise
    return output/'receipt.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--out', required=True)
    for arm in ARMS: parser.add_argument('--'+arm+'-manifest-sha256', required=True)
    args = parser.parse_args()
    print(plot(args.root, args.out, {arm: getattr(args, arm+'_manifest_sha256') for arm in ARMS}))
