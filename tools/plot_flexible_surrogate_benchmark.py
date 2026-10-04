#!/usr/bin/env python3
"""Present complete flexible-benchmark observer results without reading journals.

Only saved scalar metrics and completed controller metadata are consumed.
Internal patches are coarse contact descriptors, not native registry or binding
basins. Cached controls and new flexible chains remain separate stream records.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

CONTROLS = ('local', 'm4', 'rigid_surrogate_1', 'rigid_surrogate_8', 'rigid_surrogate_flat8')
FLEXIBLE = ('flexible_m1', 'flexible_m8', 'flexible_flat8')
ARMS = CONTROLS + FLEXIBLE
LABELS = ('local', 'm4', 'rigid\nguided 1', 'rigid\nguided 8', 'rigid\nflat 8',
          'flexible\nguided 1', 'flexible\nguided 8', 'flexible\nflat 8')
STARTS = ('source', 'proposal_prepared')
ORGANIZATION = (
    ('internal_contact_fraction', ('internal_organization', 'internal_contact_fraction'), 'Internal pair contact occupancy'),
    ('internal_patch_changes_per_cpu', ('internal_organization', 'patch_set_activity', 'direct_nonempty_changes_per_full_CPU_second'),
     'Internal patch-set changes / CPU s\n(nonempty to nonempty)'),
    ('internal_patch_returns_per_cpu', ('internal_organization', 'patch_set_activity', 'completed_nonempty_returns_per_full_CPU_second'),
     'Nonempty internal patch-set returns / CPU s'),
    ('external_contact_fraction', ('external_only', 'any_contact_fraction'), 'Any external contact occupancy'))
EFFICIENCY = (
    ('external_changes_per_cpu', ('external_activity', 'direct_nonempty_changes_per_full_CPU_second'),
     'External edge-set changes / CPU s\n(nonempty to nonempty)'),
    ('external_returns_per_cpu', ('external_activity', 'completed_nonempty_returns_per_full_CPU_second'),
     'Nonempty external edge-set returns / CPU s'),
    ('contact_fingerprint_ess_per_cpu', ('fingerprint_ess', 'apparent_ess_per_sampling_CPU_second'),
     'Whole contact-fingerprint ESS / CPU s'),
    ('external_presence_ess_per_cpu', ('external_only', 'ess', 'apparent_ess_per_sampling_CPU_second'),
     'External-edge presence ESS / CPU s'))
METRICS = ORGANIZATION + EFFICIENCY


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def record(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path))


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def authenticate(root):
    """Require a separate, successful one-job controller before reading metrics."""
    refs={}
    def load(relative):
        path=(root/relative).resolve();require(path.is_relative_to(root),'Escaping completion path')
        refs[str(path)]=sha(path)
        return json.loads(path.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
    plan=load('execution-plan.json');claim=load('execution/claim.json')
    summary=load('execution/summary.json');status=load('execution/status.json')
    require(plan['schema']=='native-class-physical-execution-v1' and plan['root']==str(root)
        and plan['maximum_workers']==plan['threads']==1 and len(plan['jobs'])==1,'Wrong analysis controller plan')
    plan_hash=refs[str(root/'execution-plan.json')]
    require(claim['schema']==plan['schema'] and claim['plan_sha256']==plan_hash
        and claim['maximum_workers']==claim['threads']==1 and claim['retries']==claim['replacements']==0,
        'Controller claim differs')
    keys=('complete','passed','failure','active','unstarted','completed','plan_sha256')
    require(all(summary[k]==status[k] for k in keys) and summary['complete'] is True and summary['passed'] is True
        and summary['failure'] is None and summary['active'] is None and summary['unstarted']==[]
        and summary['plan_sha256']==plan_hash and len(summary['completed'])==1
        and not (root/'execution/failure.json').exists() and not (root/'analysis/failure.json').exists(),
        'Independent analysis execution is incomplete, failed or undrained')
    job=plan['jobs'][0]
    require(job['id']=='whole-contact-analysis' and job['phase']=='statistics'
        and job['terminal']==dict(path=str(root/'analysis/analysis.json'),success_contract='complete'),
        'Wrong declared analysis terminal')
    directory=Path('execution/jobs')/('000-'+job['id'])
    attempt=load(directory/'attempt.json');process=load(directory/'process.json')
    done=load(directory/'success.json');exit_record=load(directory/'exit.json')
    require(done==exit_record==summary['completed'][0] and attempt['job']==job
        and all(done[k]==job[k] for k in ('id','population','phase','argv'))
        and done['success'] is True and done['child_started'] is True and done['child_drained'] is True
        and done['returncode']==0 and done['error'] is None and done['timeout'] is False
        and done['retries']==done['replacements']==0 and done['success_contract']=='complete'
        and type(done['pid']) is int and done['pid']>0
        and all(process[k]==done[k] for k in ('id','pid','birth_ticks','argv')),
        'Analysis job lifecycle differs')
    protocol=load('protocol.json');prepared=load('preparation.json')
    require(protocol['schema']=='flexible-surrogate-analysis-dispatch-v1'
        and prepared['schema']=='flexible-surrogate-analysis-preparation-v1'
        and prepared['complete'] is True and prepared['launched'] is False
        and prepared['execution_plan']==dict(path=str(root/'execution-plan.json'),sha256=plan_hash)
        and prepared['protocol']==dict(path=str(root/'protocol.json'),sha256=refs[str(root/'protocol.json')])
        and plan['files'].get(str(root/'protocol.json'))==refs[str(root/'protocol.json')]
        and claim['preparation_receipt']==dict(path=str(root/'preparation.json'),sha256=refs[str(root/'preparation.json')])
        and not (root/'preparation-failure.json').exists(),'Analysis preparation provenance differs')
    result=load('analysis/analysis.json');manifest=load('analysis/manifest.json')
    require(done['terminal']==dict(path=str(root/'analysis/analysis.json'),sha256=refs[str(root/'analysis/analysis.json')])
        and manifest['complete'] is True and manifest['files']['analysis.json']==done['terminal']['sha256'],
        'Completed result hash differs')
    require(result['schema']=='flexible-surrogate-analysis-v1' and result['complete'] is True
        and result['new_chains']==24 and result['reused_control_chains']==40 and len(result['chains'])==64
        and result['new_geometry_endpoints']==110616 and result['old_geometry_queries']==result['new_physical_draws']==0
        and result['native_observer'] is False and protocol['new_chains']==24 and protocol['cached_control_chains']==40,
        'Incomplete or unexpected whole-benchmark result')
    declaration=protocol['analysis_plan'];path=Path(declaration['path']).resolve()
    require(sha(path)==declaration['sha256']==plan['files'].get(str(path)),
        'Unbound frozen analysis declaration')
    refs[str(path)]=declaration['sha256']
    require(json.loads(path.read_text())==result['analysis_plan'],
        'Result differs from frozen analysis declaration')
    return result,refs


def number(value, label, fraction=False):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Invalid saved '+label)
    if fraction: require(value <= 1, 'Fraction exceeds one: '+label)
    return value


def count(value, label):
    require(type(value) is int and value >= 0, 'Invalid saved counter '+label)
    return value


def close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), 'Saved arithmetic differs: '+label)


def saved_values(result):
    rows = []; seen = set()
    for chain in result['chains']:
        job = chain['job']; identity = tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))
        require(identity not in seen, 'Duplicate stream identity'); seen.add(identity)
        require(chain['reused_control'] is (job['arm'] in CONTROLS), 'Wrong new/cached origin')
        metrics = chain['metrics']; cpu = number(metrics['full_sampler_cpu_seconds'], 'CPU')
        require(cpu > 0 and metrics['production_samples'] == 4096, 'Wrong sample count/CPU denominator')
        internal = metrics['internal_organization']; activity = internal['patch_set_activity']
        external = metrics['external_activity']
        require(internal['transition_baseline_block'] == 512, 'Wrong internal transition baseline')
        for value in (internal, activity, external):
            require(value['production_samples'] == 4096 and value['full_sampler_cpu_seconds'] == cpu,
                    'Production residence or full CPU differs')
        close(metrics['internal_contact_fraction'], internal['internal_contact_fraction'], 'internal occupancy')
        close(activity['nonempty_fraction'], internal['internal_contact_fraction'], 'internal token occupancy')
        close(external['nonempty_fraction'], metrics['external_only']['any_contact_fraction'], 'external occupancy')
        require(internal['attachments'] == activity['entering_nonempty']
                and internal['detachments'] == activity['leaving_nonempty'], 'Internal edge event counts differ')
        for group in (activity, external):
            changes = count(group['direct_nonempty_changes'], 'direct changes')
            count(group['entering_nonempty'], 'attachments'); count(group['leaving_nonempty'], 'detachments')
            close(group['direct_nonempty_changes_per_full_CPU_second'], changes/cpu, 'change rate')
            close(group['completed_nonempty_returns_per_full_CPU_second'],
                  len(group['completed_nonempty_returns'])/cpu, 'return rate')
        values = {}
        for name, path, _ in METRICS:
            value = metrics
            for key in path: value = value[key]
            if value is None: require('ess' in name, 'Only undefined ESS may be null')
            else: number(value, name, fraction=name.endswith('_fraction'))
            values[name] = value
        for value in (metrics['fingerprint_ess'], metrics['external_only']['ess'], internal['patch_set_ess']):
            ess, rate = value['apparent_ess'], value['apparent_ess_per_sampling_CPU_second']
            require((ess is None) == (rate is None), 'Inconsistent undefined ESS')
            if ess is not None:
                number(ess, 'ESS'); number(rate, 'ESS/CPU'); close(rate, ess/cpu, 'ESS CPU denominator')
        fractions = [number(r['fraction'], 'internal patch-set occupancy', True) for r in internal['patch_set_occupancy']]
        require(fractions and len({r['fingerprint'] for r in internal['patch_set_occupancy']}) == len(fractions),
                'Empty/duplicate patch-set occupancy')
        close(math.fsum(fractions), 1., 'patch-set occupancy total')
        occupied_frames = values['external_contact_fraction']*4096
        close(occupied_frames, round(occupied_frames), 'occupied-frame count')
        rows.append(dict(job=job, origin='cached control' if chain['reused_control'] else 'new flexible',
            reused_control=chain['reused_control'], full_sampler_cpu_seconds=cpu, production_samples=4096,
            **values, external_contact_frames=round(occupied_frames),
            internal_attachments=internal['attachments'], internal_detachments=internal['detachments'],
            unique_internal_patch_sets=len(fractions), internal_patch_changes=activity['direct_nonempty_changes'],
            internal_patch_returns=len(activity['completed_nonempty_returns']),
            external_changes=external['direct_nonempty_changes'], external_returns=len(external['completed_nonempty_returns']),
            internal_patch_set_ess=internal['patch_set_ess']['apparent_ess'],
            contact_fingerprint_ess=metrics['fingerprint_ess']['apparent_ess'],
            external_presence_ess=metrics['external_only']['ess']['apparent_ess'],
            singleton_fingerprint_fraction=number(metrics['singleton_fingerprint_fraction'], 'singleton fingerprint fraction', True)))
    expected = {(0, arm, start, stream) for arm in ARMS for start in STARTS for stream in range(4)}
    require(seen == expected and len(rows) == 64, 'All64 chains must remain distinct: 24 new and40 cached')
    return sorted(rows, key=lambda r:(STARTS.index(r['job']['initialization']), ARMS.index(r['job']['arm']), r['job']['stream']))


def saved_agreement(result):
    """Copy saved same-stream initialization comparisons, without new estimates."""
    rows = []; seen = set()
    for pair in result['comparisons']['descriptive_paired_comparisons']:
        left, right = pair['left'], pair['right']
        if left['arm'] != right['arm']: continue
        require(left['context_index'] == right['context_index'] == 0 and left['stream'] == right['stream']
                and {left['initialization'], right['initialization']} == set(STARTS), 'Mismatched initialization comparison')
        identity = (left['arm'], left['stream']); require(identity not in seen, 'Duplicate initialization comparison'); seen.add(identity)
        internal, external = pair['internal_organization'], pair['external_only']
        difference = internal['contact_fraction_difference']
        require(type(difference) in (int, float) and math.isfinite(difference) and abs(difference) <= 1,
                'Invalid initialization contact difference')
        rows.append(dict(arm=left['arm'], stream=left['stream'],
            internal_patch_set_occupancy_TV=number(internal['patch_set_occupancy_total_variation'], 'internal TV', True),
            external_set_occupancy_TV=number(external['environment_total_variation'], 'external TV', True),
            internal_contact_fraction_absolute_difference=abs(difference),
            source_start_comparison=dict(left=left['initialization'], right=right['initialization'],
                                         saved_right_minus_left_contact_difference=difference)))
    require(seen == {(arm, stream) for arm in ARMS for stream in range(4)} and len(rows) == 32,
            'All32 initialization comparisons are required')
    return sorted(rows, key=lambda r:(ARMS.index(r['arm']), r['stream']))


def plot(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Fresh plot output required')
    source = record(__file__); result, inputs = authenticate(root)
    rows = saved_values(result); agreement = saved_agreement(result)
    # Import render dependencies only after completed metadata validation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7'); markers = ('o', 's', '^', 'D')
    legend = [Line2D([0], [0], color=colors[s], marker=markers[s], lw=.8, label=f'Stream {s}') for s in range(4)]
    legend += [Patch(facecolor='#eeeeee', label='Cached 40 controls'), Patch(facecolor='#eaf3fa', label='New 24 flexible chains')]
    def decorate(ax):
        ax.axvspan(-.5, 4.5, color='#eeeeee', zorder=0)
        ax.axvspan(4.5, 7.5, color='#eaf3fa', zorder=0)
        ax.axvline(4.5, color='#777777', lw=.6)
        ax.set_xticks(range(8), LABELS, fontsize=8); ax.set_xlim(-.4, 7.4)
        ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    def draw(ax, by_arm, stream):
        # Do not join historical and new groups across the origin boundary.
        for indices in (range(5), range(5, 8)):
            ax.plot([i+(stream-1.5)*.1 for i in indices],
                    [np.nan if by_arm[ARMS[i]] is None else by_arm[ARMS[i]] for i in indices],
                    color=colors[stream], marker=markers[stream], ms=4.5, lw=.8, alpha=.9)
    output.mkdir(parents=True)
    try:
        for name, metrics, title in (
            ('contact-organization', ORGANIZATION, 'Internal contact-patch reorganization and external occupancy'),
            ('contact-efficiency', EFFICIENCY, 'External contact changes and finite-record contact ESS')):
            fig, axes = plt.subplots(2, 4, figsize=(23, 10), sharey='col')
            for row_index, start in enumerate(STARTS):
                group = [r for r in rows if r['job']['initialization'] == start]
                for column, (key, _, label) in enumerate(metrics):
                    ax = axes[row_index, column]; decorate(ax)
                    for stream in range(4):
                        draw(ax, {r['job']['arm']:r[key] for r in group if r['job']['stream'] == stream}, stream)
                    for x, arm in enumerate(ARMS):
                        missing = [str(r['job']['stream']) for r in group if r['job']['arm'] == arm and r[key] is None]
                        if missing: ax.text(x, -.23, 'undefined\n'+','.join(missing), transform=ax.get_xaxis_transform(),
                                            ha='center', va='top', fontsize=7, color='#555555', clip_on=False)
                    if row_index == 0: ax.set_title(label, fontsize=10, pad=11)
                    if column == 0: ax.set_ylabel('Source start' if start == 'source' else 'Proposal-prepared start', fontsize=11)
            for column, (key, _, label) in enumerate(metrics):
                defined = [r[key] for r in rows if r[key] is not None]; ax = axes[0, column]
                if key.endswith('_fraction'): ax.set_ylim(-.025, 1.025)
                elif 'ess' in key and defined and min(defined) > 0:
                    ax.set_yscale('log'); ax.set_ylim(.6*min(defined), 1.5*max(defined))
                    ax.set_title(label+' · log scale', fontsize=10, pad=11)
                else:
                    upper = 1.12*max(defined) if defined and max(defined) > 0 else 1.
                    ax.set_ylim(-.04*upper, upper)
                    if not defined: ax.set_yticks([])
                for row_index in range(2):
                    subset = [r[key] for r in rows if r['job']['initialization'] == STARTS[row_index] and r[key] is not None]
                    if not subset: axes[row_index, column].text(.5, .5, 'ESS undefined for all streams', transform=axes[row_index, column].transAxes, ha='center', fontsize=9)
                    elif max(subset) == 0: axes[row_index, column].text(.5, .5, 'All observed values are zero', transform=axes[row_index, column].transAxes, ha='center', fontsize=9)
            fig.suptitle(title+' · starts and streams kept separate', fontsize=15, y=.985)
            fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .943), ncol=6, frameon=False)
            fig.subplots_adjust(left=.055, right=.985, top=.83, bottom=.25, hspace=.66, wspace=.26)
            fig.text(.055, .035,
                '4096 production endpoints per chain; transition events include block 512 as baseline and retain every rejection. Rates use full sampler CPU, including warmup.\n'
                'Patches are coarse atom-contact labels: changes can be threshold flicker, not native registry, binding-basin transitions, or physical kinetics.\n'
                'ESS is a finite-record contact descriptor; constant descriptors remain undefined. Rare-contact presence ESS can be high after one occupied frame.\n'
                'Read ESS alongside external occupancy (contact-organization panel; occupied-frame counts in plotted-values.json) and nonempty returns.\n'
                'Stream markers are offset horizontally for visibility. Shared RNG roles do not create additional independent replicates. Target: 2 mobile + 262 fixed tetramers; no equilibrium/assembly claim.',
                fontsize=9.5, va='bottom')
            for extension in ('png', 'svg'): fig.savefig(output/f'{name}.{extension}', dpi=180, facecolor='white')
            plt.close(fig)
        definitions = (('internal_patch_set_occupancy_TV', 'Internal patch-set occupancy TV'),
                       ('external_set_occupancy_TV', 'External edge-set occupancy TV'),
                       ('internal_contact_fraction_absolute_difference', 'Absolute internal-contact occupancy difference'))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
        for ax, (key, title) in zip(axes, definitions):
            decorate(ax)
            for stream in range(4): draw(ax, {r['arm']:r[key] for r in agreement if r['stream'] == stream}, stream)
            ax.set_ylim(-.025, 1.025); ax.set_title(title, fontsize=10)
        axes[0].set_ylabel('Difference between source and prepared starts')
        fig.suptitle('Initialization differences · same stream, separate starts', fontsize=14, y=.98)
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .925), ncol=6, frameon=False)
        fig.subplots_adjust(left=.055, right=.985, top=.79, bottom=.24, wspace=.2)
        fig.text(.055, .055, 'Saved comparisons, no trajectories pooled. Zero difference can mean both runs stayed unbound; agreement does not establish convergence.\n'
                 'Internal patches do not identify native registry. Cached 40 controls and new 24 flexible chains remain distinct.', fontsize=10)
        for extension in ('png', 'svg'): fig.savefig(output/f'initialization-differences.{extension}', dpi=180, facecolor='white')
        plt.close(fig)
        write(output/'plotted-values.json', dict(schema='flexible-surrogate-presentation-values-v1',
            metric_paths={name:['metrics', *path] for name, path, _ in METRICS}, rows=rows,
            initialization_agreement=agreement, cached_controls=40, new_flexible_chains=24,
            interpretation='Saved observer scalars only. Occupied-frame counts are the saved fraction times 4096; initialization contact differences are displayed as magnitudes. No geometry or statistical metric is re-estimated.'))
        for path, digest in inputs.items(): require(sha(path) == digest, 'Completion input changed during plotting')
        require(record(__file__) == source, 'Plot source changed during rendering')
        receipt = dict(schema='flexible-surrogate-contact-plot-v1', complete=True, source=source, input_sha256=inputs,
            outputs={p.name:record(p) for p in output.iterdir() if p.is_file()}, chains=64,
            cached_control_chains=40, new_flexible_chains=24, initialization_comparisons=32,
            null_ess_policy='Undefined stays null in table and appears as a labelled gap; never zero-filled.',
            matplotlib_version=matplotlib.__version__, numpy_version=np.__version__, scientific_journals_read=0,
            new_geometry_queries=0, new_statistics_estimated=0, streams_or_starts_pooled=False, native_registry=False)
        write(output/'plot-receipt.json', receipt); return receipt
    except BaseException as error:
        write(output/'failure.json', dict(complete=False, error=repr(error), source=source, input_sha256=inputs)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(plot(args.root, args.output), indent=2))
