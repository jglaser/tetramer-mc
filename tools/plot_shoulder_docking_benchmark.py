#!/usr/bin/env python3
"""Plot retained shoulder states alongside descriptive ESS and coverage checks.

Reads completed audits only. No physical replay, accepted-state filtering,
thinning, fitting, stationarity threshold, or automatic speedup estimate.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT/'runs/ab-shoulder-docking-pilot-12x5000-20260921'
MODES = ('local', 'c0', 'c09')
MODE_TITLES = ('Local', 'Posterior c = 0', 'Posterior c = 0.9')
STARTS = ('direct', 'geometry')
LABELS = ('direct', 'mixture', 'geometry', 'inner_remainder', 'outer_shoulder')
LABEL_TITLES = ('Direct contact', 'Mixture contact', 'Geometry contact', 'Inner remainder', 'Outer shoulder')
LABEL_COLORS = ('#286fab', '#c06540', '#258b75', '#9382b1', '#c9ccd1')
START_COLORS = dict(direct=LABEL_COLORS[0], geometry=LABEL_COLORS[2])
START_MARKERS = dict(direct='o', geometry='s')
OFFSETS = {('direct', 0): -.20, ('direct', 1): -.09, ('geometry', 0): .09, ('geometry', 1): .20}
SCOPE = ('One mobile tetramer conditional on strict original 1<q<2 and two fixed AB neighbors. '
    'Every post-burn retained cycle, including repeats, is used. Apparent contact ESS is descriptive until '
    'occupancy, initialization and stationarity checks agree; no global convergence, assembly or kinetic-rate claim.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def near(a, b):
    require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), f'Saved statistic differs: {a}, {b}')


def load_completed(analysis_path, campaign):
    """Bind saved observables to completed audit/source bytes; do not replay MC."""
    result = read(analysis_path); manifest_path = campaign/'manifest.json'
    manifest = read(manifest_path); status_path = campaign/'status.json'; status = read(status_path)
    require(result['passed'] is True and manifest['schema'] == 'conditional-shoulder-docking-v1', 'Need completed conditional shoulder audit')
    require(status['complete'] is True and status['running'] is False, 'Physical campaign is not terminal success')
    require(sha(manifest_path) == result['campaign_manifest_sha256'] and
            sha(status_path) == result['terminal_status_sha256'], 'Audited campaign or terminal status changed')
    cycles, burn = manifest['cycles'], manifest['burn_cycles']
    require(type(cycles) is int and type(burn) is int and 0 <= burn < cycles
            and cycles-burn >= 2 and manifest['sample_every'] == 1, 'Invalid fixed retained-cycle schedule')
    retained, half = cycles-burn, (cycles-burn)//2
    require(tuple(manifest['physical_labels']) == LABELS and manifest['attempts_per_cycle'] == 3, 'Physical labels or slots differ')
    require(manifest['target_region']['window'] == dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False), 'Original shoulder target differs')
    require(result['physical_reference'] == manifest['physical_reference'], 'Frozen importance reference changed')
    inputs = {str(analysis_path): sha(analysis_path), str(manifest_path): sha(manifest_path), str(status_path): sha(status_path)}
    archive = {'analysis.json': analysis_path, 'campaign-manifest.json': manifest_path, 'terminal-status.json': status_path,
               'plot_shoulder_docking_benchmark.py': Path(__file__).resolve()}
    for name, digest in manifest['input_sha256'].items():
        path = campaign/'provenance'/name
        require(sha(path) == digest, f'Frozen campaign input changed: {name}')
        inputs[str(path)] = digest
        if path.suffix == '.py': archive['observer-sources/'+name] = path
    observer = campaign/'provenance/analyze_shoulder_docking_benchmark.py'
    require(sha(observer) == result['analyzer_sha256'], 'Completed audit used a different observer')
    reference = manifest['physical_reference']; reference_path = Path(reference['source_path'])
    require(sha(reference_path) == reference['source_sha256'] == manifest['input_sha256']['physical-reference-assessment.json'], 'Frozen independent reference identity changed')
    inputs[str(reference_path)] = reference['source_sha256']
    archive['importance-reference.json'] = reference_path
    probabilities = np.array([reference['regions'][name]['probability'] for name in LABELS])
    require(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0), 'Invalid reference probabilities')
    near(probabilities.sum(), 1.)
    reference_se = np.array([reference['regions'][name]['observed_delta_method_SE'] for name in LABELS])
    require(np.all(np.isfinite(reference_se)) and np.all(reference_se >= 0), 'Invalid observed reference errors')
    jobs = {j['id']: j for j in manifest['jobs']}
    require(len(jobs) == len(manifest['jobs']) == len(result['runs']) == len(status['jobs']) == 12, 'Need all 12 distinct pilot runs')
    require({j['id'] for j in status['jobs']} == set(jobs), 'Terminal run identities differ')
    require(all(j['status'] == 'complete' and type(j['exit_code']) is int and j['exit_code'] == 0 for j in status['jobs']), 'A physical job failed')
    require({r['id'] for r in result['runs']} == set(jobs), 'Summary run identities differ')
    records = []
    for summary in result['runs']:
        job = jobs[summary['id']]; path = analysis_path.parent/'runs'/job['id']/'analysis.json'; detail = read(path)
        require(detail['passed'] is True and all(detail[key] == value for key, value in summary.items()), 'Per-run audit disagrees with completed summary')
        require(all(detail[key] == job[key] for key in ('id', 'mode', 'start', 'replicate', 'seed')), 'Run identity/stream differs')
        require(detail['burn_cycles'] == burn and detail['retained_cycles'] == retained, 'Post-burn record changed')
        require(detail['audit']['all_moves_replayed'] == cycles*manifest['attempts_per_cycle'], 'Incomplete attempted-move audit')
        require(detail['audit']['independent_geometry_frames'] == retained, 'Incomplete contact-frame audit')
        inputs[str(path)] = sha(path); archive['runs/'+job['id']+'/analysis.json'] = path
        # Identity checks only; no density, overlap or acceptance calculation.
        for filename, digest in detail['source_sha256'].items():
            source = Path(job['directory'])/filename
            require(sha(source) == digest, f'Previously audited source bytes changed: {source}')
            inputs[str(source)] = digest
        require(sha(job['config']) == job['config_sha256'], 'Frozen per-run configuration changed')
        inputs[job['config']] = job['config_sha256']
        archive['configs/'+job['id']+'.json'] = Path(job['config'])
        rows = detail['q_rows']
        require([r['cycle'] for r in rows] == list(range(cycles+1)), 'Retained timeline is incomplete or thinned')
        kept = rows[burn+1:]
        labels = np.array([LABELS.index(r['label']) for r in kept], dtype=int)
        require(all(1 < r['q'] < 2 for r in kept), 'A retained frame is outside the audited shoulder')
        occupancy = np.bincount(labels, minlength=len(LABELS))/len(labels)
        first = np.bincount(labels[:half], minlength=len(LABELS))/half
        last = np.bincount(labels[half:], minlength=len(LABELS))/(retained-half)
        for i, name in enumerate(LABELS):
            near(occupancy[i], detail['occupancy'][name]); near(first[i], detail['first_half_occupancy'][name]); near(last[i], detail['last_half_occupancy'][name])
        half_tv = .5*float(np.abs(first-last).sum()); reference_tv = .5*float(np.abs(occupancy-probabilities).sum())
        near(half_tv, detail['half_occupancy_total_variation']); near(reference_tv, detail['reference_comparison']['total_variation'])
        cpu = detail['sampler_cpu_seconds_after_burn']
        near(cpu, rows[-1]['sampler_cpu_seconds']-rows[burn]['sampler_cpu_seconds'])
        require(math.isfinite(cpu) and cpu > 0, 'Invalid full post-burn CPU denominator')
        contact = detail['contact_incidence']; ess = contact['joint_apparent_effective_count']
        require(contact['samples'] == ess['samples'] == len(kept) and contact['stride_cycles'] == 1, 'Contact descriptor uses a different record')
        require([r['cycle'] for r in contact['rows']] == [r['cycle'] for r in kept], 'Contact-frame cycles differ')
        apparent = ess['apparent_ess']; rate = ess['apparent_ess_per_sampler_cpu_second']
        if apparent is None:
            require(rate is None and ess.get('reason'), 'Constant descriptor must remain unresolved')
        else:
            require(math.isfinite(apparent) and 0 < apparent <= len(kept), 'Invalid apparent contact ESS')
            near(rate, apparent/cpu)
        records.append(dict(id=job['id'], mode=job['mode'], start=job['start'], replicate=job['replicate'], seed=job['seed'],
            labels=labels.tolist(), occupancy=occupancy.tolist(), first_half_occupancy=first.tolist(), last_half_occupancy=last.tolist(),
            half_total_variation=half_tv, reference_total_variation=reference_tv, full_postburn_CPU_seconds=cpu,
            apparent_joint_contact_ESS=apparent, apparent_joint_contact_ESS_per_CPU_second=rate,
            unresolved_reason=ess.get('reason'), contact_descriptor=contact['descriptor']))
    expected = {(mode, start, replicate) for mode in MODES for start in STARTS for replicate in (0, 1)}
    require({(r['mode'], r['start'], r['replicate']) for r in records} == expected, 'Missing method/initialization replicate')
    records.sort(key=lambda r: (MODES.index(r['mode']), STARTS.index(r['start']), r['replicate']))
    initialization = {}
    for mode in MODES:
        means = {start: np.mean([r['occupancy'] for r in records if r['mode'] == mode and r['start'] == start], axis=0) for start in STARTS}
        initialization[mode] = .5*float(np.abs(means['direct']-means['geometry']).sum())
        near(initialization[mode], result['initialization_comparison'][mode]['initialization_mean_total_variation'])
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    return dict(complete=True, records=records, cycles=list(range(burn+1, cycles+1)), burn_cycles=burn,
        total_cycles=cycles, retained_cycles=retained,
        reference_probabilities=probabilities.tolist(), reference_observed_SE=reference_se.tolist(),
        reference_source=reference, initialization_total_variation=initialization, scope=SCOPE,
        efficiency_conclusion=result['efficiency_conclusion']), inputs, archive


def axes_style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=grid, color='#ccd0d5', alpha=.45, linewidth=.7)
    ax.set_axisbelow(True)


def xpos(record):
    return MODES.index(record['mode'])+OFFSETS[(record['start'], record['replicate'])]


def start_handles():
    return [Line2D([], [], marker=START_MARKERS[s], color=START_COLORS[s], linestyle='none', markersize=6,
                   label=f'{s.capitalize()} start (two runs/mode)') for s in STARTS]


def figure(data):
    """All points are individual runs; no best-run selection or ESS pooling."""
    records = data['records']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10})
    fig = plt.figure(figsize=(15.6, 12.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.35, 1., 1.15], left=.105, right=.985, top=.895, bottom=.16, hspace=.65)
    timeline = fig.add_subplot(outer[0])
    array = np.array([r['labels'] for r in records], dtype=int)
    timeline.imshow(array, origin='upper', aspect='auto', interpolation='none', cmap=ListedColormap(LABEL_COLORS),
        vmin=-.5, vmax=4.5, extent=(data['cycles'][0]-.5, data['cycles'][-1]+.5, len(records)-.5, -.5))
    timeline.set_yticks(range(12), [f"{MODE_TITLES[MODES.index(r['mode'])]} · {r['start']} {r['replicate']+1}" for r in records], fontsize=8)
    for boundary in (3.5, 7.5): timeline.axhline(boundary, color='white', linewidth=2.5)
    timeline.set_xlabel(f"Retained cycle endpoint (fixed burn: cycles 0–{data['burn_cycles']:,}; every later cycle retained)")
    timeline.set_title('A  Physical-label histories, including repeated states', loc='left', fontweight='bold', pad=12)
    timeline.spines[['top', 'right']].set_visible(False)
    occupancy_grid = outer[1].subgridspec(1, 5, wspace=.18)
    occupancy_axes = []
    for i, title in enumerate(LABEL_TITLES):
        ax = fig.add_subplot(occupancy_grid[i]); occupancy_axes.append(ax)
        probability, error = data['reference_probabilities'][i], data['reference_observed_SE'][i]
        ax.axhspan(max(0., probability-error), min(1., probability+error), color='#555b63', alpha=.14, zorder=0)
        ax.axhline(probability, color='#535861', linestyle='--', linewidth=1.1, zorder=1)
        for r in records:
            ax.scatter(xpos(r), r['occupancy'][i], marker=START_MARKERS[r['start']], s=28,
                color=START_COLORS[r['start']], edgecolor='white', linewidth=.4, zorder=3)
        ax.set(xlim=(-.43, 2.43), ylim=(-.025, 1.025), xticks=range(3), xticklabels=['Local', 'c = 0', 'c = 0.9'])
        ax.set_title(title, color=LABEL_COLORS[i] if i != 4 else '#616772', pad=7)
        axes_style(ax)
        if i: ax.tick_params(labelleft=False)
        else: ax.set_ylabel('Fraction of all retained cycles')
    pos = occupancy_axes[0].get_position()
    fig.text(pos.x0, pos.y1+.041, 'B  Individual-run occupancies against the frozen independent importance reference', weight='bold', fontsize=10)
    fig.text(pos.x0, pos.y1+.022, 'Dashed line / gray band: finite reference mean ± one observed delta-method SE; no MCMC error bars inferred.', fontsize=8.5, color='#505760')

    bottom = outer[2].subgridspec(1, 2, wspace=.24)
    ess_ax, disagreement_ax = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])
    rates = [r['apparent_joint_contact_ESS_per_CPU_second'] for r in records if r['apparent_joint_contact_ESS_per_CPU_second'] is not None]
    if rates:
        low, high = min(rates), max(rates)
        if high/low > 30:
            ess_ax.set_yscale('log'); ess_ax.set_ylim(low/1.7, high*1.8)
        else:
            ess_ax.set_ylim(0., high*1.2)
    else:
        ess_ax.set_ylim(0, 1); ess_ax.set_yticks([])
    for r in records:
        value = r['apparent_joint_contact_ESS_per_CPU_second']; x = xpos(r)
        if value is None:
            ess_ax.scatter(x, -.105, marker='x', color=START_COLORS[r['start']], s=36,
                transform=ess_ax.get_xaxis_transform(), clip_on=False, linewidth=1.5)
        else:
            ess_ax.scatter(x, value, marker=START_MARKERS[r['start']], color=START_COLORS[r['start']], s=42, edgecolor='white', linewidth=.5, zorder=3)
    ess_ax.set_title('C  Apparent joint atomic-contact ESS / full post-burn CPU', loc='left', fontweight='bold', pad=12)
    ess_ax.set_ylabel('Apparent contact ESS / sampling CPU second')
    ess_ax.set(xlim=(-.45, 2.45), xticks=range(3), xticklabels=MODE_TITLES)
    axes_style(ess_ax)
    ess_ax.text(.0, -.25, 'A+B atom-incidence trace statistic; four individual runs per mode.\nConstant descriptor: × below axis means unresolved, not zero or infinite ESS.',
        transform=ess_ax.transAxes, fontsize=8.2, va='top', color='#505760')
    for r in records:
        x, color, marker = xpos(r), START_COLORS[r['start']], START_MARKERS[r['start']]
        disagreement_ax.scatter(x-.018, r['reference_total_variation'], marker=marker, s=36, color=color, edgecolor=color, linewidth=.7, zorder=3)
        disagreement_ax.scatter(x+.018, r['half_total_variation'], marker=marker, s=43, facecolor='none', edgecolor=color, linewidth=1.2, zorder=3)
    disagreement_ax.scatter(range(3), [data['initialization_total_variation'][m] for m in MODES], marker='D', s=38, color='#24272b', zorder=4)
    disagreement_ax.set_title('D  Occupancy discrepancies remain separate from ESS', loc='left', fontweight='bold', pad=12)
    disagreement_ax.set(xlim=(-.45, 2.45), ylim=(-.025, 1.025), xticks=range(3), xticklabels=MODE_TITLES,
                       ylabel='Total variation of five-label occupancy')
    axes_style(disagreement_ax)
    disagreement_ax.text(.0, -.25, 'Filled: run vs reference. Open: first vs second retained half.\nBlack diamond: initialization means (two runs per start); no pass threshold.',
        transform=disagreement_ax.transAxes, fontsize=8.2, va='top', color='#505760')
    fig.suptitle('Conditional shoulder sampling: contact relaxation and occupancy coverage', fontsize=16, y=.982)
    fig.text(.54, .951, f"{len(records)} independent runs · two AB neighbors fixed · strict original 1 < q < 2 · {len(data['cycles']):,} retained cycles after fixed burn", ha='center', fontsize=10)
    legend = [Patch(facecolor=c, label=t) for c,t in zip(LABEL_COLORS, LABEL_TITLES)]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.53, .938), ncol=5, frameon=False, fontsize=9)
    fig.legend(handles=start_handles(), loc='lower center', bbox_to_anchor=(.54, .007), ncol=2, frameon=False, fontsize=9)
    fig.text(.54, .047, 'All cycle endpoints and self-loops retained. Apparent ESS is descriptive until occupancy, initialization and stationarity checks agree.', ha='center', fontsize=9)
    fig.text(.54, .031, 'Atomic contact incidence is not crystallographic patch registration. This conditional experiment does not test assembly or physical kinetic rates.', ha='center', fontsize=8.6, color='#505760')
    return fig


def plot(analysis_path, campaign, out):
    analysis_path, campaign, out = [Path(p).resolve() for p in (analysis_path, campaign, out)]
    require(not out.exists(), 'Fresh figure output directory required')
    data, inputs, archives = load_completed(analysis_path, campaign)
    fig = figure(data)
    out.mkdir(parents=True); provenance = out/'provenance'; provenance.mkdir()
    for name, path in archives.items():
        target = provenance/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        require(sha(target) == inputs[str(path)], 'Figure source changed during rendering')
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(out/f'shoulder-docking-benchmark.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    write(out/'plot-data.json', data)
    write(out/'provenance.json', dict(complete=True, input_sha256=inputs,
        archived_sha256={str(p.relative_to(provenance)): sha(p) for p in provenance.rglob('*') if p.is_file()},
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__, scope=SCOPE,
        verification='Completed audit/terminal status, source byte identities and saved-observable arithmetic checked; no physical audit replay.'))
    print(out/'shoulder-docking-benchmark.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); plot(args.analysis, args.campaign, args.out)
