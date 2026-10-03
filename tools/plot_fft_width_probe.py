#!/usr/bin/env python3
"""Plot completed FFT width controls using audited summaries only; no sampling.

Refuses incomplete/unreviewed inputs and never reads proposal or bath ledgers.
Historical tau=1 controls are descriptive: their streams and clouds differ.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO/'results'
SCALES = [.125, .25, .5, 1.]
METHODS = ['unguided', 'm4']
COLORS = {'unguided': '#377ba1', 'm4': '#bd7826'}
LABELS = {'unguided': 'Unguided', 'm4': 'Overlap guide (m=4)'}
HISTORICAL = {
    'passive_unguided': ('factorized-dimer-probe-20261002', '3facf648519de1e32433c0d1514f66b47716bc65400c804b87414bba22e9a450'),
    'physical_unguided': ('factorized-dimer-physical-20261002', '9b7fc8916c8f3e32a15dc65c2b9f025488e39b7b4de44179b9bf069a82999c8b'),
    'passive_m4': ('auxiliary-overlap-probe-20261003', '0f8a299d5efd945466fecdc54530b886c41006a94f693d42794e4d8f690b06c3'),
    'physical_m4': ('auxiliary-overlap-physical-20261003', '2d091e2241202cc38555b3914a7cb3244a1bcb8c55b9d4758d0dbe72eff65c55'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def reviewed_analysis(path, files, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, 'Historical analysis hash changed: '+str(path))
    report = json.loads(path.read_text())
    require(report.get('complete') is True and report.get('passed') is True and not report.get('failures'),
            'Analysis did not pass: '+str(path))
    review_path = path.parent/'completed-review.json'
    review = json.loads(review_path.read_text())
    require(review.get('complete') is True and review.get('passed') is True and
            review.get('output_hashes', {}).get('analysis.json') == digest,
            'Missing matching completed review: '+str(path))
    files[str(path)] = digest
    files[str(review_path)] = sha(review_path)
    return report


def log_value(value):
    if value == '-inf':
        return -math.inf
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid saved log factor')
    return float(value)


def rows_for(report, atlas, method):
    rows = [row for row in report['rows'] if row['atlas'] == atlas and row['method'] == method]
    keys = [(r['case'], r['attempt']) for r in rows]
    require(len(rows) == len(set(keys)) == 256, 'Missing, duplicate or extra outer rows')
    cases = {r['case'] for r in rows}
    require(len(cases) == 8 and set(keys) == {(case, j) for case in cases for j in range(32)},
            'Different fixed source/slot allocation')
    return rows


def summarize_pair(passive, physical, tau, method):
    pmap = {(r['case'], r['attempt']): r for r in passive}
    qmap = {(r['case'], r['attempt']): r for r in physical}
    require(pmap.keys() == qmap.keys(), 'Passive/physical keys differ')
    candidates, accepts, alpha = 0, 0, []
    full, bath = [], []
    gained = lost = 0
    for key, physical_row in qmap.items():
        p = pmap[key]
        exists = p['candidate'] is not None
        require(type(physical_row['candidate']) is bool and physical_row['candidate'] == exists,
                'Candidate/null mismatch')
        require(type(physical_row['accepted']) is bool and (exists or not physical_row['accepted']),
                'Null acquired an acceptance')
        a = physical_row['acceptance_probability']
        require(type(a) in (int, float) and math.isfinite(a) and 0 <= a <= 1,
                'Invalid conditional acceptance')
        require(exists or a == 0, 'Null acquired a conditional acceptance probability')
        candidates += exists
        accepts += physical_row['accepted']
        alpha.append(a)
        if exists:
            f = log_value(physical_row.get('log_full_f_correction', physical_row['log_proposal_correction']))
            b = log_value(physical_row['log_depletion_factor'])
            require(math.isclose(f, log_value(p['candidate']['log_reverse_forward']), rel_tol=2e-10, abs_tol=2e-7),
                    'Audited passive/physical full-F term differs')
            full.append(f)
            bath.append(b)
        if physical_row['accepted']:
            gained += physical_row['gained_contact_edges']
            lost += physical_row['lost_contact_edges']
    return dict(tau=tau, method=method, historical=tau == 1., outer_attempts=256,
        candidates=candidates, candidate_fraction=candidates/256,
        accepted=accepts, sum_conditional_alpha=math.fsum(alpha),
        median_total_full_f=statistics.median(full) if full else None,
        median_bath_logweight=statistics.median(bath) if bath else None,
        accepted_gained_contact_edges=gained, accepted_lost_contact_edges=lost)


def prepare_data(passive_path, physical_path):
    files = {}
    passive = reviewed_analysis(passive_path, files)
    physical = reviewed_analysis(physical_path, files)
    require(passive['schema'] == 'fft-width-independent-analysis-v1', 'Wrong fresh passive analysis')
    require(physical['schema'] == 'auxiliary-overlap-physical-independent-audit-v1', 'Wrong fresh physical analysis')
    require(passive['summary']['outer_attempts'] == physical['summary']['outer_attempts'] == 1536,
            'Fresh allocation incomplete')
    historical = {name: reviewed_analysis(RESULTS/directory/'analysis.json', files, digest)
                  for name, (directory, digest) in HISTORICAL.items()}
    entries = []
    for tau in SCALES:
        for method in METHODS:
            if tau == 1.:
                atlas = 'blind_fft512slots'
                source_method = 'factorized' if method == 'unguided' else 'm4'
                prows = rows_for(historical['passive_'+method], atlas, source_method)
                qrows = rows_for(historical['physical_'+method], atlas, source_method)
            else:
                atlas = 'blind_fft512slots_tau'+str(tau).replace('.', 'p')
                prows = rows_for(passive, atlas, method)
                qrows = rows_for(physical, atlas, method)
                require(all(r['tau'] == tau for r in prows), 'Wrong covariance scale')
            entries.append(summarize_pair(prows, qrows, tau, method))
    return entries, files


def axes_style(ax):
    ax.set_xticks(range(4), ['0.125', '0.25', '0.5', '1\nhistorical'])
    ax.set_xlim(-.5, 3.5)
    ax.set_xlabel('Covariance width multiplier τ; covariance = τ²Σ')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e3e7eb', linewidth=.7)
    ax.axvspan(2.5, 3.5, color='#eff1f3', zorder=0)


def draw(entries, output):
    groups = {(r['tau'], r['method']): r for r in entries}
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10}):
        fig, axes = plt.subplots(2, 2, figsize=(13.7, 11.4))
        fig.subplots_adjust(left=.075, right=.965, bottom=.26, top=.80, wspace=.30, hspace=.54)
        for ax in axes.flat:
            axes_style(ax)
        for method, offset in [('unguided', -.19), ('m4', .19)]:
            rows = [groups[tau, method] for tau in SCALES]
            xs = [i+offset for i in range(4)]
            color = COLORS[method]
            axes[0, 0].bar(xs, [100*r['candidate_fraction'] for r in rows], width=.34,
                           color=color, label=LABELS[method], zorder=3)
            for x, row in zip(xs, rows):
                axes[0, 0].annotate(str(row['candidates']), (x, 100*row['candidate_fraction']),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
            axes[0, 1].bar(xs, [r['accepted'] for r in rows], width=.34, color=color, zorder=3)
            for x, row in zip(xs, rows):
                if row['accepted'] == 0:
                    axes[0, 1].plot(x, 0, 'o', ms=5, color=color, markerfacecolor='white', zorder=4)
                axes[0, 1].annotate(f"{row['accepted']}/256", (x, row['accepted']),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8.5)
            axes[1, 0].bar(xs, [r['sum_conditional_alpha'] for r in rows], width=.34, color=color, zorder=3)
            for x, row in zip(xs, rows):
                value = row['sum_conditional_alpha']
                if value == 0:
                    axes[1, 0].plot(x, 0, 'o', ms=5, color=color, markerfacecolor='white', zorder=4)
                axes[1, 0].annotate(f'{value:.3g}', (x, value), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8.5)
            for key, marker, term_offset, term_label in [
                ('median_total_full_f', 'o', -.055, 'Full F: root + internal'),
                ('median_bath_logweight', 's', .055, 'Realized bath logweight')]:
                for x, row in zip(xs, rows):
                    value = row[key]
                    if value is None:
                        continue
                    require(math.isfinite(value), 'Infinite candidate median needs explicit plot design')
                    axes[1, 1].plot(x+term_offset, value, marker, color=color, markersize=6,
                                   markerfacecolor=color if marker == 'o' else 'white', zorder=4)
        axes[0, 0].set(ylabel='Candidates / all attempts (%)', ylim=(0, 108), yticks=[0, 25, 50, 75, 100])
        axes[0, 0].set_title('A   Geometrically eligible candidates', loc='left', pad=12)
        accepts_max = max(r['accepted'] for r in entries)
        axes[0, 1].set(ylabel='Realized accepts / 256 outers', ylim=(0, max(3, accepts_max*1.20+1)))
        axes[0, 1].set_title('B   Physical acceptance, including all nulls', loc='left', pad=12)
        axes[1, 0].set_yscale('symlog', linthresh=1e-6, linscale=.55)
        alpha_max = max(r['sum_conditional_alpha'] for r in entries)
        axes[1, 0].set(ylabel='Σ conditional α  (diagnostic, not a rate)', ylim=(0, max(1e-5, alpha_max*12)))
        axes[1, 0].set_title('C   Conditional acceptance mass', loc='left', pad=12)
        axes[1, 1].axhline(0, color='#66717e', linestyle='--', linewidth=.8)
        axes[1, 1].set_ylabel('Candidate median log factor')
        axes[1, 1].set_title('D   Proposal and bath contributions', loc='left', pad=12)
        axes[1, 1].legend(handles=[
            Line2D([], [], marker='o', linestyle='None', color='#4e5964', label='Full F: root + internal'),
            Line2D([], [], marker='s', linestyle='None', color='#4e5964', markerfacecolor='white', label='Bath logweight')],
            loc='best', frameon=False, fontsize=9)
        fig.suptitle('Narrower FFT contact proposals: feasibility and physical acceptance',
                     x=.055, y=.966, ha='left', fontsize=17, fontweight='bold')
        fig.text(.055, .925, 'Blind FFT atlas · 256 fixed-source outers per arm · radius 1.4 Å · activity 0.0275 Å⁻³ · 500 μM.', fontsize=11)
        fig.text(.055, .894, 'Every covariance entry is scaled; centers, weights and 50% uniform defense are unchanged. No physical state trajectory is propagated.', fontsize=10, color='#394451')
        fig.legend(handles=[Line2D([], [], color=COLORS[m], linewidth=7, label=LABELS[m]) for m in METHODS],
                   loc='upper left', bbox_to_anchor=(.055, .872), ncol=2, frameon=False, fontsize=10)
        fig.text(.055, .065,
            'τ = 1 is an archived, unpaired control: different proposal streams, guidance clouds and timing. Fresh widths share case-slot clouds.\n'
            'Panel C sums min(1, exp(log MH ratio)) conditional on saved poses and realized bath clouds; it is not an event rate or equilibrium weight.\n'
            'Panel D uses all surviving candidates in each arm; candidate populations differ. Medians are not additive; no confidence intervals are shown.\n'
            'Full MH logratio also includes the m=4 auxiliary-target correction once. Fixed reset tests do not establish mixing, contact ESS or assembly.',
            fontsize=9.3, color='#394451', linespacing=1.65)
        fig.savefig(output, dpi=180, metadata={'Title': 'FFT covariance width passive and physical controls'})
        plt.close(fig)


def serial(value):
    if isinstance(value, float) and value == -math.inf:
        return '-inf'
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passive', type=Path, default=RESULTS/'fft-width-probe-20261003/analysis.json')
    parser.add_argument('--physical', type=Path, default=RESULTS/'fft-width-physical-20261003/analysis.json')
    parser.add_argument('--output', type=Path, default=REPO/'docs/figures/fft-width-probe.png')
    parser.add_argument('--binding', type=Path, default=RESULTS/'fft-width-physical-20261003/plot-binding.json')
    parser.add_argument('--receipt', type=Path, default=RESULTS/'fft-width-physical-20261003/plot-receipt.json')
    args = parser.parse_args()
    require(not any(p.exists() for p in (args.output, args.binding, args.receipt)), 'Refusing to overwrite a plotting attempt')
    entries, files = prepare_data(args.passive, args.physical)
    files[str(Path(__file__).resolve())] = sha(__file__)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            files[str(path)] = sha(path)
    write_new(args.binding, dict(schema='fft-width-plot-binding-v1', complete=True,
        recorded_utc=datetime.now(timezone.utc).isoformat(), input_hashes=files,
        matplotlib_version=matplotlib.__version__, python=sys.version,
        definition='All256outersperarm; freshtau.125,.25,.5 and historicaltau1; no new geometric queries or probability draws.'))
    receipt = dict(schema='fft-width-plot-receipt-v1', complete=False, passed=False,
        binding_sha256=sha(args.binding), failures=[], arms=serial(entries))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        draw(entries, args.output)
        require(all(sha(path) == digest for path, digest in files.items()), 'Plot input changed during rendering')
        receipt.update(complete=True, passed=True, output=str(args.output.resolve()), output_sha256=sha(args.output))
    except Exception as error:
        receipt['failures'].append(str(error))
    write_new(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
