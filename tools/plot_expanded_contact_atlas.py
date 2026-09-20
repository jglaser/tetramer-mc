#!/usr/bin/env python3
"""Plot fixed-prefix sensitivity and disjoint contact weights from a completed audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point(ax, x, row, color, lower_limit, marker='o', markersize=6):
    """Transform Q +/- SE, retaining a visible open lower limit when SE >= Q."""
    value, error = row['logQ'], row['row_RSE']
    if value is None:
        ax.annotate('unresolved', (x, lower_limit+.15), rotation=90,
                    ha='center', va='bottom', fontsize=8, color=color)
        return
    assert math.isfinite(value) and error is not None and math.isfinite(error) and error >= 0
    lower = value+math.log1p(-error) if error < 1 else lower_limit+.12
    upper = value+math.log1p(error)
    ax.errorbar(x, value, yerr=np.array([[max(0., value-lower)], [upper-value]]),
                fmt=marker, color=color, markersize=markersize, capsize=4, elinewidth=1.6, zorder=4)
    if error >= 1:
        ax.annotate('', xy=(x, lower_limit+.03), xytext=(x, lower_limit+.25),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.3))


def limits(rows):
    values = [r['logQ'] for r in rows if r['logQ'] is not None]
    assert values, 'Cannot plot a wholly unobserved panel'
    lows, highs = [], []
    for row in rows:
        if row['logQ'] is None:
            continue
        y, rse = row['logQ'], row['row_RSE']
        assert rse is not None and math.isfinite(rse) and rse >= 0
        lows.append(y+math.log1p(-rse) if rse < 1 else y-2.)
        highs.append(y+math.log1p(rse))
    span = max(1., max(highs)-min(lows))
    return min(lows)-.16*span, max(highs)+.25*span


def plot(analysis_path, out):
    assert not out.exists(), 'Use a fresh figure directory'
    result = read(analysis_path)
    assert result['complete'] and result['original_q_window'] == dict(
        minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
    assert result['prefix_counts'] == [8192, 16384, 32768]
    protocol_path = analysis_path.parent/'provenance/protocol.json'
    assert sha(protocol_path) == result['archived_sha256']['protocol.json']
    protocol = read(protocol_path)
    physical = protocol['physical']
    assert physical['depletant_radius'] == 1.5 and physical['reservoir_density'] == .035
    assert physical['capture_radius'] == 18. and len(physical['fixed_poses']) == 2
    arms = {c['arm']:c for c in result['campaigns']}
    assert set(arms) == {'narrow', 'broad'}
    references = {r['neighborhood']:r['historical_multiscale_reference']
                  for r in result['reference_comparisons'] if r['radius_A'] == 2.}
    assert set(references) == {'old_peak', 'new_peak'}
    assert result['center_separation']['two_radius_2_balls_disjoint']
    final_keys = ('old_ball', 'new_ball', 'outside_both')
    prefix_rows = {}; final_rows = {}
    for name, campaign in arms.items():
        declared = next(a for a in protocol['arms'] if a['name'] == name)
        assert campaign['root'] == declared['output'] and declared['populations'] == 8
        assert declared['samples_per_population'] == 32768
        prefix_rows[name] = [p['physical']['full'] for p in campaign['prefixes']]
        assert [p['samples_per_population'] for p in campaign['prefixes']] == result['prefix_counts']
        assert [p['draws'] for p in prefix_rows[name]] == [8*n for n in result['prefix_counts']]
        assert prefix_rows[name][-1] == campaign['physical']['full']
        final_rows[name] = [campaign['disjoint_threeway']['physical'][key] for key in final_keys]
        assert campaign['disjoint_threeway']['partition_checks']['physical']['complete_row_partition']
    colors = dict(narrow='#14795c', broad='#bc722a', reference='#657181')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6), gridspec_kw={'width_ratios':[1., 1.25]})
    left_rows = [row for rows in prefix_rows.values() for row in rows]
    right_rows = [row for rows in final_rows.values() for row in rows]+list(references.values())
    for ax, rows in zip(axes, (left_rows, right_rows)):
        ax.set_ylim(*limits(rows)); ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False); ax.set_ylabel(r'$\log Q$')
    for name, offset in (('narrow', -.035), ('broad', .035)):
        rows = prefix_rows[name]; x = np.arange(3)+offset
        axes[0].plot(x, [np.nan if r['logQ'] is None else r['logQ'] for r in rows],
                     color=colors[name], alpha=.65, lw=1.5, zorder=2)
        for xpos, row in zip(x, rows):
            point(axes[0], xpos, row, colors[name], axes[0].get_ylim()[0])
    axes[0].set_xticks(range(3), ['8,192', '16,384', '32,768'])
    axes[0].set_xlim(-.35, 2.4)
    axes[0].set_xlabel('Unconditional draws per population × 8 populations')
    axes[0].set_title('Full region: sensitivity to fixed sample prefixes', fontsize=11.5, pad=23)
    axes[0].text(.5, 1.018, 'The three points share samples; they are correlated.',
                 transform=axes[0].transAxes, ha='center', fontsize=8.8, color='#4b5560')
    for name, offset in (('narrow', -.14), ('broad', .14)):
        for x, row in enumerate(final_rows[name]):
            point(axes[1], x+offset, row, colors[name], axes[1].get_ylim()[0])
    for x, label in enumerate(('old_peak', 'new_peak')):
        point(axes[1], x, references[label], colors['reference'], axes[1].get_ylim()[0], marker='D', markersize=5)
    axes[1].set_xticks(range(3), ['Old contact\nρ ≤ 2 Å', 'New contact\nρ ≤ 2 Å', 'Outside both\nremaining poses'])
    axes[1].set_xlim(-.5, 2.5)
    axes[1].set_title('Final estimates: three disjoint pieces', fontsize=11.5, pad=23)
    axes[1].text(.5, 1.018, 'Old + new + outside = full 2 ≤ q < 5 region',
                 transform=axes[1].transAxes, ha='center', fontsize=8.8, color='#4b5560')
    handles = [Line2D([], [], color=colors[name], marker='o', lw=1.4,
                       label=f'{name.capitalize()} geometry: σ = {sd} Å')
               for name, sd in (('narrow', '0.2'), ('broad', '0.4'))]
    handles.append(Line2D([], [], color=colors['reference'], marker='D', lw=0,
                          label='Historical local reference (calibration)'))
    fig.suptitle('Expanded contact atlas: global sensitivity and local calibration', fontsize=14, y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), frameon=False, ncol=3, fontsize=9)
    fig.text(.5, .165, 'Fixed AB neighbors • depletant radius 1.5 Å • activity 0.035 Å⁻³ • capture radius 18 Å • normalized Haar measure',
             ha='center', fontsize=9)
    fig.text(.5, .116, 'Bars show log(Q ± one observed row SE), not tail confidence bounds. Down arrows indicate Q − SE ≤ 0.',
             ha='center', fontsize=8.7)
    fig.text(.5, .072, 'Historical local samples informed training. The two proposal arms remain separate; no arm or nested-region pooling.',
             ha='center', fontsize=8.7)
    fig.text(.5, .028, 'Complete proposal support and apparently stable prefixes cannot exclude unseen high-weight contacts or establish assembly.',
             ha='center', fontsize=8.7)
    fig.tight_layout(rect=(0., .215, 1., .86), w_pad=3.)
    out.mkdir(parents=True)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'expanded-contact-atlas.{extension}', dpi=180, bbox_inches='tight')
    files = {'input-analysis.json':analysis_path, 'input-protocol.json':protocol_path, 'plotter.py':Path(__file__)}
    for name, path in files.items():
        shutil.copy2(path, out/name)
    provenance = dict(input_sha256={str(p):sha(p) for p in files.values()},
        archived_sha256={name:sha(out/name) for name in files},
        output_sha256={f'expanded-contact-atlas.{ext}':sha(out/f'expanded-contact-atlas.{ext}') for ext in ('png','svg')},
        prefix_plotted_rows=prefix_rows, final_disjoint_plotted_rows=final_rows,
        historical_radius_two_reference_rows=references,
        error_bars='One observed row SE on linear Q mapped to log; lower arrows mean Q-SE<=0, not a finite lower bound.',
        interpretation='Correlated fixed prefixes and separate width arms; matched historical references are training-used calibration. No global convergence, mixing or assembly claim.')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    plt.close(fig)
    print(out/'expanded-contact-atlas.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); plot(args.analysis.resolve(), args.out.resolve())
