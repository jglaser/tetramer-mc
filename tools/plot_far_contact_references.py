#!/usr/bin/env python3
"""Show local contact calibration separately from complete-domain estimates."""
import argparse
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_expanded_contact_atlas import limits, point, read, sha
from prepare_cayley_rms_cover import write, require


def plot(local_path, candidate_path, flat_path, out):
    require(not out.exists(), 'Use a fresh figure directory')
    local, candidates, flat = [read(p) for p in (local_path, candidate_path, flat_path)]
    require(all(a['complete'] for a in (local, candidates, flat)), 'Completed audits required')
    window = dict(minimum=5., maximum=37., lower_inclusive=True, upper_inclusive=False)
    require(local['original_q_window'] == flat['q_window'] == window, 'Different physical windows')
    for name, digest in local['archived_sha256'].items():
        require(sha(local_path.parent/'provenance'/name) == digest, 'Local archive changed')
    for name, digest in local['input_sha256'].items():
        require(sha(Path(name)) == digest, 'Local input changed')
    require(local['selection_audit']['candidate_sha256'] == sha(candidate_path), 'Candidate selection changed')
    require(sha(flat_path.parent/'protocol.json') == flat['protocol_sha256'], 'Flat protocol changed')
    require(sha(Path(flat['full_density_audit_path'])) == flat['full_density_audit_sha256'], 'Flat audit changed')
    cfg = read(local_path.parent/'provenance/config.json')
    flat_protocol = read(flat_path.parent/'protocol.json')
    require(all(cfg[k] == v for k, v in flat_protocol['physical_AB'].items()), 'Different AB physical targets')
    require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and len(cfg['fixed_poses']) == 2,
            'Unexpected physical parameters')
    campaigns = sorted(local['campaigns'], key=lambda c: c['radius_A'])
    require([c['radius_A'] for c in campaigns] == [.5, 1., 2.], 'Missing local radii')
    balls = [c['physical']['full'] for c in campaigns]
    shells = [c['physical'][f'radial_{i}'] for i, c in enumerate(campaigns)]
    combined = local['independent_shell_sum']['physical']['full']
    outside = local['independent_shell_sum']['physical']['old_r_gt_3']
    historical = {float(width): row for c in candidates['campaigns']
                  for width, row in c['far_statistics_by_width'].items()}
    require(set(historical) == {1., 2., 4.}, 'Missing historical width diagnostics')
    # Keep historical field extraction explicit; fail rather than substitute an
    # all-other (q>=1) statistic for the distant q>=5 region.
    rows = []
    for width in (1., 2., 4.):
        row = historical[width]
        rows.append(dict(logQ=row['logQ'], row_RSE=row['relative_se']))
    broad = dict(logQ=flat['physical']['log_normalizer'], row_RSE=flat['physical']['relative_SE'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), gridspec_kw={'width_ratios': [1.05, 1.1]})
    green, amber, gray = '#16735b', '#bb7429', '#56677d'
    for ax, panel_rows in zip(axes, (balls+shells+[combined, outside], rows+[broad, combined])):
        ax.set_ylim(*limits(panel_rows)); ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False); ax.set_ylabel(r'$\log[Q/(1\,\mathrm{Å}^3)]$')
    for i, (ball, shell) in enumerate(zip(balls, shells)):
        point(axes[0], i-.11, ball, green, axes[0].get_ylim()[0])
        point(axes[0], i+.11, shell, amber, axes[0].get_ylim()[0], marker='s')
    point(axes[0], 3, combined, green, axes[0].get_ylim()[0], marker='D')
    point(axes[0], 4, outside, gray, axes[0].get_ylim()[0], marker='D')
    axes[0].set_xticks(range(5), ['R = 0.5\n[0, 0.5]', 'R = 1\n(0.5, 1]', 'R = 2\n(1, 2]', 'Shell sum\nR ≤ 2', 'Shell sum\noutside old R3'])
    axes[0].set_xlim(-.5, 4.6)
    axes[0].set_title('Independent calibration around the discovered contact', fontsize=11, pad=25)
    axes[0].text(.5, 1.022, 'Top labels: geometric ball radius (Å). Lower labels: disjoint shells.',
                 transform=axes[0].transAxes, ha='center', fontsize=8.2, color='#4b5560')
    for i, row in enumerate(rows+[broad]):
        point(axes[1], i, row, gray if i < len(rows) else amber, axes[1].get_ylim()[0])
    q, rse = combined['logQ'], combined['row_RSE']
    if q is not None:
        axes[1].axhline(q, color=green, lw=1.3)
        if rse < 1:
            axes[1].axhspan(q+math.log1p(-rse), q+math.log1p(rse), color=green, alpha=.12)
        axes[1].text(.02, .96, 'Green: measured local R ≤ 2 contribution\nA subset of the far region; not added to these estimates.',
                     transform=axes[1].transAxes, color=green, va='top', fontsize=8.3)
    axes[1].set_xticks(range(4), ['Old guide\nwidth 1', 'Old guide\nwidth 2', 'Old guide\nwidth 4', 'Flat cover\n262,144 draws'])
    axes[1].set_xlim(-.5, 3.6)
    axes[1].set_title('Full far region: proposal sensitivity remains unresolved', fontsize=11, pad=25)
    axes[1].text(.5, 1.022, 'Each point targets all captured q ≥ 5, with both AB neighbors fixed.',
                 transform=axes[1].transAxes, ha='center', fontsize=8.2, color='#4b5560')
    handles = [Line2D([], [], color=color, marker=marker, lw=0, label=label)
               for color, marker, label in [(green, 'o', 'Whole local ball'), (amber, 's', 'Matching-radius shell'),
                  (green, 'D', 'Sum of independent disjoint shells')]]
    fig.suptitle('Narrow far contacts: complete support does not ensure resolved weight', fontsize=14, y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .17, 'Depletant radius 1.5 Å • activity 0.035 Å⁻³ • capture radius 18 Å • center volume × normalized SO(3) Haar', ha='center', fontsize=9)
    fig.text(.5, .118, 'Local campaigns: 4 × 16,384 draws per radius. Historical discovery draws are not reused as local observations.', ha='center', fontsize=8.7)
    fig.text(.5, .07, 'Bars show log(Q ± one observed row SE). Nested whole balls and overlapping full-domain estimates are never added.', ha='center', fontsize=8.7)
    fig.text(.5, .023, 'Observed errors cannot bound missed narrow contacts. The local band is an estimate, not a rigorous lower confidence bound.', ha='center', fontsize=8.7)
    fig.tight_layout(rect=(0, .22, 1, .86), w_pad=2.5)
    out.mkdir(parents=True)
    for ext in ('png', 'svg'): fig.savefig(out/f'far-contact-references.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    sources = dict(local=local_path, candidates=candidate_path, flat=flat_path,
        plotter=Path(__file__), plot_helpers=Path(__file__).with_name('plot_expanded_contact_atlas.py'))
    for name, path in sources.items(): shutil.copy2(path, out/f'{name}{path.suffix}')
    write(out/'provenance.json', dict(input_sha256={str(p): sha(p) for p in sources.values()},
        archived_sha256={f'{n}{p.suffix}': sha(out/f'{n}{p.suffix}') for n, p in sources.items()},
        output_sha256={f'far-contact-references.{ext}': sha(out/f'far-contact-references.{ext}') for ext in ('png', 'svg')},
        plotted=dict(balls=balls, shells=shells, local_sum=combined, outside_old_R3=outside, historical=rows, flat=broad),
        scope='Observed standard errors; separate local and complete-domain estimands. No unseen-tail or mixing claim.'))
    print(out/'far-contact-references.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('local', 'candidates', 'flat', 'out'): p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); plot(args.local.resolve(), args.candidates.resolve(), args.flat.resolve(), args.out.resolve())
