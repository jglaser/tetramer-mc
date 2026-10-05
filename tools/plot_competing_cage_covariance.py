"""Plot frozen training/heldout moment and coverage diagnostics; no new fits."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ellipse(ax, mean, covariance, color, label, linestyle='-'):
    values, vectors = np.linalg.eigh(covariance)
    if values[0] < -1e-10*max(1., abs(values[-1])):
        raise ValueError('Invalid projected covariance')
    width, height = 2*np.sqrt(chi2.ppf(.9, 2)*np.maximum(values[::-1], 0))
    direction = vectors[:, -1]
    ax.add_patch(Ellipse(mean, width, height,
        angle=np.degrees(np.arctan2(direction[1], direction[0])),
        fill=False, color=color, linewidth=1.8, linestyle=linestyle, label=label))
    ax.scatter(*mean, color=color, s=20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if sha(args.report) != args.expected_sha256:
        raise ValueError('Changed frozen fit report')
    report = json.loads(args.report.read_bytes())
    if not (report['schema'] == 'competing-cage-covariance-v1'
            and report['complete'] and report['passed']
            and report['heldout_fit_samples'] == 0
            and [c['cage_id'] for c in report['cages']] == [0, 1]):
        raise ValueError('Completed two-cage report required')
    if args.out.exists():
        raise ValueError('Fresh output directory required')
    args.out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    scales = np.asarray(report['scale_vector'])
    colors = ['#4477aa', '#ee7733', '#cc3311']
    for column, cage in enumerate(report['cages']):
        top, bottom = axes[:, column]
        fit = cage['fit']
        if fit is None:
            top.text(.5, .5, 'Chart failure: no fit', ha='center', va='center')
            bottom.axis('off')
            continue
        cov = np.asarray(fit['covariance'])/np.outer(scales, scales)
        _, vectors = np.linalg.eigh(cov)
        projection = vectors[:, -2:][:, ::-1]
        mean = np.asarray(fit['mean'])/scales
        ellipse(top, [0., 0.], projection.T@cov@projection,
                colors[0], 'Frozen training Gaussian')
        diagnostics = cage['likelihood_diagnostics']
        for stream, color in zip([2, 3], colors[1:]):
            rows = [r for r in cage['per_stream']
                    if r['stream'] == stream and r['phase'] == 'production']
            if len(rows) != 1:
                raise ValueError('Missing heldout stream')
            row = rows[0]
            if row['coordinate_status'] != 'complete':
                continue
            moment = row['map_coordinates_angstrom']
            m = np.asarray(moment['mean'])/scales
            c = np.asarray(moment['covariance'])/np.outer(scales, scales)
            ellipse(top, (m-mean)@projection, projection.T@c@projection,
                    color, f'Heldout stream {stream}: moments', '--')
        top.autoscale_view()
        top.set_aspect('equal', adjustable='datalim')
        top.set_xlabel('Training PC1 (scaled coordinates)')
        top.set_ylabel('Training PC2 (scaled coordinates)')
        top.set_title(f'Cage {cage["cage_id"]}: 90% Gaussian moment ellipses')
        top.legend(frameon=False, fontsize=8)
        bottom.plot([0, 1], [0, 1], color='#888888', linestyle=':', label='Gaussian expectation')
        for key, label, color in [('train', 'Training streams 0 + 1', colors[0]),
                                  ('2', 'Heldout stream 2', colors[1]),
                                  ('3', 'Heldout stream 3', colors[2])]:
            d = diagnostics[key]
            if 'nominal_gaussian_coverage' not in d:
                continue
            coverage = d['nominal_gaussian_coverage']
            bottom.plot([v['probability'] for v in coverage],
                        [v['fraction'] for v in coverage], 'o-', color=color, label=label)
        bottom.set(xlim=(.45, 1.01), ylim=(-.02, 1.02),
                   xlabel='Nominal six-dimensional Gaussian probability',
                   ylabel='Observed residence fraction inside ellipsoid',
                   title='Heldout coverage; no parameter updates')
        bottom.grid(alpha=.2)
        bottom.legend(frameon=False, fontsize=8)
    fig.suptitle('Two competing contact environments: frozen proposal fits\n'
                 'Fixed protein context, 1.5 Å depletants, z = 0.035 Å⁻³', fontsize=13)
    fig.savefig(args.out/'comparison.png', dpi=180)
    fig.savefig(args.out/'comparison.pdf')
    plt.close(fig)
    if sha(args.report) != args.expected_sha256:
        raise ValueError('Report changed during rendering')
    receipt = dict(complete=True, passed=True, source_report=str(args.report.resolve()),
        source_report_sha256=args.expected_sha256, source_sha256=sha(__file__),
        outputs={name: sha(args.out/name) for name in ('comparison.png', 'comparison.pdf')},
        new_fits=0, new_geometry_queries=0, new_clouds=0,
        scope='Top: projected Gaussian moment ellipses, not measured density contours. '
              'Bottom: all retained production states, including rejection residence. '
              'Heldout mismatch diagnoses a proposal; serial dependence and fixed context '
              'prevent interpreting these fractions as equilibrium or assembly evidence.')
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
