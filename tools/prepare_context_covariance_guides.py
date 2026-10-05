"""Freeze full/diagonal source guides from previously audited training moments.

No new poses, geometry queries, density evaluations, or depletant clouds.
The covariance describes saved trajectory residence, not equilibrium fluctuations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_moments(mean, covariance, scales):
    mean, covariance, scales = map(lambda x: np.asarray(x, float), (mean, covariance, scales))
    require(mean.shape == (6,) and covariance.shape == (6, 6) and scales.shape == (6,)
            and all(np.isfinite(x).all() for x in (mean, covariance, scales)) and (scales > 0).all(),
            'Invalid training moments or reporting scales')
    require(np.allclose(covariance, covariance.T, atol=1e-20, rtol=1e-12),
            'Asymmetric training covariance')
    scaled = .5*(covariance+covariance.T)/np.outer(scales, scales)
    eigenvalues = np.linalg.eigvalsh(scaled)
    require(eigenvalues[0] >= -1e-12*max(1., abs(eigenvalues[-1])),
            'Training covariance is not positive semidefinite')
    epsilon = max(1e-10, 1e-6*float(np.trace(scaled))/6)
    full = (scaled+epsilon*np.eye(6))*np.outer(scales, scales)
    diagonal = np.diag(np.diag(full))
    np.linalg.cholesky(full)
    return mean, full, diagonal, epsilon


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--expected-report-sha256', required=True)
    p.add_argument('--fit-plan', type=Path, required=True)
    p.add_argument('--expected-fit-plan-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(sha(args.report) == args.expected_report_sha256
            and sha(args.fit_plan) == args.expected_fit_plan_sha256, 'Changed frozen training evidence or fit plan')
    report = json.loads(args.report.read_bytes())
    plan = json.loads(args.fit_plan.read_bytes())
    require(plan['schema'] == 'context-covariance-fit-plan-v1'
            and plan['training_streams'] == [0, 1] and plan['heldout_streams'] == [2, 3]
            and plan['phase'] == 'production' and plan['samples'] == 20480
            and plan['bandwidth_multiplier'] == 1., 'Changed fixed fit allocation')
    require(report['schema'] == 'source-cage-saved-covariance-v1' and report['complete'] and report['passed']
            and report['production_states'] == 40960 and report['total_retained_states'] == 46080
            and report['chart_failures'] == 0 and report['geometry_queries'] == 0
            and report['cloud_draws'] == 0 and report['proposal_draws'] == 0, 'Incomplete saved-data reduction')
    train = [r for r in report['pooled'] if r['phase'] == 'production' and r['group'] == 'train']
    require(len(train) == 1 and train[0]['streams'] == [0, 1] and train[0]['samples'] == 20480
            and train[0]['coordinate_status'] == 'complete', 'Wrong training sample')
    moments = train[0]['map_coordinates_angstrom']
    ell = float(report['angular_length'])
    scales = [.1]*3+[ell*np.tan(np.pi/720)]*3
    require(ell > 0 and np.isfinite(ell)
            and np.allclose(scales, report['scale_vector'], atol=0, rtol=1e-14), 'Changed chart scale')
    mean, full, diagonal, epsilon = freeze_moments(moments['mean'], moments['covariance'], scales)
    args.out.mkdir()
    outputs = {}
    for arm, covariance in [('full', full), ('diagonal', diagonal)]:
        guide = dict(schema='context-covariance-frozen-guide-v1', arm=arm,
            training_streams=[0, 1], heldout_streams=[2, 3],
            reduction_sha256=args.expected_report_sha256,
            fit_plan_sha256=args.expected_fit_plan_sha256,
            source_chart=dict(angular_length=ell, covariance=covariance.tolist(),
                explicit_gaussian=dict(schema='source-gaussian-v1', mean=mean.tolist(),
                    provenance=f'Train streams 0,1 from {args.expected_report_sha256}; fit plan {args.expected_fit_plan_sha256}; {arm}.')),
            regularization=dict(scaled_ridge=epsilon, scales=scales, bandwidth_multiplier=1.,
                formula='S*(Sigma+max(1e-10,1e-6*trace(Sigma)/6)*I)*S; diagonal control retains its diagonal'),
            interpretation='Source-informed proposal from residence-weighted local controls; not an equilibrium covariance.')
        path = args.out/f'{arm}.json'
        path.write_text(json.dumps(guide, indent=2, allow_nan=False)+'\n')
        outputs[arm] = dict(path=str(path.resolve()), sha256=sha(path))
    result = dict(complete=True, passed=True, source_sha256=sha(__file__),
        report=dict(path=str(args.report.resolve()), sha256=args.expected_report_sha256),
        fit_plan=dict(path=str(args.fit_plan.resolve()), sha256=args.expected_fit_plan_sha256),
        guides=outputs, training_samples=20480, heldout_fit_samples=0,
        geometry_queries=0, clouds=0, poses_generated=0)
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
