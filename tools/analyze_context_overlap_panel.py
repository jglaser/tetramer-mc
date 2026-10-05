"""Independent count algebra for a fixed pose panel; not a regional normalizer."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import chi2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def score_counts(lower, uncertain, counts, intensity, activity, alpha=.05):
    """Two equal-intensity independent clouds, with exact Poisson count intervals.

    The unbiased estimate of log physical weight is z(L+sum(K)/(2 lambda)).
    Its exponential is NOT an unbiased weight. Positive weights use a distinct
    generating-function identity and are recorded separately below.
    """
    require(len(counts) == 2 and all(type(k) is int and k >= 0 for k in counts),
            'Exactly two complete nonnegative integer counts required')
    require(all(math.isfinite(x) for x in (lower, uncertain, intensity, activity))
            and lower >= 0 and uncertain >= 0 and intensity > 0 and activity >= 0
            and 0 < alpha < 1, 'Invalid score parameters')
    k = sum(counts)
    require(uncertain > 0 or k == 0, 'Positive hits in a zero-volume envelope')
    overlap = lower + k/(2*intensity)
    variance = activity**2*k/(4*intensity**2)
    variance_bound = activity**2*uncertain/(2*intensity)
    if uncertain == 0:
        count_interval = [0., 0.]
    else:
        count_interval = [float(.5*chi2.ppf(alpha/2, 2*k)) if k else 0.,
                          float(.5*chi2.ppf(1-alpha/2, 2*(k+1)))]
    overlap_interval = [lower+x/(2*intensity) for x in count_interval]
    log_weights = [activity*lower+k_i*math.log1p(activity/intensity) for k_i in counts]
    hi = max(log_weights)
    mean_log_weight = hi + math.log(sum(math.exp(x-hi) for x in log_weights)/2)
    return dict(overlap_volume=overlap, z_overlap=activity*overlap,
                variance_estimate=variance, variance_upper=variance_bound,
                overlap_interval=overlap_interval,
                score_interval=[activity*x for x in overlap_interval],
                count_mean_interval=count_interval, interval_alpha=alpha,
                log_weights=log_weights, log_mean_positive_weight=mean_log_weight,
                confidence_scope='Point-cloud noise conditional on this fixed pose; no pose-coverage uncertainty.')


def close(actual, expected, name):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual-expected) <= 2e-9+2e-11*abs(expected), 'Score disagreement: '+name)


def audit_rows(config, panel, rows):
    require(len(rows) == len(panel['entries']) and len(rows) > 0, 'Incomplete fixed panel')
    records = []
    totals = dict(raw_points=0, processed_points=0, overlap_points=0, clouds=0)
    for i, (entry, row) in enumerate(zip(panel['entries'], rows)):
        require(row['pose_index'] == i and row['id'] == entry['id'] and row['complete'],
                'Changed panel identity/order')
        require(row['pose'] == entry['pose'], 'Scored pose differs from frozen panel')
        require(len(row['clouds']) == 2, 'Incomplete pair of clouds')
        envelope = row['envelope']; counts = []
        if 'original_row' in entry['metadata']:
            original = entry['metadata']['original_row']
            require(row['actual'] == original['actual'] and row['region'] == original['region']
                    and row['patches'] == original['patches'], 'Saved geometry/classification differs')
            for key in ('lower_volume','upper_volume','uncertain_volume'):
                close(envelope[key],original['envelope'][key],'saved.'+key)
            for key in ('fixed_labels','retained_cells','created_cells','certified_cells'):
                require(envelope[key] == original['envelope'][key], 'Envelope structure differs')
        for cloud in row['clouds']:
            progress, weight = cloud['progress'], cloud['weight']
            planned = progress['planned_points']
            if planned is None:
                require(envelope['uncertain_volume'] == 0 and weight['raw_points'] == 0,
                        'Only a deterministic empty envelope may skip the Poisson count')
                planned = 0
            require(progress['begun'] and progress['complete']
                    and planned == progress['processed_points'] == weight['raw_points']
                    and progress['overlap_points'] == weight['overlap_points']
                    and 0 <= weight['overlap_points'] <= weight['raw_points'], 'Incomplete cloud count accounting')
            counts.append(weight['overlap_points'])
            for key in ('lower_volume', 'upper_volume', 'uncertain_volume'):
                close(weight[key], envelope[key], key)
            totals['raw_points'] += weight['raw_points']
            totals['processed_points'] += progress['processed_points']
            totals['overlap_points'] += weight['overlap_points']
            totals['clouds'] += 1
        result = score_counts(envelope['lower_volume'], envelope['uncertain_volume'], counts,
                              config['lambda'], config['activity'])
        simultaneous = score_counts(envelope['lower_volume'], envelope['uncertain_volume'], counts,
                                    config['lambda'], config['activity'], .05/len(rows))
        for key in ('overlap_volume', 'z_overlap', 'variance_estimate', 'variance_upper'):
            close(row['score'][key], result[key], key)
        for cloud, expected in zip(row['clouds'], result['log_weights']):
            close(cloud['weight']['log_weight'], expected, 'log_weight')
            close(cloud['progress']['log_weight'], expected, 'progress.log_weight')
        metadata = entry['metadata']
        log_q = metadata['log_q_balanced']
        require(type(log_q) in (int, float) and math.isfinite(log_q), 'Invalid saved density')
        for key, expected in (('log_q_balanced',log_q),
                              ('log_mean_positive_weight',result['log_mean_positive_weight']),
                              ('log_physical_importance_score',result['z_overlap']-log_q)):
            if key in row:
                close(row[key],expected,key)
        records.append(dict(id=entry['id'], pose_index=i, metadata=metadata, counts=counts,
                            **result, log_physical_importance_score=result['z_overlap']-log_q,
                            log_positive_importance_weight=result['log_mean_positive_weight']-log_q,
                            simultaneous_score_interval=simultaneous['score_interval'],
                            simultaneous_importance_interval=[x-log_q for x in simultaneous['score_interval']]))
    return records, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    require(cfg['schema'] == 'fixed-saved-pose-overlap-v1', 'Wrong scorer schema')
    require(sha(cfg['panel']['path']) == cfg['panel']['sha256'], 'Panel changed')
    panel = read(cfg['panel']['path'])
    require(panel['schema'] == 'context-fixed-pose-overlap-panel-v1', 'Wrong panel schema')
    report = read(args.result/'summary.json')
    require(report['complete'] and report['passed'], 'Physical score panel incomplete')
    producer = read(args.result/'protocol.json')
    require(producer['config_sha256'] == sha(args.config) and producer['config'] == cfg,
            'Scorer used a different configuration')
    for path, digest in producer['input_sha256'].items():
        require(sha(path) == digest, 'Scorer input changed: '+path)
    rows = [json.loads(line) for line in (args.result/'rows.jsonl').read_text().splitlines()]
    records, totals = audit_rows(cfg, panel, rows)
    require(report['poses_begun'] == report['poses_completed'] == len(rows)
            and report['clouds_begun'] == report['clouds_completed'] == 2*len(rows)
            and report['new_poses_generated'] == report['retries'] == report['replacements'] == 0
            and not report['normalizer_estimated'], 'Incomplete/changed physical allocation')
    for name in ('raw_points', 'processed_points', 'clouds_completed'):
        require(report[name] == totals['clouds' if name == 'clouds_completed' else name],
                'Summary accounting differs: '+name)
    output = dict(schema='context-overlap-score-audit-v1', complete=True, passed=True,
                  source_sha256=sha(__file__), inputs={str(p.resolve()):sha(p) for p in
                  (args.config, Path(cfg['panel']['path']), args.result/'summary.json', args.result/'rows.jsonl')},
                  records=records, totals=totals,
                  scope='Fixed selected-pose score diagnostic. No regional normalizer, equilibrium occupancy, ESS, or assembly conclusion.',
                  intervals='Garwood count intervals; simultaneous 95% family uses Bonferroni over all fixed poses. Point noise only.',
                  region_mass_estimated=False, new_clouds=0, new_poses=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(complete=True, passed=True, poses=len(records), totals=totals)))


if __name__ == '__main__':
    main()
