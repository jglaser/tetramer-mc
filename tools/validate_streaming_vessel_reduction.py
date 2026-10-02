#!/usr/bin/env python3
"""Reuse completed audit receipts to check only the new streaming reductions.

No density, geometry, native classifier, sampler, or Poisson call is made. The
old full-geometry audits authenticate saved validity/contact/support labels.
This is not an independent replacement audit of those labels or a new population.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import time

from analyze_mobile_native_pocket import local_sources
from analyze_r4_smc_control import Ledger, read, require, write
from audit_hard_free_vessel_streaming import CLASSES, attempt_batches, row_counts
from streaming_weight_moments import RegionMoments


def compare(left, right, path=''):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), 'Reduction keys differ: '+path)
        return max((compare(left[k], right[k], path+'/'+k) for k in left), default=0.)
    if isinstance(left, float):
        require(isinstance(right, (float, int)) and math.isfinite(left) and math.isfinite(right),
                'Nonfinite comparison: '+path)
        error = abs(left-right)
        require(error <= 2e-12+5e-10*max(abs(left), abs(right)), 'Reduction differs: '+path)
        return error/max(1., abs(left), abs(right))
    require(left == right, 'Reduction metadata differs: '+path)
    return 0.


def validate(validation, out):
    validation, out = Path(validation).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh reduction-validation output required')
    ledger = Ledger(); started = time.process_time()
    for path in local_sources(__file__).values(): ledger.bind(path)
    previous = read(ledger.bind(validation))
    require(previous['schema'] == 'hard-free-vessel-registration-validation-v1' and previous['complete'] is True,
            'Completed full-vessel reference receipt required')
    results = []
    for saved in previous['independent_audits']:
        receipt = read(ledger.bind(saved['path'], saved['sha256']))
        require(receipt['complete'] is True and receipt['schema'] == 'full-vessel-hard-free-line-independent-audit-v1',
                'Incomplete or wrong original geometry audit')
        for path, digest in receipt['source_sha256'].items(): ledger.bind(path, digest)
        root = Path(receipt['population']); n = receipt['manifest']['samples']
        raw = ledger.bind(root/'samples.jsonl', receipt['samples_sha256'])
        attempts = ledger.bind(root/'attempts.jsonl', receipt['attempts_sha256'])
        summary = read(ledger.bind(root/'summary.json'))
        require(summary['complete'] is True and summary['manifest'] == receipt['manifest']
                and n == saved['checked_attempts'], 'Original population changed')
        reducers = {name: RegionMoments(n) for name in CLASSES}
        counts = dict(hard_valid=0, capture_rejected=0, wall_rejected=0, hard_rejected=0, raw_points=0)
        peak = seen = 0
        for rows in attempt_batches(raw, attempts, n, 64):
            peak = max(peak, len(rows))
            for key, value in row_counts(rows).items(): counts[key] += value
            for row in rows:
                valid = row['hard_valid']; inside = row['latent_density']['in_reference_ball']
                selected = dict(total=valid, inside_R4=valid and inside, outside_R4=valid and not inside,
                    latent_zero=valid and row['latent_density']['structural_zero'],
                    exclusion_contact=valid and row['depletion_contact'], unbound=valid and not row['depletion_contact'])
                for name, chosen in selected.items(): reducers[name].add(row, chosen)
                seen += 1
        require(seen == n and all(summary[k] == counts[k] for k in counts), 'Lost unconditional count')
        estimates = {name: reducer.result() for name, reducer in reducers.items()}
        error = compare(receipt['estimates'], estimates)
        results.append(dict(population=str(root), samples=seen, peak_rows=peak,
                            regions=len(estimates), maximum_scaled_error=error, passed=True))
    require(sum(r['samples'] for r in results) == previous['independently_audited_new_law_attempts'],
            'Original audited allocation incomplete')
    ledger.recheck()
    result = dict(schema='streaming-vessel-reduction-validation-v1', complete=True, jobs=results,
        samples=sum(r['samples'] for r in results), input_sha256=ledger.files,
        CPU_seconds=time.process_time()-started, repeated_geometry_or_density_checks=0,
        new_pose_draws=0, new_clouds=0, native_classifier_calls=0,
        scope='Only new streaming reductions, all-attempt journal accounting and counters validated against '
              'authenticated completed geometry audits. No geometry/density replay or new physical evidence. '
              'This does not by itself validate the complete new audit entry point.')
    out.parent.mkdir(parents=True, exist_ok=True); write(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = validate(args.validation, args.out)
    print({k: result[k] for k in ('complete', 'samples', 'CPU_seconds')})
