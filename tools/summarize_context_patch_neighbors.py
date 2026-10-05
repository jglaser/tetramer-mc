"""Read completed contact summaries only; no geometry, trajectories, or sampling."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    for path, digest in protocol['inputs'].items():
        require(sha(path) == digest, 'Changed frozen summary input: ' + path)
    rows = []
    for source in protocol['comparisons']:
        data = read(source['comparison'])
        status = read(source['status'])
        require(data['complete'] and data['passed'] and status['complete'] and status['passed'],
                'Comparison not complete and passed')
        require(len(status['completed']) == 1, 'Unexpected comparison stages')
        done = status['completed'][0]
        require(done['success'] and done['child_drained'] and done['returncode'] == 0
                and done['terminal']['path'] == source['comparison']
                and done['terminal']['sha256'] == sha(source['comparison']), 'Comparison receipt differs')
        require(len(data['streams']) == 24, 'All24 compared streams must be retained')
        require(all(r['patch_ess']['samples'] == 10240 for r in data['streams']),
                'Unexpected unconditional production denominator')
        for entry in data['equal_stream_average_start_comparisons']:
            if entry['cadence'] != 'all_attempts':
                continue
            arm = entry['arm']
            require(entry['streams_per_start'] == 4, 'Exactly four streams per start required')
            streams = [r for r in data['streams'] if r['identity']['arm'] == arm]
            require(len(streams) == 8, 'Missing arm stream')
            starts = sorted({r['identity']['start'] for r in streams})
            require(set(starts) == {'saved_body77', 'highest_original_prior_valid_neighbor_distinct_center'},
                    'Unexpected preparations')
            for start in starts:
                require({r['identity']['stream'] for r in streams if r['identity']['start'] == start}
                        == set(range(4)), 'Missing independent stream')
            groups = defaultdict(list)
            for token in entry['patch_marginals']['tokens']:
                key = token['token']
                require(len(key) == 4 and key[0] < key[1] and 77 in key[:2], 'Malformed global patch token')
                left, right = token['left'], token['right']
                require(0 <= left <= 1 and 0 <= right <= 1, 'Invalid marginal probability')
                require(math.isclose(token['right_minus_left'], right-left, rel_tol=0, abs_tol=1e-14),
                        'Saved token difference inconsistent')
                groups[tuple(key[:2])].append(token)
            count = 0
            total_absolute = 0.0
            pair_rows = []
            for pair, tokens in sorted(groups.items()):
                differences = [abs(t['right']-t['left']) for t in tokens]
                count += len(tokens)
                total_absolute += sum(differences)
                pair_rows.append(dict(
                    body_pair=list(pair), token_union_size=len(tokens),
                    mean_absolute_marginal_difference=sum(differences)/len(tokens),
                    maximum_absolute_marginal_difference=max(differences),
                    expected_active_patch_tokens_left=sum(t['left'] for t in tokens),
                    expected_active_patch_tokens_right=sum(t['right'] for t in tokens),
                    left_always_right_never=sum(t['left'] == 1 and t['right'] == 0 for t in tokens),
                    right_always_left_never=sum(t['right'] == 1 and t['left'] == 0 for t in tokens),
                    complete_tokens=tokens))
            require(math.isclose(total_absolute/count, entry['patch_marginals']['mean_absolute_difference'],
                                 rel_tol=1e-12, abs_tol=1e-14), 'Grouped discrepancy does not reconstruct full summary')
            rows.append(dict(
                comparison_name=source['name'], comparison_sha256=sha(source['comparison']),
                arm=arm, correlation=None if arm == 'local' else source['atlas_correlation'],
                local_control_reused=arm == 'local',
                left_start='saved_body77', right_start='highest_original_prior_valid_neighbor_distinct_center',
                streams_per_start=4, retained_attempts_per_stream=10240,
                unconditional_retained_attempts_per_start=40960,
                averaged_equal_streams=True, conditioning_on_neighbor_presence=False,
                full_patch_mean_absolute_difference=total_absolute/count,
                neighbor_set_total_variation=entry['neighbor_set']['total_variation'],
                exact_fingerprint_total_variation=entry['fingerprint']['total_variation'],
                exact_fingerprint_support_intersection=entry['fingerprint']['intersection'],
                pairs=pair_rows))
    require(len(rows) == 6, 'All three arms from both comparisons required, including reused local control')
    old_local = next(r for r in rows if r['comparison_name'] == 'rho0' and r['arm'] == 'local')
    new_local = next(r for r in rows if r['comparison_name'] == 'rho095' and r['arm'] == 'local')
    require(old_local['pairs'] == new_local['pairs'], 'Reused local observations differ')
    report = dict(complete=True, passed=True, schema='context-patch-neighbor-saved-summary-reduction-v1',
        protocol_sha256=sha(args.protocol), source_sha256=sha(__file__), input_sha256=protocol['inputs'],
        scope=protocol['scope'], definitions=protocol['definitions'], rows=rows,
        rows_count=len(rows), geometry_queries=0, physical_draws=0, raw_journals_read=0,
        limitations=['These are empirical residence-weighted patch indicators, not established equilibrium probabilities.',
            'Expected active patch-token counts are a coarse contact-pattern descriptor, not physical area, exclusion-overlap volume, depletion free energy, or native registry.',
            'All retained production attempts, including rejections and Other configurations, remain in denominators.',
            'Local controls appear in both comparisons for provenance and are reused observations, not independent repetitions.'])
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True, passed=True, rows=len(rows), output_sha256=sha(args.out))))


if __name__ == '__main__':
    main()
