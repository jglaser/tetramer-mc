#!/usr/bin/env python3
"""Decompose completed FFT-width reset results using saved rows only.

Requires an explicitly bound completed-review receipt. Reads no partial run,
draws no data, evaluates no geometry/density, and never changes an MH decision.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import diagnose_auxiliary_overlap_physical as prior

SCHEMA = 'fft-width-physical-saved-decomposition-v1'
TERMS = prior.TERMS
require, sha, read, write_new = prior.require, prior.sha, prior.read, prior.write_new
close, distribution, contact_set = prior.close, prior.distribution, prior.contact_set


def number(value):
    if value == '-inf':
        return -math.inf
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid log value')
    return float(value)


def log_serial(value):
    return '-inf' if value == -math.inf else value


def density_fractions(edge, alpha=.5):
    """Contributions at a fixed pose; these are not the draw mixture weights."""
    learned, uniform, full = [number(edge[key]) for key in ('log_learned', 'log_uniform', 'log_full')]
    require(math.isfinite(full), 'Undefined source mixture-density contributions')
    lg, lu = math.log1p(-alpha) + learned - full, math.log(alpha) + uniform - full
    pg, pu = math.exp(lg), math.exp(lu)
    close(pg + pu, 1., 'Source full F differs from its two mixture contributions')
    require(max(pg, pu) <= 1. + 2e-10, 'Invalid source mixture contribution')
    return dict(log_learned=log_serial(learned), log_uniform=log_serial(uniform), log_full=full,
                learned_fraction=pg, uniform_fraction=pu,
                learned_log_fraction=log_serial(lg), uniform_log_fraction=log_serial(lu))


def verified_inputs(run, expected_review):
    hashes = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'Changed bound input: ' + str(path))
        hashes[str(path)] = digest
        return path

    # Authenticate the terminal review before reading either scientific ledger.
    paths = {'review': bind(run / 'completed-review.json', expected_review)}
    receipt = read(paths['review'])
    require(receipt['complete'] and receipt['passed'], 'Physical review is incomplete')
    for key, filename in [('analysis', 'analysis.json'), ('plan', 'config.json'), ('raw', 'execution/attempts.jsonl')]:
        paths[key] = bind(run / filename, receipt['output_hashes'][filename])
    physical, plan = read(paths['analysis']), read(paths['plan'])
    require(physical['complete'] and physical['passed'] and not physical['failures'], 'Physical audit is incomplete')
    require(plan['schema'] == 'fft-width-physical-reset-v1' and plan['total_outer'] == 1536 and
            plan['baseline'] == {} and plan['baseline_cache'] is None, 'Wrong width-only allocation')
    require(plan['depletant_radius'] == 1.4 and plan['activity'] == .0275 and plan['lambda'] == 1.76,
            'Changed diagnostic conditions')
    require(physical['reused_baseline_rows'] == [], 'Historical outcomes mixed into new width rows')
    for name in ('analysis.json', 'attempts.jsonl', 'config.json', 'root-review.json'):
        record = plan['passive'][name]
        paths['passive_' + name] = bind(record['path'], record['sha256'])
    record = plan['candidate_ledger']
    paths['cache'] = bind(record['path'], record['sha256'])
    for key in ('cache', 'passive_analysis.json', 'passive_attempts.jsonl', 'passive_config.json', 'passive_root-review.json'):
        require(physical['checked_input_hashes'][str(paths[key])] == hashes[str(paths[key])],
                'Completed physical audit does not authenticate ' + key)
    passive, review, config = [read(paths['passive_' + name]) for name in ('analysis.json', 'root-review.json', 'config.json')]
    require(passive['complete'] and passive['passed'] and not passive['failures'] and review['complete'] and review['passed'],
            'Passive review is incomplete')
    require(review['output_hashes']['analysis.json'] == hashes[str(paths['passive_analysis.json'])] and
            passive['input_hashes']['attempts.jsonl'] == hashes[str(paths['passive_attempts.jsonl'])] and
            passive['input_hashes']['config.json'] == hashes[str(paths['passive_config.json'])], 'Passive binding differs')
    require(config['schema'] == 'fft-width-screen-v1' and config['master_seed'] == 6100300401 and
            [a['tau'] for a in config['scaled_atlases']] == [.125, .25, .5], 'Changed width family')
    imports = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file():
            path = bind(filename)
            imports[name] = dict(path=str(path), sha256=hashes[str(path)])
    bind(__file__)
    bind(prior.__file__)
    return paths, hashes, imports, physical, passive, plan, config


def summarize(rows):
    result = prior.summarize(rows)
    accepted = [r for r in rows if r['accepted']]
    result.update(accepted_with_external_contacts=sum(r['new_external_contacts'] > 0 for r in accepted),
        accepted_with_gained_external_edges=sum(r['gained_external_edges'] > 0 for r in accepted),
        accepted_with_lost_external_edges=sum(r['lost_external_edges'] > 0 for r in accepted),
        accepted_gained_external_edges=sum(r['gained_external_edges'] for r in accepted),
        accepted_lost_external_edges=sum(r['lost_external_edges'] for r in accepted),
        accepted_net_external_edge_change=sum(r['gained_external_edges'] - r['lost_external_edges'] for r in accepted),
        accepted_internal_only_endpoints=sum(r['internal_only_endpoints'] for r in accepted),
        accepted_detachments_to_no_external_contact=sum(r['old_external_contacts'] > 0 and r['new_external_contacts'] == 0 for r in accepted))
    return result


def analyze(paths, physical, passive, plan, config):
    raw_lines = paths['raw'].read_text().splitlines()
    source_lines = paths['passive_attempts.jsonl'].read_text().splitlines()
    cache_lines = paths['cache'].read_text().splitlines()
    require(len(raw_lines) == len(source_lines) == len(cache_lines) == len(physical['rows']) == len(passive['rows']) == 1536,
            'Lost unconditional outer rows')
    rows, sources, seen = [], {}, set()
    z, lam = plan['activity'], plan['lambda']
    for i, (raw_line, source_line, cache_line, audited, passive_row) in enumerate(zip(
            raw_lines, source_lines, cache_lines, physical['rows'], passive['rows'])):
        raw, source, cached = read_line(raw_line), read_line(source_line), read_line(cache_line)
        require(raw['cached'] == cached and raw['index'] == cached['index'] == audited['index'] == i, 'Wrong cached physical row')
        require(cached['passive_row_sha256'] == hashlib.sha256(source_line.encode()).hexdigest() and
                cached['candidate'] == source['outcome']['candidate'] and cached['old'] == source['old'], 'Wrong passive source row')
        ai, ci = source['atlas_index'], source['case_index']
        atlas, tau = config['scaled_atlases'][ai]['name'], config['scaled_atlases'][ai]['tau']
        case, method, attempt = source['case']['name'], source['method'], source['attempt']
        key = ai, ci, method, attempt
        require(key not in seen, 'Duplicate outer')
        seen.add(key)
        require(source['tau'] == tau and source['atlas'] == atlas and source['status'] == 'completed', 'Changed width row')
        for row in (audited, passive_row):
            require((row['atlas'], row['case'], row['method'], row['attempt']) == (atlas, case, method, attempt), 'Wrong audit join')
        require(cached['atlas'] == atlas and cached['case'] == source['case'] and cached['method'] == method and
                raw['accepted'] == audited['accepted'] and raw['gate_failure'] is None, 'Wrong physical decision')
        source_key = ai, ci
        saved_source = dict(atlas=atlas, tau=tau, case=case, case_index=ci, members=source['outcome']['members'],
                            anchor=source['case']['anchor'], old_coordinates=source['old_coordinates'], old_edges=source['old_edges'])
        if source_key not in sources:
            sources[source_key] = dict(**saved_source, occurrences=0, methods=Counter(),
                root=density_fractions(source['old_edges'][0]), internal=density_fractions(source['old_edges'][1]))
        item = sources[source_key]
        require(all(item[k] == v for k, v in saved_source.items()), 'Repeated source density changed')
        item['occurrences'] += 1
        item['methods'][method] += 1
        candidate = cached['candidate'] is not None
        require(candidate == audited['candidate'] == (passive_row['candidate'] is not None), 'Candidate/null disagreement')
        old_contacts = contact_set(source['old_contacts'])
        internal_edge = tuple(sorted((source['case']['root'], source['case']['child'])))
        require(internal_edge in old_contacts, 'Missing old internal contact')
        row = dict(index=i, atlas=atlas, tau=tau, case=case, case_index=ci, attempt=attempt, method=method,
            candidate=candidate, accepted=audited['accepted'], proposal_status=cached['proposal_status'],
            old_contacts=len(old_contacts), old_external_contacts=len(old_contacts) - 1,
            acceptance_probability=audited['acceptance_probability'], source_key=[ai, ci])
        if not candidate:
            require(raw['status'] == 'proposal_null' and raw['gate'] is None and not row['accepted'] and
                    raw['log_ratio'] is None and row['acceptance_probability'] == 0., 'Null has a physical decision')
            row.update({key: None for key in TERMS})
            row.update(new_external_contacts=None, gained_external_edges=None, lost_external_edges=None,
                gained_external_contact_pairs=None, lost_external_contact_pairs=None, gained=None, lost=None,
                variance_estimate=None, internal_only_endpoints=None, pair_overlap_volume_change_estimate_A3=None,
                old_edges=None, new_edges=None)
            rows.append(row)
            continue
        contacts = contact_set(passive_row['candidate']['contacts'])
        require(internal_edge in contacts, 'Missing new internal contact')
        gained, lost = contacts - old_contacts, old_contacts - contacts
        external = len(contacts) - 1
        require(external == passive_row['candidate']['external_contacts'] == audited['external_contacts'] and
                len(gained) == audited['gained_contact_edges'] and len(lost) == audited['lost_contact_edges'], 'Changed contact fingerprint')
        diagnostic = cached['candidate']['diagnostics']
        old_edges, new_edges = diagnostic['old_edges'], diagnostic['new_edges']
        require(len(old_edges) == len(new_edges) == 2 and old_edges == source['old_edges'], 'Changed source F or edge order')
        root = number(old_edges[0]['log_full']) - number(new_edges[0]['log_full'])
        internal = number(old_edges[1]['log_full']) - number(new_edges[1]['log_full'])
        full = root + internal
        close(full, number(diagnostic['log_reverse_forward']), 'Wrong full-F sum')
        close(full, number(audited['log_full_f_correction']), 'Wrong audited full-F')
        require(diagnostic['selection_log_reverse_forward'] == diagnostic['log_tree_coordinate_jacobian'] == 0, 'Additional coordinate/selection term')
        auxiliary = 0.
        if method == 'm4':
            guidance = cached['guidance']
            auxiliary = 4 * (math.log1p(guidance['old_count']) - math.log1p(guidance['new_count']))
            close(auxiliary, number(guidance['aux_log_correction']), 'Wrong guide correction')
        else:
            require(method == 'unguided' and cached['guidance'] is None, 'Wrong unguided correction')
        close(auxiliary, number(audited['log_auxiliary_correction']), 'Wrong audited auxiliary term')
        close(full + auxiliary, number(raw['q_correction']), 'Wrong complete proposal correction')
        close(full + auxiliary, number(source['complete_log_correction']), 'Wrong passive correction')
        aggregate = raw['gate']['aggregate']
        g, l = aggregate['gained'], aggregate['lost']
        require(type(g) is int and type(l) is int and min(g, l) >= 0 and g == audited['gained'] and l == audited['lost'], 'Changed bath counts')
        require(g == sum(leg['gained'] for leg in raw['gate']['legs']) and l == sum(leg['lost'] for leg in raw['gate']['legs']), 'Wrong bath leg sum')
        log_weight = math.log1p(z / lam) * (g - l)
        close(log_weight, number(aggregate['log_weight']), 'Wrong realized bath weight')
        close(log_weight, number(audited['log_depletion_factor']), 'Wrong audited bath factor')
        log_ratio = full + auxiliary + log_weight
        close(log_ratio, number(raw['log_ratio']), 'Wrong complete MH sum')
        close(log_ratio, number(audited['log_ratio']), 'Wrong audited MH sum')
        close(math.exp(min(0., log_ratio)), row['acceptance_probability'], 'Changed saved alpha')
        require(row['accepted'] == (raw['log_uniform'] < min(0., log_ratio)), 'Changed saved MH decision')
        ell = z * (g / lam - l / (lam + z))
        variance = z * z * (g / (lam * lam) + l / ((lam + z) ** 2))
        penalty = ell - log_weight
        require(penalty >= -1e-10 and variance >= 0, 'Invalid bath diagnostic')
        internal_only = len(old_contacts) == 1 and external == 0
        row.update(new_external_contacts=external, gained_external_edges=len(gained), lost_external_edges=len(lost),
            gained_external_contact_pairs=sorted(gained), lost_external_contact_pairs=sorted(lost),
            log_root_f_correction=root, log_internal_f_correction=internal, log_full_f_correction=full,
            log_auxiliary_correction=auxiliary, log_depletion_factor=log_weight, log_ratio=log_ratio,
            gained=g, lost=l, log_physical_bath_ratio_estimate=ell, variance_estimate=variance,
            standard_error_estimate=math.sqrt(variance), estimated_auxiliary_log_penalty=penalty,
            internal_only_endpoints=internal_only, pair_overlap_volume_change_estimate_A3=ell / z if internal_only else None,
            old_edges=old_edges, new_edges=new_edges)
        rows.append(row)
    require(seen == {(a, c, m, j) for a in range(3) for c in range(8) for m in ('unguided', 'm4') for j in range(32)}, 'Lost allocation')
    require(len(sources) == 24 and all(s['occurrences'] == 64 and s['methods'] == {'unguided': 32, 'm4': 32} for s in sources.values()),
            'Lost deduplicated sources')
    total = summarize(rows)
    for key in ('outer_attempts', 'candidates', 'accepted'):
        require(total[key] == physical['summary'][key], 'Changed physical total ' + key)
    close(total['sum_conditional_acceptance_probability'], physical['summary']['sum_conditional_acceptance_probability'], 'Changed alpha total')
    for key, expected in [('accepted_gained_external_edges', 'accepted_gained_contact_edges'), ('accepted_lost_external_edges', 'accepted_lost_contact_edges')]:
        require(total[key] == physical['summary'][expected], 'Changed accepted contacts')
    groups = defaultdict(list)
    for row in rows:
        groups[row['tau'], row['method']].append(row)
    source_rows = [sources[key] for key in sorted(sources)]
    source_summary = []
    for tau in (.125, .25, .5):
        selected = [s for s in source_rows if s['tau'] == tau]
        source_summary.append(dict(tau=tau, contexts=len(selected), **{edge: dict(
            learned_fraction=distribution([s[edge]['learned_fraction'] for s in selected]),
            uniform_fraction=distribution([s[edge]['uniform_fraction'] for s in selected]),
            learned_dominant_contexts=sum(s[edge]['learned_fraction'] > .5 for s in selected),
            displayed_learned_fraction_underflow_contexts=sum(s[edge]['learned_fraction'] == 0. and s[edge]['learned_log_fraction'] != '-inf' for s in selected))
            for edge in ('root', 'internal')}))
    return dict(rows=rows, summary=total, internal_only_summary=summarize([r for r in rows if r['internal_only_endpoints']]),
        comparisons=[dict(tau=tau, method=method, summary=summarize(rs),
            internal_only=summarize([r for r in rs if r['internal_only_endpoints']])) for (tau, method), rs in sorted(groups.items())],
        accepted_rows=[r for r in rows if r['accepted']], deduplicated_source_rows=source_rows,
        deduplicated_source_summary=source_summary)


def read_line(line):
    return prior.json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--binding', type=Path)
    args = parser.parse_args()
    output = args.output or args.run / 'saved-count-decomposition.json'
    binding_path = args.binding or args.run / 'saved-count-decomposition-binding.json'
    require(not output.exists() and not binding_path.exists(), 'Refusing to overwrite a prior diagnosis')
    paths, hashes, imports, physical, passive, plan, config = verified_inputs(args.run, args.review_sha256)
    binding = dict(schema=SCHEMA + '-binding', recorded_utc=datetime.now(timezone.utc).isoformat(),
        input_hashes=hashes, imports=imports, python=sys.version, exploratory=True,
        allocation=dict(saved_outer_rows=1536, deduplicated_width_context_sources=24, new_count_queries=0,
                        new_geometry_queries=0, new_density_queries=0, new_clouds=0, new_pose_draws=0),
        formulas=dict(log_physical_bath_ratio_estimate='z*(G/lambda-L/(lambda+z))',
            variance_estimate='z^2*(G/lambda^2+L/(lambda+z)^2)', estimated_auxiliary_log_penalty='ell-realized logW',
            learned_source_density_fraction='(1-alpha)*G(old)/F(old)', uniform_source_density_fraction='alpha*U(old)/F(old)'),
        internal_only='Old contact set contains only the internal root-child edge and new external contacts equal zero.')
    write_new(binding_path, binding)
    result = dict(schema=SCHEMA, complete=False, passed=False, binding_sha256=sha(binding_path), failures=[])
    try:
        result.update(analyze(paths, physical, passive, plan, config))
        require(all(sha(path) == digest for path, digest in hashes.items()), 'Bound source or data changed during diagnosis')
        result.update(complete=True, passed=True, input_hashes=hashes, new_count_queries=0,
            new_geometry_queries=0, new_density_queries=0, new_clouds=0, new_pose_draws=0,
            limitations=[
                'Saved fixed-context candidate arithmetic; no equilibrium free energy, ESS, native classification, or assembly conclusion.',
                'All 1536 outers and nulls remain in unconditional summaries; internal-only summaries condition on both candidate endpoints having no external contacts.',
                'Root/internal F terms use old-minus-new complete mixture log densities in anchor-root, root-child order. The m4 auxiliary-target term is separate.',
                'Bath ell and variance assume the intended independent untruncated Poisson thinning law; ell is not an exactly known physical logratio.',
                'The nonnegative correlated estimate ell-realized logW estimates the auxiliary bath Jensen penalty; no exp(ell) or replacement MH test is used.',
                'Source fractions describe contributions to the mixture density at each fixed old pose, not learned/uniform draw probabilities or physical weights.',
                'Source entries are deduplicated over attempts and methods, yielding 24 width-context entries; contexts and paired arms are dependent.',
                'Tiny source density fractions may underflow to zero in display; their log fractions are retained. Such zeros do not prove absent Gaussian support.',
                'Contact gains/losses count exclusion edges incident to the moved pair. Accepted gains do not establish native registry or trajectory assembly.',
                'Only the new .125/.25/.5 widths are decomposed here; historical tau1 data are not replayed or mixed into these rows.'])
    except Exception as error:
        result['failures'].append(f'{type(error).__name__}: {error}')
    write_new(output, result)
    print(prior.json.dumps({key: result.get(key) for key in ('complete', 'passed', 'failures', 'summary', 'deduplicated_source_summary')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
