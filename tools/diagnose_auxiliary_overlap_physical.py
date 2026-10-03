#!/usr/bin/env python3
"""Explore authenticated saved MH terms/counts without sampling or geometry work.

The binding is written before the row loop. Every outer and null is retained;
candidate-only bath estimates are descriptive, never a replacement MH weight.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / 'results/auxiliary-overlap-physical-20261003'
ANALYSIS_SHA = '2d091e2241202cc38555b3914a7cb3244a1bcb8c55b9d4758d0dbe72eff65c55'
RECEIPT_SHA = '2e5b59aac3de15b21d3e98e8f834c73317a9eae1ab430402be34d209e5d2ebe5'
SCHEMA = 'auxiliary-overlap-physical-saved-count-decomposition-v1'
TERMS = ('log_root_f_correction', 'log_internal_f_correction',
         'log_full_f_correction', 'log_auxiliary_correction',
         'log_depletion_factor', 'log_ratio', 'log_physical_bath_ratio_estimate',
         'standard_error_estimate', 'estimated_auxiliary_log_penalty')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')


def close(a, b, name):
    require(math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-10), name)


def verified_inputs(run):
    """Resolve/authenticate the data chain before any candidate evaluation."""
    hashes = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'Changed input: ' + str(path))
        hashes[str(path)] = digest
        return path

    def record(item):
        return bind(item['path'], item['sha256'])

    paths = {'analysis': bind(run / 'analysis.json', ANALYSIS_SHA),
             'receipt': bind(run / 'completed-review.json', RECEIPT_SHA)}
    physical, receipt = read(paths['analysis']), read(paths['receipt'])
    require(physical['complete'] and physical['passed'] and not physical['failures'] and
            receipt['complete'] and receipt['passed'], 'Incomplete physical inputs')
    require(receipt['output_hashes']['analysis.json'] == ANALYSIS_SHA, 'Unbound physical analysis')
    paths['plan'] = bind(run / 'config.json', receipt['output_hashes']['config.json'])
    paths['guided_raw'] = bind(run / 'execution/attempts.jsonl', receipt['output_hashes']['execution/attempts.jsonl'])
    plan = read(paths['plan'])
    require(plan['activity'] == .0275 and plan['lambda'] == 1.76 and
            plan['depletant_radius'] == 1.4, 'Changed diagnostic conditions')
    for name in ('analysis.json', 'attempts.jsonl', 'protocol.json', 'root-review.json'):
        paths['passive_' + name] = record(plan['passive'][name])
    paths['guided_cache'] = record(plan['candidate_ledger'])
    paths['baseline_cache'] = record(plan['baseline_cache'])
    for name in ('analysis.json', 'completed-review.json', 'execution/attempts.jsonl'):
        paths['baseline_' + name] = record(plan['baseline'][name])
    baseline_receipt = read(paths['baseline_completed-review.json'])
    baseline_analysis = read(paths['baseline_analysis.json'])
    require(baseline_receipt['complete'] and baseline_receipt['passed'] and
            baseline_analysis['complete'] and baseline_analysis['passed'] and
            not baseline_analysis['failures'], 'Incomplete baseline')
    for name in ('analysis.json', 'execution/attempts.jsonl'):
        require(baseline_receipt['output_hashes'][name] ==
                plan['baseline'][name]['sha256'], 'Unbound baseline ' + name)
    paths['baseline_plan'] = bind(paths['baseline_analysis.json'].parent / 'config.json',
                                  baseline_receipt['output_hashes']['config.json'])
    baseline_plan = read(paths['baseline_plan'])
    for name in ('analysis.json', 'attempts.jsonl', 'root-review.json'):
        paths['old_passive_' + name] = record(baseline_plan['passive'][name])
        require(baseline_analysis['checked_input_hashes'][str(paths['old_passive_' + name])] ==
                hashes[str(paths['old_passive_' + name])], 'Historical passive input not audited')
    passive_protocol = read(paths['passive_protocol.json'])
    paths['passive_baseline_cache'] = record(passive_protocol['baseline_cache'])
    for prefix in ('passive_', 'old_passive_'):
        audit, review = read(paths[prefix + 'analysis.json']), read(paths[prefix + 'root-review.json'])
        require(audit['complete'] and audit['passed'] and not audit['failures'] and
                review['complete'] and review['passed'], 'Incomplete passive audit')
        require(review['output_hashes']['analysis.json'] == sha(paths[prefix + 'analysis.json']),
                'Passive review does not bind analysis')
        require(audit['input_hashes']['attempts.jsonl'] == sha(paths[prefix + 'attempts.jsonl']),
                'Passive audit does not bind contacts ledger')
    for key in ('guided_cache', 'baseline_cache', 'passive_baseline_cache',
                'passive_analysis.json', 'passive_attempts.jsonl'):
        require(physical['checked_input_hashes'][str(paths[key])] == hashes[str(paths[key])],
                'Input not authenticated by completed physical audit: ' + key)
    # No project modules are imported. Freeze executing source, formula precedent,
    # and every loaded file-backed standard-library import before the row loop.
    bind(__file__)
    paths['formula_precedent'] = bind(REPO / 'tools/diagnose_factorized_dimer_counts.py')
    imports = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file():
            path = bind(filename)
            imports[name] = {'path': str(path), 'sha256': hashes[str(path)]}
    return paths, hashes, imports, physical, plan


def distribution(values):
    values = sorted(values)
    require(all(math.isfinite(x) for x in values), 'Nonfinite diagnostic value')

    def quantile(p):
        if not values:
            return None
        pos = p * (len(values) - 1)
        lower = int(pos)
        return values[lower] + (pos - lower) * (values[min(lower + 1, len(values) - 1)] - values[lower])

    return dict(count=len(values), mean=math.fsum(values) / len(values) if values else None,
                **{name: quantile(p) for name, p in
                   (('min', 0), ('p05', .05), ('median', .5), ('p95', .95), ('max', 1))})


def summarize(rows):
    candidates = [r for r in rows if r['candidate']]
    summary = dict(outer_attempts=len(rows), candidates=len(candidates),
                   proposal_nulls=len(rows) - len(candidates), accepted=sum(r['accepted'] for r in rows),
                   sum_conditional_acceptance_probability=math.fsum(r['acceptance_probability'] for r in rows),
                   negative_bath_estimates=sum(r['log_physical_bath_ratio_estimate'] < 0 for r in candidates),
                   root_penalty=sum(r['log_root_f_correction'] < -1e-10 for r in candidates),
                   internal_penalty=sum(r['log_internal_f_correction'] < -1e-10 for r in candidates),
                   root_zero=sum(abs(r['log_root_f_correction']) <= 1e-10 for r in candidates),
                   internal_larger_absolute_term=sum(abs(r['log_internal_f_correction']) >
                                                     abs(r['log_root_f_correction']) for r in candidates),
                   old_external_contacts=distribution([r['old_external_contacts'] for r in candidates]),
                   new_external_contacts=distribution([r['new_external_contacts'] for r in candidates]),
                   **{key: distribution([r[key] for r in candidates]) for key in TERMS})
    return summary


def contact_set(edges):
    return {tuple(sorted(edge)) for edge in edges}


def analyze(paths, physical, plan):
    passive = read(paths['passive_analysis.json'])['rows']
    old_passive = read(paths['old_passive_analysis.json'])['rows']
    old_physical = read(paths['baseline_analysis.json'])['rows']
    old_passive_raw = paths['old_passive_attempts.jsonl'].read_bytes().splitlines(keepends=True)
    old_physical_raw = paths['baseline_execution/attempts.jsonl'].read_bytes().splitlines(keepends=True)
    require(b''.join(line for line in old_physical_raw if json.loads(line)['cached']['method'] == 'factorized') ==
            paths['baseline_cache'].read_bytes(), 'Baseline physical cache is not the exact historical subset')
    require(b''.join(line for line in old_passive_raw if json.loads(line)['method'] == 'factorized') ==
            paths['passive_baseline_cache'].read_bytes(), 'Passive baseline cache is not the exact historical subset')
    old_passive_by_key = {(r['atlas'], r['case'], r['attempt']): r for r in old_passive if r['method'] == 'factorized'}
    old_physical_by_index = {r['index']: r for r in old_physical if r['method'] == 'factorized'}
    guided_raw = paths['guided_raw'].read_text().splitlines()
    guided_passive = paths['passive_attempts.jsonl'].read_text().splitlines()
    guided_cache = paths['guided_cache'].read_text().splitlines()
    baseline_raw = paths['baseline_cache'].read_text().splitlines()
    baseline_passive = paths['passive_baseline_cache'].read_text().splitlines()
    require(len(guided_raw) == len(guided_passive) == len(guided_cache) == len(passive) ==
            len(physical['rows']) == 1536, 'Lost guided unconditional rows')
    require(len(baseline_raw) == len(baseline_passive) == len(physical['reused_baseline_rows']) ==
            len(old_passive_by_key) == len(old_physical_by_index) == 768, 'Lost baseline unconditional rows')
    rows = []
    z, lam = plan['activity'], plan['lambda']
    groups = [('guided', guided_raw, guided_passive, physical['rows'], passive),
              ('baseline', baseline_raw, baseline_passive, physical['reused_baseline_rows'], None)]
    for kind, raw_lines, passive_lines, audits, passive_audits in groups:
        seen = set()
        for position, (line, passive_line, audited) in enumerate(zip(raw_lines, passive_lines, audits)):
            raw, source = json.loads(line), json.loads(passive_line)
            cached = raw['cached']
            key = (audited['atlas'], audited['case'], audited['attempt'])
            prior = passive_audits[position] if passive_audits is not None else old_passive_by_key[key]
            method = audited['method']
            require((key, method) not in seen, 'Duplicate outer')
            seen.add((key, method))
            for item in (cached, source):
                require((item['atlas'], item['case']['name'], item['attempt']) == key,
                        'Wrong physical/passive contact join')
                require(item['method'] == ('factorized' if kind == 'baseline' else method), 'Wrong method join')
            require((prior['atlas'], prior['case'], prior['attempt'], prior['method']) ==
                    (*key, source['method']), 'Wrong passive analysis join')
            require(cached['passive_row_sha256'] == hashlib.sha256(passive_line.encode()).hexdigest() and
                    cached['candidate'] == source['outcome']['candidate'] and cached['old'] == source['old'],
                    'Changed cached passive candidate')
            require(raw['index'] == audited['index'] == cached['index'] and
                    raw['accepted'] == audited['accepted'] and raw['gate_failure'] is None, 'Wrong physical row')
            if kind == 'guided':
                require(cached == json.loads(guided_cache[position]), 'Changed guided cache')
            else:
                old = old_physical_by_index[audited['index']]
                require(all(audited[k] == v for k, v in old.items() if k != 'method'), 'Changed historical result')
            candidate = cached['candidate'] is not None
            require(candidate == audited['candidate'] == (prior['candidate'] is not None), 'Candidate/null mismatch')
            old_contacts = contact_set(source['old_contacts'])
            internal_edge = tuple(sorted((cached['case']['root'], cached['case']['child'])))
            require(internal_edge in old_contacts, 'Old internal contact missing')
            row = dict(index=audited['index'], atlas=audited['atlas'], case=audited['case'],
                       attempt=audited['attempt'], method=method, reused=kind == 'baseline',
                       candidate=candidate, accepted=audited['accepted'],
                       proposal_status=cached['proposal_status'], old_contacts=len(old_contacts),
                       old_external_contacts=len(old_contacts) - 1,
                       acceptance_probability=audited['acceptance_probability'])
            if not candidate:
                require(raw['status'] == 'proposal_null' and raw['gate'] is None and
                        raw['log_ratio'] is None and not raw['accepted'] and
                        audited['acceptance_probability'] == 0., 'Null has physical decision')
                row.update({key: None for key in TERMS})
                row.update(new_external_contacts=None, gained_external_edges=None, lost_external_edges=None,
                           gained=None, lost=None, variance_estimate=None, internal_only_endpoints=None,
                           pair_overlap_volume_change_estimate_A3=None)
                rows.append(row)
                continue
            contacts = contact_set(prior['candidate']['contacts'])
            change = prior['contact_change']
            require(internal_edge in contacts and len(old_contacts) == change['old_contacts'] and
                    len(contacts) == change['new_contacts'], 'Wrong source or endpoint contact count')
            external = len(contacts) - 1
            gained_edges, lost_edges = len(contacts - old_contacts), len(old_contacts - contacts)
            require(external == prior['candidate']['external_contacts'] == audited['external_contacts'] and
                    gained_edges == change['gained_contacts'] == audited['gained_contact_edges'] and
                    lost_edges == change['lost_contacts'] == audited['lost_contact_edges'], 'Wrong contact change')
            diagnostic = cached['candidate']['diagnostics']
            old_edges, new_edges = diagnostic['old_edges'], diagnostic['new_edges']
            require(len(old_edges) == len(new_edges) == 2, 'Expected root and internal edges')
            root = old_edges[0]['log_full'] - new_edges[0]['log_full']
            internal = old_edges[1]['log_full'] - new_edges[1]['log_full']
            close(sum(e['log_full'] for e in old_edges), diagnostic['full_old_log_density'], 'Old full F sum')
            close(sum(e['log_full'] for e in new_edges), diagnostic['full_new_log_density'], 'New full F sum')
            full = root + internal
            close(full, diagnostic['log_reverse_forward'], 'Edge corrections do not sum to full F')
            close(full, audited['log_full_f_correction'], 'Full F differs from physical decision')
            require(diagnostic['selection_log_reverse_forward'] == diagnostic['log_tree_coordinate_jacobian'] == 0,
                    'Unaccounted selection or Jacobian correction')
            auxiliary = 0.
            if kind == 'guided':
                guidance = cached['guidance']
                auxiliary = cached['m'] * (math.log1p(guidance['old_count']) - math.log1p(guidance['new_count']))
                close(auxiliary, guidance['aux_log_correction'], 'Wrong auxiliary target ratio')
                close(auxiliary, raw['auxiliary_correction'], 'Wrong saved auxiliary ratio')
                close(full, raw['full_f_correction'], 'Wrong saved full F')
                close(full + auxiliary, cached['complete_log_correction'], 'Incomplete cached correction')
            close(auxiliary, audited['log_auxiliary_correction'], 'Wrong analyzed auxiliary ratio')
            close(full + auxiliary, raw['q_correction'], 'Wrong complete proposal correction')
            close(full + auxiliary, audited['log_proposal_correction'], 'Wrong analyzed proposal correction')
            aggregate = raw['gate']['aggregate']
            g, lost = aggregate['gained'], aggregate['lost']
            require(type(g) is int and type(lost) is int and min(g, lost) >= 0 and
                    g == audited['gained'] and lost == audited['lost'], 'Changed saved bath counts')
            for count in ('gained', 'lost'):
                require(aggregate[count] == sum(leg[count] for leg in raw['gate']['legs']), 'Wrong path count sum')
            log_weight = math.log1p(z / lam) * (g - lost)
            close(log_weight, aggregate['log_weight'], 'Wrong bath count weight')
            close(log_weight, audited['log_depletion_factor'], 'Wrong analyzed bath count weight')
            log_ratio = full + auxiliary + log_weight
            close(log_ratio, raw['log_ratio'], 'Incomplete MH sum')
            close(log_ratio, audited['log_ratio'], 'Wrong analyzed MH sum')
            # This is the already saved MH alpha, not exp(ell) or a new MH test.
            close(math.exp(min(0., log_ratio)), audited['acceptance_probability'], 'Wrong saved conditional alpha')
            require(raw['accepted'] == (raw['log_uniform'] < min(0., log_ratio)), 'Changed historical MH decision')
            ell = z * (g / lam - lost / (lam + z))
            variance = z * z * (g / (lam * lam) + lost / ((lam + z) ** 2))
            penalty = ell - log_weight
            require(penalty >= -1e-10 and variance >= 0, 'Impossible saved-count diagnostic')
            internal_only = len(old_contacts) == 1 and external == 0
            row.update(new_external_contacts=external, gained_external_edges=gained_edges,
                       lost_external_edges=lost_edges, log_root_f_correction=root,
                       log_internal_f_correction=internal, log_full_f_correction=full,
                       log_auxiliary_correction=auxiliary, log_depletion_factor=log_weight,
                       log_ratio=log_ratio, gained=g, lost=lost,
                       log_physical_bath_ratio_estimate=ell, variance_estimate=variance,
                       standard_error_estimate=math.sqrt(variance),
                       estimated_auxiliary_log_penalty=penalty, internal_only_endpoints=internal_only,
                       pair_overlap_volume_change_estimate_A3=ell / z if internal_only else None)
            rows.append(row)
    guided = [r for r in rows if not r['reused']]
    baseline = [r for r in rows if r['reused']]
    for selected, expected, n in ((guided, physical['summary'], 324),
                                  (baseline, physical['reused_baseline_summary'], 511)):
        summary = summarize(selected)
        require(summary['candidates'] == n and summary['accepted'] == expected['accepted'] == 0,
                'Changed complete physical allocation')
        close(summary['sum_conditional_acceptance_probability'], expected['sum_conditional_acceptance_probability'],
              'Changed unconditional conditional-alpha sum')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['atlas'], row['method']].append(row)
    top = sorted((r for r in guided if r['candidate']), key=lambda r: r['acceptance_probability'], reverse=True)[:2]
    alpha_sum = summarize(guided)['sum_conditional_acceptance_probability']
    return dict(rows=rows, summary=dict(guided=summarize(guided), baseline=summarize(baseline)),
                internal_only_summary={name: summarize([r for r in selected if r['internal_only_endpoints']])
                                       for name, selected in (('guided', guided), ('baseline', baseline))},
                comparisons=[dict(atlas=a, method=m, summary=summarize(rs),
                                  internal_only=summarize([r for r in rs if r['internal_only_endpoints']]))
                             for (a, m), rs in sorted(grouped.items())],
                top_two_guided_conditional_alpha=dict(rows=top,
                    fraction_of_all_guided_conditional_alpha=math.fsum(r['acceptance_probability'] for r in top) / alpha_sum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=RUN / 'saved-count-decomposition.json')
    parser.add_argument('--binding', type=Path, default=RUN / 'saved-count-decomposition-binding.json')
    args = parser.parse_args()
    require(not args.output.exists() and not args.binding.exists(), 'Refusing to overwrite a prior attempt')
    result = dict(schema=SCHEMA, complete=False, passed=False, failures=[])
    try:
        paths, hashes, imports, physical, plan = verified_inputs(args.run)
        binding = dict(schema=SCHEMA + '-preanalysis-binding', complete=True,
                       recorded_utc=datetime.now(timezone.utc).isoformat(),
                       input_hashes=hashes, imports=imports, python=sys.version,
                       executable=str(Path(sys.executable).resolve()),
                       python_executable_sha256=sha(Path(sys.executable).resolve()),
                       exploratory=True, new_clouds=0, new_pose_draws=0,
                       new_native_classifications=0, new_geometry_evaluations=0,
                       endpoint_stratum='old_contacts == 1 and new_external_contacts == 0',
                       formulas=dict(log_ratio='z*(G/lambda-L/(lambda+z))',
                                     variance='z^2*(G/lambda^2+L/(lambda+z)^2)',
                                     auxiliary_Jensen_penalty='ell - saved_log_weight'))
        write_new(args.binding, binding)
        result['binding_sha256'] = sha(args.binding)
        result.update(analyze(paths, physical, plan))
        require(all(sha(path) == digest for path, digest in hashes.items()), 'Input or source changed during analysis')
        result.update(complete=True, passed=True, exploratory=True,
                      new_clouds=0, new_pose_draws=0, new_native_classifications=0, new_geometry_evaluations=0,
                      input_hashes=hashes, conditions=dict(depletant_radius_A=1.4, activity_A_minus3=.0275,
                                                        intensity_A_minus3=1.76, concentration_uM=500),
                      limitations=[
                          'Exploratory saved-candidate arithmetic, conditional on these fixed poses and path orders; not an equilibrium or association free energy.',
                          'The count logratio and variance estimators assume the intended untruncated independent Poisson thinning law; geometry is inherited from authenticated passive audits.',
                          'The count estimate ell is not the exactly known physical bath logratio. Zero counts do not prove zero volume or zero uncertainty.',
                          'ell minus the realized bath logweight is a nonnegative, correlated estimate of the auxiliary Jensen penalty, not an exact bath free energy.',
                          'No exp(ell), new MH decision, certified acceptance bound, cloud, proposal, native classification or geometry rerun is used.',
                          'The both-endpoints-internal-only stratum uses audited old contacts == 1 and new external contacts == 0; nulls remain in unconditional summaries with no energy estimate.',
                          'Guided arms share proposal prefixes; baseline decisions are cached historical outcomes. Conditional-alpha sums are not independent success counts.',
                          'Root/internal full-F terms use the saved mixture log_full densities in root then internal edge order; auxiliary-target correction remains separate.',
                          'All results concern the 1.4 A, activity 0.0275 A^-3, 500 uM diagnostic regime, separate from the original-condition panel; they cannot refute the model.'])
    except Exception as error:
        result['failures'].append(str(error))
    write_new(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('rows', 'comparisons', 'input_hashes')}, indent=2, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
