#!/usr/bin/env python3
"""Frozen, deterministic three-axis replay of the completed passive line pilot.

This is posthoc geometry/density diagnosis. It draws no poses or Poisson clouds
and does not estimate a physical normalizer, equilibrium occupancy or mixing.
Prepare first, then run the archived helper; analyze is deterministic and can
be repeated in a new receipt without rerunning the Rust geometry queries.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.special import logsumexp
from analyze_contact_line_audit import Reconstructor, audit, contains, interval_masses


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify(root, table):
    for name, digest in table.items():
        require(sha(root / name) == digest, 'Changed frozen file: ' + name)


def prepare(pilot, out):
    pilot, out = pilot.resolve(), out.resolve()
    require(not out.exists(), 'A new output directory is required')
    state = read(pilot / 'status.json')
    require(state['complete'] and state['physical_jobs_launched'] == 0 and
            state['new_Poisson_clouds'] == 0, 'Original passive pilot incomplete')
    verify(pilot, read(pilot / 'freeze.json')['files'])
    require(sha(pilot / 'analysis.json') == state['analysis_sha256'], 'Changed original analysis')
    for job in state['jobs'] + [state['coverage_job']]:
        require(job['status'] == 'complete' and job['returncode'] == 0, 'Incomplete original job')
        verify(Path(job['directory']), job['output_sha256'])
    common = pilot / 'common'
    original_guide = read(common / 'line92.json')
    require(original_guide['raw_translation_axes'] == [0] and
            original_guide['contact_widths_A'] == [.02, .1, .5] and
            original_guide['conditional_probability'] == .5 and
            original_guide['defensive_uniform_shell_probability'] == .5 and
            original_guide['minimum_conditional_mass'] == 1e-12 and
            original_guide['contact_neighbor_indices'] == [0, 1] and
            len(original_guide['gaussian_components']) == 92, 'Original design differs')
    new_guide = copy.deepcopy(original_guide)
    new_guide['raw_translation_axes'] = [0, 1, 2]
    ledger, probes, sources = [], [], {}

    def add(record, group, population, metadata=None, fixed_x=None):
        original_id = record['id']
        identifier = f'{group}/{population}/{original_id}'
        entry = dict(id=identifier, group=group, population=population,
                     original_id=original_id, original_row=record)
        if metadata is not None:
            entry['source_metadata'] = metadata
        if fixed_x is not None:
            entry['fixed_x_row'] = fixed_x
        ledger.append(entry)
        probes.append(dict(id=identifier, latent=record['latent']))

    def source(path):
        sources[str(path)] = sha(path)
        return rows(path)

    for population in range(4):
        name = f'r{population:02}'
        old = source(pilot / 'runs' / 'baseline92' / name / 'samples.jsonl')
        require(len(old) == 64 and [r['id'] for r in old] == list(range(64)), 'Baseline allocation differs')
        for record in old:
            add(record, 'baseline256', name)
    critical_meta = read(common / 'preparation' / 'axis-diagnostics.json')['rows']
    critical = source(pilot / 'runs/baseline92/r00/probes.jsonl')
    fixed = {r['id']: r for r in source(pilot / 'runs/line92/r00/probes.jsonl')}
    require(len(critical) == len(critical_meta) == len(fixed) == 78, 'Critical allocation differs')
    for record, meta in zip(critical, critical_meta):
        require(record['id'] == meta['id'] and record['latent'] == meta['source']['u'], 'Critical identity differs')
        add(record, 'critical78', meta['source']['arm'] + '/' + meta['source']['id'], meta, fixed[record['id']])
    coverage_meta = read(common / 'coverage-selection.json')['rows']
    coverage = source(pilot / 'runs/line92/coverage/probes.jsonl')
    require(len(coverage) == len(coverage_meta) == 128, 'Coverage allocation differs')
    for record, meta in zip(coverage, coverage_meta):
        require(record['id'] == meta['id'] and record['latent'] == meta['latent'], 'Coverage identity differs')
        add(record, 'breadth128', meta['coverage_class'], meta, record)
    require(len(probes) == 462 and len({r['id'] for r in probes}) == 462, 'Lost or duplicate probes')
    out.mkdir(parents=True)
    inputs = out / 'inputs'
    inputs.mkdir()
    for name in ('contact-line-guide-audit', 'source-bundle.json', 'region.json', 'shape.json', 'line92.json'):
        shutil.copy2(common / name, inputs / name)
    shutil.copy2(Path(__file__), inputs / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('analyze_contact_line_audit.py'), inputs / 'analyze_contact_line_audit.py')
    for name in ('protocol.json', 'freeze.json', 'status.json', 'analysis.json'):
        shutil.copy2(pilot / name, inputs / ('original-' + name))
    config = read(common / 'config.json')
    config['shape'] = str(inputs / 'shape.json')
    write(inputs / 'config.json', config)
    write(inputs / 'axes92.json', new_guide)
    write(inputs / 'ledger.json', dict(rows=ledger, original_source_sha256=sources))
    with (inputs / 'probes.jsonl').open('x') as stream:
        for record in probes:
            stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
    bundle = read(inputs / 'source-bundle.json')
    for name, entry in bundle['files'].items():
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Bad Rust source bundle: ' + name)
    require((inputs / 'source-bundle.json').read_bytes() in (inputs / 'contact-line-guide-audit').read_bytes(),
            'Frozen source bundle is not embedded in executable')
    command = [str(inputs / 'contact-line-guide-audit'), '--config', str(inputs / 'config.json'),
               '--region', str(inputs / 'region.json'), '--importance-guide', str(inputs / 'axes92.json'),
               '--out', str(out / 'audit'), '--samples', '0', '--seed', '610050001',
               '--probes', str(inputs / 'probes.jsonl')]
    write(out / 'protocol.json', dict(schema='contact-line-axes-diagnostic-v1', scope=__doc__,
          created=time.time(), original_pilot=str(pilot), axes=[0, 1, 2], widths_A=[.02, .1, .5],
          uniform_probability=.5, conditional_probability=.5, minimum_conditional_mass=1e-12,
          allocation=dict(baseline256=256, critical78=78, breadth128=128),
          fresh_draws=0, new_Poisson_clouds=0, maximum_CPU_workers=1,
          seed_unused_because_samples_zero=610050001, command=command,
          binary_sha256=sha(inputs / 'contact-line-guide-audit'),
          source_bundle_sha256=sha(inputs / 'source-bundle.json'),
          source_sha256={name: entry['sha256'] for name, entry in bundle['files'].items()},
          executable=sys.executable, numpy_version=np.__version__,
          metrics=['Positive hard-free two-contact intervals by axis/width and any-axis union',
                   'Per-component mass-floor fallback and original selected-component availability',
                   'Complete q: all three axes versus reconstructed single axes and original mixture',
                   'Retrospective paired M2 on the same critical78, separately by original source arm/population',
                   'Broad saved-class coverage and CPU; no unseen-mass or mixing inference']))
    write(out / 'freeze.json', dict(files={str(p.relative_to(out)): sha(p)
          for p in sorted(out.rglob('*')) if p.is_file()}))
    return out


def run(out):
    out = out.resolve()
    verify(out, read(out / 'freeze.json')['files'])
    require(sha(__file__) == sha(out / 'inputs' / Path(__file__).name), 'Use the archived helper')
    protocol = read(out / 'protocol.json')
    require(protocol['fresh_draws'] == 0 and protocol['command'][protocol['command'].index('--samples') + 1] == '0', 'Nonzero sample allocation')
    write(out / 'execution-claim.json', dict(started=time.time(), pid=os.getpid(),
          protocol_sha256=sha(out / 'protocol.json'), freeze_sha256=sha(out / 'freeze.json'), command=protocol['command']))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with (out / 'execution.log').open('x') as log:
        result = subprocess.run(protocol['command'], stdout=log, stderr=subprocess.STDOUT, check=False)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(out / 'execution.json', dict(returncode=result.returncode, wall_seconds=time.monotonic() - started,
          child_CPU_seconds=(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime),
          finished=time.time(), samples_requested=0))
    require(result.returncode == 0, 'Geometry audit failed; preserve outputs, do not retry')
    verify(out, read(out / 'freeze.json')['files'])


def distribution(values):
    values = np.asarray(values, float)
    return dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()),
                mean=float(values.mean()))


def analyze(out):
    out = out.resolve()
    verify(out, read(out / 'freeze.json')['files'])
    require(not (out / 'analysis.json').exists(), 'Analysis receipt already exists')
    start = time.process_time()
    independent = audit(out / 'audit')
    write(out / 'independent-audit.json', independent)
    inputs = out / 'inputs'
    recon = Reconstructor(read(inputs / 'region.json'), read(inputs / 'axes92.json'),
                          read(inputs / 'config.json'), read(inputs / 'shape.json'))
    ledger = read(inputs / 'ledger.json')['rows']
    observed = rows(out / 'audit/probes.jsonl')
    require(len(observed) == len(ledger) == 462 and not rows(out / 'audit/samples.jsonl'), 'Replay allocation differs')
    diagnostic = []
    max_old_x_error = 0.
    max_mixture_error = 0.
    for record, entry in zip(observed, ledger):
        require(record['id'] == entry['id'] and record['latent'] == entry['original_row']['latent'], 'Replay changed pose or order')
        old = entry['original_row']
        for key in ('hard_valid', 'shell_valid', 'capture_valid', 'width_contacts'):
            require(record[key] == old[key], 'Replay changed original predicate: ' + key)
        require(abs(record['baseline_log_density'] - old['baseline_log_density']) < 1e-10, 'Baseline density changed')
        x = recon.raw(record['latent'])
        gaussian = recon.gaussian_logs(record['latent'])
        uniform = math.log(recon.alpha) - recon.logvolume if record['shell_valid'] else -math.inf
        axes = []
        logs = []
        original_component = (old.get('draw') or {}).get('component') if entry['group'] == 'baseline256' else None
        for axis, detail in zip(recon.axes, record['density_details']['axes']):
            means, sigmas = recon.conditional(x, axis)
            factor = np.zeros(len(gaussian))
            widths = []
            for width, intervals in zip(recon.widths, recon.geometry_sets(detail, x, axis)):
                mass = interval_masses(intervals, means, sigmas)
                active = mass > recon.floor
                factor[~active] += 1.
                if contains(intervals, x[axis]):
                    factor[active] += 1 / mass[active]
                widths.append(dict(width_A=width, positive_interval=any(i['upper'] > i['lower'] for i in intervals),
                      any_component_usable=bool(active.any()), usable_components=int(active.sum()),
                      selected_component_usable=bool(active[original_component]) if original_component is not None else None,
                      component_mass_max=float(mass.max()), length_A=sum(i['upper'] - i['lower'] for i in intervals)))
            correction = 1 - recon.beta + recon.beta * factor / len(recon.widths)
            log_q = float(np.logaddexp(uniform, math.log1p(-recon.alpha) + logsumexp(gaussian + np.log(correction))))
            logs.append(log_q)
            counts = Counter(detail.get('core_counts', {}))
            for wd in detail.get('widths', []):
                for query in wd['contact_queries']:
                    counts.update(query)
            axes.append(dict(axis=axis, empty_reason=detail.get('empty_reason'), widths=widths,
                             log_q=log_q, geometry_counts=dict(counts)))
        mixture = float(logsumexp(logs) - math.log(3))
        error = abs(mixture - record['log_proposal_density'])
        max_mixture_error = max(max_mixture_error, error)
        require(error < 2e-7, 'Three-axis law is not the equal mixture of single-axis laws')
        if 'fixed_x_row' in entry:
            error = abs(logs[0] - entry['fixed_x_row']['log_proposal_density'])
            max_old_x_error = max(max_old_x_error, error)
            require(error < 2e-7, 'Fixed-x reconstruction differs from immutable prior output')
        diagnostic.append(dict(id=record['id'], group=entry['group'], population=entry['population'],
          original_component=original_component, baseline_log_q=record['baseline_log_density'],
          three_axis_log_q=record['log_proposal_density'], axes=axes,
          hard_valid=record['hard_valid'], target_domain_valid=record['hard_valid'] and record['capture_valid'] and record['shell_valid'],
          density_cpu_seconds=record['density_cpu_seconds']))

    def availability(subset):
        result = dict(rows=len(subset), hard_target_valid=sum(r['target_domain_valid'] for r in subset), by_axis={})
        for axis in range(3):
            result['by_axis'][str(axis)] = dict(
                empty_reasons=dict(Counter(r['axes'][axis]['empty_reason'] or 'hard_free_line' for r in subset)),
                positive_interval_by_width=[sum(r['axes'][axis]['widths'][w]['positive_interval'] for r in subset) for w in range(3)],
                any_component_usable_by_width=[sum(r['axes'][axis]['widths'][w]['any_component_usable'] for r in subset) for w in range(3)],
                any_width_usable=sum(any(w['any_component_usable'] for w in r['axes'][axis]['widths']) for r in subset),
                selected_component_by_width=[sum(r['axes'][axis]['widths'][w]['selected_component_usable'] is True for r in subset) for w in range(3)],
                geometry_counts=dict(sum((Counter(r['axes'][axis]['geometry_counts']) for r in subset), Counter())))
        result['selected_component_rows'] = sum(r['original_component'] is not None for r in subset)
        result['any_axis_usable_by_width'] = [sum(any(r['axes'][a]['widths'][w]['any_component_usable'] for a in range(3)) for r in subset) for w in range(3)]
        result['any_axis_any_width_usable'] = sum(any(w['any_component_usable'] for a in r['axes'] for w in a['widths']) for r in subset)
        result['rescued_relative_to_x_by_width'] = [sum(not r['axes'][0]['widths'][w]['any_component_usable'] and any(r['axes'][a]['widths'][w]['any_component_usable'] for a in (1, 2)) for r in subset) for w in range(3)]
        return result

    availability_table = {'all462': availability(diagnostic)}
    for group in ('baseline256', 'critical78', 'breadth128'):
        subset = [r for r in diagnostic if r['group'] == group]
        availability_table[group] = availability(subset)
        availability_table[group]['populations'] = {p: availability([r for r in subset if r['population'] == p]) for p in sorted({r['population'] for r in subset})}
    by_id = {r['id']: r for r in diagnostic}

    def moment(subset):
        log_terms = {name: [] for name in ('baseline92', 'fixed_x', 'axes_012')}
        for entry in subset:
            source = entry['source_metadata']['source']
            record = by_id[entry['id']]
            qs = [record['baseline_log_q'], record['axes'][0]['log_q'], record['three_axis_log_q']]
            numerator = sum(source['paired_log_weights']) + source['log_q']
            for name, q in zip(log_terms, qs):
                log_terms[name].append(numerator - q)
        totals = {name: float(logsumexp(values)) for name, values in log_terms.items()}
        return dict(rows=len(subset), log_contribution_sums=totals,
                    fixed_x_to_baseline_M2=math.exp(totals['fixed_x'] - totals['baseline92']),
                    axes_to_baseline_M2=math.exp(totals['axes_012'] - totals['baseline92']),
                    axes_to_fixed_x_M2=math.exp(totals['axes_012'] - totals['fixed_x']),
                    contribution_ESS={name: math.exp(2 * totals[name] - logsumexp(2 * np.asarray(values))) for name, values in log_terms.items()},
                    largest_fraction={name: math.exp(max(values) - totals[name]) for name, values in log_terms.items()})

    moments = {}
    for arm in ('baseline', 'expanded'):
        subset = [r for r in ledger if r['group'] == 'critical78' and r['source_metadata']['source']['arm'] == arm]
        moments[arm] = moment(subset)
        moments[arm]['populations'] = {p: moment([r for r in subset if r['source_metadata']['source']['id'] == p]) for p in sorted({r['source_metadata']['source']['id'] for r in subset})}
    coverage = {}
    for group in ('baseline256', 'critical78', 'breadth128'):
        subset = [r for r in diagnostic if r['group'] == group]
        classes = {'all': subset}
        classes.update({p: [r for r in subset if r['population'] == p] for p in sorted({r['population'] for r in subset})})
        coverage[group] = {name: dict(rows=len(rs), axes_to_baseline_log_ratio=distribution([r['three_axis_log_q'] - r['baseline_log_q'] for r in rs]),
                    axes_to_fixed_x_log_ratio=distribution([r['three_axis_log_q'] - r['axes'][0]['log_q'] for r in rs])) for name, rs in classes.items()}
    require(min(r['three_axis_log_q'] - r['baseline_log_q'] for r in diagnostic) >= math.log(.5) - 1e-8, 'Defensive support floor violated')
    execution = read(out / 'execution.json')
    summary = read(out / 'audit/summary.json')
    analysis = dict(schema='contact-line-axes-analysis-v1', complete=True, samples=0, probes=462, new_Poisson_clouds=0,
        scope='Posthoc saved-pose geometry and exact proposal densities only. Any-axis union is diagnostic, not the selection law; the proposal chooses one axis uniformly.',
        availability=availability_table, log_density_changes=coverage, retrospective_paired_M2=moments,
        M2_scope='Same 78 selected saved competing55 contributions, split by original proposal arm/population. No new physical weights, normalizer, unseen-tail bound or population convergence claim.',
        independent_audit=dict(maximum_errors=independent['maximum_errors'], max_single_axis_vs_old_error=max_old_x_error,
                               max_equal_axis_mixture_error=max_mixture_error),
        CPU=dict(execution=execution, query_CPU_seconds=summary['probe_total_cpu_seconds'],
                 complete_density_CPU_seconds=sum(r['density_cpu_seconds'] for r in diagnostic),
                 density_by_group={g:sum(r['density_cpu_seconds'] for r in diagnostic if r['group']==g) for g in ('baseline256','critical78','breadth128')},
                 independent_audit_seconds=independent['analysis_cpu_seconds'], analysis_total_seconds=time.process_time()-start),
        protocol_sha256=sha(out / 'protocol.json'), freeze_sha256=sha(out / 'freeze.json'),
        physics_conclusion='Unresolved; this diagnostic does not open physical or assembly gates.')
    write(out / 'row-diagnostics.json', dict(rows=diagnostic))
    write(out / 'analysis.json', analysis)
    verify(out, read(out / 'freeze.json')['files'])
    write(out / 'receipt.json', dict(complete=True, outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'inputs' not in p.parts},
                                    helper_sha256=sha(__file__)))
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        require(args.pilot is not None, '--pilot is required for preparation')
        print(prepare(args.pilot, args.out))
    elif args.action == 'run':
        run(args.out)
    else:
        result = analyze(args.out)
        print(json.dumps({key:result[key] for key in ('complete','probes','retrospective_paired_M2','CPU')}, indent=2))


if __name__ == '__main__':
    main()
