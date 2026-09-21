#!/usr/bin/env python3
"""Summarize audited four-body controls without rerunning physical or trace audits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from native_graph_consistency import NativeGraphConsistency


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(campaign, out):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    assert not out.exists(), 'Use a fresh comparison output'
    protocol, status = read(root/'protocol.json'), read(root/'status.json')
    assert protocol['schema'] == 'matched-mobile-four-body-growth-controller-v1'
    assert status['complete'] and status['phase'] == 'complete' and status['protocol_sha256'] == sha(root/'protocol.json')
    for name, digest in read(root/'freeze.json')['files'].items(): assert sha(root/name) == digest
    assert len(status['jobs']) == 8 and all(j['status'] == 'complete' and j['returncode'] == 0 for j in status['jobs'])
    reports, traces, sources = [], [], {}
    for arm in ('original', 'coverage'):
        folder = root/arm; manifest = read(folder/'manifest.json')
        assessment_path = folder/'assessment/analysis.json'
        assert status['audits'][arm]['returncode'] == 0 and sha(assessment_path) == status['audits'][arm]['analysis_sha256']
        assessment = read(assessment_path)
        assert assessment['complete'] and len(assessment['runs']) == 4
        assert assessment['manifest_sha256'] == sha(folder/'manifest.json')
        assert assessment['terminal_status_sha256'] == sha(folder/'status.json')
        assert assessment['analyzer_sha256'] == manifest['observer_sha256']
        sources[str(assessment_path)] = sha(assessment_path)
        catalogue = folder/'provenance/native-pair-motifs.json'
        assert sha(catalogue) == manifest['input_sha256']['native-pair-motifs.json']
        checker = NativeGraphConsistency(read(catalogue)['motifs'], 4)
        sources[str(catalogue)] = sha(catalogue)
        for result, job in zip(assessment['runs'], manifest['jobs']):
            assert result['passed'] and (result['id'], result['seed']) == (job['id'], job['seed'])
            terminal = next(j for j in status['jobs'] if j['arm'] == arm and j['id'] == job['id'])
            for name, digest in result['source_sha256'].items():
                path = Path(job['directory'])/name
                assert sha(path) == digest == terminal['output']['files'][name]
                sources[str(path)] = digest
            growth = result['four_body_growth']; assert growth['body_count'] == 4 and growth['tracked_body_index'] == 3
            assert len(growth['pair_labels']) == 6 and len(result['rows']) == protocol['sweeps']+1
            assert result['audit']['all_move_records'] == 6*protocol['sweeps']
            outcomes = growth['outcomes']; native = outcomes['native']; contact = outcomes['nonspecific']; entry = outcomes['instantaneous_entry']
            support = growth['supported_monomer_bonds_by_saved_sweep']
            first_growth = native['first_observed']['all_four_connected']
            coherent = {}
            for name, field in [('retained', 'registered_keys'), ('entry', 'instantaneous_entry_keys')]:
                first_coherent = None; inconsistent = {}; inconsistent_updates = 0
                for event in result['graph_history']:
                    keys = event[field]; check = checker.check(keys)
                    if check['consistent_connected'] and first_coherent is None:
                        first_coherent = dict(serial=event['serial'], sweep=event['sweep'], source=event['source'], initial_state=event['serial'] == -1)
                    if not check['consistent']:
                        inconsistent_updates += 1
                        signature = str(sorted(keys))
                        inconsistent.setdefault(signature, dict(first_serial=event['serial'], first_sweep=event['sweep'], keys=keys, check=check))
                coherent[name] = dict(first_consistent_connected_four=first_coherent, inconsistent_observations=inconsistent_updates,
                    incompatible_graph_witnesses=list(inconsistent.values()), initial=checker.check(result['graph_history'][0][field]),
                    final=checker.check(result['graph_history'][-1][field]))
            reports.append(dict(arm=arm, id=result['id'], start=result['start'], replicate=result['replicate'], seed=result['seed'],
                sampler_cpu_seconds=result['sampler_cpu_seconds'],
                first_D_contact=contact['first_observed']['tracked_attached'],
                first_D_native_entry=entry['first_observed']['tracked_attached'],
                first_native_connected_four=first_growth,
                observed_formation_of_native_four=first_growth is not None and not first_growth['initial_state'],
                final_native_component_size=result['rows'][-1]['native']['largest_component_size'],
                final_D_strict_supported_bonds=support[-1]['tracked_entry_count'],
                final_D_retained_supported_bonds=support[-1]['tracked_retained_count'],
                maximum_D_strict_supported_bonds=max(r['tracked_entry_count'] for r in support),
                catalogue_consistency=coherent,
                final_registered_keys=result['rows'][-1]['registered_keys'],
                outcomes=outcomes, native_candidates=result['native_candidates'], branch_counts=result['branch_counts'],
                graph_metrics=result['graph_metrics'], audit=result['audit']))
            traces.append(dict(arm=arm, id=result['id'], start=result['start'], replicate=result['replicate'],
                sweep=[r['sweep'] for r in result['rows']],
                native_size=[r['native']['largest_component_size'] for r in result['rows']],
                contact_size=[r['nonspecific']['largest_component_size'] for r in result['rows']],
                D_strict_bonds=[r['tracked_entry_count'] for r in support],
                D_retained_bonds=[r['tracked_retained_count'] for r in support]))
    assert len(reports) == 8
    result = dict(schema='mobile-four-body-growth-comparison-v1', complete=True, campaign=str(root),
        protocol_sha256=sha(root/'protocol.json'), status_sha256=sha(root/'status.json'), analyzer_sha256=sha(__file__),
        source_sha256=sources, sweeps=protocol['sweeps'], runs=reports, traces=traces,
        scope='All eight native-informed, nonequilibrium four-body controls retained. First formation excludes initial attached states. Native registry uses entry/retention thresholds; strict entry and monomer support are separate observables. Short trajectories and apparent ESS do not establish stationary efficiency, equilibrium association, physical rates, or template-free assembly.')
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Four-mobile-tetramer attachment and retention', '', result['scope'], '',
        '| Atlas | Start | Repeat | First D contact | First D native entry | First native connected4 | First catalogue-consistent entry4 | Final largest native group | Final D strict/retained monomer bonds | CPU s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def first(value):
        return 'not observed' if value is None else 'initial' if value['initial_state'] else str(value['sweep'])
    for r in reports:
        lines.append('| '+' | '.join([r['arm'], r['start'], str(r['replicate']), first(r['first_D_contact']),
            first(r['first_D_native_entry']), first(r['first_native_connected_four']),
            first(r['catalogue_consistency']['entry']['first_consistent_connected_four']), str(r['final_native_component_size']),
            f"{r['final_D_strict_supported_bonds']}/{r['final_D_retained_supported_bonds']}", f"{r['sampler_cpu_seconds']:.2f}"])+' |')
    lines += ['', 'All event serials, kernels, contact losses/returns and partner exchanges, ABC triangle survival, initial-state censoring, '
        'strict versus retained bonds, proposal corrections for new native candidates, and per-run graph diagnostics are in analysis.json. '
        'A four-body connected graph is sufficient for the accessibility endpoint; a complete six-edge clique is not required. '
        'A separate catalogue test requires that every cycle admit a consistent ideal SE(3) pose assignment, to 1e-6 Å and 1e-6 radians. '
        'Multiple motif labels on a pair are alternatives. Measured poses are never repaired; failed cycle checks remain explicit.', '',
        '![Every run, every saved sweep](four-body-growth.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    plot(traces, out)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for name in ('native_graph_consistency.py', 'test_native_graph_consistency.py'):
        (out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    (out/'freeze.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()}, indent=2)+'\n')
    print(json.dumps(dict(complete=True, out=str(out), analysis_sha256=sha(out/'analysis.json'))))
    return result


def plot(traces, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, sharey=True, layout='constrained')
    for r in traces:
        row = 2*int(r['start'] == 'retained_motif8')+r['replicate']; column = int(r['arm'] == 'coverage')
        ax = axes[row, column]
        ax.step(r['sweep'], r['contact_size'], where='post', color='#9b9b9b', lw=1.1, label='largest exclusion-connected group')
        ax.step(r['sweep'], r['native_size'], where='post', color='#2459a6', lw=1.2, label='largest native-connected group')
        ax.plot(r['sweep'], r['D_strict_bonds'], color='#cb6a20', lw=.8, alpha=.85, label='D: strict supported monomer bonds')
        ax.set_title(f"{r['arm']} · {r['start']} · repeat {r['replicate']}", fontsize=10)
        ax.grid(axis='y', alpha=.2)
        if column == 0: ax.set_ylabel('Body count / monomer bonds')
        if row == 3: ax.set_xlabel('MC sweep')
    ymax = max(max(r[field]) for r in traces for field in ('native_size', 'contact_size', 'D_strict_bonds'))
    axes[0, 0].set_ylim(0, ymax+.4)
    axes[0, 0].legend(loc='upper right', fontsize=7)
    fig.suptitle('Growth beyond a native triangle: all eight mobile controls\nNative-informed proposals; these trajectories do not establish equilibrium', fontsize=12)
    for suffix in ('png', 'svg'): fig.savefig(out/f'four-body-growth.{suffix}', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--campaign', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); summarize(args.campaign, args.out)
