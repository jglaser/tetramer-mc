#!/usr/bin/env python3
"""Plot all twelve audited mobile preparations without pooling trajectories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_observer(campaign, assessment, manifest, result, recovery):
    if recovery is None:
        assert result['analyzer_sha256'] == manifest['observer_sha256']
        return None
    recovery = Path(recovery).resolve()
    binding, status = read(recovery/'recovery.json'), read(recovery/'status.json')
    assert binding['schema'] == 'mobile-posterior-separate-audit-recovery-v1'
    assert binding['physical_reruns'] == 0 and Path(binding['source_campaign']).resolve() == campaign
    assert binding['source_campaign_manifest_sha256'] == result['manifest_sha256']
    assert binding['source_terminal_status_sha256'] == result['terminal_status_sha256']
    assert binding['original_observer_sha256'] == manifest['observer_sha256'] == result['original_frozen_observer_sha256']
    assert binding['recovery_observer_sha256'] == result['analyzer_sha256']
    assert result['observer_execution'] == 'separate-recovery-assessment'
    assert result['observer_reference'] == binding['observer_reference']
    assert status['complete'] and not status['running'] and status['exit_code'] == 0
    assert Path(status['assessment']).resolve() == assessment
    assert status['analysis_sha256'] == sha(assessment/'analysis.json')
    for name, digest in binding['code_sha256'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(recovery/'provenance'/relative) == digest
    assert binding['code_sha256']['analyze_mobile_posterior_pilot.py'] == result['analyzer_sha256']
    reference = binding['observer_reference']
    folder = recovery/'reference'
    assert Path(reference['reference']).resolve() == folder
    assert sha(folder/'reference-recovery.json') == reference['recovery_manifest_sha256']
    for name, digest in reference['original_reference_sha256'].items():
        assert sha(folder/name) == sha(campaign/'provenance/reference'/name) == digest
    addition = reference['added_coordinates']
    assert sha(folder/addition['path']) == addition['sha256']
    assert sha(reference['monomer_shape']) == reference['monomer_shape_sha256']
    return dict(package=str(recovery), binding_sha256=sha(recovery/'recovery.json'),
                status_sha256=sha(recovery/'status.json'), observer_reference=reference)


def plot(campaign, assessment, out, recovery=None):
    campaign, assessment, out = [Path(p).resolve() for p in (campaign, assessment, out)]
    assert not out.exists()
    manifest, result = read(campaign/'manifest.json'), read(assessment/'analysis.json')
    assert result['complete'] and result['manifest_sha256'] == sha(campaign/'manifest.json')
    assert result['terminal_status_sha256'] == sha(campaign/'status.json')
    recovery_record = validate_observer(campaign, assessment, manifest, result, recovery)
    modes, starts = ('capture_only', 'c0', 'c09'), ('dispersed', 'preassociated')
    runs = sorted(result['runs'], key=lambda r: (starts.index(r['start']), modes.index(r['mode']), r['replicate']))
    assert len(runs) == len({r['id'] for r in runs}) == 12
    assert {(r['start'], r['mode'], r['replicate']) for r in runs} == {(s, m, r) for s in starts for m in modes for r in (0, 1)}
    jobs = {j['id']: j for j in manifest['jobs']}
    for run in runs:
        assert run['passed']
        assert run['analyzer_sha256'] == result['analyzer_sha256']
        if recovery_record is not None:
            assert run['observer_reference'] == result['observer_reference']
        assert [row['sweep'] for row in run['rows']] == list(range(manifest['sweeps']+1))
        detail = assessment/'runs'/run['id']/'analysis.json'
        assert read(detail) == run
        for name, digest in run['source_sha256'].items():
            assert sha(Path(jobs[run['id']]['directory'])/name) == digest
    native = np.array([[r['native']['largest_component_size'] for r in run['rows']] for run in runs])
    contact = np.array([[r['nonspecific']['largest_component_size'] for r in run['rows']] for run in runs])
    assert np.isin(native, (1, 2, 3)).all() and np.isin(contact, (1, 2, 3)).all()
    labels = [f"{'Dispersed' if r['start'] == 'dispersed' else 'Preassociated'} · "
              f"{dict(capture_only='capture', c0='redraw', c09='transport')[r['mode']]} · {r['replicate']+1}" for r in runs]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8), sharex=True, constrained_layout=True)
    cmap = ListedColormap(['#e4e7ec', '#6bb6b0', '#225e94'])
    norm = BoundaryNorm([.5, 1.5, 2.5, 3.5], cmap.N)
    for ax, values, title in zip(axes, (native, contact),
            ('Native registered component (entry/retention hysteresis)', 'Any exclusion-contact component (native contacts included)')):
        im = ax.imshow(values, interpolation='nearest', aspect='auto', cmap=cmap, norm=norm,
                       extent=(-.5, manifest['sweeps']+.5, len(runs)-.5, -.5))
        ax.set_yticks(range(len(runs)), labels, fontsize=8)
        ax.axvline(manifest['burn_sweeps'], color='#bd673e', linestyle='--', linewidth=1.4)
        for y in (1.5, 3.5, 5.5, 7.5, 9.5):
            ax.axhline(y, color='white', linewidth=2 if y != 5.5 else 4)
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_ylabel('Independent run')
    axes[-1].set_xlabel('MC sweeps; every stored endpoint and repeat included (dashed line: fixed burn)')
    color = fig.colorbar(im, ax=axes, ticks=[1, 2, 3], fraction=.022, pad=.015)
    color.set_label('Largest component: tetramers')
    fig.suptitle('Three mobile tetramers: association and persistence\n'
                 'Native-informed frozen atlas · rd = 1.5 Å · z = 0.035 Å⁻³ · sphere radius 223.33 Å', fontsize=14)
    out.mkdir(parents=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(out/f'mobile-posterior-assembly.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    data = dict(run_ids=[r['id'] for r in runs], labels=labels, native_component=native.tolist(),
        exclusion_component=contact.tolist(), burn_sweeps=manifest['burn_sweeps'],
        scope='Every run retained. Hysteretic native registry and instantaneous exclusion contact are distinct. '
              'Connectivity is not crystallinity; persistence is not equilibrium stability or a kinetic rate.')
    (out/'plot-data.json').write_text(json.dumps(data, allow_nan=False)+'\n')
    provenance = out/'provenance'
    provenance.mkdir()
    for name, path in [('plotter.py', Path(__file__).resolve()), ('campaign-manifest.json', campaign/'manifest.json')]:
        shutil.copy2(path, provenance/name)
    report = dict(complete=True, analysis_sha256=sha(assessment/'analysis.json'),
        manifest_sha256=sha(campaign/'manifest.json'), plotter_sha256=sha(__file__),
        recovery=recovery_record,
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}, scope=data['scope'])
    (out/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(out/'mobile-posterior-assembly.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, help='Explicit separately bound observer recovery package')
    args = parser.parse_args()
    plot(args.campaign, args.assessment, args.out, args.recovery)
