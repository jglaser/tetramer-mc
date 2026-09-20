#!/usr/bin/env python3
"""Execute the frozen two-region plan with a shared CPU guard and four workers."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from prepare_smc_normalizer_atlas import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan_path, out = args.plan.resolve(), args.out.resolve()
    prep = plan_path.parent
    plan, fit, analysis = read(plan_path), read(prep/'report.json'), read(prep/'analysis-plan.json')
    assert sha(plan_path) == analysis['validation_plan_sha256']
    for name, digest in fit['input_sha256'].items():
        assert sha(prep/'provenance'/name) == digest, name
    for name, digest in analysis['analyzer_sha256'].items():
        assert sha(prep/'provenance'/name) == digest == sha(ROOT/'tools'/name), name
    assert sha(ROOT/'tools/run_latent_region_campaign.py') == fit['input_sha256']['run_latent_region_campaign.py']
    assert sha(plan['physical_config']) == plan['physical_config_sha256']
    assert sha(plan['executable']) == plan['executable_sha256']
    assert plan['samples_per_population'] == 16384 and plan['replicates_per_radius'] == 4
    assert plan['workers_per_campaign'] == 2 and plan['total_workers'] == 4
    assert plan['lambda_ratio'] == 64 and plan['cloud_replicates'] == 2
    assert plan['seeds'] == {'3':98621010, '5':98631010} and plan['seed_increment'] == 1009
    for key, region in plan['regions'].items():
        assert sha(region['path']) == region['sha256']
        assert not Path(plan['suggested_outputs'][key]).exists(), 'Never overwrite an existing campaign'
    reference_path = ROOT/'runs/latent-region-uniform-ball-16384-l64/assessment/analysis.json'
    reference_cpu = read(reference_path)['sampler_cpu_seconds']
    guard = 4*reference_cpu
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'supervisor.py')
    shutil.copy2(plan_path, out/'validation-plan.json')
    commands = {}
    for key in ('3', '5'):
        commands[key] = [str(ROOT.parent/'protein-nucleation/.venv/bin/python'), '-B',
            str(ROOT/'tools/run_latent_region_campaign.py'), '--out', plan['suggested_outputs'][key],
            '--config', plan['physical_config'], '--region', plan['regions'][key]['path'],
            '--binary', plan['executable'], '--samples', '16384', '--replicates', '4', '--workers', '2',
            '--seed', str(plan['seeds'][key]), '--lambda-ratio', '64', '--cloud-replicates', '2']
    manifest = dict(created_unix=time.time(), authoritative_worktree=str(ROOT),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        plan_sha256=sha(plan_path), supervisor_sha256=sha(__file__), executable_sha256=plan['executable_sha256'],
        commands=commands, reference_analysis=str(reference_path), reference_sha256=sha(reference_path),
        reference_cpu_seconds=reference_cpu, combined_cpu_guard_seconds=guard,
        guard_scope='Stop both process groups after combined observed sampler CPU exceeds4x the original single-R3 reference. Preserve all partial/failed outputs; do not analyze them as fixed-N completed estimates.')
    write(out/'manifest.json', manifest)
    processes, streams, seen_cpu = {}, {}, {}
    for key, command in commands.items():
        streams[key] = (out/f'campaign-r{key}.log').open('w')
        processes[key] = subprocess.Popen(command, cwd=ROOT, stdout=streams[key], stderr=subprocess.STDOUT,
            start_new_session=True)
    write(out/'processes.json', {key:dict(pid=p.pid, process_group=p.pid, command=commands[key]) for key,p in processes.items()})
    print(json.dumps(dict(started=True, supervisor_pid=os.getpid(), processes={k:p.pid for k,p in processes.items()},
        out=str(out), combined_cpu_guard_seconds=guard)), flush=True)
    reason = None
    try:
        while True:
            progress = {}
            for key in ('3', '5'):
                rows = []
                for rep in range(4):
                    directory = Path(plan['suggested_outputs'][key])/'runs'/f'r{rep:02d}'
                    selected = None
                    for name in ('summary.json', 'progress.json'):
                        path = directory/name
                        if path.exists():
                            try:
                                selected = read(path)
                                break
                            except json.JSONDecodeError:
                                pass
                    if selected:
                        identifier = f'{key}-{rep}'
                        seen_cpu[identifier] = max(seen_cpu.get(identifier, 0.), selected['sampler_cpu_seconds'])
                        rows.append(dict(replicate=rep, samples=selected.get('completed_draws', selected.get('samples')),
                            complete=selected.get('complete', False), cpu_seconds=seen_cpu[identifier]))
                progress[key] = rows
            cpu = sum(seen_cpu.values())
            codes = {key:p.poll() for key,p in processes.items()}
            if cpu > guard:
                reason = 'combined_cpu_guard_exceeded'
            elif any(code is not None and code != 0 for code in codes.values()):
                reason = 'campaign_process_failed'
            status = dict(running=not all(code is not None for code in codes.values()), processes=codes,
                progress=progress, combined_sampler_cpu_seconds=cpu, cpu_guard_seconds=guard, stop_reason=reason)
            write(out/'status.json', status)
            if reason or not status['running']:
                break
            time.sleep(5)
    except BaseException:
        reason = 'supervisor_interrupted'
        raise
    finally:
        if reason:
            for process in processes.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for stream in streams.values():
            stream.close()
    complete = reason is None and all(p.returncode == 0 for p in processes.values())
    if complete:
        for key in ('3', '5'):
            for rep in range(4):
                summary = read(Path(plan['suggested_outputs'][key])/'runs'/f'r{rep:02d}'/'summary.json')
                assert summary['complete'] and summary['samples'] == 16384
    final = dict(complete=complete, stop_reason=reason, combined_sampler_cpu_seconds=sum(seen_cpu.values()),
        processes={key:p.returncode for key,p in processes.items()}, plan_sha256=sha(plan_path))
    write(out/'summary.json', final)
    print(json.dumps(final), flush=True)
    if not complete:
        raise SystemExit('Frozen production stopped; partial records retained and not analyzed as completed estimates')


if __name__ == '__main__':
    main()
