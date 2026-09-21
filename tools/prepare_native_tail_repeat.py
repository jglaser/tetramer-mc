#!/usr/bin/env python3
"""Freeze a larger native complete-cover repeat without new geometry or physics draws."""
import argparse
import ast
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/native-tail-reference-preparation-20260921'
AMENDMENT = ROOT/'runs/native-tail-analyzer-amendment-v2-20260921'
BASE_SHA = 'a6d4c95f7625d108dad48255d511b8a5a8b5359b7f338cecf2f76af3b5ed5618'
AMENDMENT_SHA = '3089d8d5ab47c7ba0064a4578bdef74c442ffb7e89c47c4cfd42a246022ca73f'
SEED, POPULATIONS, SAMPLES, WORKERS = 114501010, 16, 262144, 16
PREFIX_POPULATIONS = (4, 8, 16)
LAW_FILES = ('config.json', 'model.json', 'old-model.json', 'region.json', 'complete-cover-proof.json')
LAW_FIELDS = ('schema', 'original_q_window', 'old_chart_reporting_edges', 'reporting_masks',
              'old_chart_tail_rule', 'sampling', 'weight', 'old_chart_role', 'completeness',
              'executable_sha256', 'config_sha256', 'region_sha256', 'model_sha256',
              'old_model_sha256', 'exact_cover_proof_sha256')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def require(condition, message):
    if not condition: raise ValueError(message)


def validate_repeat(base, repeated):
    """Check the statistical law and exact approved allocation independently of CLI."""
    require(all(repeated[k] == base[k] for k in LAW_FIELDS), 'Physical law or reporting masks changed')
    old, new = base['proposed_campaign'], repeated['proposed_campaign']
    require(new['samples_per_population'] == SAMPLES and new['populations'] == POPULATIONS
            and new['workers'] == WORKERS, 'Changed repeat allocation')
    require(new['seed_base'] == SEED and new['seeds'] == [SEED+1009*i for i in range(POPULATIONS)],
            'Changed repeat seed schedule')
    require(len(set(new['seeds'])) == POPULATIONS and not set(new['seeds']).intersection(old['seeds'])
            and base['geometry_probe_seed'] not in new['seeds'], 'Reused seed')
    require(new['lambda_ratio'] == old['lambda_ratio'] == 64.
            and new['cloud_replicates'] == old['cloud_replicates'] == 2, 'Changed cloud law')
    require(SAMPLES*POPULATIONS == 32*old['samples_per_population']*old['populations'], 'Wrong budget ratio')
    require(repeated['geometry_probe_count'] == 0 and repeated['geometry_probe_seed'] is None,
            'Repeat must not schedule new geometry probes')
    require(repeated['analysis_prefixes'] == [dict(populations=n, samples=n*SAMPLES,
                population_ids=[f'r{i:02d}' for i in range(n)]) for n in PREFIX_POPULATIONS],
            'Changed predeclared population prefixes')


def runner_siblings(path):
    tree = ast.parse(Path(path).read_text())
    return sorted({node.args[0].value for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'with_name' and len(node.args) == 1
                   and isinstance(node.args[0], ast.Constant)})


def mock_launcher(command, declared):
    """Exercise the unchanged launcher, replacing *every* kernel call with a stub."""
    calls = []
    with tempfile.TemporaryDirectory(prefix='native-tail-repeat-preflight-') as directory:
        mock_output = Path(directory)/'campaign'
        argv = list(command[1:]); argv[argv.index('--out')+1] = str(mock_output)
        def no_kernel(args, **kwargs):
            require(Path(args[0]).name == 'latent-region-normalizer', 'Unexpected subprocess in launcher')
            calls.append(list(args))
            return subprocess.CompletedProcess(args, 0)
        with patch.object(sys, 'argv', argv), patch('subprocess.run', side_effect=no_kernel), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(argv[0], run_name='__main__')
        master = read(mock_output/'manifest.json')
        require(len(calls) == len(master['jobs']) == declared['populations'], 'Wrong number of launcher jobs')
        require([job['seed'] for job in master['jobs']] == declared['seeds'], 'Launcher changed seeds')
        require(all(job['samples'] == declared['samples_per_population'] for job in master['jobs']), 'Launcher changed N')
        require(master['workers'] == declared['workers'] and master['lambda_ratio'] == declared['lambda_ratio']
                and master['cloud_replicates'] == declared['cloud_replicates'], 'Launcher changed allocation or bath')
        for args in calls:
            flags = dict(zip(args[1::2], args[2::2]))
            require(int(flags['--samples']) == declared['samples_per_population']
                    and float(flags['--lambda-ratio']) == declared['lambda_ratio']
                    and int(flags['--cloud-replicates']) == declared['cloud_replicates'], 'Changed kernel arguments')
        require(sorted(int(args[args.index('--seed')+1]) for args in calls) == declared['seeds'], 'Wrong kernel seeds')
        require(not (mock_output/'runs').exists(), 'Mock launcher unexpectedly created physical runs')
        require(len(list(mock_output.glob('r*-status.json'))) == declared['populations'], 'Missing mocked terminal statuses')
        return dict(complete=True, actual_kernel_executions=0, mocked_jobs=len(calls),
                    samples_per_population=declared['samples_per_population'], seeds=declared['seeds'],
                    all_runner_source_copies_succeeded=True)


def prepare(out, campaign, audit):
    out, campaign, audit = [Path(p).resolve() for p in (out, campaign, audit)]
    require(len({out, campaign, audit}) == 3 and all(not p.exists() for p in (out, campaign, audit)), 'Use distinct fresh output paths')
    require(sha(BASE/'protocol.json') == BASE_SHA and sha(AMENDMENT/'amendment.json') == AMENDMENT_SHA,
            'Upstream protocol or amendment changed')
    base, amendment = read(BASE/'protocol.json'), read(AMENDMENT/'amendment.json')
    for name, digest in read(BASE/'freeze.json').items():
        require(sha(BASE/name) == digest, 'Changed base preparation: '+name)
    for name, digest in amendment['base_preparation_sha256'].items():
        require(sha(BASE/name) == digest, 'Changed amended base input: '+name)
    for directory, hashes in ((BASE, base['archived_sha256']), (AMENDMENT, amendment['archived_sha256'])):
        for name, digest in hashes.items():
            require(sha(directory/'provenance'/name) == digest, 'Changed upstream archive: '+name)
    sources = {name:BASE/'provenance'/name for name in base['archived_sha256']}
    sources.update({name:AMENDMENT/'provenance'/name for name in amendment['archived_sha256']})
    # Help output embeds its old script paths; keep it under explicit inherited names.
    for name in ('runner-help.txt', 'analyzer-help.txt'):
        sources['inherited-'+name] = sources.pop(name)
    sources.update({'repeat-origin-protocol.json':BASE/'protocol.json',
                    'repeat-origin-freeze.json':BASE/'freeze.json',
                    'repeat-origin-report.json':BASE/'report.json',
                    'repeat-origin-geometry-probes.jsonl':BASE/'geometry-probes.jsonl',
                    'repeat-origin-amendment.json':AMENDMENT/'amendment.json',
                    Path(__file__).name:Path(__file__),
                    'test_native_tail_repeat.py':Path(__file__).with_name('test_native_tail_repeat.py')})
    archive = out/'provenance'; archive.mkdir(parents=True)
    for name, source in sources.items(): shutil.copy2(source, archive/name)
    for name in LAW_FILES: shutil.copy2(BASE/name, out/name)
    cfg = read(out/'config.json')
    require(sha(cfg['shape']) == sha(archive/'shape.json') == read(out/'region.json')['shape_sha256'], 'Changed config shape')
    frozen = copy.deepcopy(base)
    frozen.update(created_utc=datetime.now(timezone.utc).isoformat(), production_launched=False,
        geometry_probe_count=0, geometry_probe_seed=None,
        geometry_gate='No new geometry probes or refits. Reuse the byte-identical complete cover and archived pilot geometry evidence.',
        repeat_of=dict(preparation=str(BASE), protocol_sha256=BASE_SHA, analyzer_amendment=str(AMENDMENT),
                       analyzer_amendment_sha256=AMENDMENT_SHA, budget_multiplier=32,
                       inherited_geometry_report_sha256=sha(BASE/'report.json'),
                       inherited_geometry_probes_sha256=sha(BASE/'geometry-probes.jsonl')))
    physical = [sys.executable, str(archive/'run_latent_region_campaign.py'),
        '--out', str(campaign), '--config', str(out/'config.json'), '--region', str(out/'region.json'),
        '--binary', str(archive/'latent-region-normalizer'), '--samples', str(SAMPLES),
        '--replicates', str(POPULATIONS), '--workers', str(WORKERS), '--seed', str(SEED),
        '--lambda-ratio', '64', '--cloud-replicates', '2']
    analysis = [sys.executable, str(archive/'analyze_native_tail_reference.py'),
                '--preparation', str(out), '--campaign', str(campaign), '--out', str(audit)]
    frozen['proposed_campaign'] = dict(output=str(campaign), samples_per_population=SAMPLES,
        populations=POPULATIONS, workers=WORKERS, seed_base=SEED, seeds=[SEED+1009*i for i in range(POPULATIONS)],
        lambda_ratio=64., cloud_replicates=2, command_argv_after_python=physical[1:])
    frozen['analysis_command_argv_after_python'] = analysis[1:]
    frozen['physical_launch_command_argv'], frozen['analysis_command_argv'] = physical, analysis
    frozen['analysis_prefixes'] = [dict(populations=n, samples=n*SAMPLES,
        population_ids=[f'r{i:02d}' for i in range(n)]) for n in PREFIX_POPULATIONS]
    frozen['prefix_analysis_rule'] = ('First 4, 8 and 16 complete equal-size populations, in declared order; '
        'merge original first/second moments under this one law and preserve all zeros. Prefixes are nested '
        'and correlated. The raw auditor remains unchanged; a separate comparison layer reconstructs '
        'these prefixes and verifies full reconstruction. Pilot and targeted-reference samples stay separate.')
    validate_repeat(base, frozen)
    checks = dict(complete=True, no_physical_or_audit_execution=True, no_new_geometry_probes=True,
        no_refits=True, physical_law_files_byte_identical={name:sha(out/name) for name in LAW_FILES},
        total_draws=SAMPLES*POPULATIONS, budget_multiplier=32,
        analyzer_sha256=sha(archive/'analyze_native_tail_reference.py'),
        runner_sha256=sha(archive/'run_latent_region_campaign.py'),
        runner_required_siblings=runner_siblings(archive/'run_latent_region_campaign.py'))
    for name in checks['runner_required_siblings']: require((archive/name).is_file(), 'Missing runner dependency: '+name)
    require(checks['analyzer_sha256'] == amendment['archived_sha256']['analyze_native_tail_reference.py'], 'Changed analyzer')
    require(checks['runner_sha256'] == amendment['archived_sha256']['run_latent_region_campaign.py'], 'Changed runner')
    for name in LAW_FILES: require(sha(out/name) == sha(BASE/name), 'Changed law file: '+name)
    for label, command in (('runner', physical), ('analyzer', analysis),
                           ('executable', [str(archive/'latent-region-normalizer')])):
        result = subprocess.run(command[:2]+['--help'] if label != 'executable' else command+['--help'],
                                text=True, capture_output=True, check=True)
        (archive/(label+'-help.txt')).write_text(result.stdout)
        flags = [arg for arg in command if arg.startswith('--')]
        require(all(flag in result.stdout for flag in flags), 'Undeclared '+label+' CLI flag')
    checks['mocked_launcher'] = mock_launcher(physical, frozen['proposed_campaign'])
    # Frozen tests validate the emitted cover, masks, Poisson law and repeat contract.
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(archive),
                             '-p', 'test_native_tail*.py', '-v'], text=True, capture_output=True)
    (archive/'tests.txt').write_text(result.stdout+result.stderr)
    require(result.returncode == 0, 'Frozen tests failed; inspect provenance/tests.txt')
    checks['frozen_tests_returncode'] = result.returncode
    for name, source in sources.items(): require(sha(archive/name) == sha(source), 'Source changed while preparing: '+name)
    frozen['input_sha256'] = {str(source):sha(source) for source in sources.values()}
    frozen['archived_sha256'] = {p.name:sha(p) for p in sorted(archive.iterdir()) if p.is_file()}
    write(out/'preflight.json', checks)
    frozen['preflight_sha256'] = sha(out/'preflight.json')
    write(out/'protocol.json', frozen)
    write(out/'freeze.json', {p.name:sha(p) for p in sorted(out.glob('*.json'))})
    require(not campaign.exists() and not audit.exists(), 'Preflight created production or audit output')
    print(json.dumps(dict(complete=True, preparation=str(out), protocol_sha256=sha(out/'protocol.json'),
                         preflight_sha256=sha(out/'preflight.json'),
                         physical_launch_command_argv=physical, analysis_command_argv=analysis), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--campaign-out', type=Path, required=True)
    parser.add_argument('--audit-out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.out, args.campaign_out, args.audit_out)
