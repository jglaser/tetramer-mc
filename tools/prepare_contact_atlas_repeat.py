#!/usr/bin/env python3
"""Freeze larger independent allocations without changing either atlas law."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from prepare_cayley_rms_cover import read, write, sha, require

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'runs/ab-intermediate-expanded-atlas-preparation-20260920'
SEEDS = dict(narrow=103101010, broad=103201010)
POPULATIONS, SAMPLES = 16, 65536
PREFIXES = [16384, 32768, 65536]


def prepare(out):
    require(not out.exists(), 'Use a fresh repeat preparation directory')
    source = read(ORIGINAL/'protocol.json'); seal = read(ORIGINAL/'freeze.json')
    require(source['schema'] == 'intermediate-expanded-contact-atlas-v1', 'Wrong original protocol')
    require(sha(ORIGINAL/'protocol.json') == seal['protocol_sha256'], 'Original protocol changed')
    require(sha(ORIGINAL/'config.json') == seal['config_sha256'], 'Original config changed')
    for name, digest in seal['archived_sha256'].items():
        require(sha(ORIGINAL/'provenance'/name) == digest, 'Original preparation archive changed')
    require(set(seal['model_sha256']) == set(SEEDS), 'Wrong original proposal arms')
    for name, digest in seal['model_sha256'].items():
        require(sha(ORIGINAL/f'model-{name}.json') == digest, 'Original model changed')
    require(sha(Path(source['executable'])) == source['executable_sha256'], 'Physical executable changed')
    cfg = read(ORIGINAL/'config.json'); shape = Path(cfg['shape'])
    require(sha(shape) == read(ORIGINAL/'model-narrow.json')['shape_sha256'], 'Original shape changed')
    prior_seeds = {seed for arm in source['arms'] for seed in arm['seeds']}
    seeds = [base+1009*i for base in SEEDS.values() for i in range(POPULATIONS)]
    require(len(set(seeds)) == len(seeds) and not prior_seeds.intersection(seeds), 'Repeated production stream')
    archive = out/'provenance'; archive.mkdir(parents=True)
    inputs = {'original-protocol.json':ORIGINAL/'protocol.json', 'original-freeze.json':ORIGINAL/'freeze.json',
              'input-config.json':ORIGINAL/'config.json', 'shape.json':shape,
              'prepare_contact_atlas_repeat.py':Path(__file__)}
    for name, path in inputs.items(): shutil.copy2(path, archive/name)
    repeated_cfg = copy.deepcopy(cfg); repeated_cfg['shape'] = str(archive/'shape.json')
    write(out/'config.json', repeated_cfg)
    for name in SEEDS: shutil.copy2(ORIGINAL/f'model-{name}.json', out/f'model-{name}.json')
    protocol = copy.deepcopy(source)
    protocol['created_utc'] = datetime.now(timezone.utc).isoformat()
    protocol['repeat_source'] = dict(path=str(ORIGINAL/'protocol.json'), sha256=seal['protocol_sha256'],
        model_sha256=copy.deepcopy(seal['model_sha256']),
        rule='Both proposal files are byte-identical to the original models. Launch with the original model source paths because their embedded provenance names the original protocol. Only independent streams, fixed N and population count change.')
    protocol['prefix_counts'] = PREFIXES
    protocol['prefix_rule'] = 'First 16384, 32768 and all 65536 unconditional draws inside each of sixteen independent populations; prefixes are correlated. The first prefix has the same aggregate draw count as the old complete campaign. No adaptive stop, refit or row replacement.'
    for arm in protocol['arms']:
        name = arm['name']; arm.update(samples_per_population=SAMPLES, populations=POPULATIONS,
            seeds=[SEEDS[name]+1009*i for i in range(POPULATIONS)],
            output=str(ROOT/f'runs/ab-intermediate-expanded-atlas-{name}-16x65536-l64-20260920'),
            model_source_path=str(ORIGINAL/f'model-{name}.json'))
        require(not Path(arm['output']).exists(), 'Repeat campaign output already exists')
    protocol['selection_scope'] = 'Historical models are held unchanged. The larger independent repeat budget was selected after the earlier run; all current budgets and prefixes are fixed before drawing. Old and new complete campaigns may be compared as independent frozen-proposal measurements, without pooling during validation.'
    protocol['input_sha256'] = {str(path):sha(path) for path in inputs.values()}
    protocol['archived_sha256'] = {name:sha(archive/name) for name in inputs}
    write(out/'protocol.json', protocol)
    frozen = dict(protocol_sha256=sha(out/'protocol.json'), config_sha256=sha(out/'config.json'),
        model_sha256={name:sha(out/f'model-{name}.json') for name in SEEDS}, archived_sha256=protocol['archived_sha256'])
    require(frozen['model_sha256'] == seal['model_sha256'], 'Repeat model bytes changed')
    write(out/'freeze.json', frozen)
    write(out/'report.json', dict(complete=True, no_new_geometry_or_physical_draws=True,
        original_protocol_sha256=seal['protocol_sha256'], repeat_protocol_sha256=frozen['protocol_sha256'],
        identical_model_sha256=frozen['model_sha256'], independent_new_seeds=seeds,
        total_unconditional_draws=2*POPULATIONS*SAMPLES,
        scope='Allocation-only independent repeat. Reuses validated models, physical kernel and analysis regions; no new fitting or probe-based tuning.'))
    print(out/'protocol.json')
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.out.resolve())
