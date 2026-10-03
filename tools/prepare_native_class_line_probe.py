#!/usr/bin/env python3
"""Freeze a proposal-only class-line diagnostic; never launch or draw samples."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from prepare_hard_free_line_score import OLD92_SHA, REGION_SHA, SHAPE_SHA, read, require, sha, write

COMPILED_SHA = 'dbb3c3e32259f506b9c979ebb9d1fd773cb78507c1f808fd8dd0ed319e2eade4'
SELECTED_SHA = '1aa84a153db220a69bf5bad5602ccf1f26f059921f9e8836cb426cf25d5fdc72'
SAVED_SHA = '2c0fb60eaedb159553655a32edc8886744873272545f3ce9befe246a63ef7e06'
CHANNELS = [dict(class_='hard_free', probability=.2),
            dict(class_='contact_without_native', probability=.2),
            dict(class_='contact_without_native', probability=.2, orthant=22),
            dict(class_='contact_without_native', probability=.2, orthant=62),
            dict(class_='native', probability=.2, orthant=55)]


def channels(arm):
    require(arm in ('hard_free', 'class'), 'Unknown proposal arm')
    records = CHANNELS if arm == 'class' else [dict(class_='hard_free', probability=1.)]
    return [dict(**{'class': v['class_']}, **{k: x for k, x in v.items() if k != 'class_'}) for v in records]


def jobs():
    result = []
    for stream in range(4):
        for arm in ('hard_free', 'class'):
            name = f'native-class-line-proposal-only-20261003/{arm}/{stream}'
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
            result.append(dict(id=f'{arm}-r{stream:02}', arm=arm, stream=stream,
                               samples=128, seed=seed, seed_namespace=name))
    require(len({j['seed'] for j in result}) == 8, 'Seed collision')
    return result


def select_probes(selected, saved):
    require(len(selected) == 8 and Counter(v['group'] for v in selected) ==
            {'competing22': 2, 'competing62': 2, 'old-native55': 2, 'remaining-native55': 2},
            'Fixed critical inventory changed')
    result = [dict(id='critical:'+v['id'], latent=v['latent'], group=v['group'], source=v) for v in selected]
    breadth = [v for v in saved['rows'] if v['group'] == 'breadth']
    for kind in ('native_R5', 'native_complement', 'competing', 'invalid'):
        candidates = [v for v in breadth if v['metadata']['coverage_class'] == kind]
        require(len(candidates) == 32, 'Original breadth inventory changed')
        result += [dict(id='breadth:'+str(v['id']), latent=v['latent'], group='breadth:'+kind,
                        source=v) for v in candidates[:8]]
    require(len(result) == 40 and len({v['id'] for v in result}) == 40 and
            len({tuple(v['latent']) for v in result}) == 40, 'Duplicate saved probes')
    return result


def prepare(out, repository):
    out, repository = Path(out).absolute(), Path(repository).absolute()
    require(not out.exists(), 'Preparation must be new')
    prior = Path('/vast/xvg/tetramer-mc-runs/contact-arc-score-preparation-20261001')
    score = Path('/vast/xvg/tetramer-mc-runs/hard-free-line-score-20261001/common')
    compiled_path = repository/'runs/native-entry-compiled-20260924/compiled.json'
    selected_path = repository/'results/protein-angular-probe-preparation-20261002/selected-probes.json'
    bindings = {}
    def bind(path, digest=None):
        actual = sha(path)
        require(digest is None or actual == digest, 'Input hash changed: '+str(path))
        bindings[str(path)] = actual
        return path
    for name, digest in [('original92.json', OLD92_SHA), ('region.json', REGION_SHA), ('shape.json', SHAPE_SHA)]:
        bind(prior/'common'/name, digest)
    bind(compiled_path, COMPILED_SHA); bind(selected_path, SELECTED_SHA)
    bind(prior/'saved-scores.json', SAVED_SHA); bind(score/'config.json'); bind(score/'xyz.json')
    probes = select_probes(read(selected_path), read(prior/'saved-scores.json'))
    config = read(score/'config.json'); guide = read(score/'xyz.json'); compiled = read(compiled_path)
    require(config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035, 'Physical target changed')
    require(compiled['fixed_poses'] == config['fixed_poses'], 'Compiled scaffold differs')
    require(guide['region_sha256'] == REGION_SHA and len(guide['gaussian_components']) == 92 and
            guide['defensive_uniform_shell_probability'] == .5 and guide['conditional_probability'] == 1 and
            guide['raw_translation_axes'] == [0, 1, 2] and guide['minimum_conditional_mass'] == 1e-12,
            'Original guide changed')
    original = read(prior/'common/original92.json')
    require(guide['gaussian_components'] == original['gaussian_components'], 'Gaussian mixture changed')
    (out/'common').mkdir(parents=True); (out/'guides').mkdir()
    for name in ('region.json', 'shape.json', 'original92.json'):
        shutil.copy2(prior/'common'/name, out/'common'/name)
    shutil.copy2(compiled_path, out/'common/compiled-native.json')
    config['shape'] = str(out/'common/shape.json'); write(out/'common/config.json', config)
    for arm in ('hard_free', 'class'):
        value = copy.deepcopy(guide)
        value.update(schema='defensive-native-class-line-guide-v1', class_channels=channels(arm),
                     compiled_native=dict(path=str(out/'common/compiled-native.json'), sha256=COMPILED_SHA),
                     shape_sha256=SHAPE_SHA, fixed_poses=config['fixed_poses'],
                     capture_center=config['capture_center'], capture_radius=config['capture_radius'], depletant_radius=1.5)
        write(out/'guides'/f'{arm}.json', value)
    write(out/'selected-probes.json', probes)
    with (out/'probes.jsonl').open('x') as f:
        for v in probes: f.write(json.dumps(dict(id=v['id'], latent=v['latent']), allow_nan=False)+'\n')
    write(out/'allocation.json', dict(schema='native-class-line-proposal-diagnostic-v1', jobs=jobs(),
        fresh_draws=1024, development_probes=40, development_guide='class', maximum_workers=1,
        new_Poisson_clouds=0, physical_weight_estimates=0, alpha=.5, conditional_probability=1.,
        minimum_conditional_mass=1e-12, channels=channels('class'),
        native55_scope='All complete native entry in original orthant55; old-R5 remainder is NOT a proposal restriction.',
        draw_scope='All 1024 unconditional draws retained, including invalid/exterior and all fallbacks. No retries or automatic extension.',
        saved_scope='Exactly eight previously inspected difficult-stratum poses plus first eight saved breadth rows per class. Development only, not independent validation.',
        reference='Independent Python leaf-pair intervals, conditional masses, all component/axis/channel density, inverse-CDF, Jacobian, complete native classifier and hard/contact flags for every output.',
        analysis=['Unconditional class and original-orthant frequencies, four-population descriptive uncertainty',
                  'Class/hard-free/final-Normal fallback rates and masses by selected and scored branch',
                  'Saved critical and breadth density changes, separated from fresh outcomes',
                  'Draw, density, observer and independent audit CPU reported separately'],
        limits='No equilibrium mass, ESS, free-energy, assembly or model-refutation conclusion. Full-vessel and assembly gates remain closed.',
        executable_bound=False, execution_ready=False, input_sha256=bindings))
    shutil.copy2(__file__, out/'prepare.py')
    write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(out=str(out), allocation_sha256=sha(out/'allocation.json'), freeze_sha256=sha(out/'freeze.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path(__file__).absolute().parents[1])
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.repository), indent=2))
