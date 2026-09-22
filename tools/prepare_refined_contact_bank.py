#!/usr/bin/env python3
"""Freeze the declared held-out winner without generating physical samples."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from score_contact_refinements import (CANDIDATES, STRATA, OLD_BANK_SHA256, ROOT,
    fit_candidates, mixture_guide, require, read, sha, write, local_source_closure)
from run_contact_confirmation import PINS, validate_package
from run_entry_shell_reference_campaign import inside, file_hashes


def verify_freeze(directory):
    directory = Path(directory).resolve()
    frozen = read(directory/'freeze.json')['files']
    require('freeze.json' not in frozen, 'Self-referential freeze')
    for name, digest in frozen.items():
        require(sha(inside(directory, name)) == digest, 'Frozen source changed: '+name)
    return frozen


def choose_specification(scoring):
    """Use only the training grid/rule declared before any fresh draws."""
    declared = read(scoring/'declared-candidates.json')
    scores = read(scoring/'cross-validation.json')
    require(declared['candidates'] == list(CANDIDATES) and declared['strata'] == list(STRATA),
            'Declared refinement grid changed')
    name = scores['training_ranking'][0]
    matches = [spec for spec in CANDIDATES if spec['name'] == name]
    require(len(matches) == 1, 'Unknown selected candidate')
    return copy.deepcopy(matches[0]), scores


def guide_variants(old, candidates):
    bank = mixture_guide(old, candidates)
    wide = copy.deepcopy(bank)
    for component in wide['gaussian_components']:
        component['covariance'] = (4*np.asarray(component['covariance'])).tolist()
    defensive = copy.deepcopy(bank)
    defensive['defensive_uniform_shell_probability'] = .2
    return dict(bank=bank, wide=wide, alpha02=defensive)


def prepare(scoring, base, partition_plan, out):
    scoring, base, partition_plan, out = map(lambda p: Path(p).resolve(),
                                           (scoring, base, partition_plan, out))
    require(not out.exists(), 'Fresh preparation directory required')
    verify_freeze(scoring); verify_freeze(base)
    for name, digest in PINS.items():
        require(sha(base/name) == digest, 'Physical source changed: '+name)
    status = read(scoring/'status.json')
    require(status['schema'] == 'contact-refinement-score-v1' and status['complete'] is True
            and status['new_physical_samples'] == status['classifiers_rerun'] == status['audits_replayed'] == 0,
            'Require completed saved-data-only scoring')
    require(sha(base/'guide-bank.json') == OLD_BANK_SHA256, 'Retained guide changed')
    # The exact helpers used to fit held-out models also fit the full model.
    for filename in ('score_contact_refinements.py', 'prepare_contact_bank_guides.py'):
        require(sha(ROOT/'tools'/filename) == sha(scoring/'source'/filename), 'Fitting helper changed')
    specification, scores = choose_specification(scoring)
    metadata = read(scoring/'training-metadata.json')
    require(metadata['all_original_attempts_retained'] is True
            and metadata['total_unconditional_draws'] == 131072
            and metadata['region_sha256'] == PINS['region.json'], 'Training target/allocation differs')
    with np.load(scoring/'training-poses.npz', allow_pickle=False) as datafile:
        data = {name: datafile[name] for name in datafile.files}
    require(len(data['u']) == 131072 and len(metadata['populations']) == 8, 'Training rows lost')
    components, new_metadata = fit_candidates(data, metadata['populations'],
        metadata['latent_geometric_floor'], specification)
    require(len(components) == 24, 'Every population needs all three strata')
    guides = guide_variants(read(base/'guide-bank.json'), components)
    anchors = copy.deepcopy(read(base/'plan.json')['anchor_metadata'])
    for item in new_metadata:
        item['component_index'] += 56
        item['source_group'] = 'fresh-pilot'
    anchors.extend(new_metadata)
    old_plan = read(partition_plan)
    require(sha(partition_plan) == '497fa30e7389c8baf3c458060397f13a31f0227e1241f7dd38a9fff761010e50',
            'Supplementary native partition changed')
    require(sha(Path(old_plan['reference_region'])) == old_plan['reference_region_sha256'], 'Old R5 chart changed')
    out.mkdir(parents=True)
    for name in PINS:
        shutil.copy2(base/name, out/name)
    shutil.copytree(base/'native-region', out/'native-region')
    shutil.copy2(partition_plan, out/'original-native-partition-plan.json')
    write(out/'native-partition-definition.json', dict(schema='contact-confirmation-native-partition-v1',
        reference_region_sha256=old_plan['reference_region_sha256'], region_sha256=PINS['region.json'],
        native_definition_sha256=PINS['native-definition.json'], predicate=old_plan['predicate'],
        original_partition_plan_sha256=sha(partition_plan), denominators=old_plan['denominators']))
    shutil.copy2(old_plan['reference_region'], out/'old-r5-region.json')
    for name, guide in guides.items():
        write(out/f'guide-{name}.json', guide)
    # These are training artifacts, never additional independent validation data.
    shutil.copytree(scoring, out/'training')
    shutil.copy2(base/'guide-bank.json', out/'retained-guide.json')
    write(out/'selection.json', dict(specification=specification,
        scoring_freeze_sha256=sha(scoring/'freeze.json'),
        rule=scores['ranking_rule'], training_ranking=scores['training_ranking'],
        uncertainty='Observed moment tails are sparse; selection is training only, not convergence evidence.',
        next_allocation='Four fresh populations per arm; 131072 draws main, 32768 sample-size control; frozen separately before any draws.'))
    plan = dict(schema='refined-contact-bank-preparation-v1', complete=True, production_launched=False,
        **{name+'_sha256': sha(out/(name+'.json')) for name in ('config','shape','region')},
        native_definition='native-region/definition.json', native_definition_sha256=sha(out/'native-definition.json'),
        reference_region_sha256=sha(out/'old-r5-region.json'),
        supplemental_definition_sha256=sha(out/'native-partition-definition.json'),
        selection_sha256=sha(out/'selection.json'),
        guide_sha256={name: sha(out/f'guide-{name}.json') for name in guides},
        covariance_multipliers=dict(bank=1, wide=4), anchor_metadata=anchors,
        source_freeze_sha256=dict(scoring=sha(scoring/'freeze.json'), retained=sha(base/'freeze.json')),
        construction='q_new = 0.5 uniform_R4 + 0.25 G_old56 + 0.25 G_new24; new Gaussian mass equally divided over three strata and eight training populations.',
        exact_lower_bound='q_new >= 0.5 q_old for both corresponding covariance widths. No unbiased weight clipping; no Gaussian truncation or invalid-pose retries.',
        physical_samples_generated=0)
    write(out/'plan.json', plan)
    for filename, path in local_source_closure([Path(__file__)]).items():
        destination = out/'source'/filename; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write(out/'freeze.json', dict(files=file_hashes(out)))
    validate_package(out)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoring', type=Path, default=ROOT/'runs/contact-refinement-score-20260922')
    parser.add_argument('--base', type=Path, default=ROOT/'runs/reference-contact-bank-preparation-20260922')
    parser.add_argument('--partition-plan', type=Path, default=ROOT/'runs/contact-bank-reference-partition-plan-20260922/protocol.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.scoring, args.base, args.partition_plan, args.out)
    print(dict(out=str(args.out), complete=result['complete'], components=len(result['anchor_metadata'])))
