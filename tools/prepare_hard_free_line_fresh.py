#!/usr/bin/env python3
"""Freeze the fixed 2048-draw proposal-only pilot and native classifier."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
from prepare_hard_free_line_score import read,sha,write,require,OLD92_SHA,REGION_SHA,SHAPE_SHA
from run_contact_tail_pilot import verify_frozen,copy_frozen

ARMS=['baseline','xyz']
SEEDS=[610151001+1009*i for i in range(8)]
NATIVE_SHA='5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
DECLARATIONS=['protocol.json','plan.json','allocation.json','manifest.json','config.json',
              'declaration.json','prospective-plan.json','prospective-pilot-plan.json','campaign.json']
ROOTS=['/vast/xvg/tetramer-mc-runs','/vast/xvg/tetramer-mc/runs','/vast/xvg/tetramer-mc/results',
       '/vast/xvg/protein-nucleation/results','/vast/xvg/protein-nucleation-20260920/results']


def seeds_in(value):
    found=set()
    def integer(v):
        if type(v) is int:return v
        if isinstance(v,str) and v.isdecimal():return int(v)
        return None
    if isinstance(value,dict):
        for k,v in value.items():
            if 'seed' in k.lower():
                for x in v if isinstance(v,list) else [v]:
                    parsed=integer(x)
                    if parsed is not None:found.add(parsed)
            found.update(seeds_in(v))
    elif isinstance(value,list):
        for i,v in enumerate(value):
            if v=='--seed' and i+1<len(value):
                parsed=integer(value[i+1])
                if parsed is not None:found.add(parsed)
            found.update(seeds_in(v))
    return found


def seed_inventory(roots):
    require(all(Path(p).is_dir() for p in roots),'Missing campaign inventory root')
    command=['rg','--files','--hidden','--no-ignore',*roots]
    for name in DECLARATIONS:command+=['-g',name]
    result=subprocess.run(command,text=True,capture_output=True)
    require(result.returncode in (0,1),'Campaign inventory failed: '+result.stderr)
    files={};seeds=set()
    for path in sorted({str(Path(p).resolve()) for p in result.stdout.splitlines()}):
        values=seeds_in(read(path));seeds.update(values)
        files[path]=dict(sha256=sha(path),declared_seeds=sorted(values))
    require(files and not seeds.intersection(SEEDS),'Fresh seed collision or empty inventory')
    return dict(roots=roots,filename_patterns=DECLARATIONS,command=command,files=files,
        files_scanned=len(files),previous_unique_seeds=len(seeds),fresh_seeds=SEEDS,
        collisions=[],scope='Exact metadata declarations and seed command arguments in the listed /vast campaign roots; no trajectory/sample scan.')


def jobs():
    return [dict(arm=arm,id=f'r{i:02}',seed=SEEDS[4*j+i],samples=256)
            for j,arm in enumerate(ARMS) for i in range(4)]


def prepare(out,score_preparation,classifier_declaration):
    out,score_preparation,classifier_declaration=map(lambda p:Path(p).resolve(),
        (out,score_preparation,classifier_declaration))
    require(not out.exists(),'Fresh preparation directory required')
    verify_frozen(score_preparation);verify_frozen(classifier_declaration)
    require(sha(score_preparation/'common/original92.json')==OLD92_SHA and
            sha(score_preparation/'common/shape.json')==SHAPE_SHA and
            sha(score_preparation/'common/region.json')==REGION_SHA,'Original physical target changed')
    declaration=read(classifier_declaration/'declaration.json')
    require(declaration['native_definition_sha256']==NATIVE_SHA and
            sha(classifier_declaration/declaration['native_definition'])==NATIVE_SHA,
            'Complete native definition changed')
    guide=read(score_preparation/'guides/xyz.json')
    require(guide['schema']=='defensive-hard-free-line-guide-v1' and guide['raw_translation_axes']==[0,1,2] and
        guide['conditional_probability']==1 and guide['defensive_uniform_shell_probability']==.5 and
        guide['minimum_conditional_mass']==1e-12,'Frozen xyz guide changed')
    inventory=seed_inventory(ROOTS)
    out.mkdir(parents=True);(out/'common').mkdir();(out/'guides').mkdir()
    for name in ('original92.json','shape.json','region.json'):
        shutil.copy2(score_preparation/'common'/name,out/'common'/name)
    config=read(score_preparation/'common/config.json')
    require(config['depletant_radius']==1.5 and config['reservoir_density']==.035,'Physical conditions changed')
    config['shape']=str(out/'common/shape.json');write(out/'common/config.json',config)
    for arm,beta in zip(ARMS,[0.,1.]):
        candidate=copy.deepcopy(guide);candidate['conditional_probability']=beta
        write(out/'guides'/f'{arm}.json',candidate)
    copy_frozen(classifier_declaration,out/'classifier')
    write(out/'seed-inventory.json',inventory)
    write(out/'plan.json',dict(schema='hard-free-line-fresh-preparation-v1',jobs=jobs(),arms=ARMS,
        populations_per_arm=4,draws_per_population=256,total_attempted_draws=2048,probes_per_population=0,
        alpha=.5,beta={'baseline':0.,'xyz':1.},axes=[0,1,2],minimum_conditional_mass=1e-12,
        maximum_CPU_workers=1,new_Poisson_clouds=0,new_physical_mass_estimates=0,
        native_definition_sha256=NATIVE_SHA,classifier='classifier/'+declaration['native_definition'],
        classifier_source_sha256=declaration['source_sha256'],
        physical_contact='Complete sphere-union minimum surface gap < 2*rd=3A; both scaffold gaps retained.',
        classification='Exactly one complete native call and one physical exclusion-contact call per hard-valid in-domain NEW pose; invalid/exterior rows retained with null labels. No archived pose is reclassified.',
        width_contacts='Raw diagnostic sentinel is retained only in the unchanged Rust record; never interpreted as physical contact or reported as a contact count.',
        partition=['outside_R4_or_capture','inside_domain_hard_invalid','valid_native_entry',
                   'valid_contact_without_native_entry','valid_unbound_without_native_entry'],
        proposal_branches=['uniform','original_gaussian','conditioned_fallback','conditioned_success'],
        source_sha256={str(score_preparation/'freeze.json'):sha(score_preparation/'freeze.json'),
            str(classifier_declaration/'freeze.json'):sha(classifier_declaration/'freeze.json'),
            str(classifier_declaration/'declaration.json'):sha(classifier_declaration/'declaration.json')},
        seed_inventory_sha256=sha(out/'seed-inventory.json'),all_attempts_retained=True,
        no_retries=True,no_optional_stopping=True,no_autoextension=True,physical_gates_unchanged=True,
        physical_campaign_ready=False,execution_ready=False,
        diagnostics=['independent entire-line geometry/full-density/inverse-CDF reconstruction of every output',
            'unconditional and selected-branch domain/hard retention, fallback reasons and counts',
            'native/contact/both-scaffold/cycle-consistent proposal fractions, four-population SE',
            'proposal draw/density/full-loop/child CPU, audit and classifier CPU separately',
            'no importance weights, physical normalizers, equilibrium occupancy or mixing claims'],
        scope='Fresh proposal-only geometry/CPU pilot, using native labels only as a frozen observer. No depletant cloud, physical weight, assembly or convergence conclusion.'))
    shutil.copy2(__file__,out/'source.py')
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(out=str(out),plan_sha256=sha(out/'plan.json'),fresh_seeds=SEEDS,
                inventory_files=inventory['files_scanned'],previous_unique_seeds=inventory['previous_unique_seeds'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','score-preparation','classifier-declaration'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.out,a.score_preparation,a.classifier_declaration),indent=2))
