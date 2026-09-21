#!/usr/bin/env python3
"""Convert frozen guide designs to strict input schema, without fitting or sampling."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/'runs/mobile-competing-outer-diagnostic-20260921'
CAMPAIGN=ROOT/'runs/mobile-competing-outer-weights-20260921'
VALIDATION=ROOT/'runs/mobile-outer-importance-validation-20260921/validation.json'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def strict_guide(design):
    if design['schema']!='defensive-latent-shell-guide-design-v1' or design['implemented'] is not False:
        raise ValueError('Require the original unimplemented frozen guide design')
    result=dict(schema='defensive-latent-shell-guide-v1',region_sha256=design['region_sha256'],
                defensive_uniform_shell_probability=design['defensive_uniform_shell_probability'],
                gaussian_components=[{key:copy.deepcopy(c[key]) for key in ('weight','mean','covariance')}
                                     for c in design['gaussian_components']])
    for name in ('region_sha256','defensive_uniform_shell_probability'):
        assert result[name]==design[name]
    assert len(result['gaussian_components'])==len(design['gaussian_components'])
    for new,old in zip(result['gaussian_components'],design['gaussian_components']):
        assert all(new[key]==old[key] for key in ('weight','mean','covariance'))
    return result


def prepare(out):
    out=Path(out).resolve()
    if out.exists():raise ValueError('Use a fresh immutable preparation')
    for name,digest in read(DESIGN/'freeze.json').items():
        assert sha(DESIGN/name)==digest, 'Original frozen design changed'
    validation=read(VALIDATION)
    assert validation['complete']
    for name,digest in validation['source_sha256'].items():
        assert sha(ROOT/name)==digest,'Reviewed implementation changed: '+name
    assert sha(ROOT/validation['source_bundle'])==validation['source_bundle_sha256']
    archive=out/'provenance';archive.mkdir(parents=True)
    sources={Path(__file__).name:Path(__file__),'design-freeze.json':DESIGN/'freeze.json',
             'design-analysis.json':DESIGN/'analysis.json','implementation-validation.json':VALIDATION,
             'latent-region-normalizer':ROOT/'target/outer-importance-review/release/latent-region-normalizer',
             'source-bundle.json':ROOT/validation['source_bundle'],
             'physical-config.json':CAMPAIGN/'config.json','physical-preflight.json':CAMPAIGN/'preflight.json',
             'physical-protocol.json':CAMPAIGN/'protocol.json'}
    guides=[]
    for name in ('competitor-shell-5-8','competitor-shell-8-12'):
        design_path=DESIGN/f'guide-{name}.json'
        sources[f'design-{name}.json']=design_path
        region_path=CAMPAIGN/f'region-{name}.json'
        sources[f'region-{name}.json']=region_path
        design=read(design_path);guide=strict_guide(design)
        assert guide['region_sha256']==sha(region_path)
        assert len(guide['gaussian_components'])==32
        assert guide['defensive_uniform_shell_probability']==.5
        assert sum(c['weight']for c in guide['gaussian_components'])==1.
        path=out/f'guide-{name}.json';write(path,guide)
        guides.append(dict(region=name,path=str(path),guide_sha256=sha(path),
                           design_sha256=sha(design_path),region_sha256=sha(region_path),
                           components=32,scientific_parameters_unchanged=True,
                           conversion='Schema label changed; component source metadata and design-only notes removed. '
                                      'Weights, means, covariances, alpha and exact physical region hash preserved.'))
    for name,path in sources.items():shutil.copy2(path,archive/name)
    plan=dict(schema='mobile-defensive-importance-guide-preparation-v1',production_launched=False,
              model_fitted=False,new_physical_draws=0,guides=guides,
              validated_binary_sha256=sha(archive/'latent-region-normalizer'),
              validated_source_bundle_sha256=sha(archive/'source-bundle.json'),
              source_sha256={name:sha(archive/name)for name in sources},
              next_control=dict(samples_per_population=8192,independent_populations_per_shell=4,
                                cloud_replicates=2,lambda_ratio=64,defensive_uniform_probability=.5,
                                fresh_independent_seeds_required=True),
              scope='Same fixed finite 5<rho<=8 and 8<rho<=12 shells, q>1, original chart/scaffold/hard shape/bath/wall. '
                    'These guides do not represent complete competing basins and cannot bound outside mass. '
                    'The separately observed mobile return at rho18.84 is outside both targets; no shell boundary is expanded by this preparation. '
                    'The original rows are training data only and must not be reused in a guided physical estimate.')
    write(out/'plan.json',plan)
    write(out/'freeze.json',{p.name:sha(p)for p in out.iterdir()if p.is_file()})
    print(json.dumps(dict(output=str(out),plan_sha256=sha(out/'plan.json'),guides=guides,
                         production_launched=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    prepare(parser.parse_args().out)
