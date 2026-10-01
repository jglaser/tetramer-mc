#!/usr/bin/env python3
"""Freeze a no-sampling comparison of Gaussian feasibility-only line guides."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

OLD92_SHA='a00a4470d898e06f66b78dcb7f1c1b42d7c4dc76f9a111dff9ce6268e09a96c0'
REGION_SHA='924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
SHAPE_SHA='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
PROBES_SHA='55310d72d9a97a241e3e41c23cea007dd556d4db5ef6a793d78f326600a97967'
ARMS={'x':[0],'y':[1],'z':[2],'xyz':[0,1,2]}


def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def candidate(original,axes):
    require(len(original['gaussian_components'])==92 and original['defensive_uniform_shell_probability']==.5,
            'Original Gaussian guide changed')
    result=copy.deepcopy(original)
    result.update(schema='defensive-hard-free-line-guide-v1',raw_translation_axes=list(axes),
                  conditional_probability=1.,minimum_conditional_mass=1e-12)
    require('contact_widths_A' not in result and 'contact_neighbor_indices' not in result and
            'component_contact_pairs' not in result,'Feasibility guide cannot contain contact labels or widths')
    return result


def validate_inputs(probes,saved):
    require(len(probes)==len(saved)==206,'Saved allocation changed')
    require(len({p['id'] for p in probes})==206 and len({tuple(p['latent']) for p in probes})==206,
            'Repeated pose identity or geometry')
    require([p['id'] for p in probes]==[p['id'] for p in saved] and
            [p['latent'] for p in probes]==[p['latent'] for p in saved],'Saved identity/order/pose changed')
    require([p['group'] for p in saved]==['critical']*78+['breadth']*128,'Critical/breadth allocation changed')
    require(Counter(p['metadata']['source']['arm'] for p in saved[:78])=={'baseline':4,'expanded':74},
            'Original critical sources changed')
    require(Counter(p['metadata']['coverage_class'] for p in saved[78:])==
            {'native_R5':32,'native_complement':32,'competing':32,'invalid':32},'Breadth allocation changed')
    flags=Counter((p['saved_geometry']['hard_valid'],p['saved_geometry']['shell_valid'],
                   p['saved_geometry']['capture_valid']) for p in saved)
    require(flags=={(True,True,True):174,(False,True,True):32},'Original saved flags changed')


def prepare(out,previous):
    out,previous=Path(out).resolve(),Path(previous).resolve()
    require(not out.exists(),'Fresh immutable preparation required')
    bindings={}
    def bind(path,expected=None):
        digest=sha(path);require(expected is None or digest==expected,'Frozen source changed: '+str(path))
        bindings[str(path)]=digest;return path
    freeze=read(bind(previous/'freeze.json'))
    for name,digest in freeze['files'].items():bind(previous/name,digest)
    for name,digest in [('common/original92.json',OLD92_SHA),('common/region.json',REGION_SHA),
                        ('common/shape.json',SHAPE_SHA),('score-probes.jsonl',PROBES_SHA)]:bind(previous/name,digest)
    probes=rows(previous/'score-probes.jsonl');saved=read(previous/'saved-scores.json')['rows']
    validate_inputs(probes,saved)
    original=read(previous/'common/original92.json')
    out.mkdir(parents=True);(out/'common').mkdir();(out/'guides').mkdir()
    for name in ('original92.json','region.json','shape.json'):
        shutil.copy2(previous/'common'/name,out/'common'/name)
    config=read(previous/'common/config.json')
    require(config['depletant_radius']==1.5 and config['reservoir_density']==.035,'Original target changed')
    config['shape']=str(out/'common/shape.json');write(out/'common/config.json',config)
    for name in ('score-probes.jsonl','saved-scores.json','coverage-selection.json','axis-diagnostics.json'):
        shutil.copy2(previous/name,out/name)
    shutil.copy2(previous/'plan.json',out/'prior-preparation-plan.json')
    shutil.copy2(previous/'freeze.json',out/'prior-preparation-freeze.json')
    for name,axes in ARMS.items():write(out/'guides'/f'{name}.json',candidate(original,axes))
    write(out/'plan.json',dict(schema='hard-free-line-score-preparation-v1',source_sha256=bindings,
        previous_preparation=str(previous),arms=ARMS,primary_arm='xyz',secondary_arms=['x','y','z'],
        scored_guide='guides/xyz.json',secondary_densities='Derived algebraically from the same saved per-axis component logs/masses; no additional geometry.',
        queries=206,full_mixture_density_evaluations=206,derived_axis_densities=618,total_reported_candidate_densities=824,
        declared_axes_per_query=[0,1,2],axis_geometry_requests=618,Gaussian_component_axis_branches=56856,
        critical_queries=78,critical_original_sources={'baseline':4,'expanded':74},breadth_queries=128,
        breadth_classes={'native_R5':32,'native_complement':32,'competing':32,'invalid':32},
        saved_flag_counts={'hard_valid_inside_R4_and_capture':174,'hard_invalid_inside_R4_and_capture':32},
        alpha=.5,beta=1.,minimum_conditional_mass=1e-12,contact_widths=None,contact_labels=None,
        new_pose_draws=0,new_Poisson_clouds=0,new_physical_mass_estimates=0,maximum_CPU_workers=1,
        proposal='For each fixed raw translation axis, retain every component\'s original five-dimensional Gaussian marginal. Condition the remaining scalar Normal on hard-free intersection R4 intersection capture. If its mass is at most1e-12, retain the original conditional Normal without any outer-coordinate redraw.',
        density='q_axis=alpha*U_R4+(1-alpha)*sum_k weight_k*g_k*f_k_axis, where f=1 for fallback, 1/Z for feasible query when Z>floor, and0 otherwise. q_xyz=(q_x+q_y+q_z)/3; complete component sum on every axis.',
        support='Uniform R4 keeps every target-region pose supported. At beta=1 there is no global lower bound0.5old92; valid hard-free R4/capture poses instead obey q_axis>=old92 and q_xyz>=old92. Legitimate zero density outside R4 must be represented as null logq and retained.',
        required_trace=['exact id/latent, saved pose/J/flags, all three axis interval sets',
            'per-axis92 components: unweighted latent Gaussian log density, conditional mean/sigma/mass, fallback, query membership',
            'complete per-axis log q and xyz log q, original old92 log q, geometry/conditional-density CPU'],
        retrospective_moments='Separate4-row84component source and74-row92component source; sum exp(sum(paired_log_weights)+source_log_q-candidate_log_q-log(source_attempted_draws)); report old92 ratios, contribution ESS and maximum fraction without pooling sources.',
        diagnostics=['q_xyz arithmetic average of complete per-axis q; independent Schur/CDF/geometry/density reconstruction',
            'valid-pose q>=old92 and invalid-pose density changes separately; no row discarded',
            'source-separated paired moment diagnostics and breadth logq ratios in all four fixed32-row classes',
            'full score CPU and per-axis geometry cost; no sampling throughput claim'],
        selection='Exactly all206 previously frozen/inspected poses; no fitting, native filtering, resampling or adaptive choice of winning axis. xyz primary was declared before scoring.',
        no_optional_stopping=True,no_retries=True,no_autoextension=True,physical_gates_unchanged=True,
        physical_campaign_ready=False,execution_ready=False,
        execution_prerequisites=['reference geometry/normalization/fallback and pure-uniform tests',
            'bind exact reviewed executable/source bundle and independent reference',
            'freeze one-worker controller preserving every attempted query'],
        scope='Retrospective score-only proposal comparison. It cannot establish equilibrium weights, contact mixing efficiency or finite-system assembly stability.'))
    shutil.copy2(__file__,out/'source.py')
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(prepared=True,out=str(out),plan_sha256=sha(out/'plan.json'),probes_sha256=sha(out/'score-probes.jsonl'),
                source_files_bound=len(bindings),queries=206,derived_densities=824,new_pose_draws=0,new_Poisson_clouds=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.out,a.previous),indent=2))
