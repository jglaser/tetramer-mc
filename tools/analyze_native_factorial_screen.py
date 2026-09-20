#!/usr/bin/env python3
"""Conditional factorial contrast, retaining paired hard/physical population covariance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sample_variance(values):
    if len(values)<2:return None
    mean=sum(values)/len(values)
    return sum((x-mean)**2 for x in values)/(len(values)-1)


def summed_standard_error(variances):
    """A single population has no independent-population variance estimate."""
    values=list(variances)
    return None if any(value is None for value in values) else math.sqrt(sum(values))


def physical_signature(path):
    a=json.loads((path/'manifest.json').read_text())
    job=Path(a['jobs'][0]['output'])
    cfg=json.loads((job/'provenance/config.json').read_text())
    manifest=json.loads((job/'manifest.json').read_text())
    return {'shape_sha256':manifest['shape_sha256'],'metric':manifest['metric'],
            'activity':manifest['activity'],'depletant_radius':manifest['depletant_radius'],
            'capture_center':cfg['capture_center'],'capture_radius':cfg['capture_radius']},cfg['fixed_poses'],json.loads((job/'cover.json').read_text())


def target(path):
    data=json.loads((path/'assessment-streaming.json').read_text())
    assert data['all_rows_and_hashes_validated']
    rows=data['populations'];assert len({r['samples'] for r in rows})==1
    physical=data['regions']['native'];logs=[r['regions']['native']['log_normalizer'] for r in rows]
    volumes=[r['cover_volume']*r['counts']['valid']/r['samples'] for r in rows]
    mean_hard=sum(volumes)/len(volumes);logz=physical['log_normalizer']
    if logz is None or mean_hard<=0:
        return {'resolved':False,'physical':physical,'hard_volume':mean_hard,'populations':rows}
    xr=[math.exp(log-logz) if log is not None else 0. for log in logs]
    yr=[v/mean_hard for v in volumes];k=len(rows)
    vz=sample_variance(xr);vh=sample_variance(yr);vd=sample_variance([x-y for x,y in zip(xr,yr)])
    return {'resolved':True,'physical':physical,'hard_volume':mean_hard,'log_hard_volume':math.log(mean_hard),
            'log_depletion_enhancement':logz-math.log(mean_hard),'population_count':k,
            'population_logQz':logs,'population_hard_volumes':volumes,
            'variance_logQz':None if vz is None else vz/k,
            'variance_logQhard':None if vh is None else vh/k,
            'variance_log_depletion_enhancement':None if vd is None else vd/k,
            'covariance_logQz_logQhard':None if k<2 else sum((x-1)*(y-1) for x,y in zip(xr,yr))/(k*(k-1)),
            'cpu_seconds':data['cpu_seconds'],'max_population_wall_seconds':data['max_population_wall_seconds'],
            'raw_cloud_points':data['raw_cloud_points'],'sample_bytes':data['sample_bytes'],
            'assessment_path':str(path/'assessment-streaming.json'),
            'assessment_sha256':hashlib.sha256((path/'assessment-streaming.json').read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);args=parser.parse_args()
    run=json.loads((args.root/'manifest.json').read_text())
    paths={label:args.root/label for label in ['empty','B','AB']};paths['A']=Path(run['reused_A_root'])
    signatures={label:physical_signature(path) for label,path in paths.items()}
    signature=signatures['A'][0];cover=signatures['A'][2]
    assert all(s[0]==signature and s[2]==cover for s in signatures.values()),'Different physical region/bath/reference measure'
    assert signatures['empty'][1]==[]
    assert signatures['AB'][1]==signatures['A'][1]+signatures['B'][1]
    results={label:target(path) for label,path in paths.items()}
    if all(r['resolved'] for r in results.values()):
        signs={'AB':1,'empty':1,'A':-1,'B':-1}
        lz=sum(signs[t]*results[t]['physical']['log_normalizer'] for t in signs)
        lh=sum(signs[t]*results[t]['log_hard_volume'] for t in signs)
        delta=results['AB']['log_depletion_enhancement']-results['A']['log_depletion_enhancement']-results['B']['log_depletion_enhancement']
        assert abs(delta-(lz-lh))<1e-10
        sez=summed_standard_error(r['variance_logQz'] for r in results.values())
        seh=summed_standard_error(r['variance_logQhard'] for r in results.values())
        sed=summed_standard_error(results[t]['variance_log_depletion_enhancement'] for t in ['AB','A','B'])
        contrast={'resolved_as_point_estimate':True,'log_conditional_cooperativity_z':lz,
                  'conditional_cooperativity_z':math.exp(lz),'free_energy_cooperativity_kBT':-lz,
                  'observed_population_delta_SE_log_cooperativity_z':sez,
                  'log_hard_cooperativity':lh,'hard_cooperativity':math.exp(lh),
                  'observed_population_delta_SE_log_hard_cooperativity':seh,
                  'log_activity_induced_change_in_cooperativity':delta,
                  'observed_population_delta_SE_log_activity_change':sed,
                  'empty_cancels_from_activity_change':True,
                  'within_target_physical_hard_covariance_retained':True}
    else:
        contrast={'resolved_as_point_estimate':False,'reason':'At least one required target has zero observed mass; no pseudocount or log uncertainty'}
    result={'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),
            'same_physical_region_bath_shape_cover_verified':True,'targets':results,'contrast':contrast,
            'uncertainty':'Observed independent-population delta-method SEs; targetgroups independent seeds. Four populations cannot certify rare-tail convergence. Mass estimates are unbiased; their ratios/logs are not.',
            'interpretation':'Conditional native-region cooperativity includes hard support and shared-pose correlations. This is not an isolated triple-overlap depletion penalty or a global native probability.',
            'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.root/'factorial-assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
