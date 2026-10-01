#!/usr/bin/env python3
"""Arithmetic-only numerical scale check using independently audited intervals."""
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
import physical_hard_free_line_vessel as reference


def inspect(directory,audit_path):
    root=Path(directory).resolve();audit=reference.read(audit_path)
    reference.require(audit['complete'] and audit['population']==str(root),'Incomplete/mismatched full geometry audit')
    paths=[root/'samples.jsonl',root/'manifest.json',root/'config.json',root/'provenance/latent-region.json',
           root/'provenance/latent-guide.json',root/'provenance/model.json',root/'provenance/shape.json']
    bindings={str(p):reference.sha(p) for p in paths}
    for p,h in bindings.items():reference.require(audit['source_sha256'][p]==h,'Input differs from completed geometry audit')
    bindings[str(Path(audit_path).resolve())]=reference.sha(audit_path)
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    cfg=reference.read(root/'config.json');manifest=reference.read(root/'manifest.json')
    region=reference.read(root/'provenance/latent-region.json');guide=reference.read(root/'provenance/latent-guide.json')
    obj=reference.PhysicalHardFreeLineGuide(region,guide,cfg,reference.read(root/'provenance/shape.json'),
        region_sha256=manifest['latent_region_sha256'],expected_shape_sha256=manifest['shape_sha256'])
    # The baseline object supplies only independently reconstructed coordinates
    # and Jacobians. Conditional q uses the already-audited SAVED intervals.
    old=reference.original.PhysicalLatentGuide(region,dict(schema='defensive-latent-shell-guide-v1',
        region_sha256=guide['region_sha256'],defensive_uniform_shell_probability=guide['defensive_uniform_shell_probability'],
        gaussian_components=guide['gaussian_components']),region_sha256=guide['region_sha256'],expected_shape_sha256=manifest['shape_sha256'])
    poses=[r['pose']for r in rows];mapped=old.evaluate_many(poses)
    vessel=reference.vessel_reference.VesselDensity(cfg,manifest,reference.read(root/'provenance/model.json'),root/'provenance/source-bundle.json')
    vessel_logs,_=vessel.evaluate(poses)
    maxima={k:dict(error=-1.) for k in ('vessel_log_density','latent_physical_log_density','outer_mixture_log_density','valid_outer_mixture_log_density')}
    for i,(row,d) in enumerate(zip(rows,mapped)):
        if d.latent is None:latent=-math.inf
        else:
            value=reference.regional.compact_density(obj.recon,np.asarray(d.latent),row['latent_density']['hard_free_line_density'],geometry='saved-intervals')
            latent=value['log_density']-d.log_physical_jacobian
        mixed=float(reference.original.half_mixture_log_density(vessel_logs[i],latent))
        responsibility=math.log(.5)+latent-mixed
        information=dict(draw=row['draw'],outer_branch=row['outer_branch'],capture_valid=row['capture_valid'],wall_valid=row['wall_valid'],hard_valid=row['hard_valid'],
            latent_log_responsibility=responsibility if math.isfinite(responsibility) else None,
            stored_outer_log_density=row['log_proposal_density'],reference_outer_log_density=mixed,
            outer_log_density_error=abs(row['log_proposal_density']-mixed),
            physical_weight_is_zero=not row['hard_valid'],source_trace_empty_reasons=[a.get('empty_reason')for a in row['latent_density']['hard_free_line_density'].get('axes',[])])
        for field,actual,expected in [('vessel_log_density',row['log_vessel_proposal_density'],float(vessel_logs[i])),
            ('latent_physical_log_density',row['log_latent_physical_density'],latent),('outer_mixture_log_density',row['log_proposal_density'],mixed),
            ('valid_outer_mixture_log_density',row['log_proposal_density'],mixed)]:
            if field=='valid_outer_mixture_log_density' and not row['hard_valid']:continue
            actual=-math.inf if actual is None else actual
            error=0. if actual==expected else abs(actual-expected)
            reference.require(math.isfinite(error),'Density support changed')
            if error>maxima[field]['error']:
                maxima[field]=dict(error=error,recorded=actual if math.isfinite(actual) else None,
                    reference=expected if math.isfinite(expected) else None,
                    relative_log_scale_error=error/max(1.,abs(expected)),**information)
    for p,h in bindings.items():reference.require(reference.sha(p)==h,'Input changed during arithmetic check')
    return dict(population=str(root),samples=len(rows),maximum_errors=maxima,input_sha256=bindings,
        new_pose_draws=0,new_Poisson_clouds=0,new_geometry_queries=0,new_native_classifier_calls=0,
        scope='Every saved row receives independent chart, Gaussian, Normal-CDF and mixture arithmetic. Interval completeness is inherited from the bound full geometry audit; no tree or atom geometry is replayed. Component log errors are separated from complete mixture errors and valid physical weights.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    reference.require(not args.out.exists(),'Fresh diagnostic receipt required');root=args.validation
    records=[inspect(root/'toys'/path,root/f'independent-{name}.json') for path,name in
        [('analytic/activity0','activity0'),('analytic/activity0.4','activity04'),('controls/beta0/new','beta0'),('controls/alpha1/new','alpha1')]]
    value=dict(schema='hard-free-vessel-numerical-scale-v1',complete=True,records=records,source_sha256={str(Path(__file__).resolve()):reference.sha(__file__)})
    with args.out.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(args.out)
