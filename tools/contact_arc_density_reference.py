#!/usr/bin/env python3
"""Independent complete density for hard-free-arc conditioned contact guides.

Only azimuth is conditioned, at fixed Gaussian angular coordinates, selected
label and sampled radii. If its allowed probability is <= the frozen floor,
the original azimuth law is retained. No pose generator is implemented here.
The frozen distance and circle reference modules are reused without changes.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
import contact_distance_reference as distance
import contact_circle_reference as circle


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dependency_sha256():
    return {str(Path(p).resolve()):sha(p) for p in [__file__,distance.__file__,circle.__file__]}
def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def conditional_azimuth_factor(phi,allowed,law,minimum_arc_mass=1e-12):
    """Return the complete normalization/fallback correction to the old law."""
    require(math.isfinite(minimum_arc_mass) and 0<minimum_arc_mass<=1,'Invalid arc mass floor')
    mass=circle.azimuth_mass(allowed,law);fallback=mass<=minimum_arc_mass
    inside=circle.contains(allowed,phi)
    factor=0. if fallback else (-math.log(mass) if inside else -math.inf)
    return dict(arc_mass=mass,arc_fallback=fallback,query_phi_allowed=inside,log_correction=factor)


class QueryGeometryCache:
    """Pure geometry memoization; probabilities are never cached across laws.

    The cache accepts only one exact query and fixed geometry context. Two
    azimuth arms can share it, but a changed shape, scaffold, chart or query is
    rejected rather than silently reusing an unrelated circle.
    """
    def __init__(self):
        self.context=None;self.query=None;self.entries=[];self.labels={};self.hits=0
    def bind(self,context,query):
        query=tuple(float(x) for x in query)
        if self.context is None:self.context=context;self.query=query
        require(self.context==context and self.query==query,'Geometry cache belongs to a different configuration/query')
    def get(self,key,factory):
        if key in self.labels:self.hits+=1;return self.entries[self.labels[key]],True
        item=factory();item['geometry_id']=len(self.entries);item['label_tuple']=[list(p) for p in key]
        self.labels[key]=len(self.entries);self.entries.append(item)
        return item,False


class ArcDensityReference:
    def __init__(self,region,guide,config,shape,minimum_arc_mass=1e-12):
        require(math.isfinite(minimum_arc_mass) and 0<minimum_arc_mass<=1,'Invalid fixed arc floor')
        region,guide,config,shape=copy.deepcopy((region,guide,config,shape))
        self.base=distance.Reconstructor(region,guide,config,shape);self.shape=shape;self.config=config
        self.minimum_arc_mass=minimum_arc_mass
        context=dict(shape=shape,fixed_poses=config['fixed_poses'],chart=region['gaussian_chart'],fixed_neighbor=region['fixed_neighbor'])
        self.context=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.label_keys=[tuple((p['neighbor_index'],p['moving_atom'],p['fixed_atom']) for p in pair) for pair in self.base.labels]

    def _geometry(self,u,R,branch):
        coordinates=branch['coordinates'];centers=np.asarray(branch['effective_centers'])
        D,axis,e1,e2=distance.transverse_basis(*centers)
        h,rho=distance.circle_coordinates(coordinates['r1'],coordinates['r2'],D)
        center=centers[0]+h*axis
        q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]].tolist()
        geometry=circle.hard_free_arcs(self.shape,self.config['fixed_poses'],q,center,rho,[e1,e2])
        return dict(circle_center=center.tolist(),circle_radius=rho,basis=[e1.tolist(),e2.tolist()],effective_centers=centers.tolist(),
            radii=[coordinates['r1'],coordinates['r2']],query_phi=coordinates['phi'],moving_orientation=q,**geometry)

    def score(self,u,cache=None):
        started=time.process_time();u=np.asarray(u,float);require(u.shape==(6,) and np.isfinite(u).all(),'Invalid latent query')
        cache=QueryGeometryCache() if cache is None else cache;cache.bind(self.context,u)
        previous_geometry_count=len(cache.entries);previous_hits=cache.hits
        old=self.base.density(u,details=True);raw,position,R,logJ=self.base.decode(u)
        common=dict(raw_coordinates=raw.tolist(),pose=dict(position=position.tolist(),orientation=Rotation.from_matrix(R).as_quat()[[3,0,1,2]].tolist()),
            log_physical_jacobian=logJ,baseline_log_density=old['baseline_log_density'],old_distance_log_density=old['log_density'],minimum_arc_mass=self.minimum_arc_mass)
        if old.get('conditioning_disabled'):
            return dict(**common,log_density=old['log_density'],conditioning_disabled=True,branches=[],geometry_count=0,
                geometry_builds=0,geometry_cache_hits=0,components=[],score_cpu_seconds=time.process_time()-started)
        terms=[];branches=[];used=set();component_corrections={};arc_branches=0;arc_fallback_branches=0
        for old_branch in old['branches']:
            k=old_branch['component'];wi=old_branch['width_index'];old_log=old_branch['weighted_latent_log_density']
            branch=dict(component=k,width_index=wi,width_A=self.base.widths[wi],outer_fallback=old_branch['fallback'],
                old_weighted_latent_log_density=old_log,old_translation_log_density=old_branch.get('translation_log_density'),
                distance=old_branch['D'],polygon_area=old_branch['polygon_area'])
            if old_branch['fallback']:
                branch.update(outer_fallback_reason=old_branch['fallback_reason'],geometry_id=None,arc_mass=None,arc_fallback=None,query_phi_allowed=None)
                new_log=old_log
            elif not math.isfinite(old_branch['translation_log_density']):
                branch.update(geometry_id=None,arc_mass=None,arc_fallback=None,query_phi_allowed=None,new_translation_log_density=-math.inf)
                new_log=-math.inf
            else:
                geometry,_=cache.get(self.label_keys[k],lambda:self._geometry(u,R,old_branch));used.add(geometry['geometry_id'])
                # The angular law is component-specific. Its geometry is the
                # same across widths; its probability must not be inherited
                # from another Gaussian component or another azimuth arm.
                if k not in component_corrections:
                    law=dict(mode=old_branch['phi_mean'],gamma=old_branch['gamma'],
                        localized_probability=self.base.controls['localized_probability'] if old_branch['localization_enabled'] else 0.)
                    component_corrections[k]=(law,conditional_azimuth_factor(old_branch['coordinates']['phi'],geometry['allowed'],law,self.minimum_arc_mass))
                law,correction=component_corrections[k];arc_branches+=1;arc_fallback_branches+=int(correction['arc_fallback'])
                new_log=old_log+correction['log_correction']
                branch.update(geometry_id=geometry['geometry_id'],azimuth_law=law,query_phi=old_branch['coordinates']['phi'],**correction,
                    new_translation_log_density=old_branch['translation_log_density']+correction['log_correction'])
            branch['new_weighted_latent_log_density']=new_log;branches.append(branch);terms.append(new_log)
        H=float(logsumexp(terms)-math.log(len(self.base.widths)));G=float(logsumexp(self.base.gaussian_logs(u)))
        pieces=[]
        if self.base.beta<1:pieces.append(math.log1p(-self.base.beta)+G)
        if self.base.beta>0:pieces.append(math.log(self.base.beta)+H)
        uniform=math.log(self.base.alpha)-self.base.logvolume if np.linalg.norm(u)<=self.base.radius else -math.inf
        q=float(np.logaddexp(uniform,math.log1p(-self.base.alpha)+logsumexp(pieces)))
        if self.base.beta<1:require(q+1e-12>=old['baseline_log_density']+math.log1p(-self.base.beta),'Defensive old-baseline bound failed')
        components=[];_,worldmeans,angular=self.base.conditionals(raw)
        for k in range(len(self.base.weights)):
            b=[r for r in branches if r['component']==k]
            components.append(dict(component=k,angular_log_density=float(angular[k]),conditional_world_mean=worldmeans[k].tolist(),widths=b))
        return dict(**common,log_density=q,conditioned_log_density=H,component_branches=len(branches),
            outer_fallback_branches=old['fallback_component_branches'],arc_branches=arc_branches,arc_fallback_branches=arc_fallback_branches,
            branches=branches,components=components,geometry_count=len(used),geometry_builds=len(cache.entries)-previous_geometry_count,
            geometry_cache_hits=cache.hits-previous_hits,geometry_ids=sorted(used),score_cpu_seconds=time.process_time()-started)


def compare_rows(rust_rows,python_rows,guide_count):
    """Compare saved complete scores; this performs no geometry or q rerun."""
    require(len(rust_rows)==len(python_rows),'Missing score rows');maxima=Counter();counts=Counter()
    def check(actual,expected,label,atol=2e-8,rtol=2e-10):
        require(np.isfinite(np.asarray(actual,float)).all() and np.isfinite(np.asarray(expected,float)).all(),'Nonfinite '+label)
        error=float(np.max(abs(np.asarray(actual)-expected)))
        require(np.allclose(actual,expected,atol=atol,rtol=rtol),label+' differs')
        maxima[label]=max(maxima[label],error)
    def logcheck(actual,expected,label):
        if expected is None or expected==-math.inf:require(actual is None,'Nonzero '+label+' for zero branch')
        else:check(actual,expected,label,atol=2e-7,rtol=1e-11)
    for ordinal,(r,p) in enumerate(zip(rust_rows,python_rows)):
        require(r['ordinal']==ordinal and r['id']==p['id'] and r['latent']==p['latent'],'Changed score identity/order/latent')
        require(len(r['arms'])==len(p['arms'])==guide_count,'Missing proposal arm')
        for key in ('hard_valid','shell_valid','capture_valid'):require(r[key]==p[key],'Changed endpoint '+key)
        for arm in p['arms']:
            check(r['raw_coordinates'],arm['raw_coordinates'],'raw_coordinates')
            check(r['pose']['position'],arm['pose']['position'],'pose_position')
            check(distance.rotation(r['pose']['orientation']),distance.rotation(arm['pose']['orientation']),'pose_rotation')
            check(r['log_physical_jacobian'],arm['log_physical_jacobian'],'physical_log_J')
        rust_geometry=r['geometry_cache']['circles'];py_geometry=p['geometries'];links={};references=0
        require([g['id'] for g in rust_geometry]==list(range(len(rust_geometry))),'Changed geometry IDs')
        require([g['geometry_id'] for g in py_geometry]==list(range(len(py_geometry))),'Changed Python geometry IDs')
        for ai,(ra,pa) in enumerate(zip(r['arms'],p['arms'])):
            require(ra['arm_index']==ai,'Changed arm order')
            for rk,pk in [('log_proposal_density','log_density'),('old_distance_log_density','old_distance_log_density'),('baseline_log_density','baseline_log_density')]:
                check(ra[rk],pa[pk],rk,atol=2e-7,rtol=1e-11)
            if pa.get('conditioning_disabled'):
                require(ra.get('conditioning_disabled') and not ra['components'],'Disabled conditioner unexpectedly scored');continue
            require(len(ra['components'])==len(pa['components']),'Incomplete component sum')
            for rc,pc in zip(ra['components'],pa['components']):
                require(rc['component']==pc['component'],'Changed component order')
                check(rc['angular_log_density'],pc['angular_log_density'],'angular_log_density')
                check(rc['conditional_world_mean'],pc['conditional_world_mean'],'conditional_world_mean')
                require(len(rc['widths'])==len(pc['widths']),'Incomplete width sum')
                for rw,pw in zip(rc['widths'],pc['widths']):
                    require(rw['width_index']==pw['width_index'] and rw['width_A']==pw['width_A'],'Changed width order')
                    require(rw['fallback']==pw['outer_fallback'],'Outer fallback differs')
                    if 'distance' in rw:check(rw['distance'],pw['distance'],'effective_center_distance')
                    if 'polygon_area' in rw:check(rw['polygon_area'],pw['polygon_area'],'polygon_area',atol=5e-14,rtol=2e-9)
                    counts['width_branches']+=1
                    if pw['outer_fallback']:
                        require(rw['fallback_reason']==pw['outer_fallback_reason'],'Outer fallback reason differs')
                        require('geometry_id' not in rw and 'arc_mass' not in rw,'Outer fallback incorrectly conditioned');counts['outer_fallback_branches']+=1;continue
                    logcheck(rw['old_translation_log_density'],pw['old_translation_log_density'],'old_translation_log_density')
                    logcheck(rw['translation_log_density'],pw['new_translation_log_density'],'arc_translation_log_density')
                    if pw['geometry_id'] is None:
                        require('geometry_id' not in rw and 'arc_mass' not in rw,'Unsupported radius branch requested geometry');continue
                    references+=1;counts['arc_branches']+=1;counts['arc_fallback_branches']+=int(pw['arc_fallback'])
                    rid,pid=rw['geometry_id'],pw['geometry_id'];require(0<=rid<len(rust_geometry) and 0<=pid<len(py_geometry),'Unknown circle ID')
                    links.setdefault(rid,set()).add(pid)
                    check(rw['arc_mass'],pw['arc_mass'],'arc_mass',atol=1e-10,rtol=2e-7)
                    require(rw['arc_fallback']==pw['arc_fallback'],'Arc fallback differs at the frozen floor')
                    require(rw['query_phi_allowed']==pw['query_phi_allowed'],'Query hard-free membership differs')
                    angle=abs((rw['query_phi']-pw['query_phi']+math.pi)%circle.TAU-math.pi)
                    require(angle<=1e-9,'Query azimuth differs');maxima['query_phi']=max(maxima['query_phi'],angle)
                    rlaw,plaw=rw['azimuth_law'],pw['azimuth_law'];check(rlaw['localized_probability'],plaw['localized_probability'],'azimuth_localized_probability',atol=0,rtol=0)
                    if plaw['localized_probability']>0:
                        check(rlaw['gamma'],plaw['gamma'],'azimuth_gamma')
                        error=abs((rlaw['mode']-plaw['mode']+math.pi)%circle.TAU-math.pi)
                        require(error<=1e-9,'Azimuth mode differs');maxima['azimuth_mode']=max(maxima['azimuth_mode'],error)
        require(r['geometry_cache']['requests']==references,'Geometry request count differs')
        require(r['geometry_cache']['distinct_circles']==len(rust_geometry)==len(links),'Unused or missing circle geometry')
        require(set().union(*links.values())==set(range(len(py_geometry))) if links else not py_geometry,'Missing independently reconstructed geometry')
        for rid,pids in links.items():
            rg=rust_geometry[rid];rc=rg['circle']
            for pid in pids:
                pg=py_geometry[pid]
                check(rc['center'],pg['circle_center'],'circle_center');check(rc['radius'],pg['circle_radius'],'circle_radius')
                check(rc['radii'],pg['radii'],'circle_radii');check(rc['basis'],pg['basis'],'circle_basis')
                for name in ['allowed','forbidden']:
                    ri=rg['geometry'][name]['intervals'];pi=pg[name]
                    require(ri==circle.union(ri) and pi==circle.union(pi),'Noncanonical '+name+' intervals')
                    require(len(ri)==len(pi),'Changed '+name+' topology')
                    for a,b in zip(ri,pi):
                        require(a['lower_closed']==b['lower_closed'] and a['upper_closed']==b['upper_closed'],'Changed '+name+' boundary inclusion')
                        check([a['lower'],a['upper']],[b['lower'],b['upper']],name+'_endpoints',atol=1e-10,rtol=0)
        counts['queries']+=1;counts['densities']+=guide_count;counts['rust_distinct_circles']+=len(rust_geometry);counts['python_distinct_circles']+=len(py_geometry)
    return dict(complete=True,counts=dict(counts),maximum_errors=dict(maxima),tolerances=dict(log_density_absolute=2e-7,arc_mass_absolute=1e-10,arc_mass_relative=2e-7,arc_endpoints_absolute=1e-10,
        note='All fallback decisions, endpoint predicates, zero-density branches and mixture coverage must agree exactly. Full log-density comparisons additionally constrain the effect of small arc masses.'))


def audit(rust_directory,python_json):
    """Bind and compare completed Rust/Python outputs without re-evaluation."""
    root=Path(rust_directory).resolve();py_path=Path(python_json).resolve();started=time.process_time()
    manifest=read(root/'manifest.json');summary=read(root/'summary.json');reference=read(py_path)
    require(manifest['schema']=='contact-arc-density-score-v1' and summary['complete'] and summary['manifest']==manifest,'Incomplete/wrong Rust score receipt')
    require(manifest['new_pose_draws']==manifest['new_Poisson_clouds']==0,'Wrong score-only scope')
    require(reference['schema']=='independent-contact-arc-density-v1' and reference['complete'],'Incomplete Python scores')
    require(reference['minimum_arc_mass']==manifest['minimum_arc_mass'],'Changed mass floor')
    bindings={str(p):sha(p) for p in [root/'manifest.json',root/'summary.json',py_path]}
    for path,h in reference['input_sha256'].items():require(sha(path)==h,'Changed Python scoring input');bindings[path]=h
    expected=[('config.json','config_sha256'),('region.json','region_sha256'),('shape.json','shape_sha256'),('probes.jsonl','probes_sha256'),('source-bundle.json','source_bundle_sha256')]
    for name,key in expected:
        path=root/'provenance'/name;require(sha(path)==manifest[key],'Changed Rust provenance '+name);bindings[str(path)]=sha(path)
        if name!='source-bundle.json':require(manifest[key] in reference['input_sha256'].values(),'Rust/Python provenance differs '+name)
    for i,h in enumerate(manifest['guide_sha256']):
        path=root/'provenance'/f'guide-{i}.json';require(sha(path)==h and h in reference['input_sha256'].values(),'Changed/mismatched guide');bindings[str(path)]=h
    for name in ['scores','attempts']:
        path=root/(name+'.jsonl');require(sha(path)==summary[name+'_sha256'],'Changed Rust '+name);bindings[str(path)]=sha(path)
    rows=[json.loads(s) for s in (root/'scores.jsonl').read_text().splitlines()]
    attempts=[json.loads(s) for s in (root/'attempts.jsonl').read_text().splitlines()]
    probes=[json.loads(s) for s in (root/'provenance/probes.jsonl').read_text().splitlines()]
    require(len(rows)==len(attempts)==len(probes)==manifest['queries']==summary['queries']==reference['queries'],'Missing/duplicated query')
    require(manifest['arms']==summary['arms']==len(manifest['guide_sha256']) and reference['densities']==len(rows)*manifest['arms'],'Changed arm allocation')
    for ordinal,(r,a,p) in enumerate(zip(rows,attempts,probes)):
        require(a==dict(ordinal=ordinal,id=p['id'],state='begin'),'Attempt journal identity/order changed')
        require(r['id']==p['id'] and r['latent']==p['latent'],'Score query changed')
        for i,arm in enumerate(reference['rows'][ordinal]['arms']):
            require(reference['input_sha256'][arm['guide']]==manifest['guide_sha256'][i],'Changed arm guide ordering')
    result=compare_rows(rows,reference['rows'],manifest['arms'])
    require(summary['distinct_circles']==sum(r['geometry_cache']['distinct_circles'] for r in rows),'Geometry summary count differs')
    for field,expected_value in [('geometry_cpu_seconds',math.fsum(r['geometry_cache']['geometry_cpu_seconds'] for r in rows)),
            ('score_cpu_seconds',math.fsum(a['score_cpu_seconds'] for r in rows for a in r['arms']))]:
        require(math.isclose(summary[field],expected_value,abs_tol=1e-8,rel_tol=1e-10),'CPU receipt differs '+field)
    for path,h in bindings.items():require(sha(path)==h,'Input changed during comparison')
    result.update(schema='independent-contact-arc-density-audit-v1',rust_directory=str(root),python_json=str(py_path),input_sha256=bindings,
        source_bundle_sha256=manifest['source_bundle_sha256'],executable_sha256=manifest['executable_sha256'],script_sha256=sha(__file__),
        analysis_cpu_seconds=time.process_time()-started,scope='Complete saved-score comparison; no geometry, density or physical calculation rerun.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--shape',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--region',type=Path,required=True);parser.add_argument('--guide',type=Path,action='append',required=True)
    parser.add_argument('--probes',type=Path,required=True);parser.add_argument('--minimum-arc-mass',type=float,default=1e-12);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();require(not a.out.exists(),'Output must be new')
    files=[a.shape,a.config,a.region,a.probes,*a.guide];bindings={str(p.resolve()):sha(p) for p in files};bindings.update(dependency_sha256())
    shape,config,region=map(read,[a.shape,a.config,a.region]);guides=[read(p) for p in a.guide]
    require(config['fixed_poses']==region.get('physical_fixed_neighbors',[region['fixed_neighbor']]),'Changed physical neighbors')
    for region_key,config_key in [('capture_center','capture_center'),('capture_radius','capture_radius'),('depletant_radius','depletant_radius'),('activity','reservoir_density'),('physical_metric','metadata')]:
        require(region[region_key]==config[config_key],'Changed physical field: '+region_key)
    scorers=[ArcDensityReference(region,g,config,shape,a.minimum_arc_mass) for g in guides]
    probes=[json.loads(s) for s in a.probes.read_text().splitlines()];require(len({str(p['id']) for p in probes})==len(probes),'Repeated query ID')
    for guide in guides:require(guide['region_sha256']==sha(a.region),'Guide/region binding changed')
    require(region['shape_sha256']==sha(a.shape),'Shape binding changed')
    rows=[]
    for probe in probes:
        cache=QueryGeometryCache();arms=[]
        for path,scorer in zip(a.guide,scorers):arms.append(dict(guide=str(path.resolve()),**scorer.score(probe['latent'],cache)))
        raw,position,R,_=scorers[0].base.decode(probe['latent']);hard,contacts=scorers[0].base.direct_geometry(position,R)
        rows.append(dict(id=probe['id'],latent=probe['latent'],arms=arms,geometries=cache.entries,hard_valid=hard,
            shell_valid=bool(np.linalg.norm(probe['latent'])<=scorers[0].base.radius),
            capture_valid=bool(np.linalg.norm(position-np.asarray(config['capture_center']))<=config['capture_radius']),width_contacts=contacts))
    for path,h in bindings.items():require(sha(path)==h,'Frozen scoring input changed')
    result=dict(complete=True,schema='independent-contact-arc-density-v1',scope='Complete normalized arc-guide scores for archived queries only; no pose draws, depletants or physical-weight estimates.',
        minimum_arc_mass=a.minimum_arc_mass,queries=len(rows),densities=len(rows)*len(scorers),rows=rows,input_sha256=bindings)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n');print(a.out)


if __name__=='__main__':main()
