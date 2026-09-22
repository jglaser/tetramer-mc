#!/usr/bin/env python3
"""Audit NEW immutable full-R4 SMC outputs; never launch sampling or old audits.

Every initialization attempt is retained. Terminal region masses use the joint
normalizer times terminal indicator estimator; uncertainty uses whole populations.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from scipy.stats import t as student_t
from analyze_mobile_native_pocket import load_classifier, local_sources
from analyze_mobile_threshold_reference import ExclusionContact
from analyze_native_region_reference import native_q

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('total','registered_native_entry','old_R5_intersection_native','remaining_R4_native',
         'contact_no_native_entry','unbound_no_native_entry')
PRIMARY=CLASSES[:4]
STRATA={'radial':3,'angular':3,'orthant':64}
PROPOSAL_FIELDS=('translation_steps','rotation_steps_deg')
SCOPE=('Independent fixed-R4 SMC audit and whole-population estimates. Every initial attempt, '
       'zero-hit population and terminal descendant remains in the declared estimator. '
       'Terminal mass is Zhat times its indicator fraction; no descendant IID ESS or row-level '
       'standard error is reported. Intermediate beta stages are bridge measures, not final '
       'physical probabilities. Observed agreement does not bound unseen modes or authorize '
       'full-vessel or finite-assembly production.')


def require(ok,message):
    if not ok: raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def close(a,b,message='Numerical reconstruction differs',tol=2e-8):
    require(isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b)
            and abs(a-b)<=tol+2e-11*max(abs(a),abs(b)),message)
def same_pose(a,b):return a==b
def pose_key(pose):return tuple(pose['position'])+tuple(pose['orientation'])
def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:yield json.loads(line)
def line(stream,value):stream.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
def nullable_log(value):
    require(value is None or isinstance(value,(int,float)) and math.isfinite(value),'Invalid optional log weight')
    return -math.inf if value is None else float(value)
def logvalue(value):return None if value == -math.inf else float(value)


class Ledger:
    def __init__(self):self.files={}
    def bind(self,path,expected=None):
        p=Path(path).resolve();actual=sha(p)
        require(expected is None or actual==expected,'Hash mismatch: '+str(p))
        require(str(p) not in self.files or self.files[str(p)]==actual,'Source changed: '+str(p))
        self.files[str(p)]=actual;return p
    def frozen(self,root):
        root=Path(root).resolve();manifest=read(self.bind(root/'freeze.json'));entries=manifest.get('files',manifest)
        require(isinstance(entries,dict) and entries,'Missing freeze manifest')
        for name,digest in entries.items():
            p=(root/name).resolve();require(p.is_relative_to(root) and name!='freeze.json','Freeze path escapes root')
            self.bind(p,digest)
        return entries
    def recheck(self):
        for p,digest in self.files.items():require(sha(p)==digest,'Source changed during analysis: '+p)


class Chart:
    """Independent SciPy matrix/quaternion map and normalized physical Jacobian."""
    def __init__(self,region):
        self.region=region;c=region['gaussian_chart'];self.radius=float(region['mahalanobis_radius'])
        self.ell=float(c['angular_length']);self.mean=np.asarray(c['means'][0],float)
        covariance=np.asarray(c['covariances'][0],float)
        require(covariance.shape==(6,6) and np.isfinite(covariance).all() and np.max(abs(covariance-covariance.T))<1e-11,'Invalid chart covariance')
        self.lower=np.linalg.cholesky(covariance);self.logdet=float(np.log(np.diag(self.lower)).sum())
        self.fixed=np.asarray(region['fixed_neighbor']['position'],float)
        self.fixed_rotation=self.rotation(region['fixed_neighbor']['orientation'])
        self.anchor=np.asarray(c['anchors'][0]['position'],float);self.anchor_rotation=np.asarray(c['anchors'][0]['rotation'],float)
        self.log_volume=3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        self.angular_map=solve_triangular(np.linalg.cholesky(covariance[3:,3:]),self.lower[3:,:],lower=True)
    @staticmethod
    def rotation(q):
        q=np.asarray(q,float);require(q.shape==(4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1)<2e-10,'Invalid pose quaternion')
        return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    def evaluate(self,pose):
        t=np.asarray(pose['position'],float);require(t.shape==(3,) and np.isfinite(t).all(),'Invalid pose translation')
        r=self.rotation(pose['orientation']);q=Rotation.from_matrix(self.fixed_rotation.T@r@self.anchor_rotation.T).as_quat()
        if q[3]==0.:return dict(latent=None,log_jacobian=None,radius=None,inside=False)
        cayley=q[:3]/q[3]
        x=np.r_[self.fixed_rotation.T@(t-self.fixed)-self.anchor,self.ell*cayley]
        u=solve_triangular(self.lower,x-self.mean,lower=True);require(np.isfinite(u).all(),'Unrepresentable inverse chart')
        radius=float(np.linalg.norm(u));logj=self.logdet-3*math.log(self.ell)-2*math.log(math.pi)-2*math.log1p(float(cayley@cayley))
        return dict(latent=u,log_jacobian=logj,radius=radius,inside=radius<=self.radius)
    def bins(self,u):
        u=np.asarray(u,float);a=self.angular_map@u
        return dict(radial=int(np.searchsorted([2.,3.],np.linalg.norm(u),side='right')),
                    angular=int(np.searchsorted([4.,9.],float(a@a),side='right')),
                    orthant=int(sum((1<<i) for i in range(6) if u[i]>=0)))


class Proposal:
    def __init__(self,current,reference,alpha):
        require(0<alpha<=1 and (reference is not None or alpha==1),'Incomplete proposal support')
        self.current=Chart(current);self.reference=None if reference is None else Chart(reference);self.alpha=alpha
    def evaluate(self,pose):
        current=self.current.evaluate(pose);ref=None if self.reference is None else self.reference.evaluate(pose)
        terms=[]
        if current['inside']:terms.append(math.log(self.alpha)-self.current.log_volume-current['log_jacobian'])
        if self.alpha<1 and ref['inside']:terms.append(math.log1p(-self.alpha)-self.reference.log_volume-ref['log_jacobian'])
        return current,ref,float(logsumexp(terms)) if terms else -math.inf


def validate_config_identity(left,right,allowed=()):
    require(set(allowed)<=set(PROPOSAL_FIELDS),'Unapproved physical-identity exclusions')
    a=copy.deepcopy(left);b=copy.deepcopy(right);a.pop('shape',None);b.pop('shape',None)
    for name in allowed:a.pop(name,None);b.pop(name,None)
    require(a==b,'Physical configuration differs beyond documented proposal scales')


def parse_job_options(job):
    argv=job['argv'];require(len(argv)%2==1 and len(set(argv[1::2]))==len(argv[1::2]),'Malformed duplicate CLI flags')
    args=dict(zip(argv[1::2],argv[2::2]));stages=int(args['--stages'])
    require(stages>0,'Invalid fixed schedule')
    return dict(config=args['--config'],region=args['--region'],out=args['--out'],
        initial_reference_region=args.get('--initial-reference-region'),
        initial_current_probability=float(args.get('--initial-current-probability','1')),
        initial_draws=int(args['--initial-draws']),population=int(args['--population']),seed=int(args['--seed']),
        bridge=args['--bridge'].replace('-','_'),schedule=[i/stages for i in range(stages+1)],
        sweeps_per_stage=int(args['--sweeps-per-stage']),cloud_replicates=int(args['--cloud-replicates']),lambda_ratio=float(args['--lambda-ratio']))


def validate_control(protocol_path,expected_sha=None):
    protocol_path=Path(protocol_path).resolve();base=protocol_path.parent;ledger=Ledger()
    ledger.bind(protocol_path,expected_sha);frozen=ledger.frozen(base);require('protocol.json' in frozen,'Unfrozen protocol')
    p=read(protocol_path);require(p['schema']=='smc-r4-density-bridge-control-v1','Unknown SMC control schema')
    for path,digest in p['source_and_input_sha256'].items():ledger.bind(path,digest)
    binary=ledger.bind(p['binary'],p['binary_sha256']);bundle=ledger.bind(base/'common/source-bundle.json',p['source_bundle_sha256'])
    require(bundle.read_bytes() in binary.read_bytes(),'Source bundle is not embedded in the frozen executable')
    validation=ledger.bind(p['implementation_validation'],p['implementation_validation_sha256']);ledger.frozen(validation.parent)
    evidence=read(validation);require(evidence['complete'] and evidence['binary_sha256']==p['binary_sha256']
        and evidence['source_bundle_sha256']==p['source_bundle_sha256'],'Implementation validation does not bind this binary')
    config=read(p['physical_target']['config']);current=read(p['physical_target']['region']);reference=read(p['proposal']['reference_region'])
    require(current['mahalanobis_radius']==4 and current.get('minimum_mahalanobis_radius',0)==0 and current['minimum_original_q']==0
        and 'maximum_original_q' not in current,'Current target is not full R4')
    require(reference['mahalanobis_radius']==5 and reference['minimum_original_q']==1
        and reference.get('minimum_original_q_inclusive') is False and reference['capture_radius']==170,'Old R5 reporting definition changed')
    for key in ('physical_fixed_neighbors','capture_center','capture_radius','activity','depletant_radius','physical_metric','shape_sha256'):
        require(current[key]==reference[key],'R4/R5 physical target mismatch: '+key)
    require(sha(p['physical_target']['region'])==p['physical_target']['region_sha256']
        and sha(config['shape'])==p['physical_target']['shape_sha256']==current['shape_sha256'],'Physical input identity differs')
    require(current['physical_fixed_neighbors']==config['fixed_poses'] and current['physical_metric']==config['metadata']
        and current['capture_center']==config['capture_center'] and current['capture_radius']==config['capture_radius']
        and current['activity']==config['reservoir_density'] and current['depletant_radius']==config['depletant_radius'],'Config does not target current region')
    definition=base/'common/native-definition.json';d=read(definition)
    require(d['shape_sha256']==current['shape_sha256'] and d['fixed_poses']==config['fixed_poses'],'Native observer target changed')
    # The executable definition is in its complete frozen source/input directory,
    # not the convenience standalone copy under common/.
    definitions=[Path(path) for path,digest in p['source_and_input_sha256'].items()
                 if digest==sha(definition) and Path(path).name=='definition.json']
    require(len(definitions)==1,'Missing/ambiguous complete native observer path')
    definition=definitions[0];observer_config=read(definition.parent/'inputs/physical-config.json')
    allowed=()
    if 'proposal_control_difference' in p:
        change=p['proposal_control_difference'];allowed=tuple(change['changed_config_fields'])
        source=read(ledger.bind(change['source_protocol'],change['source_protocol_sha256']))
        original=read(source['physical_target']['config']);validate_config_identity(config,original,allowed)
        for name,values in change['changed_config_fields'].items():require(config[name]==values['new'] and original[name]==values['old'],'Undeclared mutation-scale value')
    validate_config_identity(config,observer_config,allowed)
    require(sha(definition.parent/'inputs/physical-config.json')==d['physical_config_sha256'],'Native physical source hash differs')
    jobs=p['jobs'];a=p['allocation']
    require(len(jobs)==a['independent_populations']==4 and len({j['seed'] for j in jobs})==4 and len({j['id'] for j in jobs})==4,'Four distinct independent populations required')
    for j in jobs:
        options=parse_job_options(j)
        require(j['argv'][0]==p['binary'] and options['out']==j['output'] and options['seed']==j['seed'],'Job path/seed differs')
        expected={'population':a['particles_each'],'initial_draws':a['unconditional_initialization_draws_each'],
            'sweeps_per_stage':a['local_moves_per_particle_per_stage'],'cloud_replicates':a['incremental_clouds_per_particle_per_stage'],
            'lambda_ratio':p['bridge']['incremental_lambda_ratio'],'schedule':p['bridge']['beta_schedule'],
            'bridge':'proposal_density','config':p['physical_target']['config'],'region':p['physical_target']['region'],
            'initial_reference_region':p['proposal']['reference_region'],'initial_current_probability':p['proposal']['current_ball_probability']}
        require(all(options[k]==v for k,v in expected.items()),'CLI differs from fixed allocation/target')
    require({k:config[k] for k in p['mutation_config']}==p['mutation_config'],'Mutation settings changed')
    ledger.recheck()
    return dict(protocol=p,protocol_path=str(protocol_path),config=config,current=current,reference=reference,
                definition=str(definition),definition_sha256=sha(definition),bindings=ledger.files)


class Observer:
    def __init__(self,context):
        self.current=Chart(context['current']);self.reference=Chart(context['reference']);self.config=context['config']
        self.classifier,self.binding=load_classifier(context['definition'])
        self.contact=ExclusionContact(read(self.config['shape']),self.config['fixed_poses'],self.config['depletant_radius'])
        self.cache=OrderedDict();self.classifier_calls=0;self.contact_calls=0
    def classify(self,pose,valid):
        if not valid:return dict(classes={name:False for name in CLASSES},classification=None,contact=None,strata=None)
        key=pose_key(pose)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        label=self.classifier.classify(pose);self.classifier_calls+=1
        contact=self.contact.classify(pose);self.contact_calls+=1
        current=self.current.evaluate(pose);reference=self.reference.evaluate(pose)
        q=float(native_q(self.config['metadata'],pose))
        old_capture=np.linalg.norm(np.asarray(pose['position'])-self.reference.region['capture_center'])<=self.reference.region['capture_radius']
        old_r5=reference['inside'] and q>1. and bool(old_capture)
        native=bool(label['native_any']);bound=bool(contact['exclusion_contact']);require(not native or bound,'Complete native entry has no exclusion contact')
        classes=dict(total=True,registered_native_entry=native,old_R5_intersection_native=native and old_r5,
            remaining_R4_native=native and not old_r5,contact_no_native_entry=bound and not native,
            unbound_no_native_entry=not bound and not native)
        result=dict(classes=classes,classification=label,contact=contact,original_q=q,old_R5_radius=reference['radius'],
            old_R5_reporting_support=bool(old_r5),strata=self.current.bins(current['latent']))
        self.cache[key]=result
        if len(self.cache)>32768:self.cache.popitem(last=False)
        return result


def systematic_parents(logs,n,offset):
    require(n>0 and 0<=offset<1/n and len(logs)>0 and np.isfinite(logs).all(),'Invalid resampling input')
    w=np.exp(np.asarray(logs)-max(logs));require((w>0).all(),'Positive resampling weight underflowed')
    cumulative=np.cumsum(w);targets=(offset+np.arange(n)/n)*float(w.sum())
    parents=np.searchsorted(cumulative,targets,side='right');require((parents<len(w)).all(),'Resampling CDF lost support')
    return parents.tolist()


def verify_ancestry(particles,saved):
    counts=Counter(int(p['initial_ancestor']) for p in particles);n=len(particles)
    require({str(k):v for k,v in counts.items()}==saved['initial_ancestor_counts'],'Initial-family counts differ')
    require(len(counts)==saved['distinct_initial_ancestors'],'Distinct family count differs')
    ess=n*n/sum(c*c for c in counts.values()) if n else 0.
    close(saved['initial_family_ESS'],ess,'Family concentration differs')
    return dict(distinct_initial_ancestors=len(counts),initial_family_ESS=ess,
        largest_initial_family_fraction=max(counts.values())/n if n else None,
        scope='Initial-family concentration only; descendants are not IID physical samples.')


def audit_pose(particle,proposal,config):
    c,r,g=proposal.evaluate(particle['pose']);require(c['inside'] and math.isfinite(g),'SMC particle left current R4/proposal support')
    require(np.linalg.norm(np.asarray(particle['pose']['position'])-config['capture_center'])<=config['capture_radius']+1e-9,'SMC particle left capture')
    require(np.max(abs(np.asarray(particle['latent'])-c['latent']))<2e-8,'Saved particle coordinates differ')
    return g


def audit_clouds(clouds,delta,lam,replicates):
    require(len(clouds)==replicates and delta>=0 and lam>0,'Incorrect cloud allocation/intensity')
    logs=[]
    for cloud in clouds:
        for field in ('raw_points','overlap_points','retained_cells','created_cells','certified_cells'):
            require(type(cloud[field]) is int and cloud[field]>=0,'Invalid Poisson count')
        require(cloud['overlap_points']<=cloud['raw_points'],'Retained count exceeds raw points')
        for field in ('lower_volume','upper_volume','uncertain_volume'):
            require(math.isfinite(cloud[field]) and cloud[field]>=0,'Invalid envelope volume')
        close(cloud['upper_volume'],cloud['lower_volume']+cloud['uncertain_volume'],'Overlap volume accounting differs')
        logw=delta*cloud['lower_volume']+cloud['overlap_points']*math.log1p(delta/lam)
        close(cloud['log_weight'],logw,'Poisson factor differs');logs.append(logw)
    for c in clouds[1:]:
        require(all(c[k]==clouds[0][k] for k in ('lower_volume','upper_volume','uncertain_volume')),'Paired cloud envelopes differ')
    return float(logsumexp(logs)-math.log(replicates))


def audit_stage(stage,previous,proposal,config,options,expected_index,log_z):
    n=options['population'];beta=options['schedule'][expected_index];db=beta-options['schedule'][expected_index-1]
    activity=beta*config['reservoir_density'];delta=db*config['reservoir_density'];lam=options['lambda_ratio']*delta if delta>0 else 1.
    require(stage['stage']==expected_index and len(stage['potentials'])==len(stage['particles'])==len(stage['parents'])==n,'Stage allocation/index differs')
    for field,value in [('beta',beta),('delta_beta',db),('activity',activity),('previous_activity',activity-delta),('delta_activity',delta),('incremental_lambda',lam)]:close(stage[field],value,'Stage path differs')
    logs=[]
    for i,row in enumerate(stage['potentials']):
        require(row['particle']==i and same_pose(row['pose'],previous[i]['pose'])
            and row['initial_ancestor']==previous[i]['initial_ancestor'],'Potential pose/lineage does not match prior endpoint')
        g=audit_pose(row,proposal,config);close(row['log_g'],g,'Full physical proposal density differs')
        correction=-db*g if options['bridge']=='proposal_density' else 0.
        close(row['deterministic_log_correction'],correction,'Density-bridge increment differs')
        weight=audit_clouds(row['clouds'],delta,lam,options['cloud_replicates'])+correction
        close(row['log_incremental_weight'],weight,'Incremental factor differs');logs.append(weight)
    inc=float(logsumexp(logs)-math.log(n));close(stage['log_Z_increment'],inc,'Normalizer increment differs');log_z+=inc
    close(stage['log_Z'],log_z,'Normalizer product differs')
    weights=np.exp(np.asarray(logs)-max(logs));close(stage['pre_resampling_weight_ESS'],float(weights.sum()**2/(weights@weights)),'Resampling ESS differs')
    parents=systematic_parents(logs,n,stage['resampling_offset']);require(parents==stage['parents'],'Systematic resampling parents differ')
    current=[copy.deepcopy(previous[i]) for i in parents]
    valid_records=stage.get('mutation_density_records',[]);seen=set();accepted=0;translations=0;rotations=0;last=(-1,-1)
    for event in valid_records:
        i=event['particle'];sweep=event['sweep'];key=(i,sweep);close(event['beta'],beta,'Mutation beta differs')
        require(type(i) is int and type(sweep) is int and 0<=i<n and 0<=sweep<options['sweeps_per_stage'] and key not in seen and key>last,'Invalid/duplicate/out-of-order mutation identity')
        seen.add(key);last=key;require(same_pose(event['old_pose'],current[i]['pose']),'Mutation history lost retained pose')
        old_g=proposal.evaluate(event['old_pose'])[2];new_c,_,new_g=proposal.evaluate(event['proposed_pose'])
        require(new_c['inside'] and math.isfinite(new_g),'Mutation candidate outside proposal/current domain')
        require(np.linalg.norm(np.asarray(event['proposed_pose']['position'])-config['capture_center'])<=config['capture_radius']+1e-9,'Mutation candidate outside capture')
        close(event['log_g_old'],old_g,'Old mutation density differs');close(event['log_g_new'],new_g,'New mutation density differs')
        correction=(1-beta)*(new_g-old_g);close(event['deterministic_log_correction'],correction,'Mutation density correction differs')
        gate=event['gate']
        for name in ('gained','lost','raw_points','retained_points'):
            require(type(gate[name]) is int and gate[name]>=0,'Invalid mutation count')
        require(gate['gained']+gate['lost']==gate['retained_points']<=gate['raw_points'],'Mutation thinning accounting differs')
        gate_log=(gate['gained']-gate['lost'])*math.log1p(1/config['poisson_lambda_ratio']) if activity>0 else 0.
        if activity==0:require(gate['gained']==gate['lost']==0,'Nonzero gate counts at zero activity')
        close(gate['log_weight'],gate_log,'Gained/lost gate differs')
        loga=min(0.,gate_log+correction);close(event['log_acceptance'],loga,'Combined bridge acceptance differs')
        logu=nullable_log(event['log_uniform']);require(logu<=0,'Invalid acceptance uniform')
        require(type(event['accepted']) is bool and event['accepted']==(logu<loga),'Acceptance decision differs')
        if event['accepted']:
            translations+=int(event['old_pose']['position']!=event['proposed_pose']['position'])
            rotations+=int(event['old_pose']['orientation']!=event['proposed_pose']['orientation'])
            current[i]['pose']=event['proposed_pose'];accepted+=1
    counts=stage['mutation_counts'];require(all(type(v) is int and v>=0 for v in counts.values()),'Invalid mutation counters')
    require(counts['attempted']==n*options['sweeps_per_stage'] and counts['attempted']==sum(counts[k] for k in ('accepted','capture_rejected','region_rejected','hard_rejected','gate_rejected')),'Mutation attempts lost')
    if options['bridge']=='proposal_density':
        require(len(valid_records)==counts['accepted']+counts['gate_rejected'] and accepted==counts['accepted'],'Mutation records/counters differ')
        require(sum(e['gate']['raw_points'] for e in valid_records)==counts['raw_points'],'Mutation raw-point count differs')
        require(translations==counts['accepted_translation_changes'] and rotations==counts['accepted_rotation_changes'],'Accepted move-type counters differ')
    for i,p in enumerate(stage['particles']):
        require(p['initial_ancestor']==previous[parents[i]]['initial_ancestor'],'Offspring initial ancestor changed')
        if options['bridge']=='proposal_density':require(same_pose(p['pose'],current[i]['pose']),'Mutation endpoint differs from accepted history')
        audit_pose(p,proposal,config)
    genealogy=verify_ancestry(stage['particles'],stage['ancestry'])
    return log_z,dict(stage=expected_index,beta=beta,log_Z=log_z,ancestry=genealogy,
        pre_resampling_weight_ESS=stage['pre_resampling_weight_ESS'],full_native_profile_available=False)


def profile(particles,log_z,observer,stage,stream,initial_classes):
    counts={name:0 for name in CLASSES};strata={family:{name:[0]*size for name in CLASSES} for family,size in STRATA.items()}
    family_classes={name:Counter() for name in CLASSES};replenished={name:0 for name in CLASSES[1:4]}
    for i,p in enumerate(particles):
        label=observer.classify(p['pose'],True);line(stream,dict(stage=stage,particle=i,initial_ancestor=p['initial_ancestor'],**label))
        for name,yes in label['classes'].items():
            if yes:
                counts[name]+=1;family_classes[name][p['initial_ancestor']]+=1
                for family,index in label['strata'].items():strata[family][name][index]+=1
                if name in replenished and not initial_classes[p['initial_ancestor']][name]:replenished[name]+=1
    n=len(particles)
    require(counts['registered_native_entry']==counts['old_R5_intersection_native']+counts['remaining_R4_native']
        and counts['total']==sum(counts[k] for k in ('registered_native_entry','contact_no_native_entry','unbound_no_native_entry')),'Terminal class partition differs')
    masses={name:None if count==0 or log_z is None else log_z+math.log(count/n) for name,count in counts.items()}
    return dict(stage=stage,particles=n,counts=counts,log_masses=masses,
        log_strata={f:{name:[None if c==0 or log_z is None else log_z+math.log(c/n) for c in a] for name,a in names.items()} for f,names in strata.items()},
        class_distinct_initial_families={name:len(c) for name,c in family_classes.items()},
        descendants_from_initially_other_class=replenished,
        measure='final physical target' if stage=='terminal' else 'fixed beta bridge target; not final physical mass')


def audit_population(context,job,out,stages_to_classify,observer=None):
    p=context['protocol'];base=Path(job['output']);out=Path(out);ledger=Ledger();options=parse_job_options(job)
    state=read(ledger.bind(base/'status.json'));require(state['complete'] is True and state['phase']=='complete','SMC population is not complete')
    summary=read(ledger.bind(base/'summary.json',state['summary_sha256']))
    manifest=read(ledger.bind(base/'manifest.json',summary['manifest_sha256']))
    require(summary['complete'] is True and summary['schema']=='latent-region-smc-summary-v1' and manifest['schema']=='latent-region-smc-v1','Unknown/incomplete SMC output')
    require(manifest['options']==options,'Executed options differ from exact frozen command')
    require(manifest['executable_sha256']==p['binary_sha256'] and manifest['source_bundle_sha256']==p['source_bundle_sha256'],'Executed binary/source closure differs')
    for name,digest in [('config.json',manifest['config_sha256']),('region.json',manifest['region_sha256']),('shape.json',manifest['shape_sha256']),('source-bundle.json',manifest['source_bundle_sha256'])]:ledger.bind(base/'provenance'/name,digest)
    require(manifest['config_sha256']==sha(options['config']) and manifest['region_sha256']==p['physical_target']['region_sha256']
        and manifest['shape_sha256']==p['physical_target']['shape_sha256'],'Executed physical inputs differ')
    if options['initial_reference_region']:
        ledger.bind(base/'provenance/initial-reference-region.json',manifest['initial_reference_sha256'])
        require(manifest['initial_reference_sha256']==sha(options['initial_reference_region']),'Executed reference chart differs')
    raw_initial=ledger.bind(base/'initialization.jsonl',summary['initialization_sha256']);raw_stages=ledger.bind(base/'stages.jsonl',summary['stages_sha256'])
    proposal=Proposal(context['current'],context['reference'],options['initial_current_probability']);config=context['config']
    observer=observer or Observer(context);out.mkdir(parents=True,exist_ok=False)
    initial=[];initial_classes=[];logs=[];valid_indices=[];branch=Counter();exterior=0
    initial_class_logs={name:[] for name in CLASSES};initial_hard_class_logs={name:[] for name in CLASSES}
    with gzip.open(out/'initial-labels.jsonl.gz','wt') as labels:
        for i,row in enumerate(jsonlines(raw_initial)):
            require(row['draw']==i and i<options['initial_draws'],'Initial draw identity/count changed')
            c,r,g=proposal.evaluate(row['pose']);require(math.isfinite(g),'Initialization lost proposal support')
            close(row['log_physical_proposal_density'],g,'Initial full mixture density differs')
            for prefix,data in [('',c),('reference_',r)]:
                if data is None:continue
                radiuskey='latent_radius' if not prefix else None
                latentkey='latent' if not prefix else 'reference_latent'
                if data['latent'] is not None:
                    require(np.max(abs(np.asarray(row[latentkey])-data['latent']))<2e-8,'Initial inverse chart differs')
                    close(row['log_physical_jacobian' if not prefix else 'reference_log_physical_jacobian'],data['log_jacobian'],'Initial Jacobian differs')
                    if radiuskey:close(row[radiuskey],data['radius'],'Initial radius differs')
            require(row['current_ball_valid']==c['inside'] and row['reference_ball_valid']==r['inside'],'Initial ball support differs')
            selected=row['selected_initial_chart'];require(selected in ('current','reference'),'Unknown initial branch');branch[selected]+=1
            source=c if selected=='current' else r;require(source['inside'],'Selected initial branch has no support')
            close(row['selected_log_physical_jacobian'],source['log_jacobian'],'Selected chart Jacobian differs')
            require(np.max(abs(np.asarray(row['selected_latent'])-source['latent']))<2e-8,'Selected chart draw differs')
            capture=np.linalg.norm(np.asarray(row['pose']['position'])-config['capture_center'])<=config['capture_radius']
            require(type(row['hard_valid']) is bool and row['capture_valid']==bool(capture),'Initial capture flag differs')
            valid=row['hard_valid'];require(not valid or capture and c['inside'],'Exterior/invalid initial row gained positive mass')
            expected=0. if valid and options['bridge']=='proposal_density' else -g if valid else -math.inf
            require(nullable_log(row['log_initial_weight'])==expected if expected in (0.,-math.inf) else abs(nullable_log(row['log_initial_weight'])-expected)<2e-8,'Operative all-attempt initialization weight differs')
            if valid:close(row['log_hard_weight'],-g,'Physical hard diagnostic differs')
            else:require(row['log_hard_weight'] is None,'Invalid initial draw lost its zero')
            label=observer.classify(row['pose'],valid);line(labels,dict(draw=i,valid=valid,selected_initial_chart=selected,**label))
            initial.append(dict(pose=row['pose'],latent=row['latent'],initial_ancestor=i));initial_classes.append(label['classes'])
            if valid:
                valid_indices.append(i);logs.append(expected)
                for name,yes in label['classes'].items():
                    if yes:initial_class_logs[name].append(expected);initial_hard_class_logs[name].append(-g)
            exterior+=int(not c['inside'])
    require(len(initial)==options['initial_draws']==summary['initial_draws'] and len(logs)==summary['initial_hits'],'Original initialization denominator/hits differ')
    stages=iter(jsonlines(raw_stages));first=next(stages,None);require(first is not None and first['stage']==0,'Missing initialization stage')
    require(first['initial_draws']==options['initial_draws'] and first['initial_hits']==len(logs),'Initialization stage denominator/hits differ')
    close(first['activity'],0.,'Initialization activity differs')
    diagnostics=[];profiles={}
    with gzip.open(out/'profile-labels.jsonl.gz','wt') as labels:
        if not logs:
            require(summary['zero_estimate'] is True and summary['log_Z'] is None and summary['Z']==0 and not summary['terminal_particles']
                and not first['particles'] and first['log_Z'] is None and first['zero_estimate'] is True
                and not first['parents'] and first['Z']==0 and summary['completed_stage']==state['completed_stage']==0
                and next(stages,None) is None,'Zero population was replaced or extended')
            verify_ancestry([],summary['ancestry'])
            diagnostics.append(dict(stage=0,beta=0.,log_Z=None,zero_estimate=True,full_native_profile_available=True))
            for stage in stages_to_classify:profiles[str(stage)]=profile([],None,observer,stage,labels,initial_classes)
            terminal=profile([],None,observer,'terminal',labels,initial_classes)
        else:
            require(summary['zero_estimate'] is False,'Positive initial population mislabeled zero')
            log_z=float(logsumexp(logs)-math.log(options['initial_draws']));close(first['log_Z'],log_z,'Initial Z lost all-M denominator')
            close(first['log_Z_increment'],log_z,'Initial normalizer increment differs')
            iw=np.exp(np.asarray(logs)-max(logs));close(first['initial_weight_ESS'],float(iw.sum()**2/(iw@iw)),'Initial weight concentration differs')
            parents=systematic_parents(logs,options['population'],first['resampling_offset'])
            expected_draws=[valid_indices[k] for k in parents];require(first['parent_initial_draws']==expected_draws,'Initial resampling changed ancestors')
            previous=first['particles'];require(len(previous)==options['population'],'Initial population size differs')
            for i,particle in enumerate(previous):require(particle==initial[expected_draws[i]],'Initial resampled pose/ancestor differs')
            genealogy=verify_ancestry(previous,first['ancestry']);diagnostics.append(dict(stage=0,beta=0.,log_Z=log_z,ancestry=genealogy,full_native_profile_available=True))
            profiles['0']=profile(previous,log_z,observer,0,labels,initial_classes)
            seen=0
            for index,stage in enumerate(stages,1):
                require(index<len(options['schedule']),'Unexpected additional annealing stage')
                log_z,diagnostic=audit_stage(stage,previous,proposal,config,options,index,log_z);previous=stage['particles'];seen=index
                if index in stages_to_classify:
                    profiles[str(index)]=profile(previous,log_z,observer,index,labels,initial_classes);diagnostic['full_native_profile_available']=True
                diagnostics.append(diagnostic)
            require(seen==len(options['schedule'])-1==summary['completed_stage']==state['completed_stage'],'Missing fixed annealing stages')
            close(summary['log_Z'],log_z,'Terminal normalizer differs');require(summary['terminal_particles']==previous,'Terminal particles differ from final stage')
            terminal=profile(previous,log_z,observer,'terminal',labels,initial_classes)
            verify_ancestry(previous,summary['ancestry'])
            if summary['Z'] is not None:
                require(summary['Z']>0 and math.isfinite(summary['Z']),'Invalid positive linear normalizer')
                close(math.log(summary['Z']),log_z,'Linear normalizer disagrees with retained log normalizer')
    ledger.recheck()
    result=dict(id=job['id'],seed=job['seed'],zero_estimate=summary['zero_estimate'],initial_draws=options['initial_draws'],
        initial_hits=len(logs),initial_branch_counts=dict(branch),exterior_initial_attempts=exterior,
        initial_class_counts={name:sum(c[name] for c in initial_classes) for name in CLASSES},
        initial_bridge_log_masses={name:float(logsumexp(a)-math.log(options['initial_draws'])) if a else None for name,a in initial_class_logs.items()},
        initial_physical_hard_log_masses={name:float(logsumexp(a)-math.log(options['initial_draws'])) if a else None for name,a in initial_hard_class_logs.items()},
        stages=diagnostics,profiles=profiles,terminal=terminal,classifier_calls=observer.classifier_calls,
        contact_calls=observer.contact_calls,source_sha256=ledger.files,sampler_cpu_seconds=summary.get('sampler_cpu_seconds'),
        scope=SCOPE,limitations='All density/count/normalizer/resampling/mutation-record stages audited. Independent atom-union hard/contact checks and complete native labels apply to valid initial poses and the declared profile, not every intermediate pose or geometric Poisson thinning event.')
    result['label_sha256']={name:sha(out/name) for name in ('initial-labels.jsonl.gz','profile-labels.jsonl.gz')}
    write(out/'population.json',result);return result


def population_statistics(log_mass_rows):
    require(len(log_mass_rows)>=2,'Whole-population uncertainty needs independent populations')
    a=np.asarray([[-math.inf if row[name] is None else row[name] for name in CLASSES] for row in log_mass_rows],float)
    require(not np.isnan(a).any() and not np.isposinf(a).any(),'Invalid population mass')
    finite=a[np.isfinite(a)];offset=float(finite.max()) if len(finite) else 0.
    x=np.exp(a-offset);require((x[np.isfinite(a)]>0).all(),'Positive population mass underflowed in covariance scaling')
    mean=x.mean(axis=0);cov=np.cov(x,rowvar=False,ddof=1)/len(x)
    estimates={};relative=[]
    for i,name in enumerate(CLASSES):
        m=float(mean[i]);se=math.sqrt(max(0.,float(cov[i,i])))
        estimates[name]=dict(log_Q=offset+math.log(m) if m>0 else None,population_relative_SE=se/m if m>0 else None,
            nonzero_populations=int((x[:,i]>0).sum()),populations=len(x),
            largest_population_fraction=float(x[:,i].max()/x[:,i].sum()) if m>0 else None,
            unresolved=None if m>0 else 'Unobserved contribution; neither zero physical mass nor an upper bound is established.')
        relative.append([float(cov[i,j]/(mean[i]*mean[j])) if mean[i]>0 and mean[j]>0 else None for j in range(len(CLASSES))])
    result=dict(estimates=estimates,covariance_of_mean=dict(class_order=list(CLASSES),log_scale=2*offset,scaled_matrix=cov.tolist(),relative_matrix=relative),
        population_count=len(x),uncertainty_scope='Arithmetic mean of independent whole-population unnormalized masses including zeros; covariance retained. No descendant IID error or ESS.')
    ni=CLASSES.index('registered_native_entry');oi=CLASSES.index('contact_no_native_entry')
    if mean[ni]>0 and mean[oi]>0:
        se=math.sqrt(max(0.,relative[ni][ni]+relative[oi][oi]-2*relative[ni][oi]));df=math.log(mean[oi]/mean[ni])
        result['free_energy_contrast']=dict(beta_F_native_minus_noentry=df,SE=se,halfwidth_95=float(student_t.ppf(.975,len(x)-1))*se)
    else:result['free_energy_contrast']=dict(unresolved='Native or no-entry terminal mass unobserved; no finite contrast or epsilon substitution.')
    return result


def aggregate(populations):
    result=population_statistics([p['terminal']['log_masses'] for p in populations]);result['strata']={}
    result['unresampled_initial_bridge']=population_statistics([p['initial_bridge_log_masses'] for p in populations])
    result['unresampled_initial_physical_hard']=population_statistics([p['initial_physical_hard_log_masses'] for p in populations])
    for family,size in STRATA.items():
        result['strata'][family]=[]
        for i in range(size):result['strata'][family].append(population_statistics([{name:p['terminal']['log_strata'][family][name][i] for name in CLASSES} for p in populations])['estimates'])
    stages=sorted(set().union(*(set(p['profiles']) for p in populations)),key=int)
    result['bridge_profiles']={stage:population_statistics([p['profiles'][stage]['log_masses'] for p in populations]) for stage in stages}
    return result


def analyze(protocol_path,out,workers=4,stage_stride=16,expected_sha=None):
    out=Path(out).resolve();require(not out.exists(),'Fresh analysis output required')
    require(type(workers) is int and 1<=workers<=4 and type(stage_stride) is int and stage_stride>0,'At most four analysis workers; positive fixed stage stride required')
    context=validate_control(protocol_path,expected_sha);p=context['protocol'];last=p['allocation']['stages_each']
    selected=sorted(set([0,last,*range(stage_stride,last+1,stage_stride)]))
    # Preflight every terminal state before creating output or constructing the classifier.
    for job in p['jobs']:
        state=read(Path(job['output'])/'status.json');require(state.get('complete') is True and state.get('phase')=='complete','All fixed SMC populations must complete before analysis')
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    write(out/'observer-plan.json',dict(schema='smc-r4-observer-plan-v1',protocol_sha256=sha(protocol_path),
        complete_native_profile_stages=selected,initial='all unconditional attempts; invalid zeros retain explicit labels',terminal='every terminal particle',
        numerical_audit='every saved stage and valid mutation record',
        scope='Current requested fixed classification profile supersedes the original outline every-stage native-label workload only; frozen physics protocol remains untouched.'))
    provenance=out/'provenance';provenance.mkdir();sources=local_sources(__file__)
    source_hashes={name:sha(path) for name,path in sources.items()}
    for name,path in sources.items():shutil.copy2(path,provenance/name)
    status=dict(complete=False,phase='auditing_and_classifying',completed_populations=[]);write(out/'status.json',status)
    try:
        records=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(audit_population,context,job,out/job['id'],selected):job for job in p['jobs']}
            for future in as_completed(futures):
                record=future.result();records.append(record);status['completed_populations'].append(record['id']);write(out/'status.json',status)
                print('Audited SMC population '+record['id'],flush=True)
        records.sort(key=lambda v:v['id']);summary=aggregate(records)
        require(all(sha(path)==digest for path,digest in context['bindings'].items()),'Frozen input changed during analysis')
        require(all(sha(sources[name])==digest for name,digest in source_hashes.items()),'Analyzer source changed during analysis')
        result=dict(schema='smc-r4-control-analysis-v1',complete=True,protocol=str(Path(protocol_path).resolve()),protocol_sha256=sha(protocol_path),
            protocol_snapshot=p,physical_config=context['config'],native_definition_sha256=context['definition_sha256'],
            populations=records,summary=summary,observer_plan=read(out/'observer-plan.json'),source_sha256=context['bindings'],
            analyzer_source_sha256=source_hashes,workers=workers,wall_seconds=time.time()-start,scope=SCOPE)
        write(out/'analysis.json',result)
        report=['# Independent R4 SMC control audit','','All declared populations were audited; no physics was launched by this analyzer.','',SCOPE,'',
                '| Region | log mean Q | Whole-population RSE | Nonzero populations |','|---|---:|---:|---:|']
        for name,estimate in summary['estimates'].items():
            q='unobserved' if estimate['log_Q'] is None else f"{estimate['log_Q']:.6f}"
            se='unresolved' if estimate['population_relative_SE'] is None else f"{estimate['population_relative_SE']:.2%}"
            report.append(f"| {name} | {q} | {se} | {estimate['nonzero_populations']}/4 |")
        report+=['','Complete initial and terminal labels are retained. Intermediate labels use the fixed profile '+str(selected)+'. All stages retain numerical and genealogy audits.','',
                 'No-entry terminal zeros cannot bound its mass. Use the separate class-balanced importance estimate; no finite-system assembly conclusion follows.']
        (out/'report.md').write_text('\n'.join(report)+'\n')
        status.update(complete=True,phase='complete',analysis_sha256=sha(out/'analysis.json'));write(out/'status.json',status)
        write(out/'freeze.json',{'files':{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}})
        return result
    except BaseException as error:
        status.update(complete=False,phase='failed',error=repr(error));write(out/'status.json',status);raise


def compare_analyses(left,right):
    require(left['complete'] and right['complete'],'Incomplete comparison input')
    a,b=left['protocol_snapshot'],right['protocol_snapshot']
    require(left['native_definition_sha256']==right['native_definition_sha256'],'Native classifier differs between controls')
    for key in ('shape_sha256','region_sha256','fixed_poses','capture_center','capture_radius','activity','depletant_radius','measure'):
        require(a['physical_target'][key]==b['physical_target'][key],'SMC physical identity differs: '+key)
    allowed=set()
    for p in (a,b):allowed.update(p.get('proposal_control_difference',{}).get('changed_config_fields',{}))
    validate_config_identity(left['physical_config'],right['physical_config'],allowed)
    require(a['proposal']['reference_region_sha256']==b['proposal']['reference_region_sha256'],'Old R5 reporting chart differs')
    require(set(j['seed'] for j in a['jobs']).isdisjoint(j['seed'] for j in b['jobs']),'SMC controls reuse population seeds')
    controls=[]
    for v in (left,right):
        allocation=v['protocol_snapshot']['allocation'];rows=v['populations']
        require(len(rows)==v['summary']['population_count']==allocation['independent_populations'],'Comparison dropped a declared population')
        require(all(r['initial_draws']==allocation['unconditional_initialization_draws_each'] for r in rows),'Comparison changed initial denominators')
        controls.append(dict(protocol=v.get('protocol'),protocol_sha256=v.get('protocol_sha256'),allocation=allocation,
            populations=[{k:r[k] for k in ('id','seed','zero_estimate','initial_draws','initial_hits')} for r in rows]))
    def mass_comparison(x,y):
        if x['log_Q'] is None or y['log_Q'] is None:
            return dict(passed=False,unresolved='One or both contributions unobserved; no zero-mass substitution.')
        difference=x['log_Q']-y['log_Q'];se=math.hypot(x['population_relative_SE'],y['population_relative_SE'])
        return dict(log_mass_difference=difference,combined_population_SE=se,
            absolute_passed=abs(difference)<=.2,three_SE_passed=abs(difference)<=3*se+1e-12,
            passed=abs(difference)<=.2 and abs(difference)<=3*se+1e-12)
    comparisons={name:mass_comparison(left['summary']['estimates'][name],right['summary']['estimates'][name]) for name in CLASSES}
    strata={family:[{name:mass_comparison(left['summary']['strata'][family][i][name],right['summary']['strata'][family][i][name])
                    for name in CLASSES} for i in range(size)] for family,size in STRATA.items()}
    x,y=left['summary']['free_energy_contrast'],right['summary']['free_energy_contrast']
    contrast=dict(passed=False,unresolved='One or both contrasts unobserved.')
    if 'SE' in x and 'SE' in y:
        difference=x['beta_F_native_minus_noentry']-y['beta_F_native_minus_noentry'];se=math.hypot(x['SE'],y['SE'])
        contrast=dict(difference=difference,combined_population_SE=se,passed=abs(difference)<=.2 and abs(difference)<=3*se+1e-12)
    return dict(schema='smc-r4-control-comparison-v1',classes=comparisons,strata=strata,controls=controls,
        documented_proposal_field_exclusions=sorted(allowed),direct_free_energy_contrast=contrast,
        primary_classes=list(PRIMARY),primary_mass_agreement=all(comparisons[n]['passed'] for n in PRIMARY),
        source_populations_kept_separate=True,scope=SCOPE+' This comparison is not a complete convergence gate.')


def compare(left_path,right_path,out):
    out=Path(out).resolve();require(not out.exists(),'Fresh comparison output required');ledger=Ledger()
    paths=[Path(left_path).resolve(),Path(right_path).resolve()]
    values=[]
    for path in paths:
        ledger.frozen(path.parent);data=read(ledger.bind(path));state=read(path.parent/'status.json')
        require(state['complete'] and state['analysis_sha256']==sha(path),'Completed analysis binding differs');values.append(data)
    result=compare_analyses(*values);ledger.recheck();out.mkdir(parents=True,exist_ok=False)
    result['source_sha256']=ledger.files;write(out/'comparison.json',result)
    write(out/'freeze.json',{'files':{'comparison.json':sha(out/'comparison.json')}});return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    a=commands.add_parser('analyze');a.add_argument('--protocol',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--workers',type=int,default=4);a.add_argument('--stage-stride',type=int,default=16);a.add_argument('--expected-protocol-sha256')
    c=commands.add_parser('compare');c.add_argument('--left',type=Path,required=True);c.add_argument('--right',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.protocol,args.out,args.workers,args.stage_stride,args.expected_protocol_sha256) if args.command=='analyze' else compare(args.left,args.right,args.out)
    print(json.dumps({'schema':result['schema'],'complete':result.get('complete',True),'physics_launched':False}))
if __name__=='__main__':main()
