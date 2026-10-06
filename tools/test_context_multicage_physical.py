"""Synthetic physical controls only; no protein poses, geometry or clouds queried."""
import copy
import hashlib
import json
import math
import struct
import unittest

import analyze_context_multicage_physical as m
from test_context_broadened_physical import score_fixture


def jobs():
    return [dict(id=f'{a}-{p}-{c}',stratum_id=f'{a}-{p}-{c}',comparison_arm=a,population_index=p,
        component=c,attempts=n) for a,counts in m.ALLOCATIONS.items() for p in range(4) for c,n in counts.items() if n]


def record(ordinal, valid=True, region='A_patch_complete', weight=1.):
    logs=[0.,-1.,2.,1.,0.,-2.,3.]
    q=float(m.log_mixture('multicage',logs))
    pose=dict(position=[float(ordinal),0.,0.],orientation=[1.,0.,0.,0.])
    original=dict(input=dict(ordinal=ordinal,proposed_pose=pose),complete=True,
        actual=dict(physical_valid=valid),region=region if valid else None,clouds=[],physical_weight_status='not_estimated')
    digest=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(ordinal=ordinal,comparison_arm='multicage',physical_valid=valid,region=region if valid else 'hard_invalid',
        source_T_complete_all_regions=valid and region=='A_patch_complete',proposed_pose=pose,
        log_q_arm=q,log_physical_weight=math.log(weight) if valid else None,
        log_u=logs[0],log_g=logs[1],log_source=dict(zip(m.CHARTS,logs[2:])),
        source_rows=dict(path='/synthetic/rows.jsonl',sha256='a'*64),canonical_source_row_sha256=digest,
        mahalanobis_squared={c:0. for c in m.CHARTS}),original


def ledger_fixture(empty_volume=False):
    cfg,panel,rows=score_fixture(0. if empty_volume else 10.,(0,0) if empty_volume else (3,5))
    cfg.update(seed=123,cloud_limits=dict(raw_per_cloud=1000000,raw_per_pose=2000000,raw_total=2000000,
        processed_per_cloud=1000000,processed_per_pose=2000000,processed_total=2000000,callback_interval=7))
    e=panel['entries'][0];r=rows[0];digest='d'*64
    e.update(ordinal=0,source_rows=dict(path='/synthetic/rows.jsonl',sha256='a'*64))
    original=dict(input=dict(ordinal=0,proposed_pose=e['pose']),actual=dict(wall_valid=True,core_valid=True,physical_valid=True),
        region='B',patches=dict(tokens=[],neighbor_labels=[16,56]),envelope=r['envelope'])
    e['metadata']['original_row']=original
    line=json.dumps(original);identity=dict(source_rows=e['source_rows'],ordinal=0,
        line_sha256=hashlib.sha256(line.encode()).hexdigest(),input=original['input'])
    r.update(saved_row_identity=identity,region='B',patches=original['patches'],actual=original['actual'])
    events=[]
    def emit(kind,**values):events.append(dict(event_index=len(events),kind=kind,**values))
    emit('setup_begun');emit('pose_begun',pose_index=0,**{k:e[k] for k in ('id','pose','source_rows','ordinal')})
    emit('saved_row_checked',pose_index=0,id=e['id'],saved_row_identity=identity)
    emit('geometry_complete',pose_index=0,wall_valid=True,core_valid=True)
    emit('patches_complete',pose_index=0,region=r['region'],patches=r['patches'])
    emit('envelope_complete',pose_index=0,envelope=r['envelope'])
    total=0
    for i,c in enumerate(r['clouds']):
        n=c['weight']['raw_points'];k=c['weight']['overlap_points'];planned=None if empty_volume else n
        limits=dict(raw_points=min(1000000,2000000-total),processed_points=min(1000000,2000000-total),callback_interval=7)
        independent_seed=hashlib.sha256(b'fixed-saved-pose-overlap-v1\x00'+digest.encode()+struct.pack('<QQ',123,0)+f'cloud{i}'.encode()).hexdigest()
        emit('cloud_begun',pose_index=0,cloud=i,seed_sha256=independent_seed,remaining_limits=limits)
        def progress(event,planned,count,hits):
            emit('cloud_progress',pose_index=0,cloud=i,event=event,progress=dict(begun=True,complete=False,
                log_weight=None,planned_points=planned,processed_points=count,overlap_points=hits))
        progress('begun',None,0,0)
        if not empty_volume:progress('count_drawn',n,0,0)
        for count in range(7,n+1,7):progress('progress',planned,count,min(k,count))
        progress('finishing',planned,n,k)
        emit('cloud_complete',pose_index=0,cloud=i,progress=c['progress'],weight=c['weight']);total+=n
    r['work']=dict(raw_points=total,processed_points=total)
    emit('pose_complete',pose_index=0,id=e['id'])
    return events,cfg,digest,panel,rows,[line]


class PhysicalTests(unittest.TestCase):
    def test_exact32_strata_and_unconditional_allocations(self):
        inventory=jobs();m.validate_inventory(inventory)
        self.assertEqual(sum(j['attempts'] for j in inventory),32768)
        for change in ('missing','duplicate','count'):
            bad=copy.deepcopy(inventory)
            if change=='missing':bad.pop()
            elif change=='duplicate':bad[-1]=bad[0]
            else:bad[-1]['attempts']-=1
            with self.assertRaises(ValueError):m.validate_inventory(bad)

    def test_complete_arm_density_reconstruction_and_seam_zero(self):
        r,_=record(0);m.validate_density(r)
        r['log_source']['cage0']=None;r['log_g']=None
        r['log_q_arm']=float(m.log_mixture('multicage',[0.,-math.inf,2.,1.,0.,-math.inf,3.]));m.validate_density(r)
        r['log_q_arm']+=.01
        with self.assertRaisesRegex(ValueError,'complete multicage arm'):m.validate_density(r)

    def test_full_panel_and_canonical_source_authentication(self):
        pairs=[record(0),record(1,False),record(2,region='unbound')];records,originals=map(list,zip(*pairs))
        entries=[dict(ordinal=i,source_rows=records[i]['source_rows'],pose=records[i]['proposed_pose'],
            metadata=dict(log_q_balanced=records[i]['log_q_arm'],original_row=originals[i])) for i in (0,2)]
        job=dict(attempts=3,empty=False);panel=dict(entries=entries)
        self.assertEqual(m.validate_panel_coverage(job,panel,records,originals),[0,2])
        bad=copy.deepcopy(records);bad[0]['canonical_source_row_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Canonical'):m.validate_panel_coverage(job,panel,bad,originals)
        panel['entries'].pop()
        with self.assertRaisesRegex(ValueError,'ALL hard-valid'):m.validate_panel_coverage(job,panel,records,originals)

    def test_primary_pooled_count_weight_secondary_arithmetic_and_score_differ(self):
        cfg,_,rows=score_fixture();r=rows[0];v=m.paired_weights(r,-4.,cfg)
        self.assertAlmostEqual(v['log_rb'],.035*2+8*math.log1p(.035/(2*2.24))+4)
        self.assertAlmostEqual(v['log_arithmetic'],r['log_mean_positive_weight']+4)
        self.assertNotAlmostEqual(v['log_rb'],v['log_arithmetic'],places=8)
        self.assertNotAlmostEqual(v['log_rb'],r['score']['z_overlap']+4,places=6)
        r['clouds'][1]['weight']['lower_volume']+=1
        with self.assertRaisesRegex(ValueError,'common certified'):m.paired_weights(r,-4.,cfg)

    def test_pooled_count_conditional_expectation_and_variance(self):
        z=.35;lam=2.;lower=.4;volume=1.3
        for n in range(15):
            conditional=sum(math.comb(n,k)/2**n*(math.exp(z*lower)*(1+z/lam)**k+
                math.exp(z*lower)*(1+z/lam)**(n-k))/2 for k in range(n+1))
            self.assertAlmostEqual(math.log(conditional),m.rao_blackwell_log_weight(lower,n,lam,z,0.),places=12)
        probability=math.exp(-2*lam*volume);first=second=0.
        for n in range(90):
            value=math.exp(m.rao_blackwell_log_weight(lower,n,lam,z,0.))
            first+=probability*value;second+=probability*value*value;probability*=2*lam*volume/(n+1)
        expected=math.exp(z*(lower+volume))
        self.assertAlmostEqual(first,expected)
        self.assertAlmostEqual(second-first*first,expected**2*math.expm1(z*z*volume/(2*lam)))

    def test_mass_zero_attempts_and_all_five_radial_partitions(self):
        rows=[record(i,False)[0] for i in range(4096)]
        rows[0]=record(0,weight=3)[0];rows[1]=record(1,region='other_contact',weight=5)[0]
        rows[1]['source_T_complete_all_regions']=True
        value=m.population_statistics(rows,1.,2.)
        self.assertAlmostEqual(value['groups']['full_domain']['physical']['mass'],8/4096)
        self.assertAlmostEqual(value['groups']['T_outside_A']['physical']['mass'],5/4096)
        self.assertEqual(value['regions']['hard_invalid']['attempted_count'],4094)
        self.assertEqual(set(value['radial_partitions']),set(m.CHARTS))
        for chart in m.CHARTS:self.assertAlmostEqual(sum(b['mass'] for b in value['radial_partitions'][chart]['full_domain']),8/4096)
        with self.assertRaisesRegex(ValueError,'denominator'):m.population_statistics(rows[:-1],1.,2.)

    def test_complete_ledger_and_zero_volume_cloud_roles(self):
        for empty in (False,True):
            args=ledger_fixture(empty);out=m.audit_ledger(*args)
            self.assertEqual(out['clouds_completed'],2);self.assertEqual(out['raw_points'],0 if empty else 40)
            if empty:self.assertAlmostEqual(m.paired_weights(args[4][0],-4.,args[1])['log_rb'],4.07)

    def test_ledger_rejects_omitted_draw_failed_suffix_or_duplicate_seed(self):
        for case in ('count','failure','seed','index'):
            args=list(ledger_fixture());events=args[0]
            if case=='count':next(e for e in events if e.get('event')=='count_drawn')['event']='progress'
            elif case=='failure':events[-1]['kind']='pose_failed'
            elif case=='seed':
                begun=[e for e in events if e['kind']=='cloud_begun'];begun[1]['seed_sha256']=begun[0]['seed_sha256']
            else:events[-1]['event_index']-=1
            with self.assertRaises(ValueError):m.audit_ledger(*args)

    def test_ledger_rejects_changed_caps_progress_and_source_line(self):
        for case in ('caps','progress','source','complete'):
            args=list(ledger_fixture());events=args[0]
            if case=='caps':next(e for e in events if e['kind']=='cloud_begun')['remaining_limits']['raw_points']-=1
            elif case=='progress':next(e for e in events if e.get('event')=='progress')['progress']['processed_points']+=1
            elif case=='source':args[-1][0]+=' '
            else:next(e for e in events if e['kind']=='cloud_complete')['progress']['complete']=False
            with self.assertRaises(ValueError):m.audit_ledger(*args)

    def test_explicit_empty_receipt_keeps_failed_allocations_distinct(self):
        job=dict(id='j',stratum_id='s',attempts=2,config={'sha256':'cfg'},panel={'sha256':'panel'})
        receipt=dict(schema='context-broadened-empty-physical-v1',complete=True,passed=True,id='j',stratum_id='s',attempts=2,
            all_hard_invalid=True,all_attempts_preserved=True,config_sha256='cfg',panel_sha256='panel',
            valid_poses=0,clouds_begun=0,clouds_completed=0,raw_points=0,processed_points=0,
            new_poses_generated=0,retries=0,replacements=0,cpu_seconds=.1)
        m.validate_empty(job,{},dict(entries=[]),receipt)
        for k in ('passed','all_attempts_preserved'):
            bad=copy.deepcopy(receipt);bad[k]=False
            with self.assertRaises(ValueError):m.validate_empty(job,{},dict(entries=[]),bad)


if __name__=='__main__':unittest.main()
