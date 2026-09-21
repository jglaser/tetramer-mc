"""Four-body graph and inverse-trace controls; no physical sampler or old audit."""
import copy
import itertools
import json
from pathlib import Path
import sys
import unittest
import numpy as np

from analyze_mobile_posterior_pilot import (FOUR_BODY_SCHEMA, affected_body_pairs,
    audit_densities, campaign_body_count, density_record, four_body_graph_flags,
    summarize_four_body_growth, supported_registered_bonds, validate_campaign_jobs,
    validate_move_indices, ChartAudit)
from mobile_posterior_metrics import summarize_graph_history
from test_analyze_mobile_posterior_pilot import model, pose, posterior_record
from test_reciprocal_pose_density import fixture, envelope, posterior_record as reciprocal_record


TRIANGLE=[(0,1,3),(0,2,7),(1,2,4)]


def history(states):
    result=[]
    for index,value in enumerate(states):
        keys,entry,near=value[:3]
        result.append(dict(serial=index-1,sweep=index,source='initial'if index==0 else'full-mixture-capture:learned',
            accepted=None if index==0 else value[3]if len(value)>3 else True,
            registered_keys=keys,instantaneous_entry_keys=entry,
            native_edges=sorted({k[:2]for k in keys}),nonspecific_edges=near,
            moving_index=None if index==0 else 3,anchor_index=None if index==0 else 2))
    return result


def summarize(rows,burn=0):
    metrics=summarize_graph_history(rows,4,burn,rows[-1]['sweep'],10.)
    frames=[dict(sweep=r['sweep'],registered_monomer_support=dict(entry_count=0,retained_count=0,
        tracked_entry_count=0,tracked_retained_count=0,entry_bonds=[],retained_bonds=[]))for r in rows]
    return summarize_four_body_growth(rows,frames,metrics,burn,12.,10.),metrics


class FourBodyObserverTests(unittest.TestCase):
    def test_explicit_grid_and_body_contract_does_not_relax_old_grids(self):
        jobs=[dict(id=f'{start}-{r}',start=start,mode='c09',replicate=r)
            for start in ('triangle_free','retained_motif8')for r in (0,1)]
        manifest=dict(schema=FOUR_BODY_SCHEMA,atlas_variant='original',body_count=4,
            tracked_body_index=3,scaffold_body_indices=[0,1,2],jobs=jobs)
        status=dict(complete=True,running=False,jobs=[dict(id=j['id'],status='complete',exit_code=0)for j in jobs])
        for variant in ('original','coverage'):
            manifest['atlas_variant']=variant
            self.assertEqual(validate_campaign_jobs(manifest,status),jobs)
            self.assertEqual(campaign_body_count(manifest),4)
        for field,value in (('body_count',3),('tracked_body_index',0),('scaffold_body_indices',[1,2,3]),('atlas_variant','legacy')):
            bad=copy.deepcopy(manifest);bad[field]=value
            with self.assertRaises(AssertionError):validate_campaign_jobs(bad,status)
        bad=copy.deepcopy(manifest);bad['jobs'][0]['mode']='c0'
        with self.assertRaises(AssertionError):validate_campaign_jobs(bad,status)
        for schema in ('mobile-frozen-posterior-pilot-campaign-v1','mobile-competing-atlas-benchmark-v1','mobile-reciprocal-atlas-benchmark-v1'):
            self.assertEqual(campaign_body_count(dict(schema=schema)),3)
            with self.assertRaises(AssertionError):campaign_body_count(dict(schema=schema,body_count=4))

    def test_all_six_pairs_and_collective_cross_pairs(self):
        pairs=set(itertools.combinations(range(4),2))
        self.assertEqual(len(pairs),6)
        for moving in range(4):
            self.assertEqual(set(affected_body_pairs(4,[moving])),{p for p in pairs if moving in p})
            self.assertEqual(len(affected_body_pairs(4,[moving])),3)
        self.assertEqual(affected_body_pairs(4,[0,1],True),[(0,2),(0,3),(1,2),(1,3)])
        self.assertEqual(affected_body_pairs(4,range(4),True),[])
        for invalid in ([4],[-1],[True],[1,1]):
            with self.assertRaises(AssertionError):affected_body_pairs(4,invalid,True)

    def test_triangle_plus_tail_and_two_triangles_are_connected_but_not_K4(self):
        full=set(itertools.combinations(range(4),2))
        for edges in ({k[:2]for k in TRIANGLE}|{(2,3)},full-{(2,3)}):
            flags=four_body_graph_flags(edges,TRIANGLE,TRIANGLE,TRIANGLE)
            self.assertTrue(flags['all_four_connected']and flags['scaffold_triangle'])
            self.assertFalse(flags['complete_K4'])
        self.assertTrue(four_body_graph_flags(full,TRIANGLE,TRIANGLE,TRIANGLE)['complete_K4'])
        flags=four_body_graph_flags({k[:2]for k in TRIANGLE},TRIANGLE,TRIANGLE,TRIANGLE)
        self.assertFalse(flags['all_four_connected']or flags['tracked_attached'])

    def test_entry_differs_from_retention_and_bond_counts_deduplicate(self):
        motif={8:dict(member_contacts=[dict(member_i=0,member_j=1,directed_class='C1'),
                dict(member_i=2,member_j=3,directed_class='C2')]),
               9:dict(member_contacts=[dict(member_i=0,member_j=1,directed_class='C1')])}
        a=dict(bodies=[2,3],members=[0,1],class_label='C1')
        b=dict(bodies=[2,3],members=[2,3],class_label='C2')
        unrelated=dict(bodies=[0,1],members=[0,1],class_label='C1')
        classification=dict(native_entry_monomer_edges=[a,a,unrelated],native_stay_monomer_edges=[a,b,a,unrelated])
        keys={(2,3,8),(2,3,9)}
        self.assertEqual(len(supported_registered_bonds(classification,keys,motif,True)),1)
        self.assertEqual(len(supported_registered_bonds(classification,keys,motif,False)),2)
        self.assertEqual(supported_registered_bonds(classification,set(),motif,False),[])

    def test_attachment_loss_return_and_partner_exchange_keep_update_axis(self):
        tri_edges=[k[:2]for k in TRIANGLE]
        tail=TRIANGLE+[(2,3,8)];other=TRIANGLE+[(0,3,4)]
        states=[(TRIANGLE,TRIANGLE,tri_edges),
            (tail,tail,tri_edges+[(2,3)]),
            (tail,TRIANGLE,tri_edges+[(2,3)]), # Stay-only D: no strict entry.
            (TRIANGLE,TRIANGLE,tri_edges),
            (other,other,tri_edges+[(0,3)]),
            (other,other,tri_edges+[(0,3)],False),
            (TRIANGLE,TRIANGLE,tri_edges),
            (tail,tail,tri_edges+[(2,3)])]
        rows=history(states);result,metrics=summarize(rows)
        self.assertEqual(len(metrics['individual_edge_apparent_ess']),12) # Both graphs × six pairs.
        native=result['outcomes']['native'];entry=result['outcomes']['instantaneous_entry']
        self.assertEqual(native['first_observed']['tracked_attached']['sweep'],1)
        self.assertEqual(entry['transitions'][2]['value'],False) # Entry loss precedes hysteretic loss.
        self.assertFalse(result['observations'][2]['instantaneous_entry']['tracked_attached'])
        self.assertTrue(result['observations'][2]['native']['tracked_attached'])
        self.assertEqual(native['tracked_event_rates']['full']['attempted_updates'],7)
        self.assertEqual(native['tracked_event_rates']['full']['counts'],dict(formed_edges=3,detached_edges=2))
        self.assertEqual([(v['lost_partner'],v['gained_partner'])for v in native['tracked_partner_exchanges']],[(2,0),(0,2)])
        self.assertEqual(len(result['observations']),8)
        self.assertGreater(native['tracked_body']['episodes']['full']['observed_environment_returns'],0)
        self.assertIsNone(native['scaffold_first_loss'])
        json.dumps(result,allow_nan=False)

    def test_retained_start_is_left_censored_and_triangle_loss_is_separate(self):
        tail=TRIANGLE+[(2,3,8)];tail_edges=[k[:2]for k in tail]
        broken=[k for k in tail if k!=(0,1,3)]
        rows=history([(tail,tail,tail_edges),(tail,tail,tail_edges,False),
            (broken,broken,[k[:2]for k in broken])])
        result,_=summarize(rows,burn=1);native=result['outcomes']['native']
        self.assertEqual(native['first_observed']['tracked_attached']['serial'],-1)
        self.assertTrue(native['first_observed']['tracked_attached']['initial_state'])
        self.assertEqual(native['tracked_event_rates']['full']['counts'],dict(formed_edges=0,detached_edges=0))
        self.assertEqual(native['scaffold_first_loss']['sweep'],2)
        self.assertTrue(result['observations'][-1]['native']['all_four_connected'])
        self.assertFalse(result['observations'][-1]['native']['scaffold_triangle'])
        self.assertEqual(native['tracked_body']['episodes']['full']['episodes'][0]['left_censored'],True)

    def test_all_twelve_moving_anchor_roles_replay_inverse_traces(self):
        cfg=dict(uniform_proposal_cube_lengths=[100.]*3,learned_uniform_weight=.1,
            frozen_posterior=dict(probability=.5,correlation=.9))
        audit=ChartAudit(model());records=[]
        for i,j in itertools.permutations(range(4),2):
            template=posterior_record(.9);state=[pose(10*k,k,-k)for k in range(4)]
            state[j]=copy.deepcopy(template['anchor']);state[i]=copy.deepcopy(template['old'])
            info=copy.deepcopy(template['proposal']);info.update(moving_index=i,anchor_index=j)
            move=dict(kind='global',moving_index=i,update_in_sweep=0,proposed_pose=template['new'],accepted=True,proposal=info)
            self.assertEqual(validate_move_indices(move,4,set()),(i,j))
            record=density_record(len(records),'frozen-posterior',move,state,audit);records.append(record)
            state[j]['position'][0]+=999 # Saved frame must remain independent of later common shifts.
            self.assertEqual(record['anchor'],template['anchor'])
        result=audit_densities(model(),records,cfg,audit)
        self.assertEqual(result['posterior_checks'],12)
        self.assertEqual(result['map_checks'],12)
        self.assertLess(max(audit.max_errors.values()),1e-10)

    def test_partial_178_base_328_virtual_atlas_replays_ordinary_and_inverse_roles(self):
        proposal=envelope(fixture(178),[True]*150+[False]*28)
        audit=ChartAudit(proposal)
        self.assertEqual(len(audit.weights),328)
        self.assertEqual((int(audit.base_indices[299]),bool(audit.inverted[299])),(149,True))
        self.assertEqual((int(audit.base_indices[300]),bool(audit.inverted[300])),(150,False))
        records=[]
        for index,(i,j)in enumerate(itertools.permutations(range(4),2)):
            source,target=((299,300),(327,1),(1,327),(300,299))[index%4]
            record=reciprocal_record(proposal,source,target,.9)
            anchor=pose(3*j,-2*j,j);state=[pose(8*k,k,-k)for k in range(4)]
            state[j]=anchor;state[i]=audit.absolute(record['old_relative'],anchor)
            info=record['proposal'];info.update(moving_index=i,anchor_index=j)
            move=dict(kind='global',moving_index=i,update_in_sweep=0,
                proposed_pose=audit.absolute(record['new_relative'],anchor),accepted=True,proposal=info)
            self.assertEqual(validate_move_indices(move,4,set()),(i,j))
            records.append(density_record(index,'frozen-posterior',move,state,audit))
        result=audit_densities(proposal,records,dict(frozen_posterior=dict(correlation=.9)),audit)
        self.assertEqual(result['posterior_checks'],12);self.assertEqual(result['map_checks'],12)
        bad=copy.deepcopy(records);bad[0]['proposal']['target_inverted']=True
        with self.assertRaises(AssertionError):audit_densities(proposal,bad,dict(frozen_posterior=dict(correlation=.9)),ChartAudit(proposal))

    def test_bad_anchor_and_duplicate_single_body_update_rejected(self):
        move=dict(kind='global',moving_index=3,update_in_sweep=0,proposed_pose=pose(),proposal=dict(anchor_index=2))
        for anchor in (3,4,-1,True):
            bad=copy.deepcopy(move);bad['proposal']['anchor_index']=anchor
            with self.assertRaises(AssertionError):validate_move_indices(bad,4,set())
        with self.assertRaises(AssertionError):validate_move_indices(move,4,{3})
        with self.assertRaises(AssertionError):validate_move_indices(move,4,{0})

    def test_actual_four_body_initial_observer_preflight_without_MC(self):
        """Use authoritative archived observer/data on just the two inert starts."""
        from analyze_mobile_posterior_pilot import read, sha
        root=Path(__file__).resolve().parents[1]
        design=root/'runs/mobile-four-body-design-20260921'
        for name,digest in read(design/'freeze.json').items():self.assertEqual(sha(design/name),digest)
        starts=read(design/'recommended-poses.json')
        definition=root/'runs/mobile-full-capture-campaign-20260921/provenance/native-region/definition.json'
        inputs=definition.parent/'inputs';reference=inputs/'reference'
        for name,digest in read(definition)['input_sha256'].items():self.assertEqual(sha(inputs/name),digest)
        sys.dont_write_bytecode=True;sys.path.insert(0,str(reference/'scripts'))
        from audit_tetramer_assembly import AtomicAssembly,pose_arrays
        from tetramer_order import TetramerOrder
        from analyze_precursor_exchange import prepare_templates
        for name in ('audit_tetramer_assembly','tetramer_order','analyze_precursor_exchange'):
            self.assertEqual(Path(sys.modules[name].__file__).resolve(),reference/'scripts'/f'{name}.py')
        shape=read(inputs/'tetramer-shape.json')
        cfg=dict(shape=str(inputs/'tetramer-shape.json'),monomer_shape=str(inputs/'monomer-shape.json'),
            native_pair_motifs=str(inputs/'native-pair-motifs.json'),rigid_members=shape['rigid_members'],
            box_lengths=[2200.]*3,seed_labels=[],depletant_radius=1.5)
        templates=prepare_templates(cfg['monomer_shape'],reference/'results/c1c3-scaffold/motifs.json',
            reference/'results/native-neighbor-classes/classification.json')
        observer=TetramerOrder(cfg,inputs,templates=templates);atomic=AtomicAssembly(cfg,inputs)
        motifs={m['id']:m for m in observer.body_motifs}
        for name,total,D in (('triangle_plus_free',10,0),('bound_retention',16,6)):
            frame=dict(poses=starts[name]);classification=observer.classify(frame)
            self.assertEqual(classification['classified_body_pairs'],6)
            self.assertTrue(classification['complete_pair_graph'])
            entry={(*m['bodies'],m['motif_id'])for m in classification['registered_tetramer_motifs']if m['entry']}
            self.assertTrue(set(TRIANGLE)<=entry)
            bonds=supported_registered_bonds(classification,entry,motifs,True)
            self.assertEqual(len(bonds),total);self.assertEqual(sum(3 in b[:2]for b in bonds),D)
            if D:self.assertIn((2,3,8),entry)
            else:self.assertFalse(any(3 in k[:2]for k in entry))
            p,q=pose_arrays(frame);checked=atomic.frame(p,q)
            self.assertTrue(checked['hard_valid'])
            if not D:self.assertFalse(any(3 in edge for edge in checked['depletion_edges']))
            self.assertEqual(supported_registered_bonds(classification,entry,motifs,False),bonds)


if __name__=='__main__':unittest.main()
