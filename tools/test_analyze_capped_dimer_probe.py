"""Deterministic corruption, stopping and actual-map-density checks; no campaigns."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analyze_capped_dimer_probe as audit
from test_dimer_destination_density import fixture


def pose(x,y=0.): return dict(position=[x,y,0.],orientation=[1.,0.,0.,0.])
def feasible(ok=True):
    return dict(internal_core_overlap=False,spectator_core_collisions=[[],[]],wall_valid=[True,True],internal_exclusion_contact=ok)
def trial(j,ok): return dict(index=j,draw=dict(candidate={'marker':j}),feasibility=feasible(ok))


class CappedAuditTests(unittest.TestCase):
    def test_streams_are_per_outer_slot_not_cap(self):
        seeds={audit.stream_seed(123,a,c,i) for a in range(3) for c in range(8) for i in range(32)}
        self.assertEqual(len(seeds),768)
        self.assertEqual(audit.stream_seed(123,0,0,0),int(audit.hashlib.sha256(b'capped-dimer-probe-v1/123/0/0/0').hexdigest()[:16],16))

    def test_stopping_retains_exhaustion_and_detects_hidden_retries(self):
        checks=audit.Checks(); source=feasible()
        outcome=dict(trial_cap=8,source_feasibility=source,status='candidate',trials=[trial(1,False),trial(2,True)],candidate={'marker':2})
        audit.verify_stopping(outcome,8,source,checks,'good');self.assertEqual(checks.failures,[])
        bad=copy.deepcopy(outcome);bad['trials'][0]['feasibility']=feasible()
        audit.verify_stopping(bad,8,source,checks,'bad')
        self.assertTrue(any(r['check']=='first-success stopping' for r in checks.failures))
        for cap in [0,1,8]:
            outcome=dict(trial_cap=cap,source_feasibility=source,status='cap_exhausted',trials=[trial(i+1,False) for i in range(cap)],candidate=None)
            checks=audit.Checks();audit.verify_stopping(outcome,cap,source,checks,'exhausted');self.assertEqual(checks.failures,[])
        outside=feasible(False);outcome=dict(trial_cap=8,source_feasibility=outside,status='source_outside_contact',trials=[],candidate=None)
        checks=audit.Checks();audit.verify_stopping(outcome,8,outside,checks,'outside');self.assertEqual(checks.failures,[])
        outcome['trials']=[dict(index=1,draw=None,feasibility=None)];outcome['source_feasibility']=source
        with self.assertRaisesRegex(ValueError,'numerical/null'):
            audit.verify_stopping(outcome,8,source,audit.Checks(),'fatal')

    def test_paired_prefixes_detect_redraws_and_rng_changes(self):
        rows=[dict(cap=k,outcome=dict(trials=[trial(1,False)]+([trial(2,True)] if k>1 else [])),raw_contacts=[[]]*(2 if k>1 else 1),rng_after_fingerprint=[k==1,2]) for k in audit.CAPS]
        checks=audit.Checks();self.assertEqual(len(audit.verify_prefixes(rows,checks,'x')),2);self.assertEqual(checks.failures,[])
        bad=copy.deepcopy(rows);bad[0]['outcome']['trials'][0]['draw']['candidate']['marker']=9
        bad[1]['rng_after_fingerprint']=[8,8]
        audit.verify_prefixes(bad,checks,'bad');self.assertEqual(len(checks.failures),2)

    def fixture(self):
        helper=audit.DimerDestinationDensity(fixture(correlated=True))
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=.2)])
        state=[pose(0.),pose(1.),pose(10.)];case=dict(name='toy',root=0,child=1,anchor=2)
        oracle=audit.DimerGeometry(shape,state,.8,100.)
        old=state[:2];anchor=state[2];new=[pose(0.,1.),pose(1.,1.)]
        old_edges=[audit.relative_pose(anchor,old[0]),audit.relative_pose(old[0],old[1])]
        new_edges=[audit.relative_pose(anchor,new[0]),audit.relative_pose(new[0],new[1])]
        edges=[]
        for old_edge,edge in zip(old_edges,new_edges):
            edges.append(dict(old_relative_pose=old_edge,branch='uniform',proposed_relative_pose=edge,null_reason=None,
                trace=dict(branch_uniform=.25,translation_uniforms=((np.asarray(edge['position'])/160.+1.)/2.).tolist(),quaternion_normals=[1.,0.,0.,0.])))
        old_d=[helper.edge_density(p,.5,160.,kind='map') for p in old_edges];new_d=[helper.edge_density(p,.5,160.,kind='map') for p in new_edges]
        a=sum(d['log_full'] for d in old_d);b=sum(d['log_full'] for d in new_d)
        diag=dict(old_edges=old_d,new_edges=new_d,full_old_log_density=a,full_new_log_density=b,
                  log_reverse_forward=a-b,log_tree_coordinate_jacobian=0.,selection_log_reverse_forward=0.)
        geometry=oracle.fingerprint([0,1],new)
        raw=dict(index=1,draw=dict(spectator=anchor,old_root=old[0],old_child=old[1],edges=edges,
                 candidate=dict(root=new[0],child=new[1],diagnostics=diag),null_reason=None),
                 feasibility=audit.feasibility(oracle,case,new,geometry))
        return helper,oracle,case,old,anchor,raw,geometry

    def test_raw_audit_uses_map_law_and_detects_density_geometry_corruption(self):
        helper,oracle,case,old,anchor,raw,geometry=self.fixture();checks=audit.Checks()
        with patch.object(helper.score,'evaluate',side_effect=AssertionError('obsolete scored factor')):
            result=audit.audit_unique_trials(helper,oracle,case,old,anchor,[raw],[geometry['contacts']],.5,160.,checks,'toy')
        self.assertEqual(checks.failures,[]);self.assertTrue(result[0]['feasible']);self.assertEqual(result[0]['external_contacts'],0)
        bad=copy.deepcopy(raw);bad['draw']['candidate']['diagnostics']['log_reverse_forward']+=.1
        bad['feasibility']['internal_core_overlap']=True
        audit.audit_unique_trials(helper,oracle,case,old,anchor,[bad],[[]],.5,160.,checks,'corrupt')
        self.assertEqual(len(checks.failures),3)

    def test_learned_generation_reconstructs_reciprocal_branch(self):
        helper=audit.DimerDestinationDensity(fixture(correlated=True));z=[.1,.2,.3,-.1,-.2,-.3]
        p=helper.decode(1,z)
        edge=dict(branch='learned',null_reason=None,proposed_relative_pose=p,
                  trace=dict(branch_uniform=.75,target_label=dict(kind='single',member=0,anchor=0,branch=1),target_latent=z))
        checks=audit.Checks();audit.decode_edge(helper,edge,.5,160.,checks,'learned');self.assertEqual(checks.failures,[])
        edge['trace']['target_latent'][0]+=.1
        audit.decode_edge(helper,edge,.5,160.,checks,'changed');self.assertTrue(checks.failures)

    def test_current_source_binding_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);files={name:dict(text='example '+name,sha256=audit.hashlib.sha256(('example '+name).encode()).hexdigest()) for name in audit.MAP_FILES}
            (path/'source-bundle.json').write_text(json.dumps(dict(files=files)))
            hashes={name:v['sha256'] for name,v in files.items()}
            binding=dict(compiled_source_bundle_sha256=audit.sha(path/'source-bundle.json'))
            config=dict(compiled_source_sha256=hashes)
            protocol=dict(map_density_source_sha256=hashes.copy(),audit_files={'analyze_capped_dimer_probe.py':dict(path=audit.__file__,sha256=audit.sha(audit.__file__))})
            self.assertEqual(audit.bind_current_source(path,config,binding,protocol),hashes)
            protocol['map_density_source_sha256']['src/docking.rs']='wrong'
            with self.assertRaisesRegex(ValueError,'Unbound map'):audit.bind_current_source(path,config,binding,protocol)

    def test_summary_counts_every_outer_state(self):
        rows=[dict(status='cap_exhausted',raw_trials=8,proposal_cpu_seconds=.2,contact_diagnostic_cpu_seconds=.1,candidate=None),
              dict(status='candidate',raw_trials=2,proposal_cpu_seconds=.1,contact_diagnostic_cpu_seconds=.03,
                   candidate=dict(external_contacts=1,both_intended_contacts=True,log_reverse_forward=-2.))]
        result=audit.summarize(rows)
        self.assertEqual(result['outer_attempts'],2);self.assertEqual(result['candidate_count'],1)
        self.assertEqual(result['raw_trials'],10);self.assertEqual(result['candidate_fraction'],.5)


if __name__=='__main__':unittest.main()
