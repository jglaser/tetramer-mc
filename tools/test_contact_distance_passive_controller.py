#!/usr/bin/env python3
"""Allocation, immutable output and failure draining checks; no sampler launch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_contact_distance_passive import SEEDS, COVERAGE_SEEDS, WIDTHS, ARMS
import run_contact_distance_passive as controller


def protocol():
    jobs=[dict(arm=ARMS[i//4][0],id=f'r{i%4:02}',seed=s,
               fresh_proposal_draws=128,archived_probe_queries=78 if i%4==0 else 0,
               probe_file='probes.jsonl' if i%4==0 else None,
               directory=f'/tmp/passive/{i}',kind='proposal_only',status='pending') for i,s in enumerate(SEEDS)]
    coverage=[dict(arm=ARMS[i][0],id='coverage',seed=seed,fresh_proposal_draws=0,
        archived_probe_queries=128,probe_file='coverage-probes.jsonl',directory=f'/tmp/passive/coverage{i}',
        kind='proposal_only',status='pending') for i,seed in enumerate(COVERAGE_SEEDS)]
    return dict(fresh_draws=1536,archived_probe_queries=234,additional_coverage_queries=384,
        total_archived_queries=618,total_output_pose_rows=2154,maximum_CPU_workers=1,new_Poisson_clouds=0,
        widths_A=WIDTHS,contact_neighbor_indices=[0,1],minimum_center_distance=1e-8,minimum_polygon_area=1e-16,
        azimuth_floors=dict(radius_floor=1e-8,projection_floor=1e-10,gamma_min=.01,gamma_max=controller.math.pi),
        arms=[dict(name=a,conditional_probability=beta,localized_probability=b) for a,beta,b in ARMS],
        keep_all_draws=True,optional_stopping=False,retries=False,larger_autoextension=False,
        jobs=jobs,coverage_jobs=coverage)


class PassiveTests(unittest.TestCase):
    def test_exact_allocation(self):
        p=protocol();controller.validate_design(p)
        self.assertEqual(len(controller.all_jobs(p)),15)

    def test_physical_or_budget_changes_rejected(self):
        for key,value in [('fresh_draws',1537),('maximum_CPU_workers',2),('new_Poisson_clouds',1),
                          ('additional_coverage_queries',128),('keep_all_draws',False),('retries',True),
                          ('optional_stopping',True),('minimum_polygon_area',1e-8)]:
            p=protocol();p[key]=value
            with self.assertRaises((AssertionError,ValueError)):controller.validate_design(p)

    def test_seed_arm_and_probe_changes_rejected(self):
        for alter in [lambda p:p['jobs'][1].update(seed=SEEDS[0]),
                      lambda p:p['jobs'][1].update(archived_probe_queries=78),
                      lambda p:p['jobs'][0].update(probe_file='coverage-probes.jsonl'),
                      lambda p:p['coverage_jobs'][0].update(fresh_proposal_draws=1),
                      lambda p:p['arms'][2].update(localized_probability=.5),
                      lambda p:p['coverage_jobs'].pop()]:
            p=protocol();alter(p)
            with self.assertRaises((AssertionError,ValueError)):controller.validate_design(p)

    def test_commands_only_use_passive_binary_and_declared_probes(self):
        root=Path('/tmp/frozen')
        for job in controller.all_jobs(protocol()):
            cmd=controller.command(root,job)
            self.assertEqual(cmd[0],'/tmp/frozen/common/contact-distance-guide-audit')
            self.assertEqual(int(cmd[cmd.index('--samples')+1]),job['fresh_proposal_draws'])
            self.assertEqual('--probes' in cmd,bool(job['archived_probe_queries']))
            if job['archived_probe_queries']:
                self.assertEqual(cmd[cmd.index('--probes')+1],str(root/'common'/job['probe_file']))

    def test_failure_preserves_partial_output_and_stops_all_remaining_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir();preparation=root/'preparation';preparation.mkdir()
            p=protocol();p.update(preparation=str(preparation),repository=str(root))
            for i,job in enumerate(controller.all_jobs(p)):
                job['directory']=str(root/f'child{i}')
            launches=[]
            def fail(steps,*args,**kwargs):
                launches.extend(steps);job=steps[0];child=Path(job['directory']);child.mkdir()
                (child/'samples.jsonl').write_text('{"attempted":0}\n')
                job['status']='failed'
                raise RuntimeError('Injected child failure')
            with patch.object(controller,'validate',return_value=p),patch.object(controller,'execute_group',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'Injected child failure'):controller.run(out,'test-hash')
                with self.assertRaises((AssertionError,ValueError)):controller.run(out,'test-hash')
            self.assertEqual(len(launches),1)
            state=json.loads((out/'status.json').read_text())
            self.assertFalse(state['complete']);self.assertEqual(state['phase'],'failed')
            self.assertEqual(controller.all_jobs(state)[0]['status'],'failed')
            self.assertTrue(all(j['status']=='not_started' for j in controller.all_jobs(state)[1:]))
            self.assertEqual((Path(launches[0]['directory'])/'samples.jsonl').read_text(),'{"attempted":0}\n')
            self.assertTrue((preparation/'execution-claim.json').exists())

    def test_attempt_and_probe_corruption_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);common=root/'common';common.mkdir();child=root/'child';child.mkdir()
            samples=[dict(kind='fresh',id=i,latent=[0.]*6) for i in range(2)]
            source=[dict(id='saved',latent=[.1]*6)]
            probes=[dict(kind='probe',**source[0])]
            job=dict(fresh_proposal_draws=2,archived_probe_queries=1,probe_file='probes.jsonl')
            def write(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write(common/'probes.jsonl',source)
            write(child/'samples.jsonl',samples);write(child/'probes.jsonl',probes)
            controller.verify_attempts(child,common,job)
            for bad in [samples[:1],[samples[0],samples[0]],samples[::-1],
                        [dict(samples[0],id=False),samples[1]]]:
                write(child/'samples.jsonl',bad)
                with self.assertRaises((AssertionError,ValueError)):controller.verify_attempts(child,common,job)
            write(child/'samples.jsonl',samples)
            for bad in [[],[dict(probes[0],latent=[.2]*6)],[dict(probes[0],id='different')]]:
                write(child/'probes.jsonl',bad)
                with self.assertRaises((AssertionError,ValueError)):controller.verify_attempts(child,common,job)

    def test_shared_executor_drains_started_child_on_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=[]
            for i in range(2):
                jobs.append(dict(id=str(i),kind='proposal_only',status='pending',directory=str(root/f'out{i}'),
                    log=str(root/f'log{i}'),command=['unused']))
            waited=[];launched=[];raised=[]
            class Child:
                pid=123
                def poll(self):return None
                def wait(self):waited.append(True);return 0
            def popen(*args,**kwargs):launched.append(True);return Child()
            def snapshot():
                if any(j['status']=='running' for j in jobs) and not raised:
                    raised.append(True);raise InterruptedError('Injected interruption')
            with self.assertRaises(InterruptedError):
                controller.execute_group(jobs,snapshot,1,root,{},popen=popen,pause=lambda _:None,
                    capacity=lambda _:dict(workers=0,physical_pids=[]),birth=lambda _:0)
            self.assertEqual(len(launched),1);self.assertEqual(len(waited),1)
            self.assertEqual([j['status'] for j in jobs],['complete','not_started'])


if __name__=='__main__':unittest.main()
