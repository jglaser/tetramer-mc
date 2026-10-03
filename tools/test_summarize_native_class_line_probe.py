"""Synthetic report/provenance controls; no geometry or physical calculations."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from run_native_class_line_probe import read, sha
from summarize_native_class_line_probe import hard_free_log_density, summarize
from test_run_native_class_line_probe import fixture, save


def guide():
    return dict(defensive_uniform_shell_probability=.5, conditional_probability=1., minimum_conditional_mass=1e-12,
                raw_translation_axes=[0], gaussian_components=[dict(weight=1.,mean=[0.]*6,covariance=np.eye(6).tolist())])


def row(i, *, saved=False):
    branch=dict(channel=0,class_mass=.25,hard_free_mass=.5,effective_mass=.25,fallback_target='class')
    return dict(id=('saved:'+str(i)) if saved else i, latent=[0.]*6, raw_coordinates=[0.]*6,
                shell_valid=True,capture_valid=True,hard_valid=False,native_decision=None,exclusion_contact_by_anchor=None,
                draw=None if saved else dict(conditional=True,channel=0,class_mass=.25,hard_free_mass=.5,
                    effective_mass=.25,fallback_target='class',geometry=dict(channels=[dict(intervals=[dict(lower=-1.,upper=1.,lower_closed=True,upper_closed=True)])])),
                density_details=dict(axes=[dict(axis=0,hard_free_intervals=[dict(lower=-1.,upper=1.,lower_closed=True,upper_closed=True)],
                    components=[dict(channels=[branch])])]),log_proposal_density=-1.,log_physical_jacobian=0.,
                draw_cpu_seconds=.01,density_cpu_seconds=.02,observer_cpu_seconds=.03)


def jsonl(path, values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def report_fixture(root):
    plan=fixture(root)
    for arm in ('hard_free','class'): save(root/'guides'/f'{arm}.json',guide())
    save(root/'selected-probes.json',[dict(id='saved:'+str(i),group='toy-development') for i in range(40)])
    freeze=read(root/'freeze.json')
    freeze['files']={name:sha(root/name) for name in freeze['files']};save(root/'freeze.json',freeze)
    plan['files']={p:sha(p) for p in plan['files']}
    plan['initial_freeze_sha256']=sha(root/'freeze.json');save(root/'execution-plan.json',plan)
    for job in plan['jobs']:
        if job['phase']!='query':continue
        directory=Path(job['terminal']).parent;(directory/'provenance').mkdir(parents=True)
        save(directory/'provenance/importance-guide.json',guide())
        jsonl(directory/'samples.jsonl',[row(i) for i in range(job['samples'])])
        jsonl(directory/'probes.jsonl',[row(i,saved=True) for i in range(job['probes'])])
        jsonl(directory/'attempts.jsonl',[dict(ordinal=i) for i in range(job['samples']+job['probes'])])
        result=dict(complete=True,samples=job['samples'],probes=job['probes'],
            samples_sha256=sha(directory/'samples.jsonl'),probes_sha256=sha(directory/'probes.jsonl'),
            manifest=dict(executable_sha256=plan['executable_sha256'],seed=job['seed'],samples=job['samples'],
                guide_sha256=sha(directory/'provenance/importance-guide.json'),latent_radius=4.))
        save(job['terminal'],result)
        paths=[directory/'summary.json',directory/'provenance/importance-guide.json',directory/'samples.jsonl',directory/'probes.jsonl',directory/'attempts.jsonl']
        audit=next(j for j in plan['jobs'] if j.get('query_id')==job['id'])
        save(audit['terminal'],dict(complete=True,passed=True,synthetic=False,samples=job['samples'],probes=job['probes'],
            executable_sha256=plan['executable_sha256'],analysis_cpu_seconds=.1,
            input_sha256={str(p.resolve()):sha(p) for p in paths}))
    save(root/'execution/claim.json',dict(plan_sha256=sha(root/'execution-plan.json')))
    refresh_done(root,plan)
    return plan


def refresh_done(root,plan):
    save(root/'execution/summary.json',dict(complete=True,passed=True,plan_sha256=sha(root/'execution-plan.json'),
        fresh_draws=1024,saved_development_probes=40,new_physical_clouds=0,
        completed=[dict(id=j['id'],terminal=j['terminal'],sha256=sha(j['terminal'])) for j in plan['jobs']]))


class SummaryTest(unittest.TestCase):
    def test_every_invalid_attempt_and_conditional_mass_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report_fixture(root);report=summarize(root)
            for arm in ('hard_free','class'):
                self.assertEqual(report['arms'][arm]['unconditional_proposal_fractions']['hard_invalid']['mean'],1.)
                self.assertEqual(sum(p['attempted'] for p in report['arms'][arm]['populations']),512)
                p=report['arms'][arm]['populations'][0]
                self.assertEqual(p['selected_conditional_masses']['0']['class_mass'],dict(count=128,minimum=.25,median=.25,maximum=.25))
                self.assertEqual(p['scored_conditional_masses']['0']['hard_free_mass']['count'],128)
            self.assertEqual(len(report['saved_development_probes']),40)
            self.assertTrue(all(r['group']=='toy-development' for r in report['saved_development_probes']))
            self.assertFalse(report['assembly_gate_open'])

    def test_audited_guide_rows_or_missing_binding_cannot_change(self):
        for mutation in ('guide','rows','missing_binding','failed_audit','completed_identity','changed_plan'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                root=Path(d);plan=report_fixture(root);directory=root/'queries/hard_free-r00'
                if mutation=='guide':
                    g=guide();g['gaussian_components'][0]['mean'][0]=1.;save(directory/'provenance/importance-guide.json',g)
                elif mutation=='rows':
                    with (directory/'samples.jsonl').open('a') as f:f.write(json.dumps(row(128))+'\n')
                elif mutation in ('missing_binding','failed_audit'):
                    path=root/'audits/hard_free-r00.json';a=read(path)
                    if mutation=='missing_binding':del a['input_sha256'][str((directory/'provenance/importance-guide.json').resolve())]
                    else:a['passed']=False
                    save(path,a);refresh_done(root,plan)
                elif mutation=='completed_identity':
                    path=root/'execution/summary.json';done=read(path);done['completed'][0]['id']='different';save(path,done)
                else:
                    plan['repository']=str(root.parent);save(root/'execution-plan.json',plan)
                with self.assertRaises(ValueError):summarize(root)

    def test_attempt_count_is_independent_of_audit_pass_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=report_fixture(root);directory=root/'queries/hard_free-r00'
            jsonl(directory/'samples.jsonl',[row(i) for i in range(127)])
            result=read(directory/'summary.json');result['samples_sha256']=sha(directory/'samples.jsonl');save(directory/'summary.json',result)
            ap=root/'audits/hard_free-r00.json';a=read(ap);a['input_sha256']={p:sha(p) for p in a['input_sha256']};save(ap,a);refresh_done(root,plan)
            with self.assertRaisesRegex(ValueError,'Attempt count'):summarize(root)

    def test_hard_free_density_averages_components_axes_and_fallbacks(self):
        g=guide();g['raw_translation_axes']=[0,1]
        g['gaussian_components']=[dict(weight=1.,mean=[0.]*6,covariance=np.eye(6).tolist()),
                                  dict(weight=3.,mean=[0.]*6,covariance=(4*np.eye(6)).tolist())]
        r=row(0)
        def axis(a, masses, intervals):
            return dict(axis=a,hard_free_intervals=intervals,components=[dict(channels=[dict(hard_free_mass=m)]) for m in masses])
        inside=[dict(lower=-1.,upper=1.,lower_closed=True,upper_closed=True)]
        outside=[dict(lower=1.,upper=2.,lower_closed=True,upper_closed=True)]
        r['density_details']['axes']=[axis(0,[.5,0.],inside),axis(1,[.25,.5],outside)]
        volume=math.pi**3*4**6/6
        phi0=(2*math.pi)**-3;phi1=phi0/64
        expected=.5/volume+.5*(.25*phi0*(2+0)/2+.75*phi1*(1+0)/2)
        self.assertAlmostEqual(hard_free_log_density(r,g),math.log(expected),places=13)
        g['defensive_uniform_shell_probability']=1.
        self.assertAlmostEqual(hard_free_log_density(r,g),-math.log(volume),places=13)
        r['shell_valid']=False;self.assertIsNone(hard_free_log_density(r,g))


if __name__=='__main__':unittest.main()
