"""Three-center estimator composition, shared-row errors, and source schemas."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_shoulder_guide_peak_references import (combine_assigned,
    same_row_exclusion,validate_population_header,ensure_original_audits)
from shoulder_union_moments import UnionMoments
from prepare_cayley_rms_cover import read,write


def estimate(values):
    values=np.asarray(values,float);q=float(values.mean());variance=float(values.var(ddof=1)/len(values))
    return dict(draws=len(values),nonzero=int(np.count_nonzero(values)),logQ=math.log(q) if q else None,
        log_variance_of_mean=math.log(variance) if variance else None,
        row_RSE=math.sqrt(variance)/q if q else None,independent_population_RSE=math.sqrt(variance)/q if q else None)


class ShoulderGuideAnalysisTests(unittest.TestCase):
    def test_guided_and_latent_summary_schemas_are_distinct(self):
        runtime=dict(seed=7,samples=10)
        guided=dict(complete=True,samples=10,samples_sha256='bound-to-prior-audit')
        validate_population_header(guided,runtime,10,7,'guided')
        latent=dict(guided,manifest=runtime)
        validate_population_header(latent,runtime,10,7,'latent')
        with self.assertRaises(ValueError):validate_population_header(dict(latent,manifest={'seed':8}),runtime,10,7,'latent')
        with self.assertRaises(ValueError):validate_population_header(guided,runtime,9,7,'guided')
        with self.assertRaises(ValueError):validate_population_header(guided,runtime,10,8,'guided')

    def test_same_row_exclusion_error_includes_covariance(self):
        before=np.array([2.,0.,4.,1.,0.,3.]);after=np.array([2.,0.,0.,1.,0.,0.])
        moments=UnionMoments()
        for b,a in zip(before,after):
            value=math.log(b) if b else None
            # Mixture before includes every positive row; priority-direct
            # removes exactly those without an after contribution.
            radii=(.8 if a else .1,.1,.8)
            moments.add(radii,value,0. if b else None,[value]*2 if b else None)
        report=moments.report();b=report['physical']['mixture_ball0p25'];a=report['physical']['mixture_assigned_ball0p25']
        cov=report['same_row_covariances']['physical']['mixture_ball0p25']['mixture_assigned_ball0p25']
        result=same_row_exclusion(b,a,cov)
        self.assertAlmostEqual(math.exp(result['removed_logQ']),(before-after).mean())
        self.assertAlmostEqual(math.exp(result['removed_log_variance_of_mean']),(before-after).var(ddof=1)/len(before))
        expected=np.cov(before,after,ddof=1)/len(before);fraction=after.mean()/before.mean()
        variance=(expected[1,1]+fraction*fraction*expected[0,0]-2*fraction*expected[0,1])/before.mean()**2
        self.assertAlmostEqual(result['assigned_fraction_delta_SE'],math.sqrt(variance))
        self.assertAlmostEqual(result['assigned_fraction'],fraction)

    def test_three_region_union_sums_independent_means_not_draw_averages(self):
        direct_row=estimate([4.,2.,0.,6.]);direct_aggregate=dict(direct_row,total_unconditional_draws=direct_row.pop('draws'),unresolved_zero_pieces=0)
        direct=dict(independent_shell_sum={k:{'ball0p5':copy.deepcopy(direct_aggregate)} for k in ('physical','hard')},
            campaigns=[dict(root='old-small',radius_A=.25,populations=[dict(seed=10,samples=2)]),dict(root='old-large',radius_A=.5,populations=[dict(seed=11,samples=2)])])
        campaigns=[];selected_arrays=[]
        for i,(owner,radius) in enumerate((('mixture',.25),('mixture',.5),('geometry',.25),('geometry',.5))):
            sample=np.array([2.,0.,3.,5.])*(i+1);selected_arrays.append(sample)
            selected=estimate(sample);fake=estimate([1000.,1000.,1000.,1000.])
            mask=f"{owner}_assigned_{'ball0p25' if radius==.25 else 'radial_1'}"
            rows={mask:selected,f'{owner}_assigned_ball0p5':fake,'full':fake}
            campaigns.append(dict(root=f'run{i}',owner=owner,radius_A=radius,populations=[dict(seed=100+i,samples=4)],physical=rows,hard=rows))
        original=copy.deepcopy((campaigns,direct));result=combine_assigned(campaigns,direct)
        expected=math.exp(direct_aggregate['logQ'])+sum(a.mean() for a in selected_arrays)
        variance=math.exp(direct_aggregate['log_variance_of_mean'])+sum(a.var(ddof=1)/len(a) for a in selected_arrays)
        self.assertAlmostEqual(math.exp(result['union']['physical']['logQ']),expected)
        self.assertAlmostEqual(math.exp(result['union']['physical']['log_variance_of_mean']),variance)
        self.assertEqual(result['union']['physical']['total_unconditional_draws'],20)
        self.assertEqual((campaigns,direct),original)
        self.assertEqual(result['source_pieces']['geometry'][1]['source_seeds'],[103])
        campaigns[0]['populations'][0]['seed']=10
        with self.assertRaises(ValueError):combine_assigned(campaigns,direct)

    def test_failed_auditor_waits_for_other_children(self):
        class Child:
            def __init__(self,pid,code):self.pid=pid;self.code=code;self.waited=False
            def wait(self):self.waited=True;return self.code
        children=[Child(11,1),Child(12,0)]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';out.mkdir();requests=[]
            for i in range(2):
                root=Path(tmp)/f'root{i}';root.mkdir();directory=root/'r00';directory.mkdir()
                write(root/'r00-status.json',dict(id='r00',returncode=0));write(directory/'summary.json',dict(complete=True,samples=10))
                requests.append((dict(name=('mixture','geometry')[i],radius_A=.25),root,dict(jobs=[dict(id='r00',samples=10,directory=str(directory))])))
            with patch('analyze_shoulder_guide_peak_references.subprocess.Popen',side_effect=children):
                with self.assertRaises(ValueError):ensure_original_audits(requests,out,True)
            state=read(out/'runner-state.json')
            self.assertTrue(all(c.waited for c in children));self.assertEqual([p['exit_code'] for p in state['processes']],[1,0])
            self.assertEqual(state['phase'],'original_audits_failed')

    def test_incomplete_later_campaign_prevents_any_auditor_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';out.mkdir();requests=[]
            for i in range(2):
                root=Path(tmp)/f'root{i}';root.mkdir();directory=root/'r00';directory.mkdir()
                write(root/'r00-status.json',dict(id='r00',returncode=i));write(directory/'summary.json',dict(complete=True,samples=10))
                requests.append((dict(name=('mixture','geometry')[i],radius_A=.25),root,dict(jobs=[dict(id='r00',samples=10,directory=str(directory))])))
            with patch('analyze_shoulder_guide_peak_references.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):ensure_original_audits(requests,out,True)
                launch.assert_not_called()


if __name__=='__main__':unittest.main()
