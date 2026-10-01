import json
from pathlib import Path
import tempfile
import unittest
import plot_hard_free_line_physical as display


def write(path,value):path.write_text(json.dumps(value)+'\n')


def fixture(root):
    (root/'comparison').mkdir();protocol=dict(total_unconditional_draws=24);write(root/'protocol.json',protocol)
    arms={}
    for ai,name in enumerate(display.ARMS):
        estimates={}
        for ri,region in enumerate(display.REGIONS):
            pops=[dict(id=f'r{i:02}',log_Qz=None if i==3 and ri==3 else 10+ri+i*.1+ai*.05) for i in range(4)]
            estimates[region]=dict(populations=pops,row_uncertainty=dict(draws=12,log_Qz=10+ri),population_uncertainty=dict(log_Qz=10+ri+.15))
        arms[name]=dict(populations=[dict(id=f'r{i:02}',seed=4*ai+i,samples=3) for i in range(4)],estimates=estimates,
            observed_importance_ESS_per_cpu_second={r:1.+ai*.1 for r in display.REGIONS})
    failed=[]
    for i in range(70):
        family='orthant' if i<64 else 'radial' if i<67 else 'angular';index=i if i<64 else (i-64)%3
        failed.append(dict(significant=True,family=family,region=display.REGIONS[0],bin=index,comparison=dict(passed=False,observed=False,reason=f'unobserved-{i:02}')))
    intervals={a:dict(observed=True,beta_F_native_minus_noentry=-1.,halfwidth_95=.4) for a in display.ARMS}
    quality={a:{r:dict(passed=False,observed=True,
        checks=dict(population_RSE=True,importance_ESS=False,largest_draw=False),
        values=dict(population_RSE=.07,importance_ESS=42,largest_draw=.031))
        for r in display.REGIONS} for a in display.ARMS}
    quality['baseline'][display.REGIONS[-1]]=dict(passed=False,observed=False,reason='Unobserved is not zero.')
    comparisons={r:dict(observed=True,passed=False,log_left_minus_right=.24,
        combined_population_log_delta_SE=.05,absolute_passed=False,SE_passed=False) for r in display.REGIONS}
    data=dict(schema='hard-free-line-physical-comparison-v1',complete=True,total_unconditional_draws=24,
        protocol_sha256=display.sha(root/'protocol.json'),arms=arms,diagnostics=dict(full_vessel_gate_open=False,assembly_gate_open=False,
            all_stratum_comparisons=failed,free_energy_intervals=intervals,quality=quality,regional_comparisons=comparisons))
    write(root/'comparison/analysis.json',data)
    state=dict(complete=True,phase='complete',jobs=[dict(status='complete',returncode=0) for _ in range(8)],
        audits=[dict(status='complete',returncode=0) for _ in range(8)],protocol_sha256=display.sha(root/'protocol.json'),
        comparison_sha256=display.sha(root/'comparison/analysis.json'))
    write(root/'status.json',state);return data,state


class CompletedPhysicalDisplayTests(unittest.TestCase):
    def test_live_or_changed_campaign_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,state=fixture(root);state['complete']=False;write(root/'status.json',state)
            with self.assertRaisesRegex(ValueError,'Wait for complete'):display.load_completed(root)
            state['complete']=True;write(root/'status.json',state)
            data['total_unconditional_draws']=999;write(root/'comparison/analysis.json',data)
            with self.assertRaisesRegex(ValueError,'Controller-bound'):display.load_completed(root)

    def test_render_all_failed_strata_and_explicit_unobserved_points(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,_=fixture(root);out=root/'display';result=display.plot(root,out)
            self.assertEqual(result['failed_material_strata_count'],70)
            self.assertEqual(len(result['unobserved_population_masses']),2)
            self.assertFalse(result['finite_assembly_resolved']);self.assertFalse(result['full_vessel_resolved'])
            report=(out/'report.md').read_text()
            for i in range(70):self.assertIn(f'unobserved-{i:02}',report)
            self.assertEqual(result['quality'],data['diagnostics']['quality'])
            self.assertEqual(result['regional_comparisons'],data['diagnostics']['regional_comparisons'])
            self.assertIn('| 0.07 | 42 | 0.031 | False |',report)
            self.assertIn('| unobserved | unobserved | unobserved | False |',report)
            self.assertIn('| 0.24 | 0.05 | False | False | False |',report)
            self.assertTrue((out/'hard-free-line-physical.png').exists())


if __name__=='__main__':unittest.main()
