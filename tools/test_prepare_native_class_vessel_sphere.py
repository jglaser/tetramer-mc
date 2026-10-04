"""Bounded math/metadata controls only; no binary launch, geometry or poses."""
import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scipy.integrate import quad
import prepare_native_class_vessel_sphere as p
import analyze_native_class_vessel_sphere as a


class SphereReferenceTests(unittest.TestCase):
    def test_radial_integrals_simpson_bound_and_normalized_haar(self):
        for z in p.ACTIVITIES:
            for lo,hi in p.REGIONS.values():
                for n in (32,4096):
                    result=p.reference_mass(z,lo,hi,n)
                    # Independent adaptive quadrature, split at exclusion cutoff.
                    exact,error=quad(lambda r:4*math.pi*r*r*math.exp(z*p.overlap(r)),lo,hi,
                        points=[1.6] if lo<1.6<hi else None,epsabs=1e-12,epsrel=1e-12)
                    self.assertLessEqual(abs(result['mass']-exact),result['quadrature_error_bound']+error+5e-12*exact)
                    if z==0:self.assertEqual(result['mass'],4*math.pi*(hi**3-lo**3)/3)
        refs=p.references()
        for z in p.ACTIVITIES:
            r=refs[str(z)]
            for left,right in [('contact','unbound'),('inside_source','outside_source')]:
                self.assertAlmostEqual(r[left]['mass']+r[right]['mass'],r['total']['mass'],places=10)
        self.assertEqual(p.overlap(1.6),0.)
        self.assertGreater(p.overlap(.6),0.)
        with self.assertRaises(ValueError):p.reference_mass(.4,.6,2.7,31)

    def test_linear_population_SE_zero_uncertainty_and_log_cap(self):
        ref=dict(mass=10.,quadrature_error_bound=0.)
        sample=[9.4,9.8,10.2,10.6]
        checked=a.reference_check(sample,ref)
        self.assertAlmostEqual(checked['standard_error'],math.sqrt(sum((x-10)**2 for x in sample)/12))
        self.assertTrue(checked['passed'])
        wrong=a.reference_check([12.]*4,ref)
        self.assertEqual(wrong['standard_error'],0.);self.assertFalse(wrong['passed'])
        wide=a.reference_check([4.,8.,12.,24.],ref)
        self.assertTrue(wide['statistical_check']);self.assertFalse(wide['magnitude_check'])
        empty=a.reference_check([0.]*4,ref)
        self.assertIsNone(empty['absolute_log_discrepancy']);self.assertFalse(empty['passed'])
        retained=a.mean_stats([0.,0.,0.,4.]);self.assertEqual(retained['mean'],1.)
        self.assertEqual(retained['count'],4)

    def test_correlated_negative_controls_retain_unconditional_target(self):
        reference=dict(references=p.references());populations=[]
        for z in p.ACTIVITIES:
            for i,scale in enumerate((.98,.995,1.005,1.02)):
                regions={}
                for name in p.REGIONS:
                    exact=reference['references'][str(z)][name]['mass']*scale
                    hard=reference['references']['0.0'][name]['mass']*scale
                    regions[name]=dict(exact=dict(mean=exact),noisy=dict(mean=exact),hard=dict(mean=hard),
                        residual=dict(mean=0.),known_conditional_residual_SE=0.)
                populations.append(dict(activity=z,population=i,regions=regions,
                    negative_controls=dict(extra_jacobian=dict(mean=regions['total']['exact']['mean']*1e-3),
                        source_censored=dict(mean=regions['inside_source']['exact']['mean']))))
        groups,checks=a.aggregate(populations,reference)
        self.assertTrue(all(c['passed'] for c in checks))
        self.assertFalse(groups['0.4']['negative_controls']['extra_jacobian']['passed'])
        self.assertTrue(groups['0.4']['negative_controls']['source_censored_inside_target']['passed'])
        damaged=copy.deepcopy(populations)
        for item in damaged:item['negative_controls']['extra_jacobian']['mean']=item['regions']['total']['exact']['mean']
        _,checks=a.aggregate(damaged,reference)
        self.assertTrue(any(not c['passed'] and 'extra_jacobian' in c['name'] for c in checks))
        damaged=copy.deepcopy(populations);damaged[0]['negative_controls']['source_censored']['mean']*=2
        with self.assertRaisesRegex(ValueError,'denominator'):a.aggregate(damaged,reference)

    def test_allocation_and_bound_path_mutation(self):
        root=Path('/tmp/unlaunched-sphere-test');jobs=p.jobs_for(root,'/synthetic/python')
        self.assertEqual(len(jobs),8);self.assertEqual(sum(j['samples'] for j in jobs),16384)
        self.assertEqual(len({j['seed'] for j in jobs}),8)
        self.assertEqual(jobs[0]['seed'],6100410001);self.assertEqual(jobs[-1]['seed'],6100411004)
        self.assertTrue(all(j['argv'][0]==str(root/'common/basin-normalizer') for j in jobs))
        plan=dict(schema=p.SCHEMA,root=str(root),launched=False,preparation_only=True,criteria=p.CRITERIA,
            total_attempts=p.TOTAL,maximum_clouds=p.MAXIMUM_CLOUDS,populations_per_activity=4,
            samples_per_population=2048,activities=list(p.ACTIVITIES),retries=0,replacements=0,extensions=0,
            jobs=copy.deepcopy(jobs),python='/synthetic/python')
        plan['jobs'][0]['argv'][2]='/wrong/config.json'
        ledger=mock.Mock()
        with self.assertRaisesRegex(ValueError,'commands/seeds'):a.check_preparation(root,plan,{},ledger)
        ledger.frozen.assert_called_once_with(root)

    def test_embedded_bundle_identity_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'binary';bundle=root/'source-bundle.json'
            text='// synthetic source only\n'
            data=dict(schema=1,files={'src/lib.rs':dict(text=text,sha256=hashlib.sha256(text.encode()).hexdigest())})
            p.write(bundle,data);binary.write_bytes(b'not executable code\0'+bundle.read_bytes());binary.chmod(0o700)
            self.assertEqual(p.bind_bundle(binary,bundle),data)
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'embedded'):p.bind_bundle(binary,bundle)
            changed=root/'bad.json';bad=copy.deepcopy(data);bad['files']['../escape.rs']=bad['files'].pop('src/lib.rs')
            p.write(changed,bad);binary.write_bytes(changed.read_bytes())
            with self.assertRaisesRegex(ValueError,'Unsafe'):p.bind_bundle(binary,changed)

    def test_gate_status_snapshot_remains_bound_after_controller_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);live=root/'status.json';snapshot=root/'analysis/gate-status.json'
            raw=b'{"complete": false, "active": {"ordinal": 16}}\n'
            live.write_bytes(raw);ledger=a.Ledger()
            value,verify=a.snapshot_gate_status(live,snapshot,ledger)
            self.assertEqual(value['active']['ordinal'],16)
            self.assertEqual(snapshot.read_bytes(),raw)
            self.assertEqual(set(ledger.files),{str(snapshot.resolve())})
            verify()  # end of the authenticated gate
            live.write_text('{"complete": true, "active": null}\n')
            ledger.recheck()  # normal driver completion cannot invalidate receipt
            with self.assertRaisesRegex(ValueError,'changed during statistics gate'):verify()
            snapshot.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'Source changed'):ledger.recheck()


if __name__=='__main__':unittest.main()
