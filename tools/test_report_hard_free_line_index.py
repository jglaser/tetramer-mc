import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from report_hard_free_line_index import factors,fresh_cost_inputs,sha


class IndexIdentityTests(unittest.TestCase):
    def test_finite_space_marginal_and_second_moments(self):
        # Two Gaussian labels, two axes, four states; target only on states0,1.
        alpha=.5;w=np.array([.3,.7]);u=np.ones(4)/4
        g=np.array([[.1,.2,.3,.4],[.3,.2,.4,.1]])
        supports=np.array([[[1,1,0,0],[1,1,0,0]],[[1,1,1,0],[1,1,0,1]]],bool)
        z=np.sum(g[None,:,:]*supports,axis=2)
        h=g[None,:,:]*supports/z[:,:,None]
        old=alpha*u+(1-alpha)*np.sum(w[:,None]*g,axis=0)
        new=alpha*u+(1-alpha)*np.mean(np.sum(w[None,:,None]*h,axis=1),axis=0)
        target=np.array([2.,3.,0.,0.]);mean=0.;second=0.;mixed_mean=0.;mixed_second=0.
        for x in (0,1):
            f=factors(alpha,w,np.log(u[x]),np.log(g[:,x]),np.log(h[:,:,x]))
            self.assertAlmostEqual(np.exp(f['log_old_mixture']),old[x],places=14)
            self.assertAlmostEqual(np.exp(f['log_new_mixture']),new[x],places=14)
            primitive_x=alpha*u[x]*(target[x]/old[x])**2
            mean+=alpha*u[x]*target[x]/old[x]
            mixed_x=0.
            for j in range(2):
                for k in range(2):
                    p=(1-alpha)*w[k]/2;weight=target[x]*z[j,k]/old[x]
                    mean+=p*h[j,k,x]*weight;primitive_x+=p*h[j,k,x]*weight**2
                    a=alpha*u[x]+(1-alpha)*g[k,x];b=alpha*u[x]+(1-alpha)*h[j,k,x]
                    weight=target[x]*a/(old[x]*b);mixed_mean+=w[k]/2*b*weight
                    mixed_x+=w[k]/2*b*weight**2
            self.assertAlmostEqual(primitive_x,target[x]**2*np.exp(f['log_second_moment_factors']['primitive_index']),places=12)
            self.assertAlmostEqual(mixed_x,target[x]**2*np.exp(f['log_second_moment_factors']['mixed_index']),places=12)
            second+=primitive_x;mixed_second+=mixed_x
        self.assertAlmostEqual(mean,target.sum(),places=14);self.assertAlmostEqual(mixed_mean,target.sum(),places=14)
        marginal_second=np.sum(target*target/new);old_second=np.sum(target*target/old)
        self.assertLessEqual(marginal_second,mixed_second);self.assertLessEqual(mixed_second,second);self.assertLessEqual(second,old_second)

    def test_fallback_identity_and_arbitrarily_small_component(self):
        g=np.array([-2.,-1000.]);f=factors(.5,[.4,.6],-3.,g,np.tile(g,(3,1)))
        for mode,v in f['ratios'].items():
            self.assertAlmostEqual(v['over_old'],1.,places=13);self.assertAlmostEqual(v['over_full'],1.,places=13)

    def test_independent_cloud_second_moment_multiplies_index_factor(self):
        # Mean-one random noise with second moment1.25, independent of the label.
        noise=np.array([.5,1.5]);f=factors(.5,[.3,.7],-2.,np.log([.1,.3]),np.log([[.3,.4],[.2,.6]]))
        values=np.array([np.exp(v) for v in f['log_second_moment_factors'].values()])
        np.testing.assert_allclose(np.mean(values[:,None]*noise[None,:]**2,axis=1),1.25*values)

    def test_invalid_component_monotonicity_is_not_assumed(self):
        with self.assertRaisesRegex(ValueError,'lost physical support'):
            factors(.5,[1.],-2.,[-1.],[[-2.]])


class FreshCostProvenanceTests(unittest.TestCase):
    def fixture(self,root):
        def write(path,x):path.write_text(json.dumps(x))
        write(root/'protocol.json',dict(fixed=True))
        pops=[]
        for arm in ('baseline','xyz'):
            for i in range(4):
                p=dict(arm=arm,population=f'r{i:02}',input_sha256={});pops.append(p)
                if arm!='xyz':continue
                directory=root/arm/p['population'];directory.mkdir(parents=True)
                sample=directory/'samples.jsonl';sample.write_text('saved timing fixture\n')
                summary=directory/'summary.json';write(summary,dict(complete=True,samples=256,samples_sha256=sha(sample)))
                p['input_sha256']={str(f):sha(f) for f in (sample,summary)}
        analysis=dict(complete=True,total_attempted_draws=2048,protocol_sha256=sha(root/'protocol.json'),populations=pops)
        write(root/'analysis.json',analysis)
        write(root/'status.json',dict(complete=True,analysis_sha256=sha(root/'analysis.json'),protocol_sha256=sha(root/'protocol.json')))

    def test_authenticates_archived_chain_without_geometry_or_row_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);bindings={}
            rows=fresh_cost_inputs(root,bindings)
            self.assertEqual(len(rows),4);self.assertEqual(len(bindings),11)
            self.assertEqual([p.parent.name for p,_ in rows],['r00','r01','r02','r03'])

    def test_analysis_protocol_and_population_tampering_rejected(self):
        for changed,expected in [('analysis.json','analysis differs'),('protocol.json','protocol binding'),
            ('xyz/r00/summary.json','timing input changed'),('xyz/r00/samples.jsonl','timing input changed')]:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as d:
                root=Path(d);self.fixture(root);path=root/changed
                path.write_text(path.read_text()+'\n')
                with self.assertRaisesRegex(ValueError,expected):fresh_cost_inputs(root,{})


if __name__=='__main__':unittest.main()
