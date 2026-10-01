"""Algebra and completed-archive plumbing; no physical geometry or sampling."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp
import report_hard_free_line_physical_index as index


def write(path,value):path.write_text(json.dumps(value,allow_nan=False)+'\n')


def guide(beta=1.):
    return dict(schema='defensive-hard-free-line-guide-v1',conditional_probability=beta,
        defensive_uniform_shell_probability=.5,gaussian_components=[{}],
        raw_translation_axes=[0,1,2],minimum_conditional_mass=1e-12)


def row(label,mass=.5,old=.2,new=.4):
    uniform = label == 'uniform'; conditional = label in ('success','fallback')
    return dict(proposal_branch='uniform-shell' if uniform else 'hard-free-line',
        proposal_component=None if uniform else 0,log_proposal_density=math.log(new),
        hard_free_line_density=dict(baseline_log_density=math.log(old)),
        hard_free_line_draw=dict(conditional=conditional,component=0,axis=1,
            conditional_mass=mass,fallback=label=='fallback'))


def completed_fixture(root):
    """Synthetic full completion chain, with no claim of physical audit validity."""
    arms = [dict(id=name,beta=float(name=='conditioned'),alpha=.5,component_count=1,samples=6) for name in index.ARMS]
    jobs=[]; records={name:[]for name in index.ARMS}; full={name:[]for name in index.ARMS}
    for ai,arm in enumerate(arms):
        (root/'audits'/arm['id']).mkdir(parents=True)
        for pi in range(4):
            pid=f'r{pi:02}';directory=root/arm['id']/pid;prov=directory/'provenance';prov.mkdir(parents=True)
            g=guide(arm['beta']);region=dict(mahalanobis_radius=4.,minimum_original_q=0.)
            write(prov/'region.json',region);write(prov/'importance-guide.json',g)
            rows=[]
            for i,label in enumerate(('uniform','success','fallback','success','uniform','fallback')):
                old=.2+.02*i;new=old*(2. if label=='success' else 1.5 if label=='uniform' else 1.) if arm['beta'] else old
                r=row(label,0. if label=='fallback' else .25,old,new)
                if not arm['beta']:
                    r['hard_free_line_draw']=dict(conditional=False)
                    r['hard_free_line_density']=dict(conditioning_disabled=True)
                u=[0.]*6;u[0]=8. if i==5 else .1*i
                valid=i<4;logs=[.1+.01*pi,.3+.01*pi];h=-math.log(new)
                r.update(draw=i,latent=u,latent_radius=float(np.linalg.norm(u)),q=1.,
                    shell_valid=i!=5,capture_valid=True,hard_valid=valid,region_valid=True,
                    log_physical_jacobian=0.,log_hard_weight=h if valid else None,
                    log_importance_weight=float(h+logsumexp(logs)-math.log(2)) if valid else None,
                    clouds=[dict(log_weight=v) for v in logs] if valid else [])
                rows.append(r)
            samples=directory/'samples.jsonl';samples.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            attempts=directory/'attempts.jsonl';attempts.write_text(''.join(json.dumps(dict(draw=i,state='begin'))+'\n' for i in range(6)))
            seed=100+ai*4+pi
            manifest=dict(schema='importance-latent-region-normalizer-v6',guide_schema=g['schema'],seed=seed,samples=6)
            write(directory/'manifest.json',manifest)
            arrays=index.read_weights(rows,6,region,arm);mom=index.paired_moments(arrays['z'],arrays['h'])
            summary=dict(complete=True,sampler_cpu_seconds=2.,samples_sha256=index.sha(samples),attempts_sha256=index.sha(attempts))
            write(directory/'summary.json',summary)
            paths=[samples,attempts,directory/'summary.json',directory/'manifest.json',prov/'region.json',prov/'importance-guide.json']
            audit=dict(schema='independent-hard-free-line-physical-audit-v1',complete=True,geometry_mode='full',samples=6,
                input_sha256={str(p):index.sha(p) for p in paths},
                estimate=dict(logQ=mom['log_Qz'],draws=6),hard_region=dict(logQ=mom['log_Q0'],draws=6))
            audit_path=root/'audits'/arm['id']/(pid+'.json');write(audit_path,audit)
            arrays['u']=arrays.pop('latents')
            arrays.update(draw=np.arange(6),source_n=np.full(6,6),log_q=np.array([r['log_proposal_density']for r in rows]),
                log_physical_jacobian=np.zeros(6),native=np.array([1,0,0,1,0,0],bool),contact=np.array([1,1,0,1,0,0],bool),
                old_R5_intersection_native=np.array([1,0,0,0,0,0],bool),remaining_R4_native=np.array([0,0,0,1,0,0],bool))
            dest=root/'comparison'/arm['id']/pid;dest.mkdir(parents=True)
            archive=dest/'records.npz';np.savez_compressed(archive,**arrays)
            labels=dest/'labels.jsonl.gz';labels.write_bytes(b'synthetic already-classified labels; never read')
            record=dict(id=pid,arm=arm['id'],seed=seed,samples=6,sampler_cpu_seconds=2.,
                records=str(archive.relative_to(root/'comparison')),records_sha256=index.sha(archive),
                labels=str(labels.relative_to(root/'comparison')),labels_sha256=index.sha(labels),
                independent_audit_sha256=index.sha(audit_path),samples_sha256=index.sha(samples))
            records[arm['id']].append(record)
            full[arm['id']].append(dict(id=pid,seed=seed,masks=index.masks(arrays),**{k:arrays[k]for k in ('z','h','pairs')}))
            jobs.append(dict(id=pid,arm=arm['id'],seed=seed,samples=6,directory=str(directory),status='complete',returncode=0))
    protocol=dict(schema='hard-free-line-physical-pilot-v1',arms=arms,jobs=jobs,total_unconditional_draws=48)
    write(root/'protocol.json',protocol)
    data=dict(schema='hard-free-line-physical-comparison-v1',complete=True,protocol_sha256=index.sha(root/'protocol.json'),
        total_unconditional_draws=48,diagnostics=dict(full_vessel_gate_open=False,assembly_gate_open=False),arms={})
    for name in index.ARMS:
        estimates,_,_=index.summarize_regions(full[name]);data['arms'][name]=dict(populations=records[name],estimates=estimates)
    write(root/'comparison/analysis.json',data)
    state=dict(complete=True,phase='complete',jobs=jobs,audits=[dict(status='complete',returncode=0)for _ in range(8)],
        protocol_sha256=index.sha(root/'protocol.json'),comparison_sha256=index.sha(root/'comparison/analysis.json'))
    write(root/'status.json',state);return state


class RetainedIndexTests(unittest.TestCase):
    def test_primitive_labels_beta_controls_and_bad_fallback(self):
        for label,mass,expected in [('uniform',.5,2.),('success',.25,.5),('fallback',0.,2.)]:
            delta,_=index.index_log_factor(row(label,mass),guide());self.assertAlmostEqual(math.exp(delta),expected)
        r=row('success');r['hard_free_line_draw']=dict(conditional=False)
        r['hard_free_line_density']=dict(conditioning_disabled=True)
        self.assertEqual(index.index_log_factor(r,guide(0.))[0],0.)
        r=row('fallback',.3)
        with self.assertRaisesRegex(ValueError,'fallback mismatch'):index.index_log_factor(r,guide())
        delta,_=index.index_log_factor(row('success',1.+2e-16),guide())
        self.assertEqual(delta,math.log(.4)-math.log(.2)+math.log(1.+2e-16))
        r=row('uniform');r['hard_free_line_draw']['conditional']=True
        with self.assertRaisesRegex(ValueError,'Uniform draw conditioned'):index.index_log_factor(r,guide())

    def test_exact_finite_state_marginal_and_rao_blackwell_identity(self):
        # Target lies in each successful conditional support. One component's
        # second axis is a fallback, exercising both kinds of primitive label.
        alpha=.5;weights=np.array([.3,.7]);u=np.full(4,.25)
        g=np.array([[.1,.2,.3,.4],[.3,.2,.4,.1]])
        supports=np.array([[[1,1,0,0],[1,1,0,0]],[[1,1,1,0],[1,1,1,1]]],bool)
        masses=np.sum(g[None,:,:]*supports,axis=2);h=g[None,:,:]*supports/masses[:,:,None]
        old=alpha*u+(1-alpha)*np.sum(weights[:,None]*g,axis=0)
        q=alpha*u+(1-alpha)*np.mean(np.sum(weights[None,:,None]*h,axis=1),axis=0)
        target=np.array([2.,3.,0.,0.]);mean=0.;m2=0.;full2=0.
        for x in (0,1):
            outcomes=[(alpha*u[x],target[x]/old[x])]
            for j in range(2):
                for k in range(2):outcomes.append(((1-alpha)*weights[k]/2*h[j,k,x],target[x]*masses[j,k]/old[x]))
            self.assertAlmostEqual(sum(prob*w for prob,w in outcomes)/q[x],target[x]/q[x])
            mean+=sum(prob*w for prob,w in outcomes);m2+=sum(prob*w*w for prob,w in outcomes)
            full2+=target[x]**2/q[x]
        self.assertAlmostEqual(mean,target.sum());self.assertGreaterEqual(m2,full2)
        # A different target outside the intersection invalidates old
        # responsibilities: this is why the unchanged vessel is forbidden.
        x=3;mass=alpha*u[x]/old[x]
        for j in range(2):
            for k in range(2):mass+=(1-alpha)*weights[k]/2*h[j,k,x]*masses[j,k]/old[x]
        self.assertLess(mass,1.)

    def test_paired_difference_uses_covariance_and_preserves_zero_denominator(self):
        a=np.log([1.,2.,3.]);same=index.paired_difference(a,a)
        self.assertEqual(same['scaled_difference_SE'],0.)
        a=np.r_[a,-np.inf];b=np.r_[np.log([2.,2.,2.]),-np.inf]
        d=index.paired_difference(a,b)
        self.assertEqual(d['draws'],4);self.assertAlmostEqual(d['scaled_difference'],0.)
        self.assertAlmostEqual(d['scaled_difference_SE']*math.exp(d['log_scale']),np.std([1.,0.,-1.,0.],ddof=1)/2)
        self.assertFalse(index.paired_difference([-np.inf]*4,[-np.inf]*4)['observed'])

    def test_completion_gate_precedes_any_result_access(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'status.json',dict(complete=False,phase='independent_audit'))
            with self.assertRaisesRegex(ValueError,'Wait for all'):index.run(root,root/'display')
            self.assertFalse((root/'display').exists())

    def test_completed_archive_preserves_pairs_denominators_and_control(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);completed_fixture(root);value=index.run(root,root/'retrospective')
            self.assertEqual(value['total_unconditional_draws'],48)
            self.assertFalse(value['actual_index_sampler_run']);self.assertEqual(value['new_geometry_queries'],0)
            for arm,data in value['arms'].items():
                self.assertEqual(data['estimates']['index']['total']['row_uncertainty']['draws'],24)
                self.assertEqual(data['estimates']['index']['total']['row_uncertainty']['nonzero'],16)
                self.assertIsNone(data['runtime']['index_only_sampler_cpu_seconds'])
                for mode in ('full','index'):
                    for p in data['estimates'][mode]['total']['populations']:
                        self.assertEqual(p['observed_importance_ESS_per_actual_full_sampler_CPU'],p['Qz_ESS']/2.)
                if arm=='baseline':
                    self.assertEqual(data['estimates']['index'],data['estimates']['full'])
                else:
                    self.assertEqual(data['runtime']['full_density_geometry_requests'],72)
                    self.assertEqual(data['runtime']['selected_draw_geometry_requests'],16)
                    self.assertNotEqual(data['estimates']['index']['total']['row_uncertainty']['log_Qz'],
                                        data['estimates']['full']['total']['row_uncertainty']['log_Qz'])
            with self.assertRaisesRegex(ValueError,'Fresh retrospective'):index.run(root,root/'retrospective')

    def test_changed_raw_trace_and_classification_archive_are_rejected(self):
        for relative in ('conditioned/r00/samples.jsonl','comparison/conditioned/r00/records.npz'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as d:
                root=Path(d);completed_fixture(root);p=root/relative
                with p.open('ab')as stream:stream.write(b'changed')
                with self.assertRaisesRegex(ValueError,'Hash changed'):index.run(root,root/'retrospective')
                self.assertFalse((root/'retrospective').exists())


if __name__=='__main__':unittest.main()
