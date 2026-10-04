"""Synthetic scalar summaries/lifecycle metadata only; no protein inputs."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import plot_surrogate_internal_native as p


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(path),sha256=sha(path))


def synthetic_result():
    chains=[]
    for ident,(arm,start,stream) in enumerate(itertools.product(p.ARMS,p.STARTS,range(4))):
        cpu=100.+10*p.ARMS.index(arm)+stream+(start=='proposal_prepared')
        frames=((1,2048,1024,3072) if start=='source' else (0,4096,2048,1024))[stream]
        fraction=frames/4096; keys=[[27,132,6]]+([[27,132,7]] if stream==3 else [])
        values={'empty':[],'native':keys}
        occupancy={k:v for k,v in (('empty',1-fraction),('native',fraction)) if v>0}
        active=0<frames<4096; multiple=active and stream==2
        events=[]
        if active:
            events.append(dict(environment='empty',departure_block=513,return_block=700,
                returned_environment_nonempty=False,passed_through_empty=False))
        if multiple:
            events.extend([dict(environment='native',departure_block=700,return_block=900,
                returned_environment_nonempty=True,passed_through_empty=True),
                dict(environment='empty',departure_block=900,return_block=1400,
                returned_environment_nonempty=False,passed_through_empty=False)])
        count=2 if multiple else 1 if active else 0
        counts=dict(enter_nonempty=count,leave_nonempty=count,direct_nonempty_changes=0,
                    completed_returns=len(events),nonempty_returns=int(multiple),completed_passages=2*count)
        value=(1024. if frames==1 else 16.+stream) if active else None
        ess=dict(samples=4096,sampling_CPU_seconds=cpu,apparent_ess=value,
            apparent_ess_per_sampling_CPU_second=None if value is None else value/cpu)
        if value is None:ess['reason']='Constant synthetic descriptor'
        descriptor=dict(environments=dict(occupancy=occupancy,completed_returns=events),environment_values=values,
            marginal_occupancy=[dict(key=k,fraction=fraction) for k in keys] if frames else [],presence_ess=ess,
            counts=counts,rates_per_full_sampler_cpu_second={k:v/cpu for k,v in counts.items()})
        chains.append(dict(job=dict(id=ident,context_index=0,arm=arm,initialization=start,stream=stream),
            reused_native_control=arm in p.CACHED,trajectory=dict(path=f'/must-not-read/{ident}.jsonl',sha256='a'*64),
            terminal=dict(path=f'/must-not-read/{ident}-terminal.json',sha256='b'*64),
            metrics=dict(schema='internal-native-metrics-v1',production_samples=4096,retained_endpoints=4609,
                full_sampler_cpu_seconds=cpu,production_window_cpu_seconds=.8*cpu,
                internal_native_attachment_fraction=fraction,internal_motifs=descriptor)))
    return dict(schema=p.observer.SCHEMA,complete=True,passed=True,allocation=p.observer.ALLOCATION,
        chains=chains,external_native_observer=False,cycle_observer=False,assembly_gate_open=False)


def fixture(root, mutate_result=None):
    root.mkdir()
    def save(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False)+'\n');return path
    source=root/'source/fake.py';source.parent.mkdir();source.write_text('# synthetic observer source\n')
    result=synthetic_result();result['source_sha256']={'fake.py':sha(source)}
    config=save('config.json',dict(physical=dict(depletant_radius=1.4,activity=.0275),contexts=[dict(root=27,child=132)]))
    declarations=[dict(id=str(c['job']['id']),job=c['job'],trajectory=c['trajectory'],terminal=c['terminal'])
                  for c in result['chains'] if not c['reused_native_control']]
    inputs={str(config):sha(config)}
    for c in declarations:
        for k in ('trajectory','terminal'):inputs[c[k]['path']]=c[k]['sha256']
    audit=save('audit-plan.json',dict(schema=p.observer.PLAN_SCHEMA,root=str(root),output=str(root/'analysis'),
        allocation=p.observer.ALLOCATION,source_sha256=result['source_sha256'],input_sha256=inputs,
        chains=declarations,config=ref(config)))
    result.update(plan_sha256=sha(audit),input_sha256=copy.deepcopy(inputs))
    if mutate_result:mutate_result(result)
    terminal=save('analysis/summary.json',result)
    protocol=save('protocol.json',dict(schema='surrogate-internal-native-dispatch-v1',root=str(root),
        allocation=p.observer.ALLOCATION,audit_plan=ref(audit),source_sha256=result['source_sha256'],
        new_jobs=[c['job'] for c in declarations],cached_control_jobs=[c['job'] for c in result['chains'] if c['reused_native_control']]))
    job=dict(id='whole-internal-native-analysis',population='context0-48-new-16-cached',phase='geometry',argv=['synthetic-observer'],
        terminal=dict(path=str(terminal),success_contract='complete_and_passed'))
    plan=save('execution-plan.json',dict(schema=p.observer.driver.SCHEMA,root=str(root),maximum_workers=1,threads=1,
        jobs=[job],files={**inputs,str(audit):sha(audit),str(protocol):sha(protocol),str(source):sha(source)}))
    prepared=save('preparation.json',dict(schema='surrogate-internal-native-preparation-v1',complete=True,launched=False,
        new_chains=48,cached_chains=16,execution_plan=ref(plan),protocol=ref(protocol),audit_plan=ref(audit)))
    claim=save('execution/claim.json',dict(schema=p.observer.driver.SCHEMA,plan_sha256=sha(plan),maximum_workers=1,threads=1,
        retries=0,replacements=0,preparation_receipt=ref(prepared)))
    done=dict(id=job['id'],population=job['population'],phase=job['phase'],argv=job['argv'],terminal=ref(terminal),
        success_contract='complete_and_passed',success=True,child_started=True,child_drained=True,returncode=0,
        error=None,timeout=False,retries=0,replacements=0,pid=123,birth_ticks=456)
    status=dict(complete=True,passed=True,failure=None,active=None,unstarted=[],completed=[done],plan_sha256=sha(plan))
    save('execution/status.json',status);save('execution/summary.json',status)
    directory='execution/jobs/000-whole-internal-native-analysis/'
    save(directory+'success.json',done);save(directory+'exit.json',done);save(directory+'attempt.json',dict(job=job))
    save(directory+'process.json',{k:done[k] for k in ('id','pid','birth_ticks','argv')})
    return result


class InternalNativePlotTests(unittest.TestCase):
    def test_all64_cached16_rare_frames_and_null_ess_preserved(self):
        data=synthetic_result();before=copy.deepcopy(data);rows=p.saved_values(data)
        self.assertEqual(len(rows),64);self.assertEqual(sum(r['reused_native_control'] for r in rows),16)
        rare=[r for r in rows if r['job']['stream']==0 and r['job']['initialization']=='source']
        self.assertTrue(all(r['native_frames']==1 and r['presence_ess']==1024. and r['nonempty_returns']==0 for r in rare))
        empty=[r for r in rows if r['job']['stream']==0 and r['job']['initialization']=='proposal_prepared']
        self.assertTrue(all(r['presence_ess'] is None and r['presence_ess_per_cpu'] is None and r['native_frames']==0 for r in empty))
        constant=[r for r in rows if r['job']['stream']==1 and r['job']['initialization']=='proposal_prepared']
        self.assertTrue(all(r['native_frames']==4096 and r['presence_ess'] is None for r in constant))
        self.assertEqual(data,before)

    def test_inventory_cpu_residence_marginal_and_return_mutations_fail(self):
        mutations=(lambda d:d['chains'].pop(),lambda d:d['chains'].__setitem__(1,copy.deepcopy(d['chains'][0])),
            lambda d:d['chains'][0].update(reused_native_control=False),
            lambda d:d['chains'][0]['metrics'].update(production_samples=4095),
            lambda d:d['chains'][0]['metrics'].update(full_sampler_cpu_seconds=0),
            lambda d:d['chains'][0]['metrics']['internal_motifs']['presence_ess'].update(apparent_ess=None),
            lambda d:d['chains'][0]['metrics']['internal_motifs']['presence_ess'].update(sampling_CPU_seconds=99.),
            lambda d:d['chains'][0]['metrics']['internal_motifs']['marginal_occupancy'][0].update(fraction=.5),
            lambda d:d['chains'][0]['metrics']['internal_motifs']['environment_values']['native'][0].__setitem__(2,14),
            lambda d:d['chains'][0]['metrics']['internal_motifs']['counts'].update(nonempty_returns=9),
            lambda d:d['chains'][0]['metrics']['internal_motifs']['rates_per_full_sampler_cpu_second'].update(completed_returns=99.))
        for mutate in mutations:
            d=synthetic_result();mutate(d)
            with self.assertRaises(ValueError):p.saved_values(d)

    def test_same_stream32_contrasts_have_signed_occupancy_and_multilabel_marginals(self):
        rows=p.saved_values(synthetic_result());comparisons=p.initialization_contrasts(rows)
        self.assertEqual(len(comparisons),32)
        for c in comparisons:
            if c['stream']==0:self.assertEqual(c['prepared_minus_source_native_fraction'],-1/4096)
            if c['stream']==1:self.assertEqual(c['prepared_minus_source_native_fraction'],.5)
        multi=[r for r in rows if r['job']['stream']==3]
        self.assertTrue(all(sum(r['motif_fractions'])==2*r['native_fraction'] for r in multi))
        with self.assertRaises(ValueError):p.initialization_contrasts(rows[:-1])

    def test_complete_metadata_source_terminal_and_input_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';expected=fixture(root);actual,inputs=p.authenticate(root)
            self.assertEqual(actual,expected);self.assertTrue(all(not name.startswith('/must-not-read/') for name in inputs))
            (root/'source/fake.py').write_text('# changed source\n')
            with self.assertRaises(ValueError):p.authenticate(root)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';fixture(root)
            with (root/'analysis/summary.json').open('a') as f:f.write(' ')
            with self.assertRaises(ValueError):p.authenticate(root)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';fixture(root,lambda d:d['input_sha256'].update({'/unbound/raw.jsonl':'c'*64}))
            with self.assertRaisesRegex(ValueError,'frozen audit declaration'):p.authenticate(root)

    def test_incomplete_gate_precedes_scientific_summary_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';fixture(root)
            for name in ('status','summary'):
                path=root/f'execution/{name}.json';v=json.loads(path.read_text());v.update(complete=False,passed=False,active={'id':'running'})
                path.write_text(json.dumps(v))
            original=Path.open
            def guarded(path,*args,**kwargs):
                if path==root/'analysis/summary.json':raise AssertionError('Read incomplete science')
                return original(path,*args,**kwargs)
            with mock.patch.object(Path,'open',guarded),self.assertRaisesRegex(ValueError,'incomplete, failed or undrained'):
                p.authenticate(root)

    def test_complete_synthetic_render_keeps_nulls_and_never_opens_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';fixture(root);out=Path(tmp)/'plots';original=Path.open
            def guarded(path,*args,**kwargs):
                if path.suffix=='.jsonl' or str(path).startswith('/must-not-read/'):
                    raise AssertionError('Raw input was opened')
                return original(path,*args,**kwargs)
            with mock.patch.object(Path,'open',guarded):receipt=p.plot(root,out)
            self.assertTrue(receipt['passed']);self.assertEqual(receipt['chains'],64)
            self.assertEqual(receipt['scientific_journals_read'],0);self.assertFalse(receipt['streams_or_starts_pooled'])
            for name in ('internal-native-efficiency','internal-native-motifs','initialization-differences'):
                self.assertGreater((out/f'{name}.png').stat().st_size,1000)
                self.assertGreater((out/f'{name}.svg').stat().st_size,1000)
            self.assertIn('undefined',(out/'internal-native-efficiency.svg').read_text())
            d=json.loads((out/'plotted-values.json').read_text());self.assertEqual(len(d['rows']),64)
            self.assertEqual(len(d['initialization_contrasts']),32)
            self.assertTrue(any(r['presence_ess_per_cpu'] is None for r in d['rows']))
            for name,r in receipt['outputs'].items():self.assertEqual(sha(out/name),r['sha256'])
            with self.assertRaisesRegex(ValueError,'Fresh plot output required'):p.plot(root,out)


if __name__=='__main__':unittest.main()
