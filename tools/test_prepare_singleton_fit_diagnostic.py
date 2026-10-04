"""Synthetic metadata only; no binary execution, fitting or geometry."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import prepare_singleton_fit_diagnostic as prepare


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))
    return dict(path=str(path.resolve()),sha256=prepare.sha(path))


def original_fixture(root):
    ref=save(root/'input.json',{})
    states=[dict(id=name,kind='source' if name=='source' else 'prepared',file=ref,poses_pointer='/poses')
            for name in ['source']+[f'prepared-{i}' for i in range(4)]]
    return dict(schema='singleton-fusion-diagnostic-manifest-v1',context='whole_27_132',coordinate_frame='sphere_centered',
        wall_center=[0.,0.,0.],output=str(root/'old-output'),config=ref,model=ref,shape=ref,members=[27,132],anchor=228,
        states=states,constructions=[dict(id=f'{s["id"]}-{i}',state=s['id'],moving=i,neighbors=[j,228])
                                   for s in states for i,j in ((27,132),(132,27))],
        oligomer=prepare.OLIGOMER,caps=prepare.CAPS,
        witness=dict(executable_sha256='a'*64,compiled_source_bundle_sha256='b'*64,cli_source_sha256='c'*64))


def build_fixture(root):
    repo=root/'repository'; entries={}
    for name in prepare.HISTORICAL_SOURCE_EDITS:
        path=repo/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('// synthetic new fit source '+name)
        entries[name]=dict(text=path.read_text(),sha256=prepare.sha(path))
    bundle=save(root/'bundle.json',dict(schema=1,files=entries))
    binary=root/'singleton-fusion-diagnostic'; binary.write_text('synthetic binary; never run'); binary.chmod(0o755)
    validation=save(root/'build-validation.json',dict(complete=True,passed=True,compiled_source_bundle=bundle,
        sources={name:dict(path=str(repo/name),sha256=e['sha256']) for name,e in entries.items()}))
    witness=dict(schema='singleton-fusion-build-witness-v1',complete=True,passed=True,production_executable_unchanged=True,
        fit_policies=prepare.POLICIES,default_kernel_unchanged=True,per_fit_trace=True,
        executable=dict(path=str(binary),sha256=prepare.sha(binary)),source_bundle=bundle,
        cli_source=dict(path=str(repo/prepare.HISTORICAL_SOURCE_EDITS[1]),sha256=entries[prepare.HISTORICAL_SOURCE_EDITS[1]]['sha256']),
        validation=validation)
    return repo,witness,save(root/'witness.json',witness)


class FitPreparationTests(unittest.TestCase):
    def test_full_cartesian_manifests_change_only_declared_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); original=original_fixture(root); before=copy.deepcopy(original); _,witness,_=build_fixture(root)
            manifests=prepare.paired_manifests(original,witness,root/'new')
            self.assertEqual([(m['fit_policy'],m['anchor']) for m in manifests],
                             [(p,a) for p in prepare.POLICIES for a in prepare.ANCHORS])
            self.assertEqual(sum(len(m['constructions']) for m in manifests),100)
            for m in manifests:
                self.assertEqual([c['moving'] for c in m['constructions']],[27,132]*5)
                self.assertEqual(m['states'],original['states'])
                self.assertTrue(all(c['neighbors'][1]==m['anchor'] for c in m['constructions']))
                for key in ('config','shape','model','oligomer','caps','coordinate_frame','wall_center'):
                    self.assertEqual(m[key],original[key])
            self.assertEqual(original,before)

    def test_only_two_exact_old_compiled_source_paths_may_use_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/prepare.HISTORICAL_SOURCE_EDITS[0]; source.parent.mkdir(parents=True)
            source.write_text('old source'); old=prepare.sha(source)
            other=root/'shape.json'; other.write_text('immutable physical input')
            files={str(source):old,str(other):prepare.sha(other)}
            entries={prepare.HISTORICAL_SOURCE_EDITS[0]:dict(text='old source',sha256=old)}
            bundle=save(root/'bundle.json',dict(schema=1,files=entries)); source.write_text('new source')
            inputs=prepare.Inputs(); checked=prepare.bundle_entries(inputs,bundle)
            substitutions=prepare.historical_inputs(inputs,files,bundle,checked,repo=root)
            self.assertEqual(len(substitutions),1)
            self.assertEqual(substitutions[0]['historical_sha256'],old)
            self.assertNotIn(str(source),inputs.files)
            self.assertIn(str(other),inputs.files)
            bad=copy.deepcopy(entries); bad[prepare.HISTORICAL_SOURCE_EDITS[0]]['text']='wrong archive'
            with self.assertRaisesRegex(ValueError,'archive entry bytes'):
                prepare.historical_inputs(prepare.Inputs(),files,bundle,bad,repo=root)
            other.write_text('changed physical input')
            with self.assertRaisesRegex(ValueError,'outside exact compiled-source allowlist'):
                prepare.historical_inputs(prepare.Inputs(),files,bundle,entries,repo=root)

    def test_build_witness_requires_fit_capabilities_and_exact_current_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo,witness,ref=build_fixture(root)
            self.assertEqual(prepare.validate_build(prepare.Inputs(),ref,repo=repo),witness)
            for key in ('fit_policies','default_kernel_unchanged','per_fit_trace'):
                bad=copy.deepcopy(witness); bad.pop(key); badref=save(root/(key+'.json'),bad)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,'fit-capable'):
                    prepare.validate_build(prepare.Inputs(),badref,repo=repo)
            source=repo/prepare.HISTORICAL_SOURCE_EDITS[0]; source.write_text('changed after validation')
            with self.assertRaisesRegex(ValueError,'Changed bound input'):
                prepare.validate_build(prepare.Inputs(),ref,repo=repo)

    def test_preparer_receipt_rejects_omitted_or_stale_helper_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=root/'prepare.py'; b=root/'driver.py'; a.write_text('a'); b.write_text('b')
            sources={str(p):prepare.sha(p) for p in (a,b)}
            ref=save(root/'validation.json',dict(complete=True,passed=True,source_before=sources,source_after=sources))
            with mock.patch.object(prepare,'source_paths',return_value=[a,b]):
                prepare.validate_tests(prepare.Inputs(),ref)
                b.write_text('changed helper')
                with self.assertRaisesRegex(ValueError,'Changed bound input'): prepare.validate_tests(prepare.Inputs(),ref)
                missing=save(root/'missing.json',dict(complete=True,passed=True,source_before={},source_after={}))
                with self.assertRaisesRegex(ValueError,'Missing preparer validation source'): prepare.validate_tests(prepare.Inputs(),missing)

    def test_materialization_is_ten_serial_jobs_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); original=original_fixture(root); repo,witness,ref=build_fixture(root)
            inputs=prepare.Inputs(); prepare.validate_build(inputs,ref,repo=repo)
            inputs.bind(prepare.driver.__file__)
            validation=save(root/'preparer-validation.json',dict(complete=True,passed=True)); inputs.load(validation)
            baseline={str(a):dict(diagnostics={'source-27':dict(fit_attempts=0,fit_returned_none=0)}) for a in prepare.ANCHORS}
            context=dict(inputs=inputs,witness=witness,witness_ref=ref,original=original,baseline=baseline,
                         historical=[{'completed':'synthetic'}],validation=validation)
            out=root/'new'; out.mkdir(); receipt=prepare.materialize(out,context)
            self.assertFalse(receipt['launched']); self.assertTrue((out/'diagnostics').is_dir())
            self.assertEqual(list((out/'diagnostics').iterdir()),[])
            plan=prepare.read(out/'execution-plan.json'); protocol=prepare.read(out/'protocol.json')
            self.assertEqual(len(plan['jobs']),10); self.assertEqual(plan['maximum_workers'],1)
            self.assertEqual(protocol['aggregate_caps'],dict(constructions=100,fits=409600,center_checks=25600,core_overlap_calls=6732800))
            self.assertEqual(protocol['baseline_diagnostics'],baseline)
            self.assertTrue(protocol['default_equivalence_required'])
            self.assertTrue(all(j['cpu_limit_seconds']==600 and j['wall_limit_seconds']==1200
                                and j['address_space_limit_bytes']==8*2**30 for j in plan['jobs']))

    def test_saved_baseline_requires_complete_five_anchor_and_fifty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); original=original_fixture(root); results=[]
            for anchor in prepare.ANCHORS:
                m=copy.deepcopy(original); m['anchor']=anchor; m['context']=f'anchor_{anchor}'; m['output']=str(root/f'anchor-{anchor}')
                for c in m['constructions']: c['neighbors'][1]=anchor
                rows=[dict(state='complete',ordinal=i,construction=c,error=None,
                           diagnostics=dict(complete=True,fit_attempts=0,fit_returned_none=0)) for i,c in enumerate(m['constructions'])]
                journal=root/f'journal-{anchor}.jsonl'; journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                results.append(dict(manifest=m,manifest_ref=save(root/f'manifest-{anchor}.json',m),
                                    journal=dict(path=str(journal),sha256=prepare.sha(journal))))
            _,baseline=prepare.validate_originals(results)
            self.assertEqual(sum(len(x['diagnostics']) for x in baseline.values()),50)
            with self.assertRaisesRegex(ValueError,'five-anchor inventory'): prepare.validate_originals(results[:-1])
            path=Path(results[-1]['journal']['path']); lines=path.read_text().splitlines(); path.write_text('\n'.join(lines[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'Missing original default baseline'): prepare.validate_originals(results)


if __name__=='__main__': unittest.main()
