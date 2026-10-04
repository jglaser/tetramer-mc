"""Synthetic continuation/adoption contracts; no scientific row is evaluated."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_native_class_support_recovery as recovery


def opaque_population(root):
    root.mkdir()
    for name in recovery.POPULATION_FILES:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'' if name == 'probes.jsonl' else ('opaque, deliberately non-JSON bytes: '+name).encode())
    return {name: dict(sha256=recovery.sha(root/name), bytes=(root/name).stat().st_size)
            for name in recovery.POPULATION_FILES}


def protocol_fixture(root):
    old = root/'original'; old.mkdir(); producer = old/'binary'; producer.write_text('never executed')
    reference = dict(path=str(producer), sha256=recovery.sha(producer))
    populations = [dict(id=f'r{i:02}', seed=100+i, audit_seed=200+i, samples=128,
        selected_ids=list(range(16)), preselection=dict(sentinel=i), directory=str(old/'queries'/f'r{i:02}')) for i in range(4)]
    return dict(schema=recovery.base.SCHEMA, root=str(old), code_directory=str(old/'code'),
        files={str(producer): recovery.sha(producer)}, source_sha256={}, python=sys.executable,
        producer=reference, config=dict(reference), guide=dict(reference), region=dict(reference),
        compiled_native=dict(reference), phase_limits=copy.deepcopy(recovery.base.PHASE_LIMITS),
        populations=populations, total_attempts=512, selected_geometry_rows=64,
        runtime=dict(sentinel='unchanged'), component_inventory=['unchanged'],
        new_Poisson_clouds=0, launched=False, launch_review_complete=False)


class RecoveryTests(unittest.TestCase):
    def test_continuation_keeps_all_seeds_ids_and_common_bindings_and_only384_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); original = protocol_fixture(root); saved = copy.deepcopy(original)
            out = root/'recovery'
            protocol = recovery.continuation_protocol(original, out, original['files'], {}, {'retained_attempts':128}, {})
            self.assertEqual(original, saved)
            for field in ('producer','config','guide','region','compiled_native','phase_limits','component_inventory','runtime'):
                self.assertEqual(protocol[field], original[field])
            for before, after in zip(original['populations'], protocol['populations']):
                self.assertEqual({k:v for k,v in before.items() if k != 'directory'},
                                 {k:v for k,v in after.items() if k != 'directory'})
            plan = recovery.execution_plan(protocol, dict(path=str(out/'protocol.json'),sha256='0'*64))
            self.assertEqual(len(plan['jobs']),17)
            self.assertIn('--adopt',plan['jobs'][0]['argv']); self.assertNotIn('--seed',plan['jobs'][0]['argv'])
            producers = [j for j in plan['jobs'] if j['phase']=='producer' and '--adopt' not in j['argv']]
            self.assertEqual([int(j['argv'][j['argv'].index('--seed')+1]) for j in producers],[101,102,103])
            self.assertEqual(sum(int(j['argv'][j['argv'].index('--samples')+1]) for j in producers),384)
            self.assertEqual(sum(j['cpu_limit_seconds'] for j in plan['jobs']),12300)
            self.assertEqual(plan['maximum_workers'],1)

    def test_adoption_copies_opaque_bytes_and_publishes_original_summary_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; inventory=opaque_population(source); destination=root/'new'
            events=[]; recovery.copy_population(source,destination,inventory,events.append)
            self.assertEqual(events[-1]['member'],'summary.json')
            for name in recovery.POPULATION_FILES:
                self.assertEqual((source/name).read_bytes(),(destination/name).read_bytes())
                self.assertEqual(recovery.sha(source/name),inventory[name]['sha256'])
            with self.assertRaisesRegex(ValueError,'fresh population'):
                recovery.copy_population(source,destination,inventory,events.append)

    def test_population_inventory_joins_stream_hashes_without_parsing_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source'; inventory=opaque_population(root)
            manifest=dict(schema=recovery.base.PRODUCER_SCHEMA,samples=128,physical_jobs=0,probes_sha256=None)
            for name,key in [('config','config'),('region','region'),('importance-guide','guide'),('shape','shape'),
                             ('compiled-native','compiled_native'),('source-bundle','source_bundle')]:
                manifest[key+'_sha256']=inventory['provenance/'+name+'.json']['sha256']
            summary=dict(manifest=manifest,complete=True,samples=128,probes=0,
                         **{name+'_sha256':inventory[name+'.jsonl']['sha256'] for name in ('samples','attempts','probes')})
            result=recovery.population_inventory(root,summary,manifest,recovery.base.Bindings())
            self.assertEqual(result,inventory)
            summary['samples_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'Changed input'):
                recovery.population_inventory(root,summary,manifest,recovery.base.Bindings())

    def test_original_driver_preserves_its_archived_self_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'code').mkdir()
            driver=root/'code/run_native_class_physical_campaign.py'
            driver.write_text('def completed_terminal(*args):\n    return __file__\n')
            recovery.write(root/'execution-plan.json',dict(files={str(driver):recovery.sha(driver)}))
            with patch.object(recovery,'ORIGINAL_PLAN',recovery.sha(root/'execution-plan.json')):
                module,_=recovery.original_driver(root,recovery.base.Bindings())
            self.assertEqual(module.completed_terminal(),str(driver))

    def test_materialization_freezes_archive_instead_of_working_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); original=protocol_fixture(root); original['storage_reserve_bytes']=1
            working=root/'working';working.mkdir()
            source={name:working/name for name in ('audit_native_class_support_pilot.py','prepare_native_class_support_recovery.py')}
            for name,path in source.items():path.write_text('# synthetic '+name+'\n')
            hashes={str(path):recovery.sha(path) for path in source.values()}
            validation=root/'validation.json'
            recovery.write(validation,dict(complete=True,passed=True,returncode=0,source_before=hashes,source_after=hashes))
            out=root/'prepared'
            with patch.object(recovery,'original_evidence',return_value=(original,dict(retained_attempts=128))), \
                 patch.object(recovery,'source_paths',return_value=source), \
                 patch.object(recovery,'CORRECTED_WORKER',hashes[str(source['audit_native_class_support_pilot.py'])]), \
                 patch.object(recovery.base,'runtime_identity',return_value=original['runtime']):
                result=recovery.prepare(root/'original',out,validation)
            self.assertTrue(result['complete']);self.assertFalse(result['launched'])
            protocol=recovery.read(out/'protocol.json');files=protocol['files']
            for name,path in source.items():
                self.assertNotIn(str(path),files)
                self.assertEqual(files[str(out/'code'/name)],hashes[str(path)])
            self.assertEqual(files[original['producer']['path']],original['producer']['sha256'])

    def test_failed_adoption_preserves_journal_and_partial_copy_without_new_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source=root/'source';inventory=opaque_population(source)
            (root/'analysis/r00').mkdir(parents=True);(root/'queries').mkdir()
            inventory['samples.jsonl']['sha256']='0'*64
            proof=dict(original_root='unused',source_directory=str(source),retained_files=inventory,
                       retained_summary=dict(path=str(source/'summary.json'),sha256=inventory['summary.json']['sha256']))
            protocol=dict(root=str(root),recovery=proof,populations=[dict(directory=str(root/'queries/r00'))])
            with patch.object(recovery.worker,'load_protocol',return_value=(protocol,{})), \
                 patch.object(recovery,'verify_live_authority'), \
                 patch.object(recovery,'original_evidence',return_value=({},proof)):
                with self.assertRaisesRegex(ValueError,'Original retained file changed'):
                    recovery.adopt(root/'protocol.json','digest')
            failure=recovery.read(root/'analysis/r00/adoption.failure.json')
            self.assertFalse(failure['complete']);self.assertEqual(failure['new_draws'],0)
            self.assertTrue((root/'analysis/r00/adoption.journal.jsonl').exists())
            self.assertTrue((root/'queries/r00/manifest.json').exists())
            self.assertFalse((root/'queries/r00/summary.json').exists())
            self.assertTrue((source/'samples.jsonl').exists())


if __name__=='__main__': unittest.main()
