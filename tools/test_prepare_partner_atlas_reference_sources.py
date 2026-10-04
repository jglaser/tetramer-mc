"""Synthetic source-cache controls only; never scan archived physical journals."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import prepare_partner_atlas_reference_sources as p


def metadata():
    expected = {name.removeprefix('reference/'):digest for name,digest in p.PINS.items() if name.startswith('reference/')}
    expected.update({'kernel-attempts.jsonl':p.JOURNAL_SHA,'source-attempts.jsonl':p.SOURCE_JOURNAL_SHA})
    source = {'tests/flexible_surrogate_stationarity.rs':p.REFERENCE_SOURCE_SHA}
    validation = dict(schema='flexible-surrogate-reference-validation-v1',complete=True,passed=True,
        child_drained=True,returncode=0,error=None,source_before=source,source_after=copy.deepcopy(source),
        production_before='protected',production_after='protected',outputs={'reference/'+k:v for k,v in expected.items()})
    counts = [dict(attempts=4096+i,void_accepted=2048) for i in range(4)]
    receipt = dict(schema='flexible-surrogate-independent-reference-receipt-v1',complete=True,passed=True,
        numerical_allocation_complete=True,scientific_checks_passed=True,error=None,retries=0,replacement_draws=0,
        output_sha256={k:v for k,v in expected.items() if k != 'receipt.json'},source_counts=counts)
    audit_validation = dict(complete=True,passed=True,child_drained=True,source_unchanged=True,
        returncode=0,error=None,audit_sha256=p.PINS['journal-audit.json'])
    audit = dict(schema='flexible-surrogate-reference-journal-audit-v1',complete=True,passed=True,
        reference_scientific_checks_passed=True,validation_sha256=p.PINS['validation.json'],
        input_sha256=expected,inventory={arm:8192 for arm in p.OLD_ARMS},source_counts=counts)
    protocol = dict(schema='flexible-surrogate-independent-reference-protocol-v1',streams=4,sources_per_stream=2048,
        arms=[[1,1.,'m1_guided'],[8,1.,'m8_guided'],[8,0.,'m8_zero']],source_sampler_exact=True,
        reference_source_sha256=p.REFERENCE_SOURCE_SHA,observable_names=p.OBSERVABLES,**copy.deepcopy(p.PHYSICAL))
    summary = dict(schema='flexible-surrogate-independent-reference-summary-v1',complete=True,
        observable_names=p.OBSERVABLES,source_counts=counts,
        stream_panel=[dict(stream=i,source_counts=c) for i,c in enumerate(counts)])
    return [validation,receipt,audit_validation,audit,protocol,summary]


def journal_rows(streams=2,per_stream=3):
    rows = []
    for stream in range(streams):
        for index in range(per_stream):
            pair = [dict(position=[.25+stream*.1+index*.01,.1,.1],orientation=[1.,0.,0.,0.]),
                    dict(position=[-.4,.2,-.2],orientation=[0.,1.,0.,0.])]
            for arm in p.OLD_ARMS:
                identity = dict(stream=stream,source_index=index,source_attempt=(index+1)*2,arm=arm)
                rows.append(dict(kind='kernel_begin',old=copy.deepcopy(pair),**identity))
                # These intentionally varied old outcomes must not select poses.
                rows.append(dict(kind='kernel_outcome',record=dict(accepted=index % 2 == 0),
                    retained=None,negative_controls=None,**identity))
    return rows


def save_rows(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); return p.sha(path)


class PartnerAtlasSourceTests(unittest.TestCase):
    def test_completed_authority_physical_target_and_38_observables(self):
        original = metadata(); p.validate_metadata(*original)
        self.assertEqual(len(p.OBSERVABLES),38)
        self.assertEqual(p.PROSPECTIVE['kernel_calls'],4*8192)
        self.assertEqual(p.PROSPECTIVE['fixed_candidates'],8192*(1+1+8+8))
        for record,key,value in [(0,'child_drained',False),(1,'passed',False),(2,'source_unchanged',False),
                                 (3,'input_sha256',{}),(4,'rd',1.5),(4,'wall_radius',3.)]:
            damaged = copy.deepcopy(original); damaged[record][key] = value
            with self.subTest(record=record,key=key),self.assertRaises(ValueError): p.validate_metadata(*damaged)
        damaged = copy.deepcopy(original); damaged[4]['observable_names'][0] = 'changed'
        with self.assertRaises(ValueError): p.validate_metadata(*damaged)
        damaged = copy.deepcopy(original); damaged[1]['source_counts'][0]['void_accepted'] = 2047
        with self.assertRaises(ValueError): p.validate_metadata(*damaged)

    def test_all_sources_preserved_independently_of_old_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sources = journal_rows(); fingerprints = []
            for variant in range(2):
                rows = copy.deepcopy(sources)
                for row in rows:
                    if row['kind'] == 'kernel_outcome': row['record']['accepted'] = bool(variant)
                journal = root/f'journal{variant}.jsonl'; output = root/f'sources{variant}.jsonl'
                digest = save_rows(journal,rows)
                result = p.extract_journal(journal,output,digest,2,3,[6,6])
                selected = [p.strict_json(line) for line in output.read_bytes().splitlines()]
                self.assertEqual(result['total_sources'],6); self.assertEqual(result['scanned_rows'],36)
                self.assertEqual(result['source_point_journal_scans'],0); self.assertEqual(result['geometry_queries'],0)
                self.assertEqual([(r['stream'],r['source_index']) for r in selected],[(s,i) for s in range(2) for i in range(3)])
                self.assertEqual([r['old'] for r in selected],[r['old'] for r in rows if r['kind'] == 'kernel_begin' and r['arm'] == 'm1_guided'])
                self.assertTrue(all(set(r) == {'kind','stream','source_index','source_attempt','old'} for r in selected))
                self.assertEqual(p.sha(output),result['sources_sha256']); fingerprints.append(result['sources_sha256'])
            self.assertEqual(*fingerprints)

    def test_missing_duplicate_reordered_and_cross_arm_sources_fail(self):
        base = journal_rows(); variants = {}
        variants['missing'] = base[:-2]
        duplicate = copy.deepcopy(base); duplicate[6:12] = copy.deepcopy(duplicate[:6]); variants['duplicate'] = duplicate
        reordered = copy.deepcopy(base); reordered[:2],reordered[2:4] = reordered[2:4],reordered[:2]; variants['reordered'] = reordered
        changed = copy.deepcopy(base); changed[2]['old'][0]['position'][0] += .01; variants['cross-arm'] = changed
        extra = copy.deepcopy(base)+copy.deepcopy(base[:2]); variants['extra'] = extra
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name,rows in variants.items():
                journal = root/(name+'.jsonl'); output = root/(name+'-sources.jsonl'); digest = save_rows(journal,rows)
                with self.subTest(name=name),self.assertRaises(ValueError): p.extract_journal(journal,output,digest,2,3,[6,6])
                self.assertTrue(output.exists())  # preserve every begun extraction

    def test_hash_endpoints_partial_tail_and_pose_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = journal_rows(1,1); journal = root/'journal.jsonl'
            digest = save_rows(journal,rows)
            with self.assertRaisesRegex(ValueError,'SHA'):
                p.extract_journal(journal,root/'hash.jsonl','0'*64,1,1,[2])
            with self.assertRaisesRegex(ValueError,'Final source attempts'):
                p.extract_journal(journal,root/'endpoint.jsonl',digest,1,1,[3])
            journal.write_bytes(journal.read_bytes().rstrip(b'\n'))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                p.extract_journal(journal,root/'tail.jsonl',p.sha(journal),1,1,[2])
            rows[0]['old'][0]['orientation'] = [0.,0.,0.,0.]; digest = save_rows(journal,rows)
            with self.assertRaisesRegex(ValueError,'quaternion'):
                p.extract_journal(journal,root/'pose.jsonl',digest,1,1,[2])
            rows = journal_rows(1,1); rows[0]['old'][0]['position'][0] = float('inf')
            digest = save_rows(journal,rows)
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                p.extract_journal(journal,root/'nonfinite.jsonl',digest,1,1,[2])

    def test_strict_json_and_bad_authority_never_open_journal(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError): p.strict_json(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root/'reference'; reference.mkdir()
            (reference/'validation.json').write_text('{}\n')
            with mock.patch.object(p,'extract_journal') as extraction:
                with self.assertRaisesRegex(ValueError,'Unauthenticated validation'):
                    p.prepare(root/'output',reference)
                extraction.assert_not_called(); self.assertFalse((root/'output').exists())

    def test_outputs_are_exclusive_and_failed_digest_leaves_partial_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); journal = root/'journal.jsonl'; output = root/'sources.jsonl'
            digest = save_rows(journal,journal_rows(1,1))
            with self.assertRaisesRegex(ValueError,'SHA'): p.extract_journal(journal,output,'0'*64,1,1,[2])
            partial = output.read_bytes(); self.assertTrue(partial)
            with self.assertRaises(FileExistsError): p.extract_journal(journal,output,digest,1,1,[2])
            self.assertEqual(output.read_bytes(),partial)


if __name__ == '__main__': unittest.main()
