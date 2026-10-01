#!/usr/bin/env python3
"""Bookkeeping tests for the passive native observer; no real pose replay."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze_contact_distance_native as observer


def row(index,kind='unbound',hard=True,shell=True,capture=True,branch='original_gaussian'):
    draw=dict(component=None if branch=='uniform' else 0,conditional=branch.startswith('conditioned'),
              fallback=branch=='conditioned_fallback')
    return dict(id=index,kind='fresh',pose=dict(kind=kind),hard_valid=hard,shell_valid=shell,capture_valid=capture,
                draw=draw,width_contacts=[[True,True],[True,True],[True,True]])


class Native:
    def __init__(self):self.calls=[]
    def classify(self,pose):
        self.calls.append(pose);native=pose['kind'].startswith('native');triangle=pose['kind']=='native_triangle'
        return dict(native_any=native,native_anchor_count=2 if triangle else int(native),
                    cooperative_entry=triangle,registry_consistent_triangle=triangle)


class Contact:
    def __init__(self):self.calls=[]
    def classify(self,pose):
        self.calls.append(pose)
        return dict(exclusion_contact=pose['kind'] in ('native','native_triangle','competing'))


class NativeObserverTests(unittest.TestCase):
    def test_all_attempts_partition_and_invalid_calls_skipped(self):
        rows=[row(0,hard=False,branch='uniform'),row(1,shell=False,branch='conditioned_success'),
              row(2,'native',branch='original_gaussian'),row(3,'competing',branch='conditioned_fallback'),
              row(4,'unbound'),row(5,'native_triangle',branch='conditioned_success')]
        native,contact,stream=Native(),Contact(),io.StringIO()
        result=observer.classify_rows(rows,native,contact,stream,dict(arm='test',population='r00',seed=1))
        saved=[json.loads(s) for s in stream.getvalue().splitlines()]
        self.assertEqual(result['attempted'],6);self.assertEqual(result['valid'],4)
        self.assertEqual(len(native.calls),4);self.assertEqual(len(contact.calls),4)
        self.assertEqual(len(saved),6);self.assertEqual([s['attempt_index'] for s in saved],list(range(6)))
        self.assertIsNone(saved[0]['classification']);self.assertIsNone(saved[1]['contact'])
        self.assertEqual(result['partition'],dict(zip(observer.DISPOSITIONS,[1,1,2,1,1])))
        self.assertEqual(result['native_both_anchors'],1);self.assertEqual(result['registry_triangle'],1)
        self.assertEqual(result['branches']['conditioned_success']['attempts'],2)
        self.assertEqual(sum(result['partition'].values()),6)
        self.assertEqual(sum(result['joint_contact_cross_counts'][0].values()),4)

    def test_native_without_contact_is_preserved_as_anomaly(self):
        result=observer.classify_rows([row(0,'native_no_contact')],Native(),Contact(),io.StringIO(),{})
        self.assertEqual(result['native_entry'],1)
        self.assertEqual(result['native_without_exclusion_contact_attempts'],[0])
        self.assertEqual(result['partition']['valid_native_entry'],1)

    def test_outside_and_hard_invalid_flags_both_retained(self):
        stream=io.StringIO();n=Native();c=Contact()
        result=observer.classify_rows([row(0,hard=False,capture=False)],n,c,stream,{})
        saved=json.loads(stream.getvalue())
        self.assertFalse(saved['physical_flags']['capture_valid']);self.assertFalse(saved['physical_flags']['hard_valid'])
        self.assertEqual(result['partition']['outside_R4_or_capture'],1)
        self.assertFalse(n.calls or c.calls)

    def test_wrong_attempt_identity_or_flags_rejected(self):
        for bad in [dict(row(0),id=1),dict(row(0),id=False),dict(row(0),hard_valid=1),dict(row(0),kind='probe')]:
            with self.assertRaises(ValueError):observer.classify_rows([bad],Native(),Contact(),io.StringIO(),{})

    def test_incomplete_passive_run_cannot_start_observer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'observer';out.mkdir();passive=root/'passive';passive.mkdir()
            (passive/'status.json').write_text(json.dumps(dict(complete=False,phase='running')))
            (passive/'protocol.json').write_text('{}')
            p=dict(passive=str(passive),passive_protocol_sha256='test')
            with patch.object(observer,'validate',return_value=p):
                with self.assertRaisesRegex(ValueError,'must be complete'):observer.run(out,'test')
            self.assertFalse((out/'launch-claim.json').exists())


if __name__=='__main__':unittest.main()
