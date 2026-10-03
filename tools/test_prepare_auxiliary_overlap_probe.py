"""Immutable baseline selection and fixed contract tests, no scientific draws."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze_auxiliary_overlap_probe as audit
import prepare_auxiliary_overlap_probe as prep


def fixture(root):
    (root/'execution').mkdir();rows=[]
    for a in range(3):
        for c in range(8):
            for j in range(32):
                for m in ['whole_joint','factorized']:rows.append(json.dumps(dict(atlas_index=a,case_index=c,attempt=j,method=m,status='completed'))+'\n')
    (root/'execution/attempts.jsonl').write_text(''.join(rows))
    (root/'analysis.json').write_text(json.dumps(dict(complete=True,passed=True,failures=[],summary=dict(outer_attempts=1536))))
    receipt=dict(schema='factorized-dimer-completed-review-v1',complete=True,passed=True,audit_exit_code=0,
        output_hashes={k:prep.sha(root/k) for k in ['analysis.json','execution/attempts.jsonl']})
    (root/'completed-review.json').write_text(json.dumps(receipt));return prep.sha(root/'completed-review.json')


class PreparationTests(unittest.TestCase):
    def test_exact_baseline_bytes_preserved_and_authenticated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);digest=fixture(root);rows,_=prep.baseline_rows(root,digest)
            self.assertEqual(len(rows),768);self.assertEqual(json.loads(rows[-1])['attempt'],31)
            original=(root/'execution/attempts.jsonl').read_bytes().splitlines(keepends=True)
            self.assertEqual(rows,original[1::2])
            (root/'execution/attempts.jsonl').write_bytes(b''.join(original[:-1]))
            with self.assertRaisesRegex(ValueError,'Changed baseline output'):prep.baseline_rows(root,digest)

    def test_incomplete_or_relabelled_baseline_cannot_be_reused(self):
        for kind in ['receipt','failed','slot','count','analysis']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixture(root);r=json.loads((root/'completed-review.json').read_text())
                if kind=='receipt':r['passed']=False
                elif kind=='analysis':
                    path=root/'analysis.json';value=json.loads(path.read_text());value['passed']=False;path.write_text(json.dumps(value));r['output_hashes']['analysis.json']=prep.sha(path)
                else:
                    path=root/'execution/attempts.jsonl';lines=path.read_text().splitlines(keepends=True)
                    if kind=='count':lines=lines[:-1]
                    else:
                        v=json.loads(lines[1]);v['status' if kind=='failed' else 'attempt']='fatal' if kind=='failed' else 8;lines[1]=json.dumps(v)+'\n'
                    path.write_text(''.join(lines));r['output_hashes']['execution/attempts.jsonl']=prep.sha(path)
                (root/'completed-review.json').write_text(json.dumps(r))
                with self.assertRaises(ValueError):prep.baseline_rows(root,prep.sha(root/'completed-review.json'))

    def test_allocation_contract_detects_extension_caps_and_probability_scope(self):
        receipt=dict(path='/unused',sha256=audit.ALLOCATION_SHA)
        c=dict(scientific_allocation=receipt,proposal_master_seed=6100203101,auxiliary_master_seed=6100300101,root_cap=32,internal_cap=32,factorized_joint_cap=1,attempts_per_context=32,cloud_raw_count=16384,factorized_order='root_first',baseline={'fixed':1})
        p=dict(scientific_allocation=receipt,proposal_master_seed=6100203101,auxiliary_master_seed=6100300101,caps=audit.CAPS,order='root_first',m_values=[1,4],
            allocation=dict(atlases=3,contexts=8,slots_per_context=32,guided_methods=['m1','m4'],new_outer_attempts=1536,reused_baseline_attempts=768,clouds=768,raw_points_per_cloud=16384,raw_cloud_points=12582912,raw_cloud_bytes=301989888,maximum_raw_edge_draws=98304,extension=False),
            density_tolerance=dict(absolute=2e-7,relative=2e-10),physical_bath_draws=0,state_updates=0,native_classification=False,cloud_law=dict(is_poisson=False,raw_count=16384),baseline_cache={'fixed':1})
        with patch.object(audit,'bound',return_value={}):
            audit.validate_contract(c,p)
            for kind in ['allocation','seed','caps','tolerance','poisson','scope','cache','receipt']:
                x,y=copy.deepcopy(c),copy.deepcopy(p)
                if kind=='allocation':y['allocation']['extension']=True
                elif kind=='seed':x['auxiliary_master_seed']+=1
                elif kind=='caps':x['root_cap']+=1
                elif kind=='tolerance':y['density_tolerance']['absolute']=1e-5
                elif kind=='poisson':y['cloud_law']['is_poisson']=True
                elif kind=='scope':y['physical_bath_draws']=1
                elif kind=='cache':x['baseline']={'other':1}
                else:x['scientific_allocation']['sha256']='wrong'
                with self.subTest(kind=kind),self.assertRaises(ValueError):audit.validate_contract(x,y)


if __name__=='__main__':unittest.main()
