"""Synthetic report-contract tests; no geometry, trajectories or physical draws."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from test_compare_context_prior_pilot import fixture
from compare_context_correlated_pilot import compare_correlations,validate_config,sha


def reports():
    old=fixture()
    for r in old:
        r.update(schema='context-prior-pilot-contact-analysis-v1',method='local' if r['identity']['arm']=='local' else 'posterior_involution',
            bindings={k:k for k in ('shape','model','fixed_context','source_state','executable','source_bundle')},
            initial_observation=dict(patch_tokens=[],neighbor_labels=[]))
        if r['identity']['arm']=='context':r['bindings']['prior']='prior'
    new=copy.deepcopy([r for r in old if r['identity']['arm']!='local'])
    for r in new:r.update(schema='context-correlated095-pilot-contact-analysis-v1',correlation=.95);r['metrics']['proposal_mechanisms']=[]
    return old,new


class Contract(unittest.TestCase):
    def test_paired_seed_is_bound_to_executed_config_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.json';p.write_text(json.dumps(dict(seed=123)))
            item=dict(config=str(p),config_sha256=sha(p),seed=123);report=dict(bindings=dict(config=sha(p)))
            validate_config(item,report)
            wrong=dict(item,seed=124)
            with self.assertRaises(ValueError):validate_config(wrong,report)
            wrong=dict(item,config_sha256='0'*64)
            with self.assertRaises(ValueError):validate_config(wrong,report)
            p.write_text(json.dumps(dict(seed=124)))
            with self.assertRaises(ValueError):validate_config(item,report)

    def test_full_inventory_reuses_controls_without_new_cpu(self):
        old,new=reports();r=compare_correlations(old,new)
        self.assertEqual(len(r['streams']),24);self.assertEqual(len(r['paired_against_same_prior_rho0']),96)
        self.assertEqual(r['new_sampler_cpu_seconds'],160.)
        self.assertEqual(r['costs']['total_full_sampler_cpu_seconds'],240.)
        self.assertTrue(all(x['ratio'] in (None,1.) for x in r['paired_against_same_prior_rho0']))
        self.assertEqual(r['correlation_by_arm'],dict(local=0.,original=.95,context=.95))

    def test_mixed_correlation_changed_context_and_missing_stream_rejected(self):
        for field in ('correlation','context','initial','inventory'):
            old,new=reports()
            if field=='correlation':new[0]['correlation']=0.
            elif field=='context':new[0]['bindings']['fixed_context']='different'
            elif field=='initial':new[0]['initial_observation']['neighbor_labels']=[16]
            else:new.pop()
            with self.assertRaises(ValueError):compare_correlations(old,new)


if __name__=='__main__':unittest.main()
