"""Synthetic authenticated-evidence checks; no physical output is consumed."""
import copy
from pathlib import Path
import tempfile
import unittest
from analyze_r4_smc_control import sha,write
from prepare_full_vessel_comparison import authenticate_gate_inputs
from test_prepare_full_vessel_comparison import reports


def frozen(path, data):
    path.parent.mkdir(parents=True,exist_ok=True);write(path,data)
    write(path.parent/'freeze.json',{'files':{path.name:sha(path)}})


def inputs(root):
    confirmation,reports_=reports();conf=root/'confirmation/analysis.json';frozen(conf,confirmation)
    paths=[];protocols={'broad':'broadsha','narrow':'narrowsha'}
    for i,(arm,protocol) in enumerate(protocols.items()):
        smc=root/arm/'analysis.json'
        frozen(smc,dict(schema='smc-r4-control-analysis-v1',complete=True,protocol_sha256=protocol,
                       populations=[{'seed':i*100+j} for j in range(4)]))
        comparison=copy.deepcopy(reports_[i]);comparison.update(inputs={'importance':str(conf),'smc':str(smc)},
            input_sha256={str(conf):sha(conf),str(smc):sha(smc)})
        path=root/(arm+'-comparison')/'analysis.json';frozen(path,comparison);paths.append(path)
    return conf,paths,protocols


class GateProvenanceTests(unittest.TestCase):
    def test_distinct_bound_controls_pass_without_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            result=authenticate_gate_inputs(*inputs(Path(d)))
            self.assertFalse(result['dispatch_performed']);self.assertEqual(len(result['smc_population_seeds']),8)

    def test_same_comparison_or_different_confirmatory_digest_rejected(self):
        from analyze_r4_smc_control import read
        with tempfile.TemporaryDirectory() as d:
            conf,paths,protocols=inputs(Path(d))
            with self.assertRaisesRegex(ValueError,'Two distinct'):authenticate_gate_inputs(conf,[paths[0]]*2,protocols)
            bad=read(paths[1]);bad['input_sha256'][str(conf)]='wrong';frozen(paths[1],bad)
            with self.assertRaisesRegex(ValueError,'different confirmation'):authenticate_gate_inputs(conf,paths,protocols)

    def test_duplicate_smc_control_even_at_distinct_comparison_paths_rejected(self):
        from analyze_r4_smc_control import read
        with tempfile.TemporaryDirectory() as d:
            conf,paths,protocols=inputs(Path(d));frozen(paths[1],read(paths[0]))
            with self.assertRaisesRegex(ValueError,'Same SMC'):authenticate_gate_inputs(conf,paths,protocols)

    def test_changed_report_without_updated_freeze_rejected(self):
        from analyze_r4_smc_control import read
        with tempfile.TemporaryDirectory() as d:
            conf,paths,protocols=inputs(Path(d));changed=read(paths[0]);changed['complete']=False;write(paths[0],changed)
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):authenticate_gate_inputs(conf,paths,protocols)

    def test_wrong_protocol_and_shared_population_seed_rejected(self):
        from analyze_r4_smc_control import read
        with tempfile.TemporaryDirectory() as d:
            conf,paths,protocols=inputs(Path(d))
            with self.assertRaisesRegex(ValueError,'protocol identities'):authenticate_gate_inputs(conf,paths,dict(broad='wrong',narrow='narrowsha'))
            comparison=read(paths[1]);smc=Path(comparison['inputs']['smc']);changed=read(smc);changed['populations'][0]['seed']=0;frozen(smc,changed)
            comparison['input_sha256'][str(smc)]=sha(smc);frozen(paths[1],comparison)
            with self.assertRaisesRegex(ValueError,'streams overlap'):authenticate_gate_inputs(conf,paths,protocols)


if __name__=='__main__':unittest.main()
