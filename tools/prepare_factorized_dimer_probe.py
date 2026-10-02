#!/usr/bin/env python3
"""Freeze the approved passive factorized-versus-whole-joint comparison; never launch."""
from datetime import datetime, timezone
import argparse
from pathlib import Path
import subprocess
import sys
import numpy
import scipy
from prepare_capped_dimer_probe import ROOT, copy, read, record, sha, write
from analyze_capped_dimer_probe import MAP_FILES, REFERENCE_CONFIG_SHA256, PANEL_SHA256

AUDIT_FILES = ['analyze_factorized_dimer_probe.py','test_analyze_factorized_dimer_probe.py',
    'analyze_capped_dimer_probe.py','test_analyze_capped_dimer_probe.py',
    'analyze_dimer_destination_probe.py','dimer_destination_density.py','dimer_destination_geometry.py',
    'normalizer_proposal_density.py','prepare_smc_normalizer_atlas.py','prepare_dimer_destination_panel.py',
    'test_dimer_destination_density.py']


def prepare(base, executable):
    reference=ROOT/'results/dimer-destination-probe-20261002/config.json'
    assert sha(reference)==REFERENCE_CONFIG_SHA256
    old=read(reference);assert old['panel']['sha256']==PANEL_SHA256
    inputs=[record(reference),old['panel'],old['shape'],old['source_config'],old['source_frame'],old['source_freeze_manifest']]+[a['model'] for a in old['atlases']]
    assert all(sha(p['path'])==p['sha256'] for p in inputs)
    assert len(old['cases'])==8 and len(old['atlases'])==3
    assert old['uniform_probability']==.5 and old['uniform_half_width']==160.
    source_files=[ROOT/p for p in ['Cargo.toml','Cargo.lock','build.rs','vendor/README.md']]+sorted((ROOT/'src').rglob('*.rs'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    bundles=[]
    for path in (ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'):
        bundle=read(path)
        if {k:v['sha256'] for k,v in bundle['files'].items()}==hashes:bundles.append(path)
    assert bundles and len({sha(p) for p in bundles})==1,'No exact compiled source bundle'
    base.mkdir();common=base/'common';common.mkdir();source=common/'source'
    source_files += [ROOT/p for p in ['examples/factorized_dimer_probe.rs','tools/prepare_factorized_dimer_probe.py','tools/prepare_capped_dimer_probe.py','tests/factorized_dimer.rs','tests/capped_dimer.rs']]
    source_files += sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())
    for p in dict.fromkeys(source_files):copy(p,source/p.relative_to(ROOT))
    audit_files={}
    for name in AUDIT_FILES:
        destination=common/'audit'/name;copy(ROOT/'tools'/name,destination);audit_files[name]=record(destination)
    frozen_exe=common/'factorized_dimer_probe';copy(executable,frozen_exe);frozen_exe.chmod(0o755)
    copy(bundles[0],common/'source-bundle.json')
    protocol=dict(schema='factorized-protein-dimer-protocol-v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        allocation=dict(atlases=3,contexts=8,outer_slots_per_context=32,methods=['whole_joint','factorized'],
            total_method_trials=1536,independent_context_slots=768,maximum_raw_edge_draws_per_method_trial=64,
            maximum_executed_raw_edge_draws=98304,extension=False),
        caps=dict(whole_joint=32,factorized_root=32,factorized_internal=32,factorized_joint=1),factorized_order='root_first',
        reference_config=record(reference),source_inputs=inputs,master_seed=6100203101,
        seed_rule='SHA256(factorized-dimer-probe-v1/{master}/{atlas-index}/{case-index}/{attempt}/{role}); first16hex as u64',
        method_rng_roles=['whole_joint','factorized'],order_role='execution_order',
        order_rule='separate order_seed lowestbit0:whole_joint then factorized; bit1:factorized then whole_joint',
        rng_fingerprint='Four terminal u64 consumed after each outcome; independent streams per method and outer slot; no shared raw prefixes claimed.',
        source='Unchanged archived sweep7400 panel: 1.4 A, z0.0275,500uM,264tetramers. Reset source every method trial.',
        decision_conditions='Original1.5 A,z0.035,approximately106.8uM remain separate; passive growth-state algorithm screen only.',
        density_law='map-factor-full-mixture-v1',uniform_probability=.5,uniform_half_width_A=160.,
        map_density_source_sha256={name:hashes[name] for name in MAP_FILES},
        factorized_source_sha256=hashes['src/factorized_dimer.rs'],audit_files=audit_files,
        density_tolerance=dict(absolute=2e-7,relative=2e-10),geometry='Independent strict atomic predicates. Frame/predicate mismatch is fatal, never a geometric retry.',
        whole_joint_condition='Full selected hard/wall validity AND internal exclusion contact.',
        factorized_conditions='Root:fixed-spectator core validity and rootwall. Internal:identity-frame mutual core validity and internal exclusion contact. Final:full selected hard/wall/contact, including childwall.',
        retries='Root-first, each edge stops at first success; first-stage exhaustion skips second. J=1: no retry of combined edge outcome. Every raw edge retained. Computational/frame failures terminate with partial ledger.',
        physical_draws=0,state_updates=0,native_classification=False,native_label_filtering=False,
        metrics=['every source context and outermethodtrial','destination feasibility perouter/perrawedge/perproposalCPU',
                 'stage exhaustion and residual final failure decomposition','fullF product ratio and exact generatedposes',
                 'external and intended-anchor contacts','proposalCPU including sourcechecks/density; separate contactdiagnosticCPU and wholeprocessCPU'],
        limitations=['Methods use independent streams; pairing is by fixed context/slot, not identical rawdraws.',
                     'Caps have equal worstcase64edge budgets, not equal expected work or time.',
                     'No physical acceptance, contactESS, equilibrium, native registry or assembly inference.',
                     'Frozen purposive contexts; report all contexts, no posthoc selection or allocation extension.',
                     'Previous primary stationarity rejection and separately matched diagnostic remain separate evidence.'])
    write(base/'protocol.json',protocol)
    config=dict(schema='factorized-dimer-screen-v1',protocol=record(base/'protocol.json'),reference_config=record(reference),master_seed=protocol['master_seed'],
        attempts_per_context=32,joint_cap=32,root_cap=32,internal_cap=32,factorized_joint_cap=1,factorized_order='root_first',
        density_law=protocol['density_law'],compiled_source_sha256=hashes,output=str((base/'execution').resolve()))
    write(base/'config.json',config)
    binding=dict(schema='factorized-dimer-probe-binding-v1',config_sha256=sha(base/'config.json'),protocol_sha256=sha(base/'protocol.json'),
        example_source_sha256=sha(ROOT/'examples/factorized_dimer_probe.rs'),compiled_source_bundle_sha256=sha(common/'source-bundle.json'),executable_sha256=sha(frozen_exe))
    write(base/'binding.json',binding)
    files=sorted(p for p in common.rglob('*') if p.is_file())+[base/'config.json',base/'protocol.json',base/'binding.json',reference]
    closure=dict(schema='factorized-dimer-prelaunch-closure-v1',complete=True,launched=False,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        files={str(p.resolve()):sha(p) for p in files},environment=dict(python=sys.version,python_executable=sys.executable,numpy=numpy.__version__,scipy=scipy.__version__,
            rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),cargo=subprocess.check_output(['cargo','--version'],text=True).strip()),
        command=[str(frozen_exe.resolve()),'--config',str((base/'config.json').resolve()),'--binding',str((base/'binding.json').resolve())],
        analysis_command=[sys.executable,str((common/'audit/analyze_factorized_dimer_probe.py').resolve()),'--run',str((base/'execution').resolve()),'--output',str((base/'analysis.json').resolve())])
    write(base/'prelaunch.json',closure)
    return dict(base=str(base),prelaunch_sha256=sha(base/'prelaunch.json'),**binding)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results/factorized-dimer-probe-20261002')
    parser.add_argument('--executable',type=Path,default=ROOT/'target-validation-line-guide/release/examples/factorized_dimer_probe')
    args=parser.parse_args();print(__import__('json').dumps(prepare(args.output,args.executable),indent=2))
