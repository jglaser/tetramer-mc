#!/usr/bin/env python3
"""Freeze the bounded paired-cap passive screen; never launch proposals."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy
import scipy
from analyze_capped_dimer_probe import MAP_FILES, REFERENCE_CONFIG_SHA256, PANEL_SHA256

ROOT=Path(__file__).resolve().parents[1]
AUDIT_FILES=['analyze_capped_dimer_probe.py','test_analyze_capped_dimer_probe.py',
             'analyze_dimer_destination_probe.py','dimer_destination_density.py','dimer_destination_geometry.py',
             'normalizer_proposal_density.py','prepare_smc_normalizer_atlas.py','prepare_dimer_destination_panel.py',
             'test_dimer_destination_density.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def record(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,value):
    with Path(path).open('x') as out:json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')
def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)


def prepare(base, executable):
    reference=ROOT/'results/dimer-destination-probe-20261002/config.json'
    assert sha(reference)==REFERENCE_CONFIG_SHA256
    old=read(reference);assert old['panel']['sha256']==PANEL_SHA256
    assert sha(old['panel']['path'])==PANEL_SHA256
    inputs=[record(reference),old['panel'],old['shape'],old['source_config'],old['source_frame'],old['source_freeze_manifest']]+[a['model'] for a in old['atlases']]
    assert all(sha(p['path'])==p['sha256'] for p in inputs)
    assert old['uniform_probability']==.5 and old['uniform_half_width']==160.
    source_files=[ROOT/p for p in ['Cargo.toml','Cargo.lock','build.rs','vendor/README.md']]+sorted((ROOT/'src').rglob('*.rs'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    bundles=[]
    for path in (ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'):
        bundle=read(path)
        if {k:v['sha256'] for k,v in bundle['files'].items()}==hashes:bundles.append(path)
    assert bundles and len({sha(p) for p in bundles})==1,'No exact compiled source bundle'
    base.mkdir();common=base/'common';common.mkdir()
    source=common/'source'
    all_sources=source_files+[ROOT/'examples/capped_dimer_probe.rs',ROOT/'tools/prepare_capped_dimer_probe.py',
        ROOT/'tests/capped_dimer.rs',ROOT/'docs/capped-dimer-conditioning.md',ROOT/'docs/docking-factor-closure.md']
    all_sources+=sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())
    for p in dict.fromkeys(all_sources):copy(p,source/p.relative_to(ROOT))
    audit_files={}
    for name in AUDIT_FILES:
        destination=common/'audit'/name;copy(ROOT/'tools'/name,destination);audit_files[name]=record(destination)
    frozen_exe=common/'capped_dimer_probe';copy(executable,frozen_exe);frozen_exe.chmod(0o755)
    copy(bundles[0],common/'source-bundle.json')
    protocol=dict(schema='capped-protein-dimer-protocol-v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        allocation=dict(atlases=3,contexts=8,outer_attempts_per_context_and_cap=32,caps=[1,8,32],
                        total_outer_attempts=2304,independent_source_seed_slots=768,
                        maximum_executed_raw_trials=31488,maximum_unique_raw_prefix_positions=24576,extension=False),
        reference_config=record(reference),source_inputs=inputs,master_seed=6100202901,
        seed_rule='SHA256(capped-dimer-probe-v1/{master}/{atlas-index}/{case-index}/{attempt}); first16hex as u64; cap excluded',
        rng_fingerprint='Four terminal u64 values consumed after each outcome; source RNG is reset per cap and outer slot.',
        source='Same archived sweep7400 panel: 1.4A, z0.0275,500uM,264tetramers. Each outer attempt resets to source.',
        decision_conditions='Original 1.5 A, z=0.035, approximately 106.8 uM remain separate; this is a growth-state diagnostic.',
        density_law='map-factor-full-mixture-v1',uniform_probability=.5,uniform_half_width_A=160.,
        map_density_source_sha256={name:hashes[name] for name in MAP_FILES},audit_files=audit_files,
        density_tolerance=dict(absolute=2e-7,relative=2e-10),geometry='Strict atomic interior predicates; threshold ambiguities retained.',
        condition='Full hard/wall validity AND internal exclusion contact; source outside contact self-loops.',
        retries='Whole-joint independent draws, first geometric success only; numerical errors terminate with ledger record.',
        physical_draws=0,state_updates=0,native_classification=False,native_label_filtering=False,
        metrics=['all outer attempts and raw trials','paired cap prefix/restart consistency','actual-map fullF density and generation reconstruction',
                 'independent all-atom geometry/fingerprints at every unique raw prefix, with bitwise duplicate checks',
                 'candidate count per raw trial and proposalCPU','external and intended anchor contacts','hard-valid contact-free trials skipped before bath',
                 'source eligibility and all eight source/anchor contexts'],
        limitations=['Increasing the cap changes the common channel rate, not success-conditional destination law.',
                     'Caps are dependent paired controls; no uncertainty estimate treating 2304 as independent.',
                     'No physical acceptance/contact ESS/thermodynamic/assembly claim; prior stationarity failure remains unresolved.',
                     'No repeated allocation after failures; separate predeclared analysis-only repairs require retained original output.'])
    write(base/'protocol.json',protocol)
    config=dict(schema='capped-dimer-screen-v1',protocol=record(base/'protocol.json'),reference_config=record(reference),
                master_seed=protocol['master_seed'],attempts_per_context=32,caps=[1,8,32],density_law=protocol['density_law'],
                compiled_source_sha256=hashes,output=str((base/'execution').resolve()))
    write(base/'config.json',config)
    binding=dict(schema='capped-dimer-probe-binding-v1',config_sha256=sha(base/'config.json'),protocol_sha256=sha(base/'protocol.json'),
                 example_source_sha256=sha(ROOT/'examples/capped_dimer_probe.rs'),compiled_source_bundle_sha256=sha(common/'source-bundle.json'),
                 executable_sha256=sha(frozen_exe))
    write(base/'binding.json',binding)
    files=sorted(p for p in common.rglob('*') if p.is_file())+[base/'config.json',base/'protocol.json',base/'binding.json',reference]
    closure=dict(schema='capped-dimer-prelaunch-closure-v1',complete=True,launched=False,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),files={str(p.resolve()):sha(p) for p in files},
        environment=dict(python=sys.version,python_executable=sys.executable,numpy=numpy.__version__,scipy=scipy.__version__,
                         rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
                         cargo=subprocess.check_output(['cargo','--version'],text=True).strip()),
        command=[str(frozen_exe.resolve()),'--config',str((base/'config.json').resolve()),'--binding',str((base/'binding.json').resolve())],
        analysis_command=[sys.executable,str((common/'audit/analyze_capped_dimer_probe.py').resolve()),'--run',str((base/'execution').resolve()),'--output',str((base/'analysis.json').resolve())])
    write(base/'prelaunch.json',closure)
    return dict(base=str(base),prelaunch_sha256=sha(base/'prelaunch.json'),**binding)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/capped-dimer-probe-20261002')
    parser.add_argument('--executable',type=Path,default=ROOT/'target-validation-line-guide/release/examples/capped_dimer_probe')
    args=parser.parse_args();print(json.dumps(prepare(args.output,args.executable),indent=2))
