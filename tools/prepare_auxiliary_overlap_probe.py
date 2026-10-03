#!/usr/bin/env python3
"""Freeze a guided passive probe and cached baseline; never sample or launch."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy
import scipy
from prepare_capped_dimer_probe import ROOT,copy,read,record,sha,write
from analyze_capped_dimer_probe import MAP_FILES,REFERENCE_CONFIG_SHA256,PANEL_SHA256

BASELINE=ROOT/'results/factorized-dimer-probe-20261002'
BASELINE_RECEIPT='738c91a6d4a588626ce62fe567238db1ebf46683a474ad3684738795d1a98a1a'
ALLOCATION=ROOT/'results/auxiliary-overlap-probe-design-20261003/allocation.json'
ALLOCATION_SHA='d768b70d07e5837218217d512a3266bb75b221916fb3ee74c7312aba15126b1f'
AUDIT_FILES=['analyze_auxiliary_overlap_probe.py','test_analyze_auxiliary_overlap_probe.py',
 'test_prepare_auxiliary_overlap_probe.py','prepare_auxiliary_overlap_probe.py',
 'analyze_factorized_dimer_probe.py','test_analyze_factorized_dimer_probe.py',
 'analyze_capped_dimer_probe.py','test_analyze_capped_dimer_probe.py',
 'analyze_dimer_destination_probe.py','dimer_destination_density.py','dimer_destination_geometry.py',
 'normalizer_proposal_density.py','prepare_smc_normalizer_atlas.py','prepare_dimer_destination_panel.py',
 'test_dimer_destination_density.py','prepare_capped_dimer_probe.py']


def require(ok,message):
    if not ok:raise ValueError(message)


def baseline_rows(root=BASELINE,expected_receipt=BASELINE_RECEIPT):
    root=Path(root);require(sha(root/'completed-review.json')==expected_receipt,'Baseline receipt changed')
    receipt=read(root/'completed-review.json')
    require(receipt['schema']=='factorized-dimer-completed-review-v1' and receipt['complete'] is True and receipt['passed'] is True and receipt['audit_exit_code']==0,'Baseline audit not complete')
    for name,digest in receipt['output_hashes'].items():require(sha(root/name)==digest,'Changed baseline output: '+name)
    analysis=read(root/'analysis.json')
    require(analysis['complete'] is True and analysis['passed'] is True and not analysis['failures'] and analysis['summary']['outer_attempts']==1536,'Incomplete baseline analysis')
    lines=[];keys=[]
    with (root/'execution/attempts.jsonl').open('rb') as stream:
        for line in stream:
            row=json.loads(line)
            if row['method']!='factorized':continue
            require(row['status']=='completed','Failed baseline row')
            keys.append((row['atlas_index'],row['case_index'],row['attempt']));lines.append(line)
    require(keys==[(a,c,i) for a in range(3) for c in range(8) for i in range(32)],'Missing/reordered baseline slots')
    require(len(lines)==768,'Baseline row count')
    return lines,receipt


def prepare(base,executable):
    require(sha(ALLOCATION)==ALLOCATION_SHA,'Scientific allocation changed')
    allocation=read(ALLOCATION)
    require(allocation['allocation']==dict(cases=8,atlases=3,slots_per_case_atlas=32,guided_m=[1,4],new_outer_trials=1536,baseline_rows_reused=768,new_baseline_draws=0),'Scientific allocation differs')
    lines,receipt=baseline_rows()
    reference=ROOT/'results/dimer-destination-probe-20261002/config.json'
    require(sha(reference)==REFERENCE_CONFIG_SHA256,'Reference config changed')
    ref=read(reference);require(ref['panel']['sha256']==PANEL_SHA256,'Panel changed')
    inputs=[record(reference),ref['panel'],ref['shape'],ref['source_config'],ref['source_frame'],ref['source_freeze_manifest']]+[a['model'] for a in ref['atlases']]
    require(all(sha(v['path'])==v['sha256'] for v in inputs),'Changed frozen source input')
    require(len(ref['cases'])==8 and len(ref['atlases'])==3 and ref['uniform_probability']==.5 and ref['uniform_half_width']==160.,'Frozen source contract changed')
    shape=read(ref['shape']['path']);r=numpy.asarray([a['radius']+ref['depletant_radius'] for a in shape['atoms']]);c=numpy.asarray([a['center'] for a in shape['atoms']]);low=(c-r[:,None]).min(axis=0);high=(c+r[:,None]).max(axis=0);volume=float(numpy.prod(high-low))
    source_files=[ROOT/p for p in ['Cargo.toml','Cargo.lock','build.rs','vendor/README.md']]+sorted((ROOT/'src').rglob('*.rs'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files};bundles=[]
    for path in (ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'):
        b=read(path)
        if {k:v['sha256'] for k,v in b['files'].items()}==hashes:bundles.append(path)
    require(bundles and len({sha(p) for p in bundles})==1,'No matching compiled source bundle')
    base=Path(base).resolve();base.mkdir();common=base/'common';common.mkdir()
    all_sources=source_files+[ROOT/p for p in ['examples/auxiliary_overlap_probe.rs','tools/prepare_auxiliary_overlap_probe.py','tests/auxiliary_overlap_threshold.rs','tests/factorized_dimer.rs']]
    all_sources+=sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())
    for p in dict.fromkeys(all_sources):copy(p,common/'source'/p.relative_to(ROOT))
    audit_files={}
    for name in AUDIT_FILES:
        p=common/'audit'/name;copy(ROOT/'tools'/name,p);audit_files[name]=record(p)
    with (common/'baseline.jsonl').open('xb') as out:out.writelines(lines)
    baseline_sources={name:record(BASELINE/name) for name in ['completed-review.json','analysis.json','prelaunch.json','config.json','protocol.json','binding.json','execution/attempts.jsonl','execution/terminal.json']}
    frozen_exe=common/'auxiliary_overlap_probe';copy(executable,frozen_exe);frozen_exe.chmod(0o755);copy(bundles[0],common/'source-bundle.json')
    copy(ALLOCATION,common/'scientific-allocation.json')
    protocol=dict(schema='auxiliary-overlap-passive-protocol-v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
      scientific_allocation=record(common/'scientific-allocation.json'),
      allocation=dict(atlases=3,contexts=8,slots_per_context=32,guided_methods=['m1','m4'],new_outer_attempts=1536,reused_baseline_attempts=768,clouds=768,raw_points_per_cloud=16384,raw_cloud_points=12582912,raw_cloud_bytes=301989888,maximum_raw_edge_draws=98304,extension=False),
      caps=dict(root=32,internal=32,joint=1),order='root_first',m_values=[1,4],proposal_master_seed=6100203101,auxiliary_master_seed=6100300101,
      proposal_seed_rule='SHA256(factorized-dimer-probe-v1/{master}/{atlas}/{case}/{attempt}/factorized); first16hex u64; reused exact baseline seed',
      auxiliary_seed_rule='SHA256(auxiliary-overlap-probe-v1/{master}/{atlas}/{case}/{attempt}/{role}); first16hex u64',auxiliary_roles=['cloud','threshold','execution_order'],
      arm_order='execution_order seed lowbit0:m1,m4; lowbit1:m4,m1',
      matching='Same cloud and threshold RNG prefix per slot. Separate proposal RNG reset to cached baseline seed; common root/internal raw-edge prefixes checked. Arms are dependent paired controls.',
      cloud_law=dict(kind='fixed-size-uniform-AABB-root-thinning',raw_count=16384,low=low.tolist(),high=high.tolist(),volume_A3=volume,effective_raw_intensity_Aminus3=16384/volume,
          membership='Closed atomic membership d_squared <= inflated_radius_squared; body-body overlap remains strict.',
          thinning='Keep root exclusion members only, independent of old internal pose. Never redraw for old_count0 or choose points by overlap.',
          storage='All raw uniforms are LEf64 triples in cloud-uniforms.bin; affine transform coordinates and retained coordinates carry hashes; kept indices saved.',
          is_poisson=False,point_frame='root body; co-transports with root during pose block'),
      threshold_law='m independent uniform integers0..=Kold; maximum; fixed across all capped edge/joint attempts; no auxiliary resampling after failure',
      auxiliary_correction='m*(log(Kold+1)-log(Knew+1)), separate from fullF; selection/tree-coordinate corrections zero for these fixed labels',
      physical_conditions=dict(depletant_radius_A=1.4,activity_Aminus3=.0275,concentration_uM=500),decision_conditions='Original1.5A/.035A^-3/~106.8uM remain separate.',
      density_law='map-factor-full-mixture-v1',uniform_probability=.5,uniform_half_width_A=160.,reference_config=record(reference),source_inputs=inputs,
      baseline_sources=baseline_sources,baseline_cache=record(common/'baseline.jsonl'),baseline_summary=receipt['allocation'],
      map_density_source_sha256={k:hashes[k] for k in MAP_FILES},guidance_source_sha256={k:hashes[k] for k in ['src/factorized_dimer.rs','src/auxiliary_overlap_threshold.rs']},audit_files=audit_files,
      density_tolerance=dict(absolute=2e-7,relative=2e-10),physical_bath_draws=0,state_updates=0,native_classification=False,
      timing='Charge entire shared cloudconstruction CPU to each guided arm; guidance setup and proposal calls also included; true campaign CPU reported separately. Baseline historical CPU, no new baseline cloud work.',
      metrics=['all unconditional outertrials and capfailures','raw-edge common prefixes','source/destination counts and threshold failures','fullF and separate auxiliary correction',
        'internal-overlap count retention','external/intended geometric contacts','perarm standaloneCPU and actualcampaignCPU'],
      limitations=['Passive guide/candidate comparison only; no physical acceptance,contactESS,native registry or assembly inference.',
        'Fixed finite cloud count is a proposal auxiliary, not an ideal-depletant bath or an exact overlap volume.',
        'Dependent paired arms; reused baseline timing from earlier run, not a simultaneous timing replicate.',
        'Cloud law is state independent; m changes threshold law but adds its complete target correction.',
        'Every failed or fatal attempt is retained; fatal computation stops allocation without replacement draws.'])
    write(base/'protocol.json',protocol)
    config=dict(schema='auxiliary-overlap-screen-v1',scientific_allocation=record(common/'scientific-allocation.json'),protocol=record(base/'protocol.json'),reference_config=record(reference),proposal_master_seed=6100203101,auxiliary_master_seed=6100300101,cloud_raw_count=16384,
      baseline=record(common/'baseline.jsonl'),attempts_per_context=32,root_cap=32,internal_cap=32,factorized_joint_cap=1,factorized_order='root_first',density_law=protocol['density_law'],compiled_source_sha256=hashes,output=str(base/'execution'))
    write(base/'config.json',config)
    binding=dict(schema='auxiliary-overlap-probe-binding-v1',config_sha256=sha(base/'config.json'),protocol_sha256=sha(base/'protocol.json'),example_source_sha256=sha(ROOT/'examples/auxiliary_overlap_probe.rs'),compiled_source_bundle_sha256=sha(common/'source-bundle.json'),executable_sha256=sha(frozen_exe));write(base/'binding.json',binding)
    files=sorted(p for p in common.rglob('*') if p.is_file())+[base/'protocol.json',base/'config.json',base/'binding.json']
    prelaunch=dict(schema='auxiliary-overlap-prelaunch-v1',complete=True,launched=False,files={str(p.resolve()):sha(p) for p in files},external_inputs={v['path']:v['sha256'] for v in inputs+list(baseline_sources.values())},
       environment=dict(python=sys.version,python_executable=sys.executable,numpy=numpy.__version__,scipy=scipy.__version__,rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),cargo=subprocess.check_output(['cargo','--version'],text=True).strip()),
       command=[str(frozen_exe),'--config',str(base/'config.json'),'--binding',str(base/'binding.json')],analysis_command=[sys.executable,str(common/'audit/analyze_auxiliary_overlap_probe.py'),'--run',str(base/'execution'),'--output',str(base/'analysis.json')])
    write(base/'prelaunch.json',prelaunch);return dict(base=str(base),prelaunch_sha256=sha(base/'prelaunch.json'),**binding)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/auxiliary-overlap-probe-20261003');p.add_argument('--executable',type=Path,default=ROOT/'target-validation-line-guide/release/examples/auxiliary_overlap_probe');a=p.parse_args();print(json.dumps(prepare(a.output,a.executable),indent=2))
