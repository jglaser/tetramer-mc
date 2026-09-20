#!/usr/bin/env python3
"""Repeat fixed-pose clouds at new partition extremes using a frozen driver."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from prepare_intermediate_local_region import ROOT,read,write,sha
from run_intermediate_fixed_pose_clouds import summarize_clouds


def run(diagnostic_path,out):
    assert not out.exists(), 'Use a fresh output directory'
    diagnostic=read(diagnostic_path);assert diagnostic['complete']
    previous=ROOT/'runs/ab-intermediate-fixed-pose-clouds-20260920'
    prior=read(previous/'protocol.json');prior_analysis=read(previous/'analysis.json')
    assert sha(previous/'protocol.json')==prior_analysis['protocol_sha256']
    for name,h in prior['archived_sha256'].items():assert sha(previous/'provenance'/name)==h
    binary=previous/'provenance/fixed-pose-clouds'
    assert sha(binary)==prior['binary_sha256']
    archive=out/'provenance';archive.mkdir(parents=True)
    for path,name in [(Path(__file__),'recheck_intermediate_partition_clouds.py'),
                      (ROOT/'tools/run_intermediate_fixed_pose_clouds.py','run_intermediate_fixed_pose_clouds.py'),
                      (diagnostic_path,'input-diagnostic.json'),(previous/'protocol.json','driver-protocol.json'),
                      (binary,'fixed-pose-clouds'),(previous/'provenance/fixed_pose_clouds.rs','fixed_pose_clouds.rs')]:
        shutil.copy2(path,archive/name)
    config=previous/'provenance/config.json';cases=[]
    for name in ('r16','r32','remainder'):
        p=next(p for p in diagnostic['pieces'] if p['name']==name)['top_poses'][0]
        clouds=p['original_clouds'];assert len(clouds)==2
        cases.append(dict(id=name,pose=p['pose'],original_q=p['q'],original_cloud_log_weights=[c['log_weight'] for c in clouds],
                          original_lower_volume=clouds[0]['lower_volume'],original_upper_volume=clouds[0]['upper_volume'],
                          clouds=256,seed=100701010+1009*len(cases),source_point=p))
    write(out/'cases.json',cases)
    write(out/'control.json',dict(config=str(config),lambda_ratio=64.,cases=cases))
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=sha(diagnostic_path),
                  cases_sha256=sha(out/'cases.json'),control_sha256=sha(out/'control.json'),
                  original_driver_protocol_sha256=sha(previous/'protocol.json'),binary_sha256=sha(binary),
                  config_sha256=sha(config),shape_sha256=sha(Path(read(config)['shape'])),
                  seeds=[c['seed'] for c in cases],clouds_per_pose=256,activity=.035,depletant_radius=1.5,lambda_ratio=64.,
                  archive_sha256={p.name:sha(p) for p in archive.iterdir()},
                  scope='New independent clouds conditional on three selected fixed poses; frozen existing kernel, original envelopes and physical model. No regional mass or unbiased-log claim.')
    write(out/'protocol.json',protocol)
    result=subprocess.run([str(archive/'fixed-pose-clouds'),str(out/'control.json'),str(out/'clouds.jsonl')],capture_output=True,text=True)
    (out/'execution.log').write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in (out/'clouds.jsonl').read_text().splitlines()];assert len(rows)==768
    estimates=[summarize_clouds(c,[r for r in rows if r['case']==c['id']]) for c in cases]
    write(out/'analysis.json',dict(complete=True,points=estimates,protocol_sha256=sha(out/'protocol.json'),
                                  cloud_rows_sha256=sha(out/'clouds.jsonl'),scope=protocol['scope']))
    print(json.dumps(estimates,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();run(args.diagnostic.resolve(),args.out.resolve())
