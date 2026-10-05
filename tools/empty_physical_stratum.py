"""Authenticate an all-hard-invalid stratum without inventing a pose or cloud."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def empty_summary(config,panel,rows):
    if not (panel['entries']==[] and panel['every_valid_pose_included'] is True
            and len(rows)==panel['attempts']==panel['hard_invalid_count'] and len(rows)>0):
        raise ValueError('Not a complete empty physical stratum')
    for i,row in enumerate(rows):
        if not (row['complete'] and row['input']['ordinal']==i
                and row['actual']['physical_valid'] is False and row['physical_zero'] is True
                and row['region'] is None and row['clouds']==[]
                and row['log_physical_contribution'] is None
                and row['physical_weight_status']=='not_estimated'):
            raise ValueError('Zero-cloud receipt would omit a valid or incomplete draw')
    return dict(schema='context-broadened-empty-physical-v1',complete=True,passed=True,
                id=panel['stratum_id'],stratum_id=panel['stratum_id'],attempts=len(rows),
                valid_poses=0,all_hard_invalid=True,all_attempts_preserved=True,
                source_rows=panel['source_rows'],clouds_begun=0,clouds_completed=0,
                raw_points=0,processed_points=0,new_poses_generated=0,retries=0,replacements=0,
                normalizer_estimated=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();started=time.process_time();cfg=read(args.config)
    if sha(cfg['panel']['path'])!=cfg['panel']['sha256']:raise ValueError('Panel hash differs')
    panel=read(cfg['panel']['path']);source=panel['source_rows']
    if sha(source['path'])!=source['sha256']:raise ValueError('Source geometry hash differs')
    rows=[json.loads(l) for l in Path(source['path']).read_text().splitlines()]
    result=empty_summary(cfg,panel,rows)
    result.update(config_sha256=sha(args.config),panel_sha256=sha(cfg['panel']['path']),cpu_seconds=time.process_time()-started)
    args.out.mkdir()
    (args.out/'rows.jsonl').write_text('')
    with (args.out/'summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
