#!/usr/bin/env python3
"""Run independent uniform-latent-ball/shell integrals of one immutable region."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"runs/basin-normalizer-importance-guided-16384-l64/provenance/config.json")
    parser.add_argument("--region",type=Path,default=ROOT/"runs/smc-normalizer-deep-far/site0/fixed-discovered-region.json")
    parser.add_argument("--binary",type=Path,default=ROOT/"target/release/latent-region-normalizer")
    parser.add_argument("--samples",type=int,default=16384)
    parser.add_argument("--replicates",type=int,default=4)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=98531010)
    parser.add_argument("--lambda-ratio",type=float,default=64.)
    parser.add_argument("--cloud-replicates",type=int,default=2)
    args=parser.parse_args()
    if min(args.samples,args.replicates,args.workers,args.cloud_replicates)<1 or args.lambda_ratio<=0:
        parser.error("Require positive counts and lambda ratio")
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use an empty fresh output directory")
    archive=out/"provenance";archive.mkdir(parents=True)
    sources={"input-config.json":args.config,"region.json":args.region,
             "latent-region-normalizer":args.binary,"launcher.py":Path(__file__),
             "analyze_latent_region.py":Path(__file__).with_name("analyze_latent_region.py"),
             "prepare_smc_normalizer_atlas.py":Path(__file__).with_name("prepare_smc_normalizer_atlas.py"),
             "prepare_deep_far_normalizer_atlas.py":Path(__file__).with_name("prepare_deep_far_normalizer_atlas.py"),
             "analyze_basin_normalizers.py":Path(__file__).with_name("analyze_basin_normalizers.py"),
             "analyze_native_region_reference.py":Path(__file__).with_name("analyze_native_region_reference.py"),
             "analyze_latent_region_shells.py":Path(__file__).with_name("analyze_latent_region_shells.py")}
    for name,path in sources.items():
        shutil.copy2(path,archive/name)
    config=json.loads(args.config.read_text())
    shape=Path(config["shape"])
    if not shape.is_absolute():shape=args.config.parent/shape
    shutil.copy2(shape,archive/"shape.json")
    config["shape"]=str(archive/"shape.json")
    write(archive/"config.json",config)
    jobs=[]
    for i in range(args.replicates):
        directory=out/"runs"/f"r{i:02d}"
        jobs.append(dict(id=f"r{i:02d}",seed=args.seed+1009*i,samples=args.samples,directory=str(directory)))
    manifest=dict(schema="uniform-latent-region-campaign-v1",created_unix=time.time(),jobs=jobs,
        workers=args.workers,physical_activity=config["reservoir_density"],lambda_ratio=args.lambda_ratio,
        cloud_replicates=args.cloud_replicates,region_sha256=sha(archive/"region.json"),
        archive_sha256={p.name:sha(p) for p in archive.iterdir()},
        source_inputs={name:str(path.resolve()) for name,path in sources.items()},
        scope="REGION ONLY. Uniform six-dimensional latent ball/shell integration of one frozen region with all prescribed physical neighbors. Independent fixed-N zeros retained; no model fitting or global mass inference.")
    write(out/"manifest.json",manifest)
    def execute(job):
        args_command=[str(archive/"latent-region-normalizer"),"--config",str(archive/"config.json"),
            "--region",str(archive/"region.json"),"--out",job["directory"],"--samples",str(job["samples"]),
            "--seed",str(job["seed"]),"--lambda-ratio",str(args.lambda_ratio),"--cloud-replicates",str(args.cloud_replicates)]
        with (out/f"{job['id']}.log").open("w") as f:
            result=subprocess.run(args_command,stdout=f,stderr=subprocess.STDOUT)
        status=dict(id=job["id"],returncode=result.returncode,completed_unix=time.time())
        write(out/f"{job['id']}-status.json",status)
        return status
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute,job) for job in jobs]):
            status=future.result();print(json.dumps(status),flush=True)
            if status["returncode"]:
                raise RuntimeError(f"{status['id']} failed; inspect log")
    print(json.dumps(dict(output=str(out),complete=True)))

if __name__=="__main__":main()
