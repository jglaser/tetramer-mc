#!/usr/bin/env python3
"""Run a fixed-budget independent native-region reference campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--seed-base", type=int, default=98581010)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cover-scales", help="Optional comma-separated geometric proposal scales; target metric unchanged")
    parser.add_argument("--cover-weights", help="Optional comma-separated positive mixture weights")
    parser.add_argument("--model", type=Path, help="Optional frozen relative-pose Gaussian guide")
    parser.add_argument("--model-weight", type=float, help="Hybrid guide-family probability (default 0.75 with --model)")
    parser.add_argument("--model-uniform-probability", type=float, help="Uniform-cube probability inside the guide (default 0.05)")
    parser.add_argument("--model-anchor-index", type=int, help="One explicit fixed-neighbor proposal anchor (default 0)")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze inputs and commands without launching populations")
    parser.add_argument("--expected-binary-sha256", help="Require this exact reviewed executable hash before launching")
    args = parser.parse_args()
    assert args.replicates > 0 and args.samples > 0 and args.workers > 0
    root, config, binary = (p.resolve() for p in (args.root, args.config, args.binary))
    assert not root.exists(), "Refuse to restart an existing campaign"
    if args.expected_binary_sha256:
        assert sha(binary)==args.expected_binary_sha256, 'Unexpected executable; do not launch'
    assert args.cover_scales is not None or args.cover_weights is None, 'Cover weights need explicit scales'
    proposal=None
    guide=None
    if args.cover_scales is not None:
        scales=[float(x) for x in args.cover_scales.split(',')]
        weights=[float(x) for x in args.cover_weights.split(',')] if args.cover_weights else [1.]*len(scales)
        assert scales and all(math.isfinite(x) and 0<x<=1 for x in scales) and 1. in scales
        assert len(weights)==len(scales) and all(math.isfinite(x) and x>0 for x in weights)
        assert math.isfinite(sum(weights))
        proposal={'scales':scales,'weights':[x/sum(weights) for x in weights]}
    if args.model is None:
        assert all(value is None for value in (args.model_weight,args.model_uniform_probability,args.model_anchor_index)), 'Guide options require --model'
    else:
        from analyze_native_region_reference import GaussianGuide
        model_path=args.model.resolve();raw_model=json.loads(model_path.read_text());cfg=json.loads(config.read_text())
        beta=.75 if args.model_weight is None else args.model_weight
        epsilon=.05 if args.model_uniform_probability is None else args.model_uniform_probability
        anchor_index=0 if args.model_anchor_index is None else args.model_anchor_index
        assert type(anchor_index) is int and 0<=anchor_index<len(cfg['fixed_poses'])
        shape=Path(cfg['shape']);shape=shape if shape.is_absolute() else config.parent/shape
        guide={'model_sha256':sha(model_path),'weight':beta,'uniform_probability':epsilon,
            'anchor_index':anchor_index,'anchor_pose':cfg['fixed_poses'][anchor_index],
            'cube_lengths':[2*cfg['capture_radius']]*3,'capture_center':cfg['capture_center']}
        GaussianGuide(guide,raw_model,cfg,sha(shape))
    (root / "provenance").mkdir(parents=True)
    archived = root / "provenance/native-region-normalizer"
    shutil.copy2(binary, archived)
    if args.expected_binary_sha256:
        assert sha(archived)==args.expected_binary_sha256, 'Frozen executable hash changed'
    shutil.copy2(config, root / "provenance/input-config.json")
    shutil.copy2(__file__, root / "provenance/runner.py")
    if guide:
        frozen_model=root/'provenance/guide-model.json'
        shutil.copy2(model_path,frozen_model)
        assert sha(frozen_model)==guide['model_sha256'], 'Guide changed while archiving'
    (root / "logs").mkdir()
    (root / "runs").mkdir()
    jobs = []
    for i in range(args.replicates):
        output = root / "runs" / f"r{i:02d}"
        seed = args.seed_base + 1009*i
        command = [str(archived), "--config", str(config), "--out", str(output),
                   "--samples", str(args.samples), "--seed", str(seed), "--lambda-ratio", "64",
                   "--cloud-replicates", "2"]
        if args.cover_scales is not None:
            command.extend(['--cover-scales',args.cover_scales])
        if args.cover_weights is not None:
            command.extend(['--cover-weights',args.cover_weights])
        if guide:
            command.extend(['--model',str(frozen_model),'--model-weight',str(guide['weight']),
                '--model-uniform-probability',str(guide['uniform_probability']),
                '--model-anchor-index',str(guide['anchor_index'])])
        jobs.append({"replicate": i, "seed": seed, "output": str(output), "command": command})
    manifest = {"protocol": "complete-native-cover-independent-v1", "jobs": jobs,
                "config_sha256": sha(config), "binary_sha256": sha(archived),
                "runner_sha256": sha(__file__), "total_unconditional_draws": args.replicates*args.samples,
                "proposal_override":proposal,
                "max_workers": args.workers, "created_utc": datetime.now(timezone.utc).isoformat(),
                "no_restart": "Never replace a zero, high-weight, or failed population according to its result"}
    if guide:
        manifest['guide_override']=guide
        manifest['model_source_path']=str(model_path)
        manifest['frozen_model_path']=str(frozen_model)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    state = {"runner_pid": os.getpid(), "complete": False, "jobs": {}}
    lock = threading.Lock()

    def persist():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = root / "runner-status.tmp"
        tmp.write_text(json.dumps(state, indent=2)+"\n")
        tmp.replace(root / "runner-status.json")

    if args.prepare_only:
        state.update(prepared_only=True,running=False)
        persist()
        print(json.dumps({'prepared_only':True,'root':str(root),'jobs':len(jobs),'guide_override':guide}),flush=True)
        return

    def run(job):
        label = f"r{job['replicate']:02d}"
        started = time.monotonic()
        with (root / "logs" / f"{label}.log").open("x") as log:
            process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT)
            with lock:
                state["jobs"][label] = {"pid": process.pid, "status": "running", "seed": job["seed"]}
                persist()
            while process.poll() is None:
                with lock:
                    progress = Path(job["output"]) / "progress.json"
                    if progress.exists():
                        try:
                            state["jobs"][label]["progress"] = json.loads(progress.read_text())
                        except json.JSONDecodeError:
                            pass
                    persist()
                time.sleep(3)
            with lock:
                result = state["jobs"][label]
                result.update(status="complete" if process.returncode == 0 else "failed",
                              exit_code=process.returncode, wall_seconds=time.monotonic()-started)
                summary = Path(job["output"]) / "summary.json"
                progress=Path(job['output'])/'progress.json'
                if progress.exists():
                    result['progress']=json.loads(progress.read_text())
                if summary.exists():
                    result["summary_sha256"] = sha(summary)
                    result["summary"] = json.loads(summary.read_text())
                persist()
            return process.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    with lock:
        state["complete"] = True
        state["success"] = all(code == 0 for code in codes)
        persist()
    print(json.dumps(state), flush=True)
    if not state["success"]:
        raise SystemExit("At least one population failed; outputs retained without replacement")


if __name__ == "__main__":
    main()
