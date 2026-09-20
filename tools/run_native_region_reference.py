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
    parser.add_argument("--expected-binary-sha256", help="Require this exact reviewed executable hash before launching")
    args = parser.parse_args()
    assert args.replicates > 0 and args.samples > 0 and args.workers > 0
    root, config, binary = (p.resolve() for p in (args.root, args.config, args.binary))
    assert not root.exists(), "Refuse to restart an existing campaign"
    if args.expected_binary_sha256:
        assert sha(binary)==args.expected_binary_sha256, 'Unexpected executable; do not launch'
    assert args.cover_scales is not None or args.cover_weights is None, 'Cover weights need explicit scales'
    proposal=None
    if args.cover_scales is not None:
        scales=[float(x) for x in args.cover_scales.split(',')]
        weights=[float(x) for x in args.cover_weights.split(',')] if args.cover_weights else [1.]*len(scales)
        assert scales and all(math.isfinite(x) and 0<x<=1 for x in scales) and 1. in scales
        assert len(weights)==len(scales) and all(math.isfinite(x) and x>0 for x in weights)
        assert math.isfinite(sum(weights))
        proposal={'scales':scales,'weights':[x/sum(weights) for x in weights]}
    (root / "provenance").mkdir(parents=True)
    archived = root / "provenance/native-region-normalizer"
    shutil.copy2(binary, archived)
    if args.expected_binary_sha256:
        assert sha(archived)==args.expected_binary_sha256, 'Frozen executable hash changed'
    shutil.copy2(config, root / "provenance/input-config.json")
    shutil.copy2(__file__, root / "provenance/runner.py")
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
        jobs.append({"replicate": i, "seed": seed, "output": str(output), "command": command})
    manifest = {"protocol": "complete-native-cover-independent-v1", "jobs": jobs,
                "config_sha256": sha(config), "binary_sha256": sha(archived),
                "runner_sha256": sha(__file__), "total_unconditional_draws": args.replicates*args.samples,
                "proposal_override":proposal,
                "max_workers": args.workers, "created_utc": datetime.now(timezone.utc).isoformat(),
                "no_restart": "Never replace a zero, high-weight, or failed population according to its result"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    state = {"runner_pid": os.getpid(), "complete": False, "jobs": {}}
    lock = threading.Lock()

    def persist():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = root / "runner-status.tmp"
        tmp.write_text(json.dumps(state, indent=2)+"\n")
        tmp.replace(root / "runner-status.json")

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
