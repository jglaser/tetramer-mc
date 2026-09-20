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
    parser.add_argument("--q-min",type=float,default=0.,help="Original-q target lower boundary")
    parser.add_argument("--q-max",type=float,default=1.,help="Original-q target upper boundary; sets the complete geometric cover")
    parser.add_argument("--q-lower-open",action="store_true",help="Exclude the lower q boundary")
    parser.add_argument("--q-upper-open",action="store_true",help="Exclude the upper q boundary")
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
    from analyze_native_region_reference import DEFAULT_Q_WINDOW,validate_q_window,window_cover_metric
    q_window=validate_q_window(dict(minimum=args.q_min,maximum=args.q_max,
        lower_inclusive=not args.q_lower_open,upper_inclusive=not args.q_upper_open))
    windowed=q_window!=DEFAULT_Q_WINDOW
    root, config, binary = (p.resolve() for p in (args.root, args.config, args.binary))
    assert not root.exists(), "Refuse to restart an existing campaign"
    if windowed:
        cfg=json.loads(config.read_text())
        source_config=config
        source_config_sha256=sha(config)
        source_shape=Path(cfg['shape'])
        if not source_shape.is_absolute():source_shape=config.parent/source_shape
        source_shape=source_shape.resolve()
        source_shape_sha256=sha(source_shape)
        cover_metric=window_cover_metric({key:cfg['metadata'][key] for key in
            ('native_poses','rigid_members','member_error_scale','angle_error_scale_deg')},q_window)
        help_text=subprocess.run([str(binary),'--help'],check=True,capture_output=True,text=True).stdout
        assert all(flag in help_text for flag in ('--q-min','--q-max','--q-lower-open','--q-upper-open')), 'Binary lacks the q-window API'
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
    if windowed:
        assert sha(root/'provenance/input-config.json')==source_config_sha256, 'Source config changed during preparation'
        frozen_shape=root/'provenance/shape.json'
        shutil.copy2(source_shape,frozen_shape)
        assert sha(frozen_shape)==source_shape_sha256, 'Source shape changed during preparation'
        # Only relocate the shape path; physical poses, metric, domain and bath
        # remain those of the immutable original input configuration.
        cfg=json.loads((root/'provenance/input-config.json').read_text())
        cfg['shape']=str(frozen_shape)
        config=root/'provenance/config.json'
        config.write_text(json.dumps(cfg,indent=2)+'\n')
        shutil.copy2(Path(__file__).with_name('analyze_native_region_reference.py'),root/'provenance/analyze_native_region_reference.py')
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
        if windowed:
            command.extend(['--q-min',str(q_window['minimum']),'--q-max',str(q_window['maximum'])])
            if args.q_lower_open:command.append('--q-lower-open')
            if args.q_upper_open:command.append('--q-upper-open')
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
    if windowed:
        manifest.update(protocol='complete-original-q-window-independent-v1',q_window=q_window,
            cover_metric=cover_metric,
            target_rule='Original q metric is unchanged; only the declared window and enclosing proposal cover differ.',
            analyzer_sha256=sha(root/'provenance/analyze_native_region_reference.py'),
            source_config_path=str(source_config),source_config_sha256=source_config_sha256,
            source_shape_path=str(source_shape),shape_sha256=source_shape_sha256,
            frozen_config_path=str(config),frozen_shape_path=str(frozen_shape))
        manifest['archive_sha256']={name:sha(root/'provenance'/name) for name in
            ['input-config.json','config.json','shape.json','native-region-normalizer','runner.py','analyze_native_region_reference.py']}
        if guide:manifest['archive_sha256']['guide-model.json']=sha(frozen_model)
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
        if windowed:
            assert sha(config)==manifest['config_sha256'], 'Physical config changed after freezing the campaign'
            assert all(sha(root/'provenance'/name)==digest for name,digest in manifest['archive_sha256'].items()), 'Frozen campaign archive changed'
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
