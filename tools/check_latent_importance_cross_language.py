#!/usr/bin/env python3
"""Run a bounded synthetic-sphere launcher/normalizer/auditor integration check.

Exactly two fresh populations of 256 draws, two clouds, and at most two workers.
This is a serialization and estimator-accounting control, not a protein campaign
or a proposal-efficiency measurement. Existing output directories are refused.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
SEEDS=[120101010,120102019]
SAMPLES=256


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fixture(directory):
    """Same analytic AO sphere/chart as tests/latent_region_importance.rs."""
    directory.mkdir()
    fixed=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])
    native=dict(position=[1.,0.,0.],orientation=[1.,0.,0.,0.])
    shape=dict(name='sphere',volume=4*math.pi*.3**3/3,atoms=[dict(center=[0.,0.,0.],radius=.3)])
    write(directory/'shape.json',shape)
    metadata=dict(native_poses=[native],rigid_members=[fixed],member_error_scale=1.,angle_error_scale_deg=15.)
    config=dict(shape=str(directory/'shape.json'),fixed_poses=[fixed],initial_pose=native,
        capture_center=[0.,0.,0.],capture_radius=4.,depletant_radius=.4,reservoir_density=2.,
        poisson_lambda_ratio=64.,translation_steps=[.1],rotation_steps_deg=[1.],rotation_probability=.5,
        local_attempts_per_cycle=1,uniform_probability=.1,seed=1,metadata=metadata)
    write(directory/'config.json',config)
    identity=lambda n:[[float(i==j) for j in range(n)] for i in range(n)]
    region=dict(fixed_neighbor=fixed,physical_fixed_neighbors=[fixed],capture_center=[0.,0.,0.],capture_radius=4.,
        shape_sha256=sha(directory/'shape.json'),activity=2.,depletant_radius=.4,physical_metric=metadata,
        minimum_original_q=0.,minimum_mahalanobis_radius=1.2,mahalanobis_radius=2.4,
        gaussian_chart=dict(shape_sha256=sha(directory/'shape.json'),angular_length=1.,coordinate_convention='anchor-body-relative',
            anchors=[dict(position=[0.,0.,0.],rotation=identity(3))],means=[[0.]*6],covariances=[identity(6)],weights=[1.]))
    write(directory/'region.json',region)
    guide=dict(schema='defensive-latent-shell-guide-v1',region_sha256=sha(directory/'region.json'),
        defensive_uniform_shell_probability=.5,gaussian_components=[dict(weight=1.,mean=[0.]*6,
            covariance=[[4.*x for x in row] for row in identity(6)])])
    write(directory/'guide.json',guide)


def run(out,validation):
    out,validation=Path(out).resolve(),Path(validation).resolve()
    assert not out.exists(), 'Use a fresh synthetic output directory'
    reviewed=read(validation)
    assert reviewed['complete'] and all(c['exit_code']==0 for c in reviewed['commands'])
    binary_paths=[name for name in reviewed['source_sha256'] if name.endswith('/latent-region-normalizer')]
    assert len(binary_paths)==1
    binary=ROOT/binary_paths[0];binary_sha=reviewed['source_sha256'][binary_paths[0]]
    assert sha(binary)==binary_sha
    bundle_path=ROOT/reviewed['source_bundle'];bundle_raw=bundle_path.read_bytes()
    assert sha(bundle_path)==reviewed['source_bundle_sha256'] and bundle_raw in binary.read_bytes()
    bundle=json.loads(bundle_raw)
    assert len(bundle['files'])==reviewed['embedded_source_files']
    for name,entry in bundle['files'].items():
        assert hashlib.sha256(entry['text'].encode()).hexdigest()==entry['sha256']==sha(ROOT/name)
    for name,digest in reviewed['source_sha256'].items():assert sha(ROOT/name)==digest
    out.mkdir(parents=True);inputs=out/'provenance';inputs.mkdir()
    shutil.copy2(validation,inputs/'reviewed-validation.json')
    shutil.copy2(bundle_path,inputs/'source-bundle.json')
    shutil.copy2(ROOT/'tests/latent_region_importance.rs',inputs/'latent_region_importance.rs')
    shutil.copy2(__file__,inputs/Path(__file__).name)
    fixture(out/'fixture')
    campaign=out/'campaign'
    launcher=[sys.executable,'-B',str(ROOT/'tools/run_latent_region_campaign.py'),
        '--out',str(campaign),'--config',str(out/'fixture/config.json'),
        '--region',str(out/'fixture/region.json'),'--binary',str(binary),
        '--importance-guide',str(out/'fixture/guide.json'),'--samples',str(SAMPLES),
        '--replicates','2','--workers','2','--seed',str(SEEDS[0]),'--lambda-ratio','64','--cloud-replicates','2']
    auditor=[sys.executable,'-B',str(campaign/'provenance/analyze_latent_region.py'),'--root',str(campaign)]
    status=dict(schema='latent-importance-cross-language-check-v1',complete=False,started=now(),
        reviewed_validation_sha256=sha(validation),binary_sha256=binary_sha,source_bundle_sha256=sha(bundle_path),
        seeds=SEEDS,samples_per_population=SAMPLES,total_samples=2*SAMPLES,workers=2,cloud_replicates=2,
        commands=[dict(role='launcher',argv=launcher),dict(role='auditor',argv=auditor)],
        fixture_sha256={p.name:sha(p) for p in (out/'fixture').iterdir()},
        scope='New synthetic sphere integration control only. Exactly two fixed-size populations; no retries and one auditor invocation. Serialization, bindings and unconditional estimator accounting are checked; 512 draws do not establish variance reduction or CPU speedup.')
    write(out/'validation.json',status)
    try:
        with (out/'launcher.log').open('xb') as log:
            result=subprocess.run(launcher,stdout=log,stderr=subprocess.STDOUT)
        status['commands'][0]['exit_code']=result.returncode;write(out/'validation.json',status)
        result.check_returncode()
        manifest=read(campaign/'manifest.json');archive=campaign/'provenance'
        assert manifest['schema']=='importance-latent-region-campaign-v1'
        assert (manifest['workers'],manifest['cloud_replicates'],manifest['lambda_ratio'])==(2,2,64.)
        assert [j['seed'] for j in manifest['jobs']]==SEEDS and all(j['samples']==SAMPLES for j in manifest['jobs'])
        assert sha(archive/'latent-region-normalizer')==binary_sha
        assert sha(archive/'region.json')==sha(out/'fixture/region.json')==manifest['region_sha256']
        assert sha(archive/'importance-guide.json')==sha(out/'fixture/guide.json')==manifest['importance_guide_sha256']
        assert sha(archive/'shape.json')==sha(out/'fixture/shape.json')
        for name,digest in manifest['archive_sha256'].items():assert sha(archive/name)==digest
        jobs=[]
        for job in manifest['jobs']:
            directory=Path(job['directory']);summary=read(directory/'summary.json');m=summary['manifest']
            assert read(campaign/f"{job['id']}-status.json")['returncode']==0
            assert summary['complete'] and summary['samples']==SAMPLES and m==read(directory/'manifest.json')
            assert m['config_sha256']==sha(archive/'config.json')==sha(directory/'provenance/input-config.json')
            assert m['executable_sha256']==binary_sha and m['source_bundle_sha256']==sha(bundle_path)
            assert sha(directory/'provenance/source-bundle.json')==sha(bundle_path)
            assert m['region_sha256']==manifest['region_sha256'] and m['importance_guide_sha256']==manifest['importance_guide_sha256']
            rows=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
            assert len(rows)==SAMPLES and [r['draw'] for r in rows]==list(range(SAMPLES))
            # A one-atom sphere admits an independent exact hard/capture check.
            for row in rows:
                center=math.hypot(*row['pose']['position'])
                assert row['capture_valid']==(center<=4.)
                assert row['hard_valid']==(center<=4. and center>=.6)
                if not row['shell_valid']:
                    assert row['log_importance_weight'] is None and row['log_hard_weight'] is None and not row['clouds']
            outside=sum(not r['shell_valid'] for r in rows)
            assert outside>0 and any(r['log_importance_weight'] is not None for r in rows)
            jobs.append(dict(id=job['id'],seed=job['seed'],samples=len(rows),outside_shell=outside,
                nonzero=sum(r['log_importance_weight'] is not None for r in rows),
                samples_sha256=sha(directory/'samples.jsonl'),summary_sha256=sha(directory/'summary.json')))
        status['jobs']=jobs;status['manifest_sha256']=sha(campaign/'manifest.json');write(out/'validation.json',status)
        with (out/'auditor.log').open('xb') as log:
            result=subprocess.run(auditor,stdout=log,stderr=subprocess.STDOUT)
        status['commands'][1]['exit_code']=result.returncode;write(out/'validation.json',status)
        result.check_returncode()
        assessment=read(campaign/'assessment/analysis.json')
        assert assessment['estimate']['draws']==2*SAMPLES and assessment['independently_reconstructed_poses']==2*SAMPLES
        assert len(assessment['populations'])==2
        assert assessment['importance_sampling']['guide_sha256']==manifest['importance_guide_sha256']
        status.update(complete=True,finished=now(),assessment_sha256=sha(campaign/'assessment/analysis.json'),
            importance_sampling_audit=assessment['importance_sampling'],
            maximum_independent_jacobian_error=assessment['independent_jacobian_reconstruction_max_error'])
        write(out/'validation.json',status)
        (out/'report.md').write_text(
            '# Synthetic latent-importance cross-language check\n\n'
            'Both 256-draw sphere populations and the single archived Python audit passed. '
            f"All 512 unconditional rows were reconstructed, including {sum(j['outside_shell'] for j in jobs)} outside-shell zeros.\n\n"
            'The reviewed executable, embedded source bundle, frozen guide, region, shape and relocated config were hash-checked. '
            'The exact single-sphere hard/capture predicates were also checked independently.\n\n'
            'This validates serialization, launcher dispatch and full-density/Jacobian accounting. '
            'The small sample count supplies no proposal variance or CPU-speed comparison.\n')
        write(out/'artifact-sha256.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        return status
    except BaseException as error:
        status.update(complete=False,finished=now(),error=repr(error));write(out/'validation.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--validation',type=Path,default=ROOT/'runs/mobile-outer-importance-validation-20260921/validation.json')
    args=parser.parse_args();result=run(args.out,args.validation)
    print(json.dumps(dict(complete=result['complete'],samples=result['total_samples'],
        outside_shell=sum(j['outside_shell'] for j in result['jobs']),assessment_sha256=result['assessment_sha256']),indent=2))


if __name__=='__main__':main()
