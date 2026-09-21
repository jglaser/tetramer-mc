#!/usr/bin/env python3
"""Freeze, run and audit exact-sphere controls for the complete AB far region.

The run-controls action has a strict sphere-only allowlist. The separate
run-protein action starts the predeclared AB jobs only after controls pass.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from scipy.integrate import quad

from analyze_native_region_reference import native_q, rotate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'runs/ab-intermediate-expanded-atlas-repeat-preparation-20260920/config.json'
BINARY = ROOT/'runs/ab-shoulder-mixture-4x16384-l64-20260920/provenance/native-region-normalizer'
BUNDLE = BINARY.parent.parent/'runs/r00/provenance/source-bundle.json'
EXPECTED_CONFIG = 'cd44972bd266888f14673b84298ac49c56084e4226bbe4fbe9f61c8cf70575eb'
EXPECTED_SHAPE = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
EXPECTED_BINARY = '3a6a2dbba0ec5234c36cf66d7c40877336f22358a6a1ea91ea8366b344ee027b'
WINDOW = dict(minimum=5., maximum=37., lower_inclusive=True, upper_inclusive=False)
BETA = .99
POSE = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def require(ok, message):
    if not ok: raise ValueError(message)
def ball(r): return 4*math.pi*r**3/3
def haar_cap(a): return (a-math.sin(a))/math.pi


def norm_upper(vector):
    """Outward bound for the exact Euclidean norm of binary64 input values.

    Decimal products/sums round upward. sqrt is correctly rounded; next_plus
    moves outward even if its nearest rounding was downward. Conversion to
    binary64 receives one further outward step. This bounds input geometry,
    not every floating-point operation of the physical collision kernel.
    """
    require(all(math.isfinite(x) for x in vector), 'finite vector required')
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        squares = sum((Decimal.from_float(float(x))**2 for x in vector), Decimal(0))
        upper = squares.sqrt().next_plus()
        return math.nextafter(float(upper), math.inf)


def capture_proof(config):
    metric = config['metadata']
    require(metric['native_poses'] == [POSE], 'proof requires the recorded identity native reference')
    require(config['capture_center'] == POSE['position'], 'proof requires coincident native/capture centers')
    members = [p['position'] for p in metric['rigid_members']]
    require(members and all(len(p) == 3 for p in members), 'member coordinates required')
    require(all(sum(p[k] for p in members) == 0. for k in range(3)), 'centered members required')
    delta, alpha = metric['member_error_scale'], metric['angle_error_scale_deg']
    radius = config['capture_radius']
    require(all(math.isfinite(x) and x > 0 for x in (delta, alpha, radius)), 'positive finite geometry scales')
    radii = [math.hypot(*p) for p in members]
    rmax = max(norm_upper(p) for p in members)
    # q <= max[(R + 2 max|m|)/delta, 180/alpha]. The explicit slack is
    # deliberately much larger than binary64 error at this geometry scale.
    slack = 4096*math.ulp(1.)*(1+radius+2*rmax)/delta
    q_nominal = max((radius+2*max(radii))/delta, 180/alpha)
    upper = math.nextafter(max((radius+2*rmax)/delta, 180/alpha)+slack, math.inf)
    require(upper < WINDOW['maximum'], 'declared qmax does not enclose complete capture')
    proof = dict(formula='q <= max((capture_radius+2*max_member_radius)/member_error_scale,180/angle_error_scale_deg)',
                 member_radii_A=radii, outward_max_member_radius_A=rmax,
                 exact_kinematic_supremum_evaluated_FP64=q_nominal,
                 q_roundoff_slack=slack, guarded_q_upper=upper,
                 strict_qmax=WINDOW['maximum'], strict_margin=WINDOW['maximum']-upper,
                 scope='Analytic real-geometry cover with outward input-norm bounds and explicit FP64 slack; collision kernel is not formal interval arithmetic.')
    if max(radii) > 0:
        j = max(range(len(members)), key=radii.__getitem__)
        m = members[j]; r = radii[j]
        # A half-turn about an axis perpendicular to the extreme member maps
        # m to -m; placing its center at -R*m/r saturates the displacement bound.
        basis = [float(k == min(range(3), key=lambda i: abs(m[i]))) for k in range(3)]
        axis = [m[1]*basis[2]-m[2]*basis[1], m[2]*basis[0]-m[0]*basis[2], m[0]*basis[1]-m[1]*basis[0]]
        axis_norm = math.hypot(*axis)
        pose = dict(position=[-radius*x/r for x in m], orientation=[0.]+[x/axis_norm for x in axis])
        q = native_q(metric, pose)
        expected = max((radius+2*r)/delta, 180/alpha)
        require(abs(q-expected) < slack, 'half-turn saturation check failed')
        require(abs(math.hypot(*pose['position'])-radius) < slack*delta, 'capture-edge check failed')
        flipped = rotate(pose['orientation'], m)
        require(math.hypot(*[a+b for a,b in zip(flipped,m)]) < slack*delta, 'half-turn member check failed')
        proof['saturating_pose'] = dict(pose=pose, member_index=j, original_q=q,
            center_radius_A=math.hypot(*pose['position']),
            qualification='Kinematic capture-edge witness only; no hard-validity or Boltzmann-weight claim.')
    return proof


def sphere_overlap(r):
    return math.pi*(4+r)*(2-r)**2/12 if r < 2 else 0.


def sphere_exact(z):
    """Core radius .5, exclusion radius 1, capture radius3, metric delta=.25.

    q=max(4r,theta/15deg), so q>=5 iff r>=1.25 OR theta>=75deg.
    q<37 is automatic inside capture. Hard validity is r>=1.
    """
    require(math.isfinite(z) and z >= 0, 'nonnegative activity required')
    inner_angular_fraction = 1-haar_cap(math.radians(75.))
    regions = [(1., 1.25, inner_angular_fraction), (1.25, 2., 1.), (2., 3., 1.)]
    pieces=[]; errors=[]
    for lo,hi,angular in regions:
        value,error=quad(lambda r: angular*4*math.pi*r*r*math.exp(z*sphere_overlap(r)),
                         lo,hi,epsabs=1e-11,epsrel=1e-12)
        pieces.append(value);errors.append(error)
    return dict(Q=sum(pieces), radial_contributions=pieces, quadrature_error_bound_estimate=sum(errors),
                inner_angular_fraction=inner_angular_fraction,
                hard_exact=ball(3)-ball(1)-haar_cap(math.radians(75.))*(ball(1.25)-ball(1)))


def dormant_model(shape_sha):
    # This component is required by the existing model schema, but epsilon=1
    # removes it from both sampling and physical density. It is not learned.
    return dict(schema=1,coordinate_convention='anchor-body-relative',angular_length=1.,shape_sha256=shape_sha,
                anchors=[dict(position=[0.]*3,rotation=[[float(i==j) for j in range(3)] for i in range(3)])],
                means=[[0.]*6],covariances=[[[float(i==j) for j in range(6)] for i in range(6)]],weights=[1.])


def runner_command(root,config,model,binary,n,seed,workers):
    return [sys.executable,str(ROOT/'tools/run_native_region_reference.py'),'--root',str(root),
        '--config',str(config),'--binary',str(binary),'--expected-binary-sha256',EXPECTED_BINARY,
        '--replicates','4','--samples',str(n),'--seed-base',str(seed),'--workers',str(workers),
        '--q-min','5','--q-max','37','--q-upper-open','--cover-scales','1',
        '--model',str(model),'--model-weight',str(BETA),'--model-uniform-probability','1',
        '--model-anchor-index','0']


def prepare(out):
    require(not out.exists(), 'fresh output required')
    require(sha(CONFIG)==EXPECTED_CONFIG and sha(BINARY)==EXPECTED_BINARY, 'changed reviewed input')
    cfg=read(CONFIG); require(sha(cfg['shape'])==EXPECTED_SHAPE, 'changed AB shape')
    require(len(cfg['fixed_poses'])==2 and cfg['depletant_radius']==1.5 and cfg['reservoir_density']==.035,
            'changed AB physical target')
    bundle=read(BUNDLE); checked={}
    for name in ['src/native_region.rs','src/bin/native-region-normalizer.rs','src/proposal.rs','src/overlap_weight.rs','src/math.rs','src/geometry.rs']:
        record=bundle['files'][name]
        require(hashlib.sha256(record['text'].encode()).hexdigest()==record['sha256']==sha(ROOT/name), 'changed archived source '+name)
        checked[name]=record['sha256']
    archive=out/'provenance';archive.mkdir(parents=True)
    for name,path in [('original-AB-config.json',CONFIG),('AB-shape.json',Path(cfg['shape'])),
                      ('native-region-normalizer',BINARY),('source-bundle.json',BUNDLE),
                      ('prepare_far_capture_control.py',Path(__file__)),
                      ('run_native_region_reference.py',ROOT/'tools/run_native_region_reference.py'),
                      ('analyze_native_region_reference.py',ROOT/'tools/analyze_native_region_reference.py')]:
        shutil.copy2(path,archive/name)
    cfg['shape']=str(archive/'AB-shape.json');dump(out/'AB-config.json',cfg)
    dump(out/'AB-dormant-model.json',dormant_model(EXPECTED_SHAPE))
    proof=capture_proof(cfg); dump(out/'capture-proof.json',proof)
    shape=out/'sphere-shape.json';dump(shape,dict(name='analytic core-radius .5 sphere',volume=ball(.5),atoms=[dict(center=[0.]*3,radius=.5)]))
    model=out/'sphere-dormant-model.json';dump(model,dormant_model(sha(shape)))
    controls=[]
    for name,z,seed in [('sphere-zero',0.,105001010),('sphere-positive',.4,105101010)]:
        toy=dict(shape=str(shape),fixed_poses=[POSE],capture_center=[0.]*3,capture_radius=3.,depletant_radius=.5,
                 reservoir_density=z,poisson_lambda_ratio=64.,endpoint_gate=dict(max_cells=127,max_depth=10,min_width=.125),
                 metadata=dict(native_poses=[POSE],rigid_members=[POSE],member_error_scale=.25,angle_error_scale_deg=15.))
        config=out/(name+'-config.json');dump(config,toy)
        campaign=out/name;command=runner_command(campaign,config,model,archive/'native-region-normalizer',16384,seed,4)
        subprocess.run(command+['--prepare-only'],check=True,stdout=subprocess.DEVNULL)
        controls.append(dict(name=name,activity=z,root=str(campaign),command=command,exact=sphere_exact(z)))
    protein_root=out/'AB-flat-prepared'
    command=runner_command(protein_root,out/'AB-config.json',out/'AB-dormant-model.json',archive/'native-region-normalizer',65536,105201010,4)
    subprocess.run(command+['--prepare-only'],check=True,stdout=subprocess.DEVNULL)
    import shlex
    (out/'AB-command.txt').write_text('# Already prepared; do not rerun the preparer over this directory.\n'
        '# The frozen manifest contains four exact executable commands. No AB jobs have been launched.\n'
        '# Equivalent fresh-directory runner invocation:\n'+shlex.join(command)+'\n')
    protocol=dict(schema='far-complete-capture-flat-control-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        q_window=WINDOW,proposal=dict(cover_weight=1-BETA,guide_weight=BETA,guide_uniform_probability=1.,cover_scales=[1.],
        convention='unconditional normalized ball/Haar cover plus cube/Haar; dormant Gaussian has exactly zero weight'),
        physical_AB={k:cfg[k] for k in ['fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density','metadata']},
        capture_proof_sha256=sha(out/'capture-proof.json'),reviewed_source_sha256=checked,
        input_sha256={str(CONFIG):EXPECTED_CONFIG,str(BINARY):EXPECTED_BINARY,str(Path(read(CONFIG)['shape'])):EXPECTED_SHAPE},
        controls=controls,protein=dict(root=str(protein_root),command=command,prepared_only=True,
            samples_per_population=65536,populations=4,seeds=[105201010+1009*i for i in range(4)]),
        acceptance_rule='All-row density/q/hash/Poisson and analytic geometry audits must pass; both analytic Q comparisons must be within six observed row SE. No retry, replacement or optional stopping.',
        uncertainty='Observed row and population errors are diagnostics, not unseen-tail guarantees. Sphere controls validate implementation, not protein sampling efficiency.',
        max_control_workers=8)
    dump(out/'protocol.json',protocol)
    # Freeze every file present before physical sampling, including all three
    # prepared manifests and executable commands. Runner status is mutable.
    sealed={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='runner-status.json'}
    dump(out/'freeze.json',dict(files=sealed))
    print(json.dumps(dict(prepared=str(out),protocol_sha256=sha(out/'protocol.json'),guarded_q_upper=proof['guarded_q_upper'],protein_launched=False)),flush=True)


def verify(out):
    frozen=read(out/'freeze.json')
    for rel,digest in frozen['files'].items():require(sha(out/rel)==digest,'changed frozen file '+rel)
    return read(out/'protocol.json')


def run_controls(out,workers):
    require(1<=workers<=8,'one to eight workers')
    p=verify(out);require(not (out/'control-run-status.json').exists(),'controls already started; no retry')
    jobs=[]
    for c in p['controls']:
        require(c['name'] in ('sphere-zero','sphere-positive'),'sphere-only launch allowlist')
        root=Path(c['root']);require(root.parent==out,'unexpected control root')
        for job in read(root/'manifest.json')['jobs']:jobs.append((c['name'],root,job))
    status=dict(complete=False,started_utc=datetime.now(timezone.utc).isoformat(),jobs={})
    lock=threading.Lock();dump(out/'control-run-status.json',status)
    def run(item):
        name,root,job=item;label=f"{name}-r{job['replicate']:02d}"
        require(not Path(job['output']).exists(),'physical output already exists')
        with (root/'logs'/f"r{job['replicate']:02d}.log").open('x') as log:
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
            with lock:
                status['jobs'][label]=dict(pid=process.pid,status='running',seed=job['seed']);dump(out/'control-run-status.json',status)
            rc=process.wait()
        with lock:
            status['jobs'][label].update(status='complete' if rc==0 else 'failed',returncode=rc);dump(out/'control-run-status.json',status)
        return rc
    with ThreadPoolExecutor(max_workers=workers) as pool: codes=list(pool.map(run,jobs))
    status.update(complete=True,success=all(rc==0 for rc in codes));dump(out/'control-run-status.json',status)
    require(status['success'],'control process failed; retain all outputs')
    for c in p['controls']:
        root=Path(c['root']);dump(root/'runner-status.json',dict(complete=True,success=True,prepared_only=False,
            jobs={k:v for k,v in status['jobs'].items() if k.startswith(c['name'])},driver=str(Path(__file__))))
    require(not any(Path(j['output']).exists() for j in read(Path(p['protein']['root'])/'manifest.json')['jobs']),'AB unexpectedly launched')
    print(json.dumps(dict(controls_terminal=True,jobs=len(jobs),protein_launched=False)),flush=True)


def audit(out):
    p=verify(out); require(read(out/'control-run-status.json').get('success'),'controls must finish successfully')
    require(not (out/'analysis.json').exists(),'preserve existing analysis')
    reports={}
    for c in p['controls']:
        root=Path(c['root']); assessment=root/'assessment-streaming.json'
        require(not assessment.exists(),'do not duplicate a completed audit')
        with (root/'audit.log').open('x') as log:
            subprocess.run([sys.executable,str(root/'provenance/analyze_native_region_reference.py'),str(root),'--workers','4'],check=True,stdout=log,stderr=subprocess.STDOUT)
        a=read(assessment); n=a['samples']; region=a['regions']['region']; Q=math.exp(region['log_normalizer']);se=Q*region['relative_SE']
        exact=c['exact']['Q']; require(abs(Q-exact)<=6*se,'analytic sphere mean differs by over six observed row SE')
        max_density_error=0.;max_envelope_violation=0.;counts=dict(q=0,capture=0,hard=0,valid=0); hashes={}
        for job in read(root/'manifest.json')['jobs']:
            directory=Path(job['output']); rows=directory/'samples.jsonl'; hashes[str(rows)]=sha(rows)
            cover=read(directory/'cover.json'); R=cover['ball_radius']
            metric=read(directory/'provenance/config.json')['metadata']
            require(R==9.25 and abs(cover['angle_cap']-math.pi)<1e-14,'toy full-Haar cover mismatch')
            with rows.open() as handle:
                for row in map(json.loads,handle):
                    pose=row['pose'];r=math.hypot(*pose['position'])
                    in_cube=all(-3<=x<3 for x in pose['position'])
                    g=(BETA/6**3 if in_cube else 0.)+((1-BETA)/ball(R) if r<=R else 0.)
                    require(g>0,'draw outside flat mixture')
                    error=abs(math.log(g)-row['log_proposal_density']);max_density_error=max(error,max_density_error)
                    require(error<2e-11,'independent epsilon-one mixture density')
                    q=native_q(metric,pose)
                    expected='q' if not 5<=q<37 else 'capture' if r>3 else 'hard' if r<1 else None
                    require(row.get('zero')==expected,'analytic sphere geometry zero reason')
                    counts[expected or 'valid']+=1
                    if expected is None:
                        C=sphere_overlap(r)
                        violation=max(row['lower_volume']-C,C-row['upper_volume'],0.)
                        max_envelope_violation=max(violation,max_envelope_violation)
                        require(violation<2e-10,'analytic overlap outside envelope')
                    else:require('log_importance_weight' not in row,'invalid draw has nonzero weight')
        require(sum(counts.values())==n and all(counts[k]>0 for k in counts),'missing zero categories or rows')
        require(a['guide_uniform_count'] if 'guide_uniform_count' in a else all(x['guide_uniform_count']>0 for x in a['populations']),'no uniform branch')
        require(all(sum(x['guide_component_counts'])==0 for x in a['populations']),'epsilon-one learned draws')
        reports[c['name']]=dict(activity=c['activity'],samples=n,estimate_Q=Q,row_SE=se,
            row_relative_SE=region['relative_SE'],population_relative_SE=a['independent_population_relative_SE'],
            exact_Q=exact,difference_in_row_SE=(Q-exact)/se,counts=counts,
            maximum_point_fraction=region['maximum_point_fraction'],weight_ESS=region['ess'],
            max_density_log_error=max_density_error,max_analytic_envelope_violation=max_envelope_violation,
            audit_sha256=sha(assessment),sample_sha256=hashes,cpu_seconds=a['cpu_seconds'])
    result=dict(complete=True,controls_passed=True,controls=reports,protocol_sha256=sha(out/'protocol.json'),
        freeze_sha256=sha(out/'freeze.json'),capture_proof=read(out/'capture-proof.json'),
        protein_launched=False,scope=p['uncertainty'])
    require(not any(Path(j['output']).exists() for j in read(Path(p['protein']['root'])/'manifest.json')['jobs']),'AB unexpectedly launched')
    dump(out/'analysis.json',result);print(json.dumps(result,indent=2),flush=True)


def run_protein(out,workers):
    require(1<=workers<=4,'one to four AB workers')
    p=verify(out);controls=read(out/'analysis.json')
    require(controls['complete'] and controls['controls_passed'] and
            controls['protocol_sha256']==sha(out/'protocol.json'), 'matching completed controls required')
    root=Path(p['protein']['root']); manifest=read(root/'manifest.json')
    require(not (out/'protein-run-status.json').exists(),'protein campaign already started; no retry')
    require(len(manifest['jobs'])==4 and manifest['total_unconditional_draws']==4*65536,'changed AB budget')
    require([j['seed'] for j in manifest['jobs']]==[105201010+1009*i for i in range(4)],'changed AB streams')
    status=dict(complete=False,started_utc=datetime.now(timezone.utc).isoformat(),
                controls_sha256=sha(out/'analysis.json'),jobs={})
    lock=threading.Lock();dump(out/'protein-run-status.json',status)
    def run(job):
        require(not Path(job['output']).exists(),'AB output already exists')
        label=f"r{job['replicate']:02d}"
        with (root/'logs'/f'{label}.log').open('x') as log:
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
            with lock:
                status['jobs'][label]=dict(pid=process.pid,status='running',seed=job['seed']);dump(out/'protein-run-status.json',status)
            rc=process.wait()
        with lock:
            status['jobs'][label].update(status='complete' if rc==0 else 'failed',returncode=rc);dump(out/'protein-run-status.json',status)
        return rc
    with ThreadPoolExecutor(max_workers=workers) as pool:codes=list(pool.map(run,manifest['jobs']))
    status.update(complete=True,success=all(rc==0 for rc in codes));dump(out/'protein-run-status.json',status)
    dump(root/'runner-status.json',dict(status,prepared_only=False,driver=str(Path(__file__))))
    require(status['success'],'AB process failed; retain every output')
    print(json.dumps(dict(protein_terminal=True,jobs=len(codes),root=str(root))),flush=True)


def audit_protein(out):
    """Reuse one complete archived audit, then independently check its extremes."""
    p=verify(out);require(read(out/'protein-run-status.json').get('success'),'AB must finish successfully')
    destination=out/'AB-analysis.json';require(not destination.exists(),'preserve AB analysis')
    root=Path(p['protein']['root']);assessment=root/'assessment-streaming.json'
    if not assessment.exists():
        with (root/'audit.log').open('x') as log:
            subprocess.run([sys.executable,str(root/'provenance/analyze_native_region_reference.py'),str(root),'--workers','4'],
                           check=True,stdout=log,stderr=subprocess.STDOUT)
    a=read(assessment);manifest=read(root/'manifest.json');cfg=read(root/'provenance/config.json')
    require(a['complete'] and a['all_rows_and_hashes_validated'] and a['independent_q_and_density_rows']==4*65536,
            'incomplete independent AB audit')
    require(a['analysis_script_sha256']==sha(root/'provenance/analyze_native_region_reference.py'),'wrong density auditor')
    require(a['q_window']==p['q_window']==WINDOW and a['samples']==4*65536,'changed AB region/budget')
    require(all(cfg[k]==v for k,v in p['physical_AB'].items()),'changed AB physical configuration')
    require(sha(root/'provenance/shape.json')==EXPECTED_SHAPE,'changed AB shape')
    require(a['guide']['uniform_probability']==1. and a['guide']['weight']==BETA,'changed flat mixture')
    cover=a['cover_mixture']['covers'][0]
    require(a['cover_mixture']['scales']==[1.] and cover['ball_radius']==74. and
            abs(cover['angle_cap']-math.pi)<1e-14,'changed complete AB cover')
    hashes={}
    for job,record in zip(manifest['jobs'],a['populations']):
        path=Path(job['output'])/'samples.jsonl';hashes[str(path)]=sha(path)
        require(hashes[str(path)]==record['sample_sha256'],'rows changed since full audit')
        require(sum(record['guide_component_counts'])==0,'epsilon-one sampled learned branch')
    from prepare_native_confirmation_atlas import AtomUnionAudit
    atom=AtomUnionAudit(read(root/'provenance/shape.json'),cfg['fixed_poses']);top=[]
    total=a['regions']['region'];log_sum=total['log_normalizer']+math.log(a['samples'])
    for point in a['top_weights'][:8]:
        gaps=atom.gaps(point['pose']);require(min(gaps)>=0,'top pose fails independent all-atom hard check')
        top.append(dict(point,minimum_atomic_gap_by_neighbor_A=gaps,
                        fraction_of_total=math.exp(point['log_importance_weight']-log_sum)))
    result=dict(complete=True,full_far_region=True,q_window=WINDOW,independent_q_and_density_rows=a['samples'],
        physical=total,population_relative_SE=a['independent_population_relative_SE'],hard=a['hard_regions']['region'],
        hard_population_relative_SE=a['hard_independent_population_relative_SE'],
        counts={key:sum(r['counts'][key] for r in a['populations']) for key in ['q','capture','hard','valid']},
        population_logQ=[r['regions']['region']['log_normalizer'] for r in a['populations']],
        population_mass_fractions=[math.exp(r['regions']['region']['log_normalizer']-total['log_normalizer'])/4
                                   if r['regions']['region']['log_normalizer'] is not None else 0. for r in a['populations']],
        CPU_seconds=a['cpu_seconds'],raw_cloud_points=a['raw_cloud_points'],top_poses=top,
        independent_atom_check_count=len(top),full_density_audit_path=str(assessment),full_density_audit_sha256=sha(assessment),
        sample_sha256=hashes,protocol_sha256=sha(out/'protocol.json'),capture_proof_sha256=sha(out/'capture-proof.json'),
        analysis_driver_sha256=sha(__file__),atom_auditor_sha256=sha(ROOT/'tools/prepare_native_confirmation_atlas.py'),
        scope='Complete physical support; observed far contact mass is dominated by a few draws. No convergence or missing-mass upper-bound claim; no pooling with local measurements.')
    dump(destination,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run-controls','audit','run-protein','audit-protein']);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();out=args.out.resolve()
    if args.action=='prepare':prepare(out)
    elif args.action=='run-controls':run_controls(out,args.workers)
    elif args.action=='run-protein':run_protein(out,args.workers)
    elif args.action=='audit-protein':audit_protein(out)
    else:audit(out)
