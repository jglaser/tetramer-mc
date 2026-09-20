#!/usr/bin/env python3
"""Prepare and audit an unchanged-binary A-neighbor q<=2 SMC reference.

The original q uses 2 Angstrom / 15 degree tolerances. Only the archived
executable's internal region scales change. The positive uniform branch is
uniform in the capture BALL, not a cube. No production is launched by prepare.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import beta

from audit_previous_smc_regions import recompute_q

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity')
EXPECTED = {
    'implementation/coordination_smc': '9422a048f80476845510de71b1e8b97813846e521a1d10040d8a4efa1c4421c9',
    'implementation/src/bin/coordination_smc.rs': 'f216b5885e063c4a3c14109424b1ddeb071114b3b43f71cc355b79f292d33487',
    'implementation/src/single_body_depletion.rs': '0388e24a6c68b54874f6200e6278dc1fda607c6d5f42558a73740ea70b1680b7',
    'implementation/src/rigid_pose_sampling.rs': '927c077af994a5b5000212f82ba407955bf0a644e25f59dc7dc52adbadad4264',
    'inputs/tetramer-shape.json': 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9',
    'inputs/geometry/environment-site0-m1.json': 'a809999cd6c004028a25007997ebbebaca4cd264b6e7c2c19c8513bb2bba29c4',
}
ORIGINAL_METRIC = {'member_error_scale': 2., 'angle_error_scale_deg': 15.}
SEED_BASE = 104001010


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')
def require(b, message):
    if not b: raise ValueError(message)


def ball_volume(r): return 4 * math.pi * r**3 / 3

def haar_cap(degrees):
    a = math.radians(degrees)
    return (a - math.sin(a)) / math.pi


def flat_density(cfg, env):
    """Constant on a SINGLE native-cap target, not a generally constant g."""
    comps = cfg['proposal_components']
    require(len(comps) == 1, 'flat target needs exactly one envelope')
    c = comps[0]
    return ((1-cfg['uniform_probability']) /
            (ball_volume(c['position_radius']) * haar_cap(c['angle_radius_deg']))
            + cfg['uniform_probability'] / ball_volume(env['capture_radius']))


def rescale_config(source, env, population, seed):
    cfg = copy.deepcopy(source)
    refs = env.get('native_poses') or [env['native_pose']]
    members = np.asarray([p['position'] for p in env['rigid_members']])
    require(len(refs) == 1, 'multiple references need a nonconstant-density audit')
    require(np.array_equal(members.mean(axis=0), np.zeros(3)), 'centered members required')
    require(source['member_error_scale'] == 2. and source['angle_error_scale_deg'] == 15., 'source metric changed')
    cfg.update(basin='native', member_error_scale=4., angle_error_scale_deg=30.,
               population=population, seed=seed, threads=1, uniform_probability=.01)
    radius = math.nextafter(4., math.inf)
    cfg['proposal_components'] = [dict(refs[0], position_radius=radius, angle_radius_deg=30., weight=1.)]
    # Explicit components override these, but keep both representations consistent.
    cfg['proposal_position_radii'] = [radius]
    cfg['proposal_angle_radii_deg'] = [30.]
    cfg['metadata'] = {
        'control': 'A-neighbor original q<=2; original native q<=1 and shoulder 1<q<=2',
        'original_metric': ORIGINAL_METRIC,
        'stored_q_to_original_q': 2.,
        'uniform_branch': 'capture ball with full normalized proper SO(3) Haar',
        'physical_target': 'H I_capture I_original_q<=2 exp(z C_A)',
        'initialization': 'fixed unconditional proposal budget; no retry of zero hits',
        'g_on_target_A_minus3': flat_density(cfg, env),
        'terminal_estimator': 'Zhat times terminal indicator mean; average populations on linear scale',
    }
    return cfg


def sphere_exact(z):
    """Core radius .5, exclusion radius 1, hard shell 1<r<3."""
    def overlap(r): return math.pi * (4+r) * (2-r)**2 / 12 if r < 2 else 0.
    inside = quad(lambda r: 4*math.pi*r*r*math.exp(z*overlap(r)), 1., 2., epsabs=1e-12)[0]
    outside = ball_volume(3.) - ball_volume(2.)
    native = haar_cap(15.) * inside
    total = haar_cap(30.) * (inside + outside)
    return {'native': native, 'shoulder': total-native, 'total': total}


def population_statistics(rows):
    """Rows [native mass, shoulder mass]; includes zeros, retains covariance."""
    x = np.asarray(rows, dtype=float)
    require(x.ndim == 2 and x.shape[1] == 2 and len(x) >= 2, 'need independent mass pairs')
    require(np.isfinite(x).all() and (x >= 0).all(), 'invalid mass')
    mean = x.mean(axis=0)
    covariance = np.cov(x, rowvar=False, ddof=1) / len(x)
    totals = x.sum(axis=1)
    require(abs(covariance.sum() - totals.var(ddof=1)/len(x)) < 1e-10 * max(1., totals.var(ddof=1)), 'lost covariance')
    values = {}
    for i, name in enumerate(['native', 'shoulder', 'total']):
        a = x[:, i] if i < 2 else totals
        m, se = float(a.mean()), float(a.std(ddof=1)/math.sqrt(len(a)))
        values[name] = {'mean_Q': m, 'Q_se': se, 'relative_se': se/m if m else None,
                        'log_mean_Q': math.log(m) if m else None, 'population_Q': a.tolist()}
    return {'regions': values, 'covariance_of_mean_native_shoulder': covariance.tolist(),
            'population_count': len(x)}


def matched_references(out):
    """Re-mask independent historical sources; do not pool their proposals."""
    source=read(out/'provenance/original-config.json'); env=read(source['environment'])
    answer={'definition':'original q<=1 and 1<q<=2, both bound and unbound; A only',
            'sources':{},'scope':'Raw hashes and positive Poisson factors checked; full proposal density reused from the archived estimator. Observed errors do not bound unseen contact mass.'}
    for name in ['basin-normalizer-importance-guided-16384-l64','basin-normalizer-mis-refined-16384-l64']:
        root=ROOT/'runs'/name; archived=read(root/'assessment/analysis.json'); cfg=read(root/'provenance/config.json')
        require(not archived['pending'],'reference unfinished')
        checked_hashes={}
        for path,expected in archived['provenance'].items():
            # Historic analysis records the then-live tool path. Its immutable
            # campaign archive is the source; the current tool may have evolved.
            actual = root/'provenance/analyze_basin_normalizers.py' if Path(path)==ROOT/'tools/analyze_basin_normalizers.py' else Path(path)
            require(sha(actual)==expected,f'reference changed {actual}')
            checked_hashes[str(actual)]=expected
        campaign=read(root/'manifest.json')
        for path,expected in campaign['archive_sha256'].items():
            require(sha(root/'provenance'/path)==expected,f'archived reference input changed {path}')
            checked_hashes[str(root/'provenance'/path)]=expected
        for key in ['fixed_poses','capture_center','capture_radius']:
            require(cfg[key]==env[key],f'reference environment differs: {key}')
        for key in ['depletant_radius','reservoir_density']:
            require(cfg[key]==source[key],f'reference physical parameter differs: {key}')
        require(sha(cfg['shape'])==sha(source['shape']),'reference shape mismatch')
        require(cfg['metadata']['rigid_members']==env['rigid_members'],'reference member geometry mismatch')
        require(cfg['metadata']['native_poses']==(env.get('native_poses') or [env['native_pose']]),'reference native list mismatch')
        require(all(cfg['metadata'][k]==v for k,v in ORIGINAL_METRIC.items()),'reference metric mismatch')
        arrays=[]; populations=[]; max_q_error=0.
        for population in archived['populations']:
            job=population['job']; path=Path(job['directory'])/'samples.jsonl'
            manifest=read(path.parent/'manifest.json')
            require(manifest['seed']==job['seed'] and manifest['samples']==job['samples'],'reference seed/budget')
            weights=np.zeros((job['samples'],2))
            with path.open() as f:
                for i,line in enumerate(f):
                    row=json.loads(line); require(row['draw']==i and i<len(weights),'reference row ordering')
                    if not row['hard_valid']:
                        require(row['log_importance_weight'] is None and not row['clouds'],'invalid row contributes')
                        continue
                    q=recompute_q(row['pose'],env,ORIGINAL_METRIC)
                    max_q_error=max(max_q_error,abs(q-row['q']))
                    require(max_q_error<1e-10,'reference q mismatch')
                    clouds=row['clouds'];require(len(clouds)==2,'reference cloud budget')
                    for c in clouds:
                        expected=manifest['activity']*c['lower_volume']+c['overlap_points']*math.log1p(manifest['activity']/manifest['lambda'])
                        require(abs(c['log_weight']-expected)<1e-10,'reference Poisson factor')
                    logw=float(logsumexp([c['log_weight'] for c in clouds])-math.log(2)-row['log_proposal_density'])
                    require(abs(logw-row['log_importance_weight'])<1e-10,'reference full weight')
                    if q<=2.:weights[i,0 if q<=1. else 1]=math.exp(logw)
            require(i+1==len(weights),'reference draw count')
            arrays.append(weights);populations.append(weights.mean(axis=0))
        x=np.concatenate(arrays); covariance=np.cov(x,rowvar=False,ddof=1)/len(x)
        stats=population_statistics(populations)
        for i,key in enumerate(['native','shoulder','total']):
            values=x[:,i] if i<2 else x.sum(axis=1)
            record=stats['regions'][key]
            require(abs(record['mean_Q']/float(values.mean())-1)<1e-12,'reference population mean mismatch')
            record['row_relative_se']=float(values.std(ddof=1)/math.sqrt(len(values))/values.mean())
            if i<2:
                require(abs(record['log_mean_Q']-archived['groups']['1.0']['estimates'][key]['logQ'])<1e-10,'original regional mean changed')
        stats.update(row_count=len(x),row_covariance_of_mean_native_shoulder=covariance.tolist(),max_q_error=max_q_error,
                     source_analysis_sha256=sha(root/'assessment/analysis.json'),source=str(root),
                     source_input_hashes=checked_hashes)
        answer['sources'][name]=stats
    answer['analyzer_sha256']=sha(__file__)
    destination=out/'matched-independent-references.json';require(not destination.exists(),'preserve reference assessment')
    dump(destination,answer);print(json.dumps({'reference_assessment':str(destination)}))


def initialization_coverage(out):
    cfg=read(out/'configs/protein-zero.json'); path=out/'results/protein-zero/summary.json';s=read(path)
    require(s['complete'] and cfg['reservoir_density']==0.,'completed zero control required')
    env=read(cfg['environment']);g=flat_density(cfg,env);n=cfg['initial_draws'];hits=s['initialization']['hits']
    volume=ball_volume(env['capture_radius']); hashes={str(path):sha(path)}
    olds=sorted((SOURCE/'configs').glob('site0-m1-r1.5-z0.035-other_adsorbed-*.json'))
    require(len(olds)==8,'eight old populations required')
    for oldpath in olds:
        old=read(oldpath);require(old['basin']=='other_adsorbed' and old['uniform_probability']==1. and old['proposal_components']==[],'old proposal is not full uniform')
        require(old['initial_draws']==n and all(old.get(k,v)==v for k,v in ORIGINAL_METRIC.items()),'old budget/metric mismatch')
        require(sha(old['shape'])==sha(cfg['shape']) and read(old['environment'])==env,'old physical geometry mismatch')
        summary_path=SOURCE/'runs'/oldpath.stem/'summary.json';saved=read(summary_path)
        require(saved['complete'] and all(abs(p['log_g']+math.log(volume))<1e-12 for p in saved['final_particles']),'old operative density mismatch')
        hashes[str(oldpath)]=sha(oldpath);hashes[str(summary_path)]=sha(summary_path)
    q=hits/n/g
    ci=[beta.ppf(.025,hits,n-hits+1)/g,beta.ppf(.975,hits+1,n-hits)/g]
    a={'old_proposal':'uniform_probability=1; explicit components=[]; draw branch always capture-ball/full-Haar; all old saved log_g=-log(Vcapture)',
       'hard_q2_mass_estimate':q,'binomial_95_percent_CI':ci,
       'old_uniform_expected_q2_hits_per_population':n*q/volume,
       'old_uniform_expected_q2_hits_95_interval':[n*x/volume for x in ci],
       'q2_cap_to_uniform_density_ratio':g*volume,
       'probability_zero_q2_hits_in_eight_old_populations_at_point_estimate':math.exp(8*n*math.log1p(-q/volume)),
       'verified_input_sha256':hashes,'driver_sha256':sha(__file__),
       'scope':'Independent hard-volume estimate predicts unconditional initialization coverage only. Old q>1 excludes native, so shoulder hits can only be fewer than q<=2 hits. Confidence intervals are binomial sampling intervals, not deterministic geometric bounds. Density ratio is not a physical mixing speedup.'}
    destination=out/'initialization-coverage-confirmed.json';require(not destination.exists(),'preserve diagnostic')
    dump(destination,a);print(json.dumps({'coverage':str(destination),'expected_q2_hits':n*q/volume}))


def prepare(out):
    require(not out.exists(), 'output must be fresh')
    for rel, expected in EXPECTED.items(): require(sha(SOURCE/rel) == expected, f'source hash changed: {rel}')
    cfgpath = SOURCE/'configs/site0-m1-r1.5-z0.035-native-00.json'
    source, env = read(cfgpath), read(SOURCE/'inputs/geometry/environment-site0-m1.json')
    require(len(env['fixed_poses']) == 1 and source['reservoir_density'] == .035 and source['depletant_radius'] == 1.5, 'physical model changed')
    require(source['population'] == 512 and source['initial_draws'] == 1048576 and source['sweeps_per_stage'] == 2, 'historical budget changed')
    require(source['schedule'] == [i/128 for i in range(129)], 'schedule changed')
    out.mkdir(parents=True)
    for name in ['configs', 'provenance', 'controls']: (out/name).mkdir()
    used = {str(SOURCE/rel): h for rel,h in EXPECTED.items()}
    used[str(cfgpath)] = sha(cfgpath)
    runtime = []
    for p in sorted((SOURCE/'runs').glob('site0-m1-r1.5-z0.035-native-*/summary.json')):
        s = read(p); require(s['complete'], 'historical run incomplete')
        logp = p.parent/'run.log'; first = json.loads(logp.open().readline())
        runtime.append({'source': str(p), 'sha256': sha(p), 'wall_seconds': s['wall_seconds'],
                        'initialization_seconds': first['wall_seconds'], 'initialization_hits': s['initialization']['hits']})
        used[str(p)] = sha(p); used[str(logp)] = sha(logp)
    require(len(runtime) == 8, 'missing runtime evidence')
    shutil.copy2(cfgpath, out/'provenance/original-config.json')
    shutil.copy2(__file__, out/'provenance/prepare_smc_shoulder_control.py')
    jobs = []
    def add(cfg, label, group, replica):
        path = out/'configs'/f'{label}.json'; dump(path, cfg)
        output = out/'results'/label
        jobs.append({'id': label, 'group': group, 'replica': replica, 'seed': cfg['seed'],
                     'population': cfg['population'], 'config': str(path), 'config_sha256': sha(path),
                     'output': str(output), 'command': [str(SOURCE/'implementation/coordination_smc'), '--config', str(path), '--out', str(output)]})
    for n, offset in [(512,0), (2048,100000)]:
        for i in range(4):
            add(rescale_config(source,env,n,SEED_BASE+offset+1009*i),f'protein-n{n}-r{i}',f'protein-n{n}',i)
    zero = rescale_config(source,env,512,SEED_BASE+200000)
    zero.update(reservoir_density=0., schedule=[0.,1.])
    zero['metadata']['physical_target'] = 'zero-activity H I_capture I_original_q<=2'
    add(zero, 'protein-zero', 'protein-zero', 0)
    pose = {'position':[0.,0.,0.], 'orientation':[1.,0.,0.,0.]}
    shape = out/'controls/sphere-shape.json'; environment = out/'controls/sphere-environment.json'
    dump(shape, {'name':'sphere-q2-control', 'volume':ball_volume(.5), 'atoms':[{'center':[0.,0.,0.],'radius':.5}]})
    sphere_env = {'fixed_poses':[pose], 'native_pose':pose, 'rigid_members':[pose], 'capture_center':[0.,0.,0.], 'capture_radius':3., 'box_lengths':[20.,20.,20.]}
    dump(environment, sphere_env)
    for zi, z in enumerate([0.,.4]):
        for i in range(4):
            cfg = rescale_config(source,sphere_env,512,SEED_BASE+300000+10000*zi+1009*i)
            cfg.update(shape=str(shape), environment=str(environment), box_lengths=[20.,20.,20.],
                       depletant_radius=.5, reservoir_density=z, reference_activity=.4,
                       schedule=[i/16 for i in range(17)], initial_draws=8192,
                       translation_steps=[.2,.6], rotation_steps_deg=[3.,15.],
                       envelope_options={'max_cells':255, 'max_node_visits':200000,'max_depth':24,'target_width':.1})
            cfg['metadata'].update(exact=sphere_exact(z), physical_target='sphere analytic q<=2 target')
            add(cfg,f'sphere-z{z}-r{i}',f'sphere-z{z}',i)
    zero_hit = copy.deepcopy(cfg)
    zero_hit.update(seed=SEED_BASE+320000, reservoir_density=0., initial_draws=128, population=16,
                    member_error_scale=.25, schedule=[0.,1.])
    zero_hit['metadata']={'control':'deliberately impossible hard region, retain zero estimate'}
    add(zero_hit,'sphere-zero-hit','zero-hit',0)
    used[str(shape)] = sha(shape); used[str(environment)] = sha(environment)
    meanwall=float(np.mean([r['wall_seconds'] for r in runtime])); init=float(np.mean([r['initialization_seconds'] for r in runtime]))
    protocol = {'schema':1,'status':'prepared_no_jobs_launched','source_hashes':used,'jobs':jobs,
       'original_metric':ORIGINAL_METRIC,'target':'one fixed tetramer A; original q<=2; capture radius18; rd1.5; z.035',
       'region_sum':'q<=1 plus 1<q<=2; boundary conventions differ from q<2 only on zero-measure surface',
       'measure':'center Angstrom^3 times normalized proper SO(3) Haar',
       'uniform_branch':'positive capture BALL/Haar (.01), not cube',
       'g_on_protein_target':flat_density(rescale_config(source,env,512,SEED_BASE),env),
       'runtime_evidence':runtime,'historical_mean_wall_seconds_n512':meanwall,
       'forecast_n2048_seconds':init+4*(meanwall-init),
       'budget_reason':'4 independent N512 populations at original fixed schedule/budget: half the original eight-population campaign. N2048 sensitivity prepared separately, not automatically run.',
       'controls_before_production':['sphere-z0.0','sphere-z0.4','zero-hit','protein-zero'],
       'reporting':'Every population including zero retained. Native and shoulder are Zhat*terminal fraction. Whole-population covariance controls sum uncertainty. No stopping or retry by outcome.',
       'limitations':'This targets A only and q<=2. It does not test the whole competing domain, AB, molecular dynamics, or assembly.',
       'archived_preparer_sha256':sha(out/'provenance/prepare_smc_shoulder_control.py')}
    require(len({j['seed'] for j in jobs}) == len(jobs), 'duplicate seeds')
    dump(out/'protocol.json',protocol)
    print(json.dumps({'out':str(out),'jobs':len(jobs),'g':protocol['g_on_protein_target'], 'historical_n512_wall_seconds':meanwall}))


def run_group(out, group, workers):
    p = read(out/'protocol.json')
    for path, expected in p['source_hashes'].items(): require(sha(path)==expected,f'source changed {path}')
    jobs = [j for j in p['jobs'] if j['group']==group]
    require(jobs and workers > 0, 'unknown group/workers')
    if group.startswith('protein-n'):
        v=read(out/'control-validation.json'); require(v['passed'] and v['protocol_sha256']==sha(out/'protocol.json'), 'controls have not passed')
    for j in jobs:
        require(sha(j['config'])==j['config_sha256'], 'config changed')
        require(not Path(j['output']).exists(), 'existing output: never retry by result')
    state = out/f'runner-{group}.json'; require(not state.exists(), 'group already launched')
    records = []
    dump(state,{'status':'launching','group':group,'jobs':jobs})
    def run(j):
        output=Path(j['output']); output.parent.mkdir(exist_ok=True)
        log=out/f'{j["id"]}.log'
        begin=time.monotonic()
        with log.open('x') as f:
            child=subprocess.Popen(j['command'],stdout=f,stderr=subprocess.STDOUT)
            dump(out/f'process-{j["id"]}.json', {'id':j['id'],'pid':child.pid,
                 'status':'running','command':j['command'],'runner_sha256':sha(__file__)})
            code=child.wait()
        record={'id':j['id'],'pid':child.pid,'exit_code':code,'elapsed_seconds':time.monotonic()-begin}
        records.append(record)
        dump(out/f'exit-{j["id"]}.json',record)
        require(code==0,f'child failed: {j["id"]}')
    with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(run,jobs))
    dump(state,{'status':'complete','group':group,'records':records})
    print(json.dumps({'group':group,'complete':True,'elapsed_sum':sum(r['elapsed_seconds'] for r in records)}))


def audit_stages(output, cfg, summary, density):
    """Reconstruct weights, normalizer, and ancestry without rerunning RNG."""
    initial = summary['initialization']
    log_z = math.log(initial['hits']/initial['draws'])
    rate=cfg['smc_lambda_ratio']*cfg['reference_activity']
    logc=math.log1p(cfg['reservoir_density']/rate)
    max_error=0.; initial_families=None; initial_family_ess=None; minimum_weight_ess=float(cfg['population'])
    native_path=[]
    previous=None
    with (output/'stages.jsonl').open() as sf, (output/'populations.jsonl').open() as pf:
        for index,t in enumerate(cfg['schedule']):
            stage=json.loads(sf.readline()); pop=json.loads(pf.readline())
            require(stage['stage']==pop['stage']==index and stage['t']==pop['t']==t,'stage ordering')
            particles=pop['particles'];require(len(particles)==cfg['population'],'stage population')
            native=[p for p in particles if p['q']<=.5]
            native_path.append({'stage':index,'t':t,'native_endpoints':len(native),
                                'native_fraction':len(native)/len(particles),
                                'native_families':len({p['family_id'] for p in native})})
            require(all(p['q']<=1+1e-10 for p in particles),'saved stage region')
            require(max(abs(p['log_g']-math.log(density)) for p in particles)<1e-10,'stage g mismatch')
            if index:
                counts=np.asarray(stage['K'])
                require(len(counts)==cfg['population'] and (counts>=0).all() and (counts==counts.astype(np.int64)).all(),'invalid counts')
                oldg=np.asarray([p['log_g'] for p in previous])
                require(np.max(np.abs(oldg-np.asarray(stage['log_g'])))<1e-12,'stage uses wrong old density')
                score=counts*logc-oldg
                weights=(t-cfg['schedule'][index-1])*score
                increment=float(logsumexp(weights)-math.log(len(weights)))
                require(np.max(np.abs(weights-np.asarray(stage['log_weights'])))<1e-11,'stage weight mismatch')
                require(abs(increment-stage['logZ_increment'])<1e-11,'stage increment mismatch')
                require(abs(stage['conditional_count_intensity']-rate*math.exp(cfg['schedule'][index-1]*logc))<1e-12,'conditional count intensity')
                require(abs(stage['z_eff']-rate*math.expm1(t*logc))<1e-12,'annealing activity')
                require(abs(stage['smc_lambda']-rate)<1e-12 and abs(stage['log_c']-logc)<1e-12,'auxiliary law mismatch')
                parents=stage['parents']
                require(len(parents)==len(particles) and all(isinstance(p,int) and 0<=p<len(previous) for p in parents),'invalid parents')
                require(all(p['family_id']==previous[a]['family_id'] for p,a in zip(particles,parents)),'lineage propagation')
                # All parent intervals must admit one common systematic offset.
                probability=np.exp(weights-logsumexp(weights)); edges=np.concatenate(([0.],np.cumsum(probability)))
                lower=max(0.,max(edges[a]-k/len(parents) for k,a in enumerate(parents)))
                upper=min(1/len(parents),min(edges[a+1]-k/len(parents) for k,a in enumerate(parents)))
                require(lower<=upper+2e-12,'parents cannot be systematic resampling')
                log_z+=increment
                expected_ess=1/float(probability@probability)
                require(abs(stage['weight_ess']-expected_ess)<1e-8,'weight ESS mismatch')
                minimum_weight_ess=min(minimum_weight_ess,expected_ess)
            else:
                require(abs(stage['logZ_increment']-log_z)<1e-12,'initial hit normalizer')
            _,family_counts=np.unique([p['family_id'] for p in particles],return_counts=True)
            family_ess=len(particles)**2/float(family_counts@family_counts)
            require(abs(family_ess-stage['family_ess'])<1e-9 and len(family_counts)==stage['distinct_families'],'family summary mismatch')
            if not index: initial_families=len(family_counts);initial_family_ess=family_ess
            max_error=max(max_error,abs(log_z-stage['logZ_total']))
            require(max_error<1e-10,'cumulative normalizer mismatch')
            previous=particles
        require(not sf.readline() and not pf.readline(),'extra stages')
    require(abs(log_z-summary['logZ'])<1e-10,'terminal normalizer mismatch')
    require(previous==summary['final_particles'],'terminal particle mismatch')
    return {'stages_checked':len(cfg['schedule']),'max_logZ_error':max_error,
            'initial_distinct_families':initial_families,'initial_family_ess':initial_family_ess,
            'minimum_weight_ess':minimum_weight_ess,
            'native_path':native_path,
            'first_native_stage':next((p['stage'] for p in native_path if p['native_endpoints']),None),
            'native_path_scope':'For the q_internal=q_original/2 control. Annealing-target coverage; not equilibrium roundtrips or independent descendant counts.',
            'stage_sha256':sha(output/'stages.jsonl'),'population_sha256':sha(output/'populations.jsonl')}


def audit_population(j):
    cfg=read(j['config']); require(sha(j['config'])==j['config_sha256'],'config hash')
    env=read(cfg['environment']); output=Path(j['output']); summary=read(output/'summary.json')
    require(summary['complete'],'unfinished population')
    require(summary['initialization']['draws']==cfg['initial_draws'],'wrong initialization budget')
    if summary['zero_estimate']:
        require(summary['Z']==0. and summary['initialization']['hits']==0,'invalid zero')
        return {'id':j['id'],'masses':[0.,0.], 'zero':True, 'summary_sha256':sha(output/'summary.json')}
    require(summary['population']==cfg['population'] and summary['completed_stage']==len(cfg['schedule'])-1,'wrong population/stage')
    points=summary['final_particles']; require(len(points)==cfg['population'],'wrong endpoint count')
    q=np.asarray([recompute_q(x['pose'],env,ORIGINAL_METRIC) for x in points])
    internal=np.asarray([recompute_q(x['pose'],env,cfg) for x in points])
    saved=np.asarray([x['q'] for x in points])
    require(np.max(np.abs(internal-saved))<1e-10,'saved q mismatch')
    require(np.max(np.abs(q-2*internal))<1e-10 and (q<=2+1e-10).all(),'rescaled region mismatch')
    density=flat_density(cfg,env)
    require(max(abs(x['log_g']-math.log(density)) for x in points)<1e-10,'nonflat g on target')
    total=math.exp(summary['logZ']); fraction=float((q<=1).mean())
    stage_audit=audit_stages(output,cfg,summary,density)
    if cfg['reservoir_density']==0.:
        identity=summary['initialization']['hits']/cfg['initial_draws']/density
        require(abs(total/identity-1)<2e-12,'zero-activity identity failed')
    if j['group'].startswith('sphere'):
        radial=np.asarray([np.linalg.norm(x['pose']['position']) for x in points])
        require((radial>1-1e-12).all() and (radial<=3+1e-12).all(),'sphere hard/capture support')
    atom_audit=None
    if j['group'].startswith('protein-n'):
        from prepare_native_confirmation_atlas import AtomUnionAudit
        shape=read(cfg['shape']);atom=AtomUnionAudit(shape,env['fixed_poses'])
        bound=max(np.linalg.norm(a['center']) for a in shape['atoms'])
        halfbox=np.asarray(cfg.get('box_lengths',env.get('box_lengths')))/2
        chosen=[]
        for name,mask in [('native',q<=1),('shoulder',q>1)]:
            indices=np.flatnonzero(mask); ordered=indices[np.argsort(q[indices])]
            for i in sorted(set(ordered[:2].tolist()+ordered[-2:].tolist())):
                pose=points[i]['pose']
                for fixed in env['fixed_poses']:
                    require((np.abs(np.asarray(pose['position'])-fixed['position'])+2*bound<halfbox).all(),
                            'nonperiodic independent atom audit not certified')
                gaps=atom.gaps(pose);require(min(gaps)>=0,'selected endpoint atom overlap')
                chosen.append({'particle':i,'region':name,'original_q':float(q[i]),'minimum_gap_A':min(gaps)})
        atom_audit={'selection':'two smallest and two largest original q endpoints in each populated region',
                    'points':chosen,'source_sha256':sha(Path(__file__).parent/'prepare_native_confirmation_atlas.py'),
                    'pose_arrays_source_sha256':sha(Path(__file__).parent/'prepare_smc_normalizer_atlas.py'),
                    'scope':'Selected endpoints only, independent all-atom nearest-distance check; periodic primary-image condition certified.'}
    return {'id':j['id'],'masses':[total*fraction,total*(1-fraction)],'zero':False,
            'logQ_total':summary['logZ'],'native_fraction':fraction,'q_range':[float(q.min()),float(q.max())],
            'family_ess':summary['family_ess'],'distinct_families':summary['distinct_families'],
            'initialization':summary['initialization'],'wall_seconds':summary['wall_seconds'],
            'max_q_error':float(np.max(np.abs(internal-saved))),'stage_audit':stage_audit,'atom_audit':atom_audit,
            'summary_sha256':sha(output/'summary.json')}


def analyze(out, controls=False, group=None):
    p=read(out/'protocol.json')
    for path, expected in p['source_hashes'].items(): require(sha(path)==expected,f'source changed {path}')
    selected=p['controls_before_production'] if controls else [group]
    report={'protocol_sha256':sha(out/'protocol.json'),'groups':{},'passed':True,
            'analyzer_sha256':sha(__file__), 'metric_auditor_sha256':sha(Path(__file__).parent/'audit_previous_smc_regions.py')}
    for name in selected:
        jobs=[j for j in p['jobs'] if j['group']==name]; require(jobs,'unknown group')
        rows=[audit_population(j) for j in jobs]
        if name=='zero-hit': require(all(r['zero'] for r in rows),'impossible control must remain zero')
        record={'populations':rows}
        if len(rows)>1:
            record.update(population_statistics([r['masses'] for r in rows]))
            totals=np.asarray([sum(r['masses']) for r in rows])
            if totals.sum():
                fractions=totals/totals.sum()
                record.update(population_mass_fractions=fractions.tolist(),
                              population_mass_ess=1/float(fractions@fractions))
            if name.startswith('sphere-z'):
                exact=sphere_exact(float(name.split('z')[1]))
                for k,value in record['regions'].items():
                    value['exact_Q']=exact[k]
                    value['pass']=abs(value['mean_Q']-exact[k])<=max(5*value['Q_se'],.03*exact[k])
                    require(value['pass'],f'analytic {name}/{k} mismatch')
            elif name.startswith('protein-n'):
                reference_path=out/'matched-independent-references.json'
                references=read(reference_path)
                record['matched_reference_sha256']=sha(reference_path)
                record['comparisons']={}
                for label, ref in references['sources'].items():
                    compare={}
                    for key,current in record['regions'].items():
                        base=ref['regions'][key]
                        se=math.hypot(current['Q_se'],base['Q_se'])
                        rowse=math.hypot(current['Q_se'],base['row_relative_se']*base['mean_Q'])
                        compare[key]={'SMC_logQ':current['log_mean_Q'], 'SMC_population_RSE':current['relative_se'],
                            'reference_logQ':base['log_mean_Q'],'reference_population_RSE':base['relative_se'],
                            'reference_row_RSE':base['row_relative_se'],
                            'difference_in_combined_population_SE':(current['mean_Q']-base['mean_Q'])/se if se else None,
                            'difference_in_SMC_population_plus_reference_row_SE':(current['mean_Q']-base['mean_Q'])/rowse if rowse else None}
                    record['comparisons'][label]=compare
        report['groups'][name]=record
    dest=out/('control-validation.json' if controls else f'assessment-{group}.json')
    require(not dest.exists(),'preserve existing assessment')
    dump(dest,report); print(json.dumps({'assessment':str(dest),'passed':True}))


def main():
    a=argparse.ArgumentParser(description=__doc__); sub=a.add_subparsers(dest='action',required=True)
    for command in ['prepare','run','analyze','references','coverage']:
        x=sub.add_parser(command); x.add_argument('--out',type=Path,required=True)
        if command=='run': x.add_argument('--group',required=True); x.add_argument('--workers',type=int,default=4)
        if command=='analyze': x.add_argument('--group'); x.add_argument('--controls',action='store_true')
    args=a.parse_args(); out=args.out.resolve()
    if args.action=='prepare': prepare(out)
    elif args.action=='run': run_group(out,args.group,args.workers)
    elif args.action=='references': matched_references(out)
    elif args.action=='coverage': initialization_coverage(out)
    else: analyze(out,args.controls,args.group)

if __name__=='__main__': main()
