#!/usr/bin/env python3
"""Offline proposal diagnostics from frozen SMC geometry and saved importance rows.

No physical draws, classifier calls or audit replay. All old populations remain
unchanged, including exterior/invalid zeros in their unconditional denominators.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.special import logsumexp
from prepare_contact_bank_guides import ROOT, read, write, sha, log_proposal, local_source_closure

AUTH_SHA = 'b5e23f739358bac7b0234d629a6744ee08b9c7c37f8ffe43dfc5c71d05b33fc7'
COMPARISON_SHA = '963b2b32d8b4dc5fe4acbaf3aed4b4a9afdf8ec224d61306dd1ab55e059c0b05'
CLASSES = ('registered_native_entry', 'old_R5_intersection_native',
           'remaining_R4_native', 'contact_no_native_entry')
SCOPE = ('Proposal-design diagnostics only. Held-out SMC descendants are correlated geometry, '
    'not IID equilibrium observations. Importance moment diagnostics reuse previously inspected '
    'failed-convergence data; they cannot validate a newly selected guide, establish coverage of '
    'unseen poses, or open the full-vessel/assembly gate. All attempted draws remain in their '
    'original denominators; original physical weights and classifiers are unchanged. '
    'A second-moment ratio is neither a variance ratio nor a measured sampling speedup.')


def require(value, message):
    if not value:
        raise ValueError(message)


def log_moment_statistics(terms, n):
    """Unconditional moments and diagnostic concentration, including unlisted zeros."""
    terms = np.asarray(terms, float)
    require(n > 0 and len(terms) <= n and not np.isnan(terms).any()
            and not np.isposinf(terms).any(), 'Invalid unconditional moment input')
    terms = terms[np.isfinite(terms)]
    if not len(terms):
        return {'log_second_moment': None, 'term_count': 0, 'contribution_ESS': 0.,
                'largest_contribution': None}
    total = float(logsumexp(terms))
    return {'log_second_moment': total - math.log(n), 'term_count': len(terms),
            'contribution_ESS': float(math.exp(2 * total - logsumexp(2 * terms))),
            'largest_contribution': float(math.exp(float(terms.max()) - total))}


def score_moments(z, pairs, logq_source, logq_targets, n):
    """E_source[w^2 q_source/q_target], with independent-cloud counterpart."""
    z, pairs, source = np.asarray(z), np.asarray(pairs), np.asarray(logq_source)
    require(pairs.shape == (len(z), 2) and source.shape == z.shape and n >= len(z),
            'Invalid cloud/denominator dimensions')
    require(np.isfinite(source).all(), 'Nonfinite source density')
    require(not np.isnan(z).any() and not np.isposinf(z).any(), 'Invalid saved weight')
    require(not np.isnan(pairs).any() and not np.isposinf(pairs).any(), 'Invalid cloud weights')
    require(np.allclose(np.logaddexp(pairs[:, 0], pairs[:, 1])-math.log(2), z,
                        rtol=0, atol=2e-10), 'Saved weight is not the two-cloud mean')
    results = {}
    for name, target in logq_targets.items():
        target = np.asarray(target)
        require(target.shape == z.shape and np.isfinite(target).all(), 'Invalid target density')
        correction = source - target
        results[name] = {
            'two_cloud_noisy': log_moment_statistics(2 * z + correction, n),
            'paired_physical': log_moment_statistics(pairs.sum(axis=1) + correction, n)}
    require('bank' in results, 'Missing common target-proposal baseline')
    for result in results.values():
        noisy = result['two_cloud_noisy']['log_second_moment']
        physical = result['paired_physical']['log_second_moment']
        require(noisy is None or noisy >= physical-1e-9, 'Cloud product exceeds squared mean')
        for kind in ('two_cloud_noisy', 'paired_physical'):
            moment = result[kind]['log_second_moment']
            base = results['bank'][kind]['log_second_moment']
            result[kind]['ratio_to_bank'] = None if moment is None else float(math.exp(moment - base))
            require(moment is None or result[kind]['ratio_to_bank'] <= 2 + 1e-9,
                    'Legacy defensive bound violated')
    mass = None if not np.isfinite(z).any() else float(logsumexp(z) - math.log(n))
    return {'unconditional_attempts': n, 'contributing_rows': int(np.isfinite(z).sum()),
            'original_log_Qz': mass, 'guides': results}


def combine_population_scores(populations):
    """Equal-size, independent source populations; never combine different cloud laws."""
    require(len(populations) == 4 and len({p['id'] for p in populations}) == 4,
            'Expected four independent source populations')
    require(len({p['source_arm'] for p in populations}) == 1, 'Cannot pool different source arms')
    require(len({p['cloud_law'] for p in populations}) == 1, 'Cannot pool different cloud laws')
    require(len({p['samples'] for p in populations}) == 1, 'Unequal-N pooling is not declared')
    n = sum(p['samples'] for p in populations)
    groups = {}
    for group in populations[0]['groups']:
        entries = [p['groups'][group] for p in populations]
        guides = {}
        for name in entries[0]['guides']:
            guides[name] = {}
            for kind in ('two_cloud_noisy', 'paired_physical'):
                pieces = [e['guides'][name][kind] for e in entries]
                finite = [v for v in pieces if v['log_second_moment'] is not None]
                if not finite:
                    guides[name][kind] = dict(log_second_moment=None, contribution_ESS=0.,
                        largest_contribution=None, ratio_to_bank=None,
                        maximum_population_fraction=None, population_ratios_to_bank=[None]*4)
                    continue
                totals = np.asarray([v['log_second_moment'] + math.log(populations[0]['samples']) for v in finite])
                logtotal = float(logsumexp(totals))
                weights = np.exp(totals - logtotal)
                guides[name][kind] = dict(log_second_moment=logtotal - math.log(n),
                    contribution_ESS=float(1 / sum(w*w/v['contribution_ESS'] for w,v in zip(weights,finite))),
                    largest_contribution=float(max(w*v['largest_contribution'] for w,v in zip(weights,finite))),
                    maximum_population_fraction=float(weights.max()),
                    population_ratios_to_bank=[v['ratio_to_bank'] for v in pieces])
        for name in guides:
            for kind in ('two_cloud_noisy', 'paired_physical'):
                value = guides[name][kind]['log_second_moment']
                base = guides['bank'][kind]['log_second_moment']
                guides[name][kind]['ratio_to_bank'] = None if value is None else math.exp(value-base)
        masses = [e['original_log_Qz'] for e in entries if e['original_log_Qz'] is not None]
        groups[group] = dict(unconditional_attempts=n,
            contributing_rows=sum(e['contributing_rows'] for e in entries),
            original_log_Qz=None if not masses else float(logsumexp(masses)-math.log(4)),
            guides=guides)
    return dict(samples=n, source_arm=populations[0]['source_arm'],
                cloud_law=populations[0]['cloud_law'], groups=groups)


def candidate_densities(u, bank, candidates):
    """Reuse the unchanged 80-component contribution exactly in each blend."""
    base = log_proposal(u, bank)
    result = {'bank': base}
    old_count = len(bank['gaussian_components'])
    for name, guide in candidates.items():
        require(guide['defensive_uniform_shell_probability'] == .5
                and guide['region_sha256'] == bank['region_sha256'], 'Candidate support changed')
        old = guide['gaussian_components'][:old_count]
        require(len(old) == old_count, 'Legacy components lost')
        for original, component in zip(bank['gaussian_components'], old):
            require(component['mean'] == original['mean'] and component['covariance'] == original['covariance']
                    and abs(component['weight']-.5*original['weight']) < 1e-14, 'Legacy component changed')
        new = guide['gaussian_components'][old_count:]
        require(new and abs(sum(c['weight'] for c in new)-.5) < 1e-12, 'New group allocation changed')
        small = dict(guide, gaussian_components=new)
        result[name] = np.logaddexp(base, log_proposal(u, small))-math.log(2)
        require(np.all(result[name] >= base-math.log(2)-1e-12), 'Pointwise defensive bound failed')
    return result


def geometry_scores(populations, bank, candidates):
    """Per-population descriptive log densities; no descendant-IID standard errors."""
    reports = []
    for population in populations:
        u = np.asarray(population['u'])
        q = candidate_densities(u, bank, candidates)
        masks = dict(all=np.ones(len(u),bool), inside_old_R5_chart=population['old_r5_inside'],
                     outside_old_R5_chart=~np.asarray(population['old_r5_inside']))
        groups = {}
        for name, mask in masks.items():
            groups[name] = {'slots': int(mask.sum()), 'guides': {key: {
                'mean_log_density_gain_vs_bank': float(np.mean((values-q['bank'])[mask])) if mask.any() else None,
                'median_log_density_gain_vs_bank': float(np.median((values-q['bank'])[mask])) if mask.any() else None,
                'fraction_density_above_bank': float(np.mean(values[mask]>q['bank'][mask])) if mask.any() else None}
                for key, values in q.items()}}
        reports.append(dict(id=population['id'], groups=groups))
    equal_population = {}
    for group in ('all', 'inside_old_R5_chart', 'outside_old_R5_chart'):
        equal_population[group] = {}
        for name in q:
            values = [r['groups'][group]['guides'][name]['mean_log_density_gain_vs_bank'] for r in reports]
            equal_population[group][name] = dict(populations=len(values),
                nonempty_populations=sum(v is not None for v in values),
                mean_log_density_gain_vs_bank=None if any(v is None for v in values) else float(np.mean(values)))
    return dict(populations=reports, equal_population=equal_population)


def saved_groups(arrays):
    valid = np.isfinite(arrays['z'])
    classes = {'registered_native_entry':valid & arrays['native'],
        'old_R5_intersection_native':valid & arrays['old_R5_intersection_native'],
        'remaining_R4_native':valid & arrays['remaining_R4_native'],
        'contact_no_native_entry':valid & arrays['contact'] & ~arrays['native']}
    require(np.array_equal(classes[CLASSES[0]],classes[CLASSES[1]]|classes[CLASSES[2]])
            and not np.any(classes[CLASSES[1]]&classes[CLASSES[2]]), 'Native partition changed')
    groups = dict(classes, total=valid, unbound=valid & ~arrays['contact'])
    for family,count in [('radial',3),('angular',3),('orthant',64)]:
        for index in range(count):
            for name,mask in classes.items():
                groups[f'{family}:{index}:{name}'] = mask & (arrays['bin_'+family]==index)
    return groups


def run(preparation, out):
    from prepare_smc_geometry_guides import load_authenticated_terminal_geometry
    from analyze_r4_smc_control import validate_config_identity
    require(not out.exists(), 'Refuse to overwrite diagnostic artifact')
    start = time.monotonic(); bindings = {}
    def bind(path, expected=None):
        path=Path(path).resolve(); digest=sha(path)
        require(expected is None or digest==expected, 'Changed source: '+str(path))
        bindings[str(path)]=digest
        return path
    plan=read(bind(preparation/'plan.json'))
    status=read(bind(preparation/'status.json'))
    require(status['complete'], 'Geometry preparation incomplete')
    # The preparation freeze binds every guide and fitting source; no mutation/retraining here.
    freeze=read(bind(preparation/'freeze.json'))
    for name,digest in freeze['files'].items():
        path=(preparation/name).resolve()
        require(path.is_relative_to(preparation), 'Preparation freeze escapes its directory')
        bind(path,digest)
    for path,digest in {**plan['input_sha256'],**plan['code_sha256']}.items(): bind(path,digest)
    authpath=ROOT/'runs/smc-completed-evidence-review-20260923/authentication.json'
    auth=read(bind(authpath,AUTH_SHA))
    comparison=ROOT/'runs/contact-confirmation-comparison-20260922'
    primary=read(bind(comparison/'analysis.json',COMPARISON_SHA))
    require(auth['input_sha256'][str(comparison/'analysis.json')]==COMPARISON_SHA
        and primary['complete'], 'Unbound completed importance source')
    campaign=Path(primary['campaign'])
    protocol=read(bind(campaign/'protocol.json',primary['protocol_sha256']))
    bank=read(bind(ROOT/'runs/refined-contact-bank-preparation-20260922/guide-bank.json',
        'd0e62d1b23d627582ca5f393af515a1ca10b8ecc01ce1a0e75c40e3e18885929'))
    candidates={name:read(bind(preparation/f'guide-{name}.json')) for name in
        ('r5-cov1','r5-cov4','r5-sign-cov1','r5-sign-cov4')}
    out.mkdir(parents=True)
    write(out/'diagnostic-plan.json',dict(schema='smc-guide-diagnostic-plan-v1',
        created=datetime.now(timezone.utc).isoformat(), candidate_ids=list(candidates),
        fit_populations=['r00','r01'],heldout_populations=['r02','r03'],
        importance_arms=list(primary['arms']),classes=CLASSES,
        strata=protocol['strata'],new_physical_draws=0,selection_rule='Report every candidate; no automatic winner',
        scope=SCOPE,source_sha256=bindings.copy()))
    heldout,mapping=load_authenticated_terminal_geometry(plan,['r02','r03'])
    heldout_scores=geometry_scores(heldout,bank,candidates)
    arms={}
    baseline_config=read(bind(ROOT/'runs/refined-contact-bank-preparation-20260922/config.json'))
    for arm in primary['arms']:
        allocation=primary['arms'][arm]['allocation']
        manifest=read(bind(campaign/arm/'manifest.json',allocation['manifest_sha256']))
        source_guide=read(bind(campaign/arm/'provenance/importance-guide.json',manifest['importance_guide_sha256']))
        config=read(bind(campaign/arm/'provenance/config.json',manifest['config_sha256']))
        validate_config_identity(config,baseline_config)
        require(config['depletant_radius']==1.5 and config['reservoir_density']==.035
            and manifest['physical_activity']==.035,'Bath changed')
        require(manifest['lambda_ratio']==allocation['lambda_ratio'] and manifest['cloud_replicates']==2
            and manifest['region_sha256']==primary['region_sha256']
            and manifest['shape_sha256']==primary['shape_sha256'],'Cloud law or physical identity changed')
        populations=[]
        for record in sorted(primary['arms'][arm]['populations'],key=lambda p:p['id']):
            job=next(j for j in protocol['jobs'] if j['arm']==arm and j['id']==record['id'])
            actual=read(bind(Path(job['directory'])/'manifest.json',record['raw_output']['manifest_sha256']))
            require(actual['lambda_ratio']==manifest['lambda_ratio'] and actual['cloud_replicates']==2
                and actual['activity']==manifest['physical_activity']
                and actual['lambda']==actual['activity']*actual['lambda_ratio']
                and actual['samples']==record['samples']==job['samples']
                and actual['seed']==record['seed']==job['seed'], 'Actual source population law changed')
            for key in ('config_sha256','shape_sha256','region_sha256','importance_guide_sha256'):
                require(actual[key]==manifest[key], 'Actual physical source changed: '+key)
            path=bind(comparison/record['records'],record['records_sha256'])
            with np.load(path,allow_pickle=False) as saved:
                arrays={k:saved[k] for k in saved.files}
            n=record['samples']; valid=np.isfinite(arrays['z'])
            require(len(valid)==n and np.array_equal(arrays['draw'],np.arange(n))
                and np.all(arrays['source_n']==n),'Original attempted rows lost')
            require(not np.any(valid & ~arrays['support']) and arrays['pairs'].shape==(n,2),
                'Support/cloud law changed')
            require(np.allclose(logsumexp(arrays['pairs'][valid],axis=1)-math.log(2),
                arrays['z'][valid],rtol=0,atol=2e-10),'Independent-cloud mean changed')
            source_error=float(np.max(abs(log_proposal(arrays['u'],source_guide)-arrays['log_q'])))
            require(source_error<2e-8,'Saved normalized source density differs')
            # Zero terms need no candidate evaluation; all remain in every original N.
            qvalid=candidate_densities(arrays['u'][valid],bank,candidates)
            q={name:np.zeros(n) for name in qvalid}
            for name,values in qvalid.items(): q[name][valid]=values
            groups={}
            for name,mask in saved_groups(arrays).items():
                groups[name]=score_moments(arrays['z'][mask],arrays['pairs'][mask],
                    arrays['log_q'][mask],{key:values[mask] for key,values in q.items()},n)
            for name in CLASSES+('total',):
                expected=next(p for p in primary['arms'][arm]['estimates'][name]['populations'] if p['id']==record['id'])
                observed=groups[name]
                require(expected['draws']==n and expected['nonzero']==observed['contributing_rows']
                    and (expected['log_Qz']==observed['original_log_Qz'] or
                         expected['log_Qz'] is not None and observed['original_log_Qz'] is not None
                         and abs(expected['log_Qz']-observed['original_log_Qz'])<2e-10),
                    'Original region mass changed: '+name)
            populations.append(dict(id=record['id'],seed=record['seed'],samples=n,source_arm=arm,
                cloud_law=f"two-independent-clouds-lambda-over-z-{allocation['lambda_ratio']:g}",
                invalid_or_exterior=int((~valid).sum()),source_density_max_error=source_error,groups=groups))
            print(f'scored {arm}/{record["id"]}: {n} original attempts; no new physics',flush=True)
        arms[arm]=dict(populations=populations,aggregate=combine_population_scores(populations))
        write(out/f'{arm}.json',arms[arm])
    closure=local_source_closure([Path(__file__),ROOT/'tools/test_score_smc_geometry_guides.py'])
    for filename,path in closure.items():
        dest=out/'source'/filename; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(bind(path),dest)
    for path,digest in bindings.items(): require(sha(path)==digest,'Source changed during scoring')
    result=dict(schema='smc-guide-design-score-v1',complete=True,scope=SCOPE,
        new_physical_draws=0,classifier_calls=0,audits_replayed=0,
        total_reused_attempts=sum(a['aggregate']['samples'] for a in arms.values()),
        heldout_geometry=heldout_scores,heldout_mapping=mapping,
        aggregates={arm:values['aggregate'] for arm,values in arms.items()},
        source_sha256=bindings,wall_seconds=time.monotonic()-start)
    write(out/'summary.json',result)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.preparation.resolve(),args.out.resolve())
