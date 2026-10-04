"""Audit completed cached-source reference arithmetic without sampling geometry.

The externally pinned runner receipt is the admission authority. Statistical
failure after a complete allocation remains auditable and is reported separately.
Saved observable vectors are reduced; their geometry is not reclassified here.
"""
from pathlib import Path
import argparse
import hashlib
import math
import audit_flexible_surrogate_reference as shared

require, integer, finite, close = shared.require, shared.integer, shared.finite, shared.close
boolean, poses, vector = shared.boolean, shared.poses, shared.vector
strict_json, sha = shared.strict_json, shared.sha
ARMS = ("direct", "m1_guided", "m8_guided", "flat8")
HORIZONS = (1, 1, 8, 8)
SCOPE = ("Saved arithmetic, decisions, source identity, retained poses, counters and observable-vector reductions. "
         "No geometry reclassification, proposal-density evaluation, RNG replay, source draw or physical sampling.")


def allocation(streams=4, sources=2048):
    return dict(arms=[dict(id=a, candidates_per_source=m) for a, m in zip(ARMS, HORIZONS)],
                kernel_calls=streams*sources*4, fixed_candidates=streams*sources*18,
                streams=streams, sources_per_stream=sources, new_source_draws=0,
                retries=0, replacements=0, extensions=0)


def protocol_meta(p):
    require(p['schema'] == 'partner-atlas-cached-reference-protocol-v1'
            and p['streams'] == 4 and p['sources_per_stream'] == 2048
            and p['prospective_reference'] == allocation()
            and p['total_outer_calls'] == 32768 and p['total_candidate_attempts'] == 147456
            and p['new_source_draws'] == 0, 'wrong fixed allocation')
    require(p['arms'] == [[1, 1., True, 'direct'], [1, 1., False, 'm1_guided'],
                          [8, 1., False, 'm8_guided'], [8, 0., False, 'flat8']], 'wrong arms')
    require(p['members'] == [0, 1] and p['selected_slot_probabilities'] == [.5, .5]
            and p['mode_probabilities'] == {'partner_atlas': .25, 'local': .75}, 'changed proposal law')
    require(p['activity'] == .5 and p['lambda'] == 2. and p['core_radius'] == .1
            and p['rd'] == .9 and p['inflated_radius'] == 1. and p['wall_radius'] == 1.6
            and p['center_radius'] == 1.5, 'changed physical target')
    require(p['body_retained_count'] == 280 and p['body_raw_count'] == 512 and p['point_volume'] == .015625
            and p['body_points_sha256'] == 'c68700a110eba1324d1d50fb5498a1be724b1a4f5693bef880cb2fb9ae8fd482',
            'changed fixed score cloud')
    require(p['atlas_cube'] == [3., 3., 3.] and p['atlas_center'] == [0., 0., 0.]
            and p['atlas_correlation'] == .6 and p['atlas_uniform_probability'] == .1, 'changed atlas settings')
    require(p['moment_se_multiplier'] == 6. and p['negative_minimum_drift'] == .0005
            and p['negative_control_arm'] == 'direct' and p['negative_controls'] == ['omitted', 'wrong_sign'],
            'changed diagnostics')
    require(len(p['observable_names']) == 38 and len(set(p['observable_names'])) == 38
            and len(p['extra_observable_names']) == 21 and len(set(p['extra_observable_names'])) == 21,
            'changed observable inventory')
    extra_names = [f'relative_R_{i}{j}{suffix}' for suffix in ('','_squared') for i in range(3) for j in range(3)]
    require(p['extra_observable_names'] == extra_names+['R0_00_squared','R1_00_squared','x0_times_R0_00'],
            'changed bounded orientation panel')
    require(p['observable_names'] == p['original_protocol']['observable_names'], 'original panel differs')
    expected_roles = []
    for stream in range(4):
        arms = []
        for arm in ARMS:
            seeds = []
            for role in ('proposal', 'inner', 'bath', 'outer'):
                payload = (b'partner-atlas-cached-iid-physical-reference-v1' + stream.to_bytes(8, 'little')
                           + arm.encode() + b'/' + role.encode())
                seeds.append(dict(role=role, seed=list(hashlib.sha256(payload).digest())))
            arms.append(dict(arm=arm, seeds=seeds))
        expected_roles.append(dict(stream=stream, arms=arms))
    require(p['role_seeds'] == expected_roles, 'changed/folded RNG roles')
    return dict(activity=.5, **{'lambda': 2.}, points_per_body=280, point_volume=.015625,
                config=dict(translation_std=.25, rotation_std_degrees=30.),
                cube=p['atlas_cube'], center=p['atlas_center'], branches=3,
                per_leg_cap=p['bath_per_leg_cap'], per_outer_cap=p['bath_per_outer_cap'])


def cube_contains(pose, meta):
    return all(abs(x-c) <= length*.5 for x, c, length in zip(pose['position'], meta['center'], meta['cube']))


def proposal_correction(step, current, slot, meta):
    """Check complete helper arithmetic, without evaluating chart densities."""
    if step['mode'] == 'local':
        require(not any(k in step for k in ('proposal_trace', 'partner_slot', 'partner_label', 'partner_pose')),
                'local attempt has atlas trace')
        require(step['proposal_log_reverse_forward'] == 0., 'asymmetric local correction')
        return 0., False
    require(step['mode'] == 'partner_atlas' and integer(step['partner_slot'], 'partner slot') == 1-slot
            and integer(step['partner_label'], 'partner label') == 1-slot and step['partner_pose'] == current[1-slot],
            'wrong current conditional partner')
    info = step['proposal_trace']
    require(info['charts'] == 'members' and info['branch'] in ('uniform', 'involution'), 'wrong atlas helper')
    if step['status'] == 'null_proposal':
        require(info['branch'] == 'involution' and type(info['null_reason']) is str
                and bool(info['null_reason']) and step['proposed'] == current
                and 'proposal_log_reverse_forward' not in step, 'malformed helper null')
        return None, False
    q = finite(info['log_reverse_forward'], 'helper correction')
    if info['branch'] == 'uniform':
        require(q == 0., 'uniform branch correction differs')
        source_support = cube_contains(current[slot], meta)
        candidate_support = cube_contains(step['proposed'][slot], meta)
        require(boolean(step['uniform_source_support'], 'uniform source support') == source_support
                and boolean(step['uniform_candidate_support'], 'uniform candidate support') == candidate_support
                and candidate_support, 'wrong immutable uniform support')
        if not source_support:
            require(step['status'] == 'zero_reverse_support' and 'proposal_log_reverse_forward' not in step,
                    'exterior source did not reject')
            return None, False
        require(step['status'] != 'zero_reverse_support', 'spurious zero reverse support')
    else:
        require(step['status'] != 'zero_reverse_support' and info['source_law'] == 'posterior'
                and info['handle'] == step['proposed'][slot], 'wrong learned handle/law')
        old_g = finite(info['full_old_member_log_density'], 'complete old G')
        new_g = finite(info['full_new_member_log_density'], 'complete new G')
        close(q, old_g-new_g, 'complete learned q')
        source, target = info['labels']['source'], info['labels']['target']
        for label in (source, target):
            require(set(label) == {'member', 'anchor', 'branch'} and integer(label['member'], 'chart member') == 0
                    and integer(label['anchor'], 'chart anchor') == 0
                    and integer(label['branch'], 'chart') < meta['branches'], 'wrong singleton/partner chart label')
        detail = info['step']
        require(integer(detail['inverse_trace']['source'], 'inverse source') == target['branch']
                and integer(detail['inverse_trace']['target'], 'inverse target') == source['branch'], 'wrong inverse labels')
        for values in (detail['source_latent'], detail['target_latent'], detail['inverse_trace']['noise']):
            vector(values, 6, 'atlas latent trace')
        close(detail['log_correction'], finite(detail['log_extended_jacobian'], 'map Jacobian')
              + finite(detail['log_auxiliary_ratio'], 'auxiliary ratio'), 'map correction')
        expanded = finite(info['expanded_log_reverse_forward'], 'expanded correction')
        close(expanded, detail['log_correction'] + finite(info['label_log_reverse_forward'], 'label correction'),
              'expanded helper correction')
        require(abs(expanded-q) <= 1e-9 + 1e-12*abs(detail['log_correction']), 'label/Jacobian cancellation differs')
    close(step['proposal_log_reverse_forward'], q, 'recorded proposal correction')
    return q, info['branch'] == 'involution' and abs(q) > 1e-9


def empty_work():
    scalar = ('attempts candidate_attempts accepted physical_decisions identities local atlas learned '
              'nonzero_atlas_corrections learned_physical_decisions learned_physical_acceptances uniform '
              'null_proposals zero_reverse_support hard_rejections mh_rejections changed_lens_candidates '
              'changed_lens_accepted raw_bath_points retained_bath_points bath_gained bath_lost').split()
    return dict.fromkeys(scalar, 0) | dict(selected_slots=[0, 0], path_orders=[0, 0])


def audit_record(record, old, retained, arm, meta, negatives, saved_drift):
    require(arm in ARMS, 'unknown arm')
    direct, horizon = arm == 'direct', HORIZONS[ARMS.index(arm)]
    strength = 0. if arm == 'flat8' else 1.
    require(record['kind'] == ('flexible_partner_atlas_direct' if direct else 'flexible_partner_atlas_chain')
            and boolean(record['inner_filter'], 'inner filter') is (not direct), 'wrong kernel variant')
    require(record['status'] in ('completed', 'identity_self_loop') and record['old'] == old
            and record['members'] == [0, 1] and record['selection_probabilities'] == [.5, .5]
            and record['mode_probabilities'] == {'partner_atlas': .25, 'local': .75}, 'wrong kernel contract/source')
    require(record['config'] == dict(meta['config'], inner_steps=horizon, guidance_strength=strength), 'wrong fixed config')
    compare(record['members'], [0, 1], 'member labels')
    integer(record['config']['inner_steps'], 'configured horizon')
    poses(old, 'source'); poses(retained, 'physical retained')
    current = old; current_score = None
    if direct:
        require(not any(k in record for k in ('old_score', 'proposed_score', 'inner_counts')), 'direct control evaluated score')
        counts = dict(attempted=0, eligible=0, hard_rejected=0, null_proposals=0, zero_reverse_support=0)
    else:
        current_score = record['old_score']; shared.score(current_score, meta, strength)
        counts = dict(attempted=0, accepted=0, hard_rejected=0, mh_rejected=0, null_proposals=0, zero_reverse_support=0)
    require(type(record['steps']) is list and len(record['steps']) == horizon, 'wrong fixed horizon')
    work = empty_work(); work['attempts'] = 1; learned_change = False; last_q = 0.
    for index, step in enumerate(record['steps']):
        require(integer(step['index'], 'step index') == index and step['old'] == current, 'broken inner residence')
        slot = integer(step['selected_slot'], 'slot')
        require(slot in (0, 1) and integer(step['selected_label'], 'label') == slot, 'wrong fair-scan label')
        proposed = step['proposed']; poses(proposed, 'proposed pair')
        require(proposed[1-slot] == current[1-slot], 'unselected member changed')
        q, nontrivial = proposal_correction(step, current, slot, meta)
        counts['attempted'] += 1; work['candidate_attempts'] += 1; work['selected_slots'][slot] += 1
        if step['mode'] == 'local': work['local'] += 1
        else:
            work['atlas'] += 1; work['uniform' if step['proposal_trace']['branch'] == 'uniform' else 'learned'] += 1
        work['nonzero_atlas_corrections'] += nontrivial
        if direct:
            require(not any(k in step for k in ('old_score', 'proposed_score', 'retained_score', 'log_u', 'log_acceptance_ratio')),
                    'direct control used inner score/coin')
        else:
            require(step['old_score'] == current_score['log_surrogate'], 'wrong source score')
        status = step['status']
        if status in ('hard_rejected', 'null_proposal', 'zero_reverse_support'):
            require(step['accepted'] is False and not any(k in step for k in ('proposed_score', 'log_u', 'log_acceptance_ratio')),
                    'rejection consumed inner decision')
            counter = {'hard_rejected':'hard_rejected', 'null_proposal':'null_proposals',
                       'zero_reverse_support':'zero_reverse_support'}[status]
            counts[counter] += 1
            work[{'hard_rejected':'hard_rejections', 'null_proposals':'null_proposals',
                  'zero_reverse_support':'zero_reverse_support'}[counter]] += 1
            require((q is None) == (status != 'hard_rejected'), 'proposal failure status differs')
        elif direct:
            require(status == 'direct_candidate' and step['accepted'] is None and q is not None,
                    'wrong direct candidate state')
            current = proposed; counts['eligible'] += 1; last_q = q; learned_change |= nontrivial
        else:
            require(status == 'completed' and q is not None, 'incomplete inner step')
            new_score = shared.score(step['proposed_score'], meta, strength)
            delta = new_score-current_score['log_surrogate']+q
            close(step['log_acceptance_ratio'], delta, 'inner score plus proposal ratio')
            log_u = finite(step['log_u'], 'inner uniform'); require(log_u < 0., 'invalid inner uniform')
            accepted = boolean(step['accepted'], 'inner decision')
            require(accepted == (log_u < min(0., delta)), 'wrong inner decision')
            counts['accepted' if accepted else 'mh_rejected'] += 1
            work['mh_rejections'] += not accepted
            if accepted:
                current, current_score = proposed, step['proposed_score']; learned_change |= nontrivial
        require(step['retained'] == current, 'wrong inner retained state')
        if not direct: require(step['retained_score'] == current_score['log_surrogate'], 'wrong retained score')
    compare(record['proposal_counts' if direct else 'inner_counts'], counts, 'attempt counters')
    require(record['proposed'] == current, 'wrong outer candidate')
    identity = current == old
    if direct: correction = 0. if identity else last_q
    else:
        require(record['proposed_score'] == current_score, 'wrong final score')
        correction = record['old_score']['log_surrogate']-current_score['log_surrogate']
    close(record['complete_log_correction'], correction, 'outer correction')
    before, after = record['budget_before'], record['budget_after']
    for budget in (before, after):
        require(set(budget) == {'raw', 'retained'}, 'wrong budget fields')
        for k,v in budget.items(): integer(v, k)
    accepted = boolean(record['accepted'], 'physical decision'); work['accepted'] = int(accepted)
    integer(record['physical_decisions'], 'physical decisions')
    if identity:
        require(record['status'] == 'identity_self_loop' and not accepted and record['physical_decisions'] == 0
                and before == after and not any(k in record for k in ('bath', 'log_u', 'log_acceptance_ratio')),
                'identity spent physical budget')
        work['identities'] = 1
    else:
        require(record['status'] == 'completed' and record['physical_decisions'] == 1, 'wrong physical gate count')
        bath = shared.path(record['bath'], old, current, [0, 1], meta)
        for leg in record['bath']['legs']:
            require(leg['raw_points'] <= meta['per_leg_cap'] and leg['retained_points'] <= meta['per_leg_cap'], 'leg budget exceeded')
        require(bath['raw_points'] <= meta['per_outer_cap'] and bath['retained_points'] <= meta['per_outer_cap'], 'outer budget exceeded')
        require(after == dict(raw=before['raw']+bath['raw_points'], retained=before['retained']+bath['retained_points']),
                'bath counters mismatch')
        ratio = bath['log_weight']+correction; close(record['log_acceptance_ratio'], ratio, 'one physical ratio')
        log_u = finite(record['log_u'], 'outer uniform'); require(log_u < 0., 'invalid outer uniform')
        require(accepted == (log_u < min(0., ratio)), 'wrong physical decision')
        work.update(physical_decisions=1, learned_physical_decisions=int(learned_change),
                    learned_physical_acceptances=int(learned_change and accepted), raw_bath_points=bath['raw_points'],
                    retained_bath_points=bath['retained_points'], bath_gained=bath['gained'], bath_lost=bath['lost'])
        work['path_orders'][0 if record['bath']['order'] == 'first_then_second' else 1] = 1
    require(retained == (current if accepted else old), 'wrong physical retained pose')
    drifts = None
    if direct:
        require(type(negatives) is dict and set(negatives) == {'omitted', 'wrong_sign'}, 'missing direct negative controls')
        density_delta = 0.; detail = record['steps'][0].get('proposal_trace', {})
        if not identity and detail.get('branch') == 'involution':
            density_delta = (math.atan(detail['full_new_member_log_density'])-math.atan(detail['full_old_member_log_density']))/math.pi
        drifts = [density_delta if accepted else 0.]
        for name in ('omitted', 'wrong_sign'):
            take = False if identity else record['log_u'] < min(0., record['bath']['aggregate']['log_weight']
                                                              - (0. if name == 'omitted' else last_q))
            control = negatives[name]
            require(type(control) is dict and set(control) == {'accepted', 'retained'}
                    and boolean(control['accepted'], f'{name} decision') == take
                    and control['retained'] == (current if take else old), f'wrong {name} negative decision/retention')
            drifts.append(density_delta if take else 0.)
        vector(saved_drift, 3, 'direct drift')
        for a,b in zip(saved_drift, drifts): close(a,b,'bounded complete-G drift')
    else: require(negatives is None and saved_drift is None, 'unexpected negative control')
    return work, after, drifts


class Moments:
    def __init__(self): self.n=0; self.sum=0.; self.sum2=0.
    def add(self, value):
        finite(value, 'moment value'); self.n += 1; self.sum += value; self.sum2 += value*value
    def report(self):
        mean = self.sum/self.n if self.n else None
        se = math.sqrt(max(0., self.sum2-self.sum*self.sum/self.n)/(self.n*(self.n-1))) if self.n>1 else None
        return dict(n=self.n, sum=self.sum, sum2=self.sum2, mean=mean, se=se)
    def check(self, name, truth=0.):
        m=self.report(); threshold=6*m['se']+1e-11 if self.n>1 else 0.
        return dict(name=name, passed=self.n>1 and abs(m['mean']-truth)<=threshold,
                    truth=truth, threshold=threshold, moments=m)


def compare(actual, expected, name='value'):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual)==set(expected), f'wrong {name} fields')
        for k,v in expected.items(): compare(actual[k],v,f'{name}/{k}')
    elif type(expected) is list:
        require(type(actual) is list and len(actual)==len(expected), f'wrong {name} length')
        for i,v in enumerate(expected): compare(actual[i],v,f'{name}/{i}')
    elif type(expected) is float: close(actual,expected,name)
    else: require(type(actual) is type(expected) and actual==expected, f'wrong {name}')


def rows(path, maximum=2*1024**2):
    with Path(path).open('rb') as stream:
        while raw := stream.readline(maximum+1):
            require(len(raw)<=maximum and raw.endswith(b'\n'), 'oversized/unterminated JSON row')
            yield strict_json(raw)


def read_sources(path, streams, per_stream):
    result={}; previous=[0]*streams
    for n,row in enumerate(rows(path,4096)):
        require(n<streams*per_stream and set(row)=={'kind','stream','source_index','source_attempt','old'}, 'wrong cache inventory/row')
        s,i=divmod(n,per_stream)
        require(row['kind']=='cached_iid_source' and integer(row['stream'],'stream')==s
                and integer(row['source_index'],'source index')==i
                and integer(row['source_attempt'],'source attempt')>previous[s], 'filtered/reordered source cache')
        poses(row['old'],'cached source'); previous[s]=row['source_attempt']; result[s,i]=row
    require(len(result)==streams*per_stream,'incomplete source cache'); return result


def reduce_journal(path, sources, streams, per_stream, meta):
    work=[[empty_work() for _ in ARMS] for _ in range(streams)]
    paired=[[[Moments() for _ in range(38)] for _ in ARMS] for _ in range(streams)]
    extra=[[[Moments() for _ in range(21)] for _ in ARMS] for _ in range(streams)]
    total=[[Moments() for _ in range(38)] for _ in ARMS]; total_extra=[[Moments() for _ in range(21)] for _ in ARMS]
    source_m=[Moments() for _ in range(38)]; source_e=[Moments() for _ in range(21)]
    drift=[[Moments() for _ in range(3)] for _ in range(streams)]; total_drift=[Moments() for _ in range(3)]
    budget=dict(raw=0,retained=0); iterator=iter(rows(path)); calls=0; baseline=None
    for begin in iterator:
        outcome=next(iterator,None); require(outcome is not None,'missing outcome')
        require(calls<streams*per_stream*4,'extra kernel call')
        source_ordinal,arm_index=divmod(calls,4); s,i=divmod(source_ordinal,per_stream); arm=ARMS[arm_index]
        require(begin['kind']=='kernel_begin' and outcome['kind']=='kernel_outcome','wrong journal pairing')
        source=sources[s,i]
        for record in (begin,outcome):
            require((integer(record['stream'],'stream'),integer(record['source_index'],'source index'),record['arm'])==(s,i,arm)
                    and integer(record['source_attempt'],'attempt')==source['source_attempt'],'filtered/reordered kernel inventory')
        require(begin['old']==source['old'] and outcome['kernel_error'] is None and outcome['audit_error'] is None,
                'wrong cached source or failed kernel')
        record=outcome['record']; require(record['budget_before']==budget,'broken campaign budget')
        one,budget,values=audit_record(record,source['old'],outcome['retained'],arm,meta,
                                      outcome['negative_controls'],outcome['bounded_density_drift'])
        before,after=outcome['source_observables'],outcome['retained_observables']
        before_e,after_e=outcome['source_extra_observables'],outcome['retained_extra_observables']
        for v,width in ((before,38),(after,38),(before_e,21),(after_e,21)): vector(v,width,'saved observable vector')
        if arm_index==0:
            baseline=(before,before_e)
            for m,v in zip(source_m,before): m.add(v)
            for m,v in zip(source_e,before_e): m.add(v)
        else: require((before,before_e)==baseline,'arm-specific source observables')
        if outcome['retained']==source['old']: require(after==before and after_e==before_e,'self-loop changed observables')
        candidate_lens=finite(outcome['candidate_internal_lens'],'candidate lens')
        if record['proposed']==source['old']: close(candidate_lens,before[10],'identity candidate lens')
        if record['accepted']: close(candidate_lens,after[10],'accepted candidate lens')
        one['changed_lens_candidates']=int(abs(candidate_lens-before[10])>1e-10)
        one['changed_lens_accepted']=int(abs(after[10]-before[10])>1e-10)
        for k,v in one.items():
            if type(v) is list: work[s][arm_index][k]=[a+b for a,b in zip(work[s][arm_index][k],v)]
            else: work[s][arm_index][k]+=v
        for j,(a,b) in enumerate(zip(after,before)):
            paired[s][arm_index][j].add(a-b); total[arm_index][j].add(a-b)
        for j,(a,b) in enumerate(zip(after_e,before_e)):
            extra[s][arm_index][j].add(a-b); total_extra[arm_index][j].add(a-b)
        if values is not None:
            for j,v in enumerate(values): drift[s][j].add(v); total_drift[j].add(v)
        calls+=1
    require(calls==streams*per_stream*4,'incomplete kernel allocation')
    return dict(work=work,paired=paired,extra=extra,total=total,total_extra=total_extra,source_m=source_m,
                source_e=source_e,drift=drift,total_drift=total_drift,budget=budget,calls=calls)


def check_summary(summary, reduced, protocol):
    r=reduced; require(summary['schema']=='partner-atlas-cached-reference-summary-v1'
                       and summary['complete'] is True and summary['error'] is None
                       and summary['completed_sources']==8192 and summary['arms']==protocol['arms'],
                       'incomplete numerical summary')
    reports=lambda moments:[m.report() for m in moments]
    panels=[dict(stream=s,work=r['work'][s],paired_changes=[reports(a) for a in r['paired'][s]],
                 paired_extra_changes=[reports(a) for a in r['extra'][s]],bounded_density_drift=reports(r['drift'][s]))
            for s in range(4)]
    compare(summary['streams'],panels,'stream reductions')
    for key,value in dict(paired_changes=[reports(a) for a in r['total']],
                          paired_extra_changes=[reports(a) for a in r['total_extra']],
                          source_moments=reports(r['source_m']),source_extra_moments=reports(r['source_e']),
                          bounded_density_drift=reports(r['total_drift'])).items(): compare(summary[key],value,key)
    require(summary['raw_bath_points']==r['budget']['raw']<=protocol['bath_raw_cap']
            and summary['retained_bath_points']==r['budget']['retained']<=protocol['bath_retained_cap'], 'final bath budget mismatch')
    require(0 <= finite(summary['cpu_seconds'],'CPU') <= protocol['cpu_limit_seconds'], 'CPU budget exceeded')
    for k in ('new_source_draws','retries','replacement_draws'): require(summary[k]==0, 'unplanned work')
    checks=[]
    for a,(name,horizon) in enumerate(zip(ARMS,HORIZONS)):
        totals={k:sum(p[a][k] for p in r['work']) for k in r['work'][0][a] if type(r['work'][0][a][k]) is int}
        for key in ('selected_slots','path_orders'):
            totals[key]=[sum(p[a][key][i] for p in r['work']) for i in range(2)]
        passed=(totals['attempts']==8192 and totals['candidate_attempts']==8192*horizon and totals['accepted']>200
                and totals['learned']>100 and totals['uniform']>10 and totals['changed_lens_accepted']>100
                and totals['nonzero_atlas_corrections']>100 and totals['hard_rejections']>0
                and totals['physical_decisions']>500 and all(n>100 for n in totals['selected_slots'])
                and all(n>100 for n in totals['path_orders']) and totals['learned_physical_decisions']>10
                and totals['learned_physical_acceptances']>0 and totals['raw_bath_points']>100
                and totals['bath_gained']>0 and totals['bath_lost']>0)
        fields=('attempts candidate_attempts accepted learned uniform changed_lens_accepted nonzero_atlas_corrections '
                'hard_rejections physical_decisions selected_slots path_orders learned_physical_decisions '
                'learned_physical_acceptances raw_bath_points bath_gained bath_lost').split()
        checks.append(dict(name=f'{name}/inventory_and_activity',passed=passed,**{k:totals[k] for k in fields}))
        checks += [m.check(f'{name}/{label}') for m,label in zip(r['total'][a],protocol['observable_names'])]
        checks += [m.check(f'{name}/{label}') for m,label in zip(r['total_extra'][a],protocol['extra_observable_names'])]
    for k in list(range(6))+[8]+list(range(17,35)):
        checks.append(r['source_m'][k].check('source/'+protocol['observable_names'][k]))
    for k in range(35,38): checks.append(r['source_m'][k].check('source/'+protocol['observable_names'][k],.25))
    for k,label in enumerate(protocol['extra_observable_names']):
        checks.append(r['source_e'][k].check('source/'+label,1/3 if 9<=k<20 else 0.))
    checks.append(r['total_drift'][0].check('direct/bounded_learned_density_drift'))
    for i,name in enumerate(('omitted','wrong_sign')):
        m=r['total_drift'][i+1].report(); threshold=max(.0005,6*m['se'])
        checks.append(dict(name=f'negative/{name}/bounded_density_drift',passed=m['n']==8192 and m['mean']>threshold,
                           threshold=threshold,moments=m,failure_meaning='unresolved negative-control sensitivity; no retuning'))
    compare(summary['checks'],checks,'diagnostic checks')
    passed=all(c['passed'] for c in checks)
    require(boolean(summary['passed'],'summary passed')==passed
            and boolean(summary['statistical_checks_passed'],'statistical passed')==passed,'inconsistent diagnostic status')
    return passed


def load(path,maximum=1024*1024):
    path=Path(path); require(path.stat().st_size<=maximum,'oversized metadata'); return strict_json(path.read_bytes())


def check_lifecycle(validation, statistical_passed):
    """A completed failed diagnostic is valid input; a killed process is not."""
    require(validation['numerical_allocation_complete'] is True and validation['receipt_bindings_valid'] is True
            and boolean(validation['scientific_receipt_passed'],'runner scientific status') == statistical_passed
            and boolean(validation['passed'],'runner passed') == statistical_passed,
            'runner numerical/scientific flags disagree')
    expected_exit = 0 if statistical_passed else 101
    numerical = validation['numerical_process']
    require(type(validation['returncode']) is int and validation['returncode'] == expected_exit
            and type(numerical['returncode']) is int and numerical['returncode'] == expected_exit
            and numerical['error'] is None
            and numerical['timeout'] is False and numerical['child_started'] is True
            and numerical['child_drained'] is True and numerical['argv'] == validation['argv'],
            'reference did not exit normally after the full allocation')
    require(all(numerical['cleanup'][k] is True for k in ('child_drained','leader_reaped','process_group_absent')),
            'numerical process group was not drained')
    require(validation['archive_before'] == validation['archive_after']
            and validation['allocation'] == allocation(), 'archive or allocation changed')


def audit_completed(root, expected_validation):
    root=Path(root).resolve(); validation_path=root/'validation.json'
    require(sha(validation_path)==expected_validation,'runner validation hash mismatch')
    validation=load(validation_path,4*1024**2)
    require(validation['schema']=='partner-atlas-reference-validation-v1' and validation['complete'] is True
            and validation['child_started'] is True and validation['child_drained'] is True
            and validation['error'] is None,'execution incomplete/undrained/infrastructure failure')
    require(validation['source_before']==validation['source_after'],'execution source changed')
    for before,after in (('production_before','production_after'),('protected_before','protected_after')):
        if before in validation or after in validation: require(validation[before]==validation[after],'protected executable changed')
    paths={name:root/'reference'/name for name in ('protocol.json','kernel-attempts.jsonl','summary.json','receipt.json')}
    bound={str(validation_path):expected_validation}
    for name,path in paths.items():
        expected=validation['outputs']['reference/'+name]
        require(path.stat().st_size<=2*1024**3 and sha(path)==expected,'unauthenticated reference output '+name)
        bound[str(path)]=expected
    receipt=load(paths['receipt.json']); protocol=load(paths['protocol.json']); summary=load(paths['summary.json'],16*1024**2)
    require(receipt['schema']=='partner-atlas-cached-reference-receipt-v1' and receipt['complete'] is True
            and receipt['numerical_allocation_complete'] is True,'reference allocation incomplete')
    check_lifecycle(validation,boolean(receipt['passed'],'claimed statistical status'))
    require(receipt['output_sha256']=={n:bound[str(p)] for n,p in paths.items() if n!='receipt.json'},'receipt output mismatch')
    require(protocol['reference_source_sha256']==validation['source_before']['tests/partner_atlas_stationarity.rs'],
            'compiled reference source differs')
    require(protocol['compiled_library_bundle_sha256']==validation['bundle_sha256'], 'compiled library bundle differs')
    require(validation['argv']==[str(root/'test-binary'),'--ignored','--exact',
                                'partner_atlas_preserves_cached_physical_sources','--test-threads=1','--nocapture'],
            'wrong numerical invocation')
    for k in ('new_source_draws','retries','replacement_draws'): require(receipt[k]==0,'unplanned reference work')
    meta=protocol_meta(protocol)
    def bound_json(ref):
        require(set(ref)=={'path','sha256'},'wrong bound-file fields')
        p=Path(ref['path']); require(p.is_absolute() and sha(p)==ref['sha256'],'changed bound input')
        bound[str(p)]=ref['sha256']; return load(p)
    source_receipt=bound_json(protocol['source_receipt'])
    require(source_receipt==protocol['source_authority'] and source_receipt['schema']=='partner-atlas-iid-source-cache-v1'
            and source_receipt['complete'] is True and source_receipt['passed'] is True
            and source_receipt['prospective_reference']==allocation()
            and source_receipt['streams']==4 and source_receipt['sources_per_stream']==2048 and source_receipt['total_sources']==8192
            and source_receipt['sources']==protocol['sources'],'cache authority mismatch')
    require(bound_json(source_receipt['original_protocol'])==protocol['original_protocol'],'changed original target metadata')
    source_path=Path(protocol['sources']['path']); source_hash=protocol['sources']['sha256']
    require(source_path.is_absolute() and source_path.stat().st_size<=32*1024**2 and sha(source_path)==source_hash,'changed source cache')
    require(receipt['source_cache_sha256']==source_hash and receipt['source_receipt_sha256']==protocol['source_receipt']['sha256'],
            'scientific receipt cache mismatch')
    bound[str(source_path)]=source_hash
    sources=read_sources(source_path,4,2048)
    reduced=reduce_journal(paths['kernel-attempts.jsonl'],sources,4,2048,meta)
    statistical_passed=check_summary(summary,reduced,protocol)
    require(boolean(receipt['passed'],'receipt passed')==statistical_passed
            and boolean(receipt['statistical_checks_passed'],'receipt statistical passed')==statistical_passed,
            'receipt diagnostic status mismatch')
    require(validation['scientific_receipt_error']==receipt['error']
            and ((receipt['error'] is None) if statistical_passed else (type(receipt['error']) is str and bool(receipt['error']))),
            'wrong diagnostic error status')
    check_lifecycle(validation,statistical_passed)
    # Detect mutation during the streaming pass, including the externally pinned admission record.
    for p,h in bound.items(): require(sha(p)==h,'input changed during audit')
    return dict(schema='partner-atlas-reference-journal-audit-v1',complete=True,passed=True,
                validation_sha256=expected_validation,input_sha256=bound,inventory={a:8192 for a in ARMS},
                candidate_attempts=sum(w['candidate_attempts'] for panel in reduced['work'] for w in panel),
                work=reduced['work'],budget=reduced['budget'],reference_statistical_checks_passed=statistical_passed,
                geometry_queries=0,new_physical_samples=0,scope=SCOPE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True); parser.add_argument('--validation-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    require(not args.out.exists(),'fresh audit output required')
    result=audit_completed(args.input,args.validation_sha256)
    result['auditor_sha256']={str(Path(__file__).resolve()):sha(__file__),
                             str(Path(shared.__file__).resolve()):sha(shared.__file__)}
    import json
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(complete=True,passed=True,reference_statistical_checks_passed=result['reference_statistical_checks_passed'])))


if __name__=='__main__': main()
