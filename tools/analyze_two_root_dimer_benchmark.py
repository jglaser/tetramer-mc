#!/usr/bin/env python3
"""Label-aware, contact-only observer for the fixed two-root m4 control.

New journals are never rewritten. A transient canonical projection goes to the
unchanged state replay; nested proposal, auxiliary and bath records retain their
selected-root order. Saved scalar arithmetic is checked, not geometric density,
RNG or Poisson-predicate reconstruction. No native or equilibrium claim follows.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time

import analyze_evolving_dimer_benchmark as previous
import analyze_root_guided_dimer_benchmark as external
import analyze_two_neighbor_singleton_benchmark as scalar

require,read,sha,canonical,write=(previous.require,previous.read,previous.sha,previous.canonical,previous.write)
ARM='two_root_m4'
POLICY=dict(schema='two-root-factorized-m4-policy-v1',root_probabilities=[.5,.5],
            selection='once_per_global_slot',internal_threshold_m=4)
SOURCE_FILES=list(dict.fromkeys(['tools/analyze_two_root_dimer_benchmark.py',
    'tools/test_analyze_two_root_dimer_benchmark.py','tools/run_native_class_physical_campaign.py']+scalar.SOURCE_FILES))


def analysis_plan():
    return dict(schema='two-root-dimer-analysis-plan-v1',source_files=SOURCE_FILES,
        new_chains=8,reused_control_chains=16,reused_control_arms=['local','m4'],context_indices=[0],
        new_retained_initial_observations=8*4609,new_production_observations=8*4096,
        maximum_mobile_related_pairs_per_endpoint=525,maximum_new_pair_classifications=8*4609*525,
        scalar_local_attempts=8*4608*4,scalar_dimer_attempts=8*4608,
        observer_limits=dict(threads=1,cpu_limit_seconds=1800,wall_limit_seconds=3600,
                             address_space_limit_bytes=16*2**30,max_record_bytes=16*2**20),
        original_plan=previous.analysis_plan(),external_plan=external.analysis_plan(),
        target='Original context0 conditional two-mobile target; all other labels fixed. No contexts pooled.',
        root_selection='State-independent fair prior, once per outer slot, retained through all retries and decision.',
        replay='Validate root identity/projections, complete saved F/auxiliary/bath arithmetic and every elementary state. Only top-level replay pose fields are canonicalized in a copy.',
        trust_boundary='No independent learned density, geometric predicates, RNG or cloud reconstruction; source-bound Rust and saved predicates remain trusted.',
        comparison='Four independent streams per initialization. Local and m4 proposal/threshold/bath/accept RNG roles are shared; root-order role is separate. Paired arms are not independent replicates.',
        external_events='Separate whole external edge sets from internal/whole fingerprint. Include empty residence; distinguish direct nonempty changes from attachment/detachment and returns through empty.',
        per_root_diagnostics='Same validated pass: all root-choice dispositions; candidate-only finite log-factor mean/range with negative infinity and edge support zeros separate. No counterfactual efficiency claim.',
        native_observer=False,new_geometry_only=True,old_geometry_queries=0,new_physical_draws=0,
        complete_inventory_required=True,trajectories_concatenated=False,assembly_gate_open=False)


def identity(job):return tuple(job[k] for k in ('context_index','arm','initialization','stream'))


def validate_inventory(config):
    expected={(0,ARM,i,s) for i in ('source','proposal_prepared') for s in range(4)}
    require(len(config['jobs'])==8 and len({j['id'] for j in config['jobs']})==8
            and {identity(j) for j in config['jobs']}==expected,'Changed complete eight-chain inventory')
    require(config['two_root_policy']==POLICY and config.get('singleton_policy') is None,'Changed two-root policy')
    require(config['physical']==scalar.PHYSICAL and config['allocation']['warmup_blocks']==512
            and config['allocation']['production_blocks']==4096,'Changed target or schedule')


def root_contract(members,anchor,cloud):
    return dict(schema='evolving-dimer-two-root-m4-v1',canonical_members=members,
        root_orders=[members,list(reversed(members))],root_probabilities=[.5,.5],anchor_label=anchor,
        selection='once_per_global_slot_retained_through_all_retries',root_order_rng_role='two_root_m4/root_order',
        selection_log_reverse_forward=0.,cloud=cloud,
        internal=dict(m=4,point_frame='selected_mobile_root_body',threshold_rng_role='m4/threshold'),
        root_guidance=False,matched_control_arm='m4',proposal_rng_role='m4/proposal',bath_rng_role='m4/bath',
        accept_rng_role='m4/accept',local_schedule='canonical members [0,1,0,1]; shared unchanged local RNG roles',
        retained_and_checkpoint_order='canonical_members',proposal_record_order='selected_members',physical_decisions_per_candidate=1)


def poses(value):
    require(type(value) is list and len(value)==2,'Expected two ordered poses')
    for pose in value:
        require(set(pose)=={'position','orientation'} and len(pose['position'])==3 and len(pose['orientation'])==4
                and all(type(x) in (int,float) and math.isfinite(x) for x in pose['position']+pose['orientation'])
                and abs(sum(x*x for x in pose['orientation'])-1.)<=1e-8,'Malformed or nonnormalized pose')
    return value


def _logsum(values):
    top=max(values)
    return -math.inf if top==-math.inf else top+math.log(math.fsum(math.exp(x-top) for x in values))


def density(record,config):
    f=config['factorized'];u,g,q=(scalar.log_value(record[k]) for k in ('log_uniform','log_learned','log_full'))
    require(u==-math.inf or math.isclose(u,-3*math.log(2*f['uniform_half_width']),rel_tol=1e-10,abs_tol=1e-8),
            'Uniform F component normalization differs')
    alpha=f['uniform_probability'];require(0<=alpha<=1,'Invalid defensive probability')
    terms=([u+math.log(alpha)] if alpha else [])+([g+math.log1p(-alpha)] if alpha<1 else [])
    scalar.close(record['log_full'],_logsum(terms),'Complete defensive F mixture differs')
    return q


def joint_predicates(value):
    require(set(value)=={'internal_core_overlap','spectator_core_collisions','wall_valid','internal_exclusion_contact'}
            and type(value['internal_core_overlap']) is bool and type(value['internal_exclusion_contact']) is bool
            and len(value['spectator_core_collisions'])==len(value['wall_valid'])==2,'Malformed joint predicates')
    hard=[scalar.hard_valid(dict(spectator_core_collisions=value['spectator_core_collisions'][i],
                                wall_valid=value['wall_valid'][i])) for i in range(2)]
    return not value['internal_core_overlap'] and all(hard),value['internal_exclusion_contact']


def check_frame(frame,world,count):
    """Compare saved predicate/count views only; do not reconstruct any pose."""
    require(frame is not None and frame['reconstructed_feasibility']==world
            and frame['reconstructed_root']==dict(spectator_core_collisions=world['spectator_core_collisions'][0],wall_valid=world['wall_valid'][0])
            and frame['internal_relative']=={k:world[k] for k in ('internal_core_overlap','internal_exclusion_contact')}
            and frame.get('root_guidance') is None,'Saved frame predicates disagree')
    values=frame['guidance']
    require(set(values)=={'relative_count','recovered_count','world_count','reconstructed_world_count'}
            and all(type(v) is int and v==count for v in values.values()),'Saved guidance frame counts disagree')


def validate_attempt(attempt,caps,g):
    passed={}
    for stage in ('root','internal'):
        draws=attempt[stage+'_draws'];require(type(draws) is list and len(draws)<=caps[stage],'Edge cap exceeded')
        success=False
        for index,draw in enumerate(draws,1):
            require(not success and draw['index']==index and draw['feasibility'] is not None
                    and draw['draw']['proposed_relative_pose'] is not None and draw['draw']['null_reason'] is None,
                    'Fatal/incomplete draw or retry after first success')
            if stage=='root':
                success=scalar.hard_valid(draw['feasibility'])
                require(draw.get('guidance_count') is None,'Unexpected root guidance')
            else:
                f=draw['feasibility']
                require(set(f)=={'internal_core_overlap','internal_exclusion_contact'}
                        and all(type(v) is bool for v in f.values()),'Malformed internal predicates')
                geometric=not f['internal_core_overlap'] and f['internal_exclusion_contact']
                count=draw.get('guidance_count')
                if geometric:
                    require(type(count) is int and 0<=count<=g['point_count'],'Missing/invalid internal guidance count')
                    success=count>=g['threshold']
                else:
                    require(count is None,'Geometrically rejected edge queried guidance');success=False
        passed[stage]=success
    if not passed['root']:
        require(attempt['status']=='root_cap_exhausted' and len(attempt['root_draws'])==caps['root']
                and not attempt['internal_draws'],'Premature/wrong root cap exhaustion')
    elif not passed['internal']:
        require(attempt['status']=='internal_cap_exhausted' and len(attempt['internal_draws'])==caps['internal'],
                'Premature/wrong internal cap exhaustion')
    else:
        hard,contact=joint_predicates(attempt['final_feasibility'])
        require(attempt['status']==('candidate' if hard and contact else 'final_rejected')
                and attempt['proposed'] is not None,'Final predicate/status mismatch')
        count=attempt['internal_draws'][-1]['guidance_count'];check_frame(attempt['frame'],attempt['final_feasibility'],count)
        require(attempt['frame']['reconstructed_root']==attempt['root_draws'][-1]['feasibility']
                and attempt['frame']['internal_relative']==attempt['internal_draws'][-1]['feasibility'],
                'Raw edge and final frame predicates disagree')
        return
    require(all(attempt.get(k) is None for k in ('proposed','final_feasibility','frame')),'Exhausted edge produced an endpoint')


def validate_factors(row,config,members,anchor):
    """Selected-root records stay untouched; only their saved scalar sums are audited."""
    p=row['proposal'];f=config['factorized'];caps={k:f[k+'_cap'] for k in ('root','internal','joint')}
    require(p['members']==members and p['anchor_label']==anchor and p['old']==row['old']
            and p['caps']==caps and p['order']==f['order']=='root_first' and p.get('root_guidance') is None,
            'Selected-root proposal identity/policy differs')
    g=p['guidance'];require(g['m']==4,'Wrong internal guidance multiplicity')
    n=scalar.nat(g['point_count'],'point count');old=scalar.nat(g['old_count'],'old guidance count')
    require(old<=n,'Impossible old guidance count')
    hard,contact=joint_predicates(p['source_feasibility']);require(hard,'Hard-invalid source must be fatal')
    check_frame(p['source_frame'],p['source_feasibility'],old)
    require((p['status']=='source_outside_domain') is (not contact),'Source contact/status mismatch')
    attempts=p['attempts'];require(type(attempts) is list and len(attempts)<=caps['joint'],'Joint cap exceeded')
    no_threshold=p['status']=='source_outside_domain' or 0 in caps.values()
    if no_threshold:
        require(not attempts and g['threshold'] is None and g['integer_draws']==[], 'Ineligible source/zero cap spent RNG')
    else:
        threshold=scalar.nat(g['threshold'],'guidance threshold')
        require(threshold<=old and len(g['integer_draws'])==4 and all(type(x) is int and 0<=x<=old for x in g['integer_draws'])
                and max(g['integer_draws'])==threshold,'Guidance threshold trace differs')
    for index,a in enumerate(attempts,1):
        require(a['index']==index and a['status'] in ('root_cap_exhausted','internal_cap_exhausted','final_rejected','candidate')
                and (a['status']!='candidate' or index==len(attempts)),'Incomplete or retried successful attempt')
        validate_attempt(a,caps,g)
    if row['status']=='proposal_self_loop':
        require(p['status'] in ('source_outside_domain','cap_exhausted') and p['candidate'] is None
                and row['accepted'] is False and all(a['status']!='candidate' for a in attempts)
                and g['new_count'] is None and g['aux_log_correction'] is None,'Incomplete/false null proposal')
        require(no_threshold or len(attempts)==caps['joint'],'Premature joint cap exhaustion')
        require(not any(k in row for k in ('proposed','canonical_proposed','bath','log_u','log_acceptance_ratio','complete_log_correction')),
                'Null proposal contains a physical decision')
        return
    require(row['status']=='completed' and p['status']=='candidate' and attempts and attempts[-1]['status']=='candidate',
            'Incomplete candidate outcome')
    candidate=p['candidate'];require([candidate['root'],candidate['child']]==row['proposed']==attempts[-1]['proposed'],
                                    'Candidate pose record differs')
    q=candidate['diagnostics'];require(len(q['old_edges'])==len(q['new_edges'])==2,'Incomplete full F record')
    oldq=sum(density(x,config) for x in q['old_edges']);newq=sum(density(x,config) for x in q['new_edges'])
    require(math.isfinite(newq),'Candidate has invalid/zero forward density')
    scalar.close(q['full_old_log_density'],oldq,'Full old F differs');scalar.close(q['full_new_log_density'],newq,'Full new F differs')
    scalar.close(q['log_reverse_forward'],oldq-newq,'Full F ratio differs')
    require(q['selection_log_reverse_forward']==q['log_tree_coordinate_jacobian']==0.,'Unexpected fixed-label/Jacobian correction')
    new=scalar.nat(g['new_count'],'new guidance count')
    require(g['threshold']<=new<=n and new==attempts[-1]['internal_draws'][-1]['guidance_count'],'Candidate violates guidance support/count')
    auxiliary=4*(math.log1p(old)-math.log1p(new));scalar.close(g['aux_log_correction'],auxiliary,'Guidance correction differs')
    correction=oldq-newq+auxiliary;scalar.close(row['complete_log_correction'],correction,'Complete correction differs')
    bath=row['bath'];require(len(bath['legs'])==len(bath['ordered_members'])==2 and set(bath['ordered_members'])==set(members),'Wrong physical path labels')
    weights=[scalar.bath_weight(leg,config) for leg in bath['legs']]
    for key in ('gained','lost','raw_points','retained_points','created_cells','retained_cells'):
        require(bath['aggregate'][key]==sum(leg[key] for leg in bath['legs']),'Physical leg accounting differs')
    weight=scalar.bath_weight(bath['aggregate'],config)
    scalar.close(bath['aggregate']['log_weight'],sum(weights),'Physical leg weights differ')
    scalar.close(row['log_acceptance_ratio'],correction+weight,'Physical MH sum differs')


def new_root_diagnostics(config,job):
    case=config['contexts'][job['context_index']]
    return {str(slot):dict(root_label=case[name],attempts=0,candidates=0,accepted=0,proposal_self_loops=0,
        factor_scope='Candidate-only saved scores; exhausted/source-outside attempts were never scored and are not zero densities. Edge support counts span two edge factors per candidate.',
        null_status_counts={},null_stage_counts={},candidate_factors={},edge_support_zeros={side:{kind:0 for kind in
            ('log_uniform','log_learned','log_full')} for side in ('old_edges','new_edges')})
        for slot,name in enumerate(('root','child'))}


def accumulate_root(target,row):
    """Bounded scalar summaries; no pose selection, reread or factor removal."""
    entry=target[str(row['root_slot'])];entry['attempts']+=1;entry['accepted']+=row['accepted']
    p=row['proposal']
    if row['status']=='proposal_self_loop':
        entry['proposal_self_loops']+=1
        entry['null_status_counts'][p['status']]=entry['null_status_counts'].get(p['status'],0)+1
        for attempt in p['attempts']:
            k=attempt['status'];entry['null_stage_counts'][k]=entry['null_stage_counts'].get(k,0)+1
        return
    entry['candidates']+=1;q=p['candidate']['diagnostics']
    values=dict(source_full_log_density=q['full_old_log_density'],destination_full_log_density=q['full_new_log_density'],
        full_log_reverse_forward=q['log_reverse_forward'],auxiliary_log_correction=p['guidance']['aux_log_correction'],
        bath_log_weight=row['bath']['aggregate']['log_weight'],complete_log_correction=row['complete_log_correction'],
        total_log_acceptance_ratio=row['log_acceptance_ratio'])
    for key,value in values.items():
        v=scalar.log_value(value)
        stat=entry['candidate_factors'].setdefault(key,dict(count=0,finite_count=0,negative_infinity_count=0,
                                                          finite_mean=None,finite_min=None,finite_max=None))
        stat['count']+=1
        if v==-math.inf:stat['negative_infinity_count']+=1;continue
        n=stat['finite_count']+1;stat['finite_count']=n
        stat['finite_mean']=v if n==1 else stat['finite_mean']*((n-1)/n)+v/n
        stat['finite_min']=v if n==1 else min(stat['finite_min'],v)
        stat['finite_max']=v if n==1 else max(stat['finite_max'],v)
    for side,kinds in entry['edge_support_zeros'].items():
        for record in q[side]:
            for kind in kinds:kinds[kind]+=scalar.log_value(record[kind])==-math.inf


def canonical_rows(rows,config,job,initial,cloud=None,root_diagnostics=None):
    require(job['arm']==ARM and config['two_root_policy']==POLICY,'Wrong arm/policy')
    c=config['contexts'][job['context_index']];members=[c['root'],c['child']];anchor=c['anchor']
    require(all(type(i) is int and i>=0 for i in members+[anchor]) and len(set(members+[anchor]))==3,'Invalid fixed labels')
    poses(initial)
    for index,row in enumerate(rows):
        require(not any(k in row for k in ('fatal_error','proposal_failure','bath_failure')),'Fatal elementary outcome cannot be omitted')
        if index==0:
            require(row['kind']=='initial' and row['fixed_source']==config['source_frame'],'Initial fixed source differs')
            bank=row['cloud'] if cloud is None else cloud
            require(row['cloud']==bank and row['root_order_contract']==root_contract(members,anchor,bank),
                    'Initial root/cloud contract differs')
        if row['kind']=='factorized_dimer':
            slot=row['root_slot'];require(type(slot) is int and slot in (0,1),'Invalid root slot')
            selected=[members[slot],members[1-slot]]
            require(row['canonical_members']==members and row['selected_members']==row['members']==selected
                    and type(row['root_order_probability']) in (int,float) and row['root_order_probability']==.5
                    and type(row['root_order_log_reverse_forward']) in (int,float) and row['root_order_log_reverse_forward']==0.
                    and row['root_order_rng_role']=='two_root_m4/root_order','Root choice identity/probability differs')
            old=poses(row['old']);canonical_old=[old[selected.index(label)] for label in members]
            require(row['canonical_old']==canonical_old,'Canonical old projection differs')
            validate_factors(row,config,selected,anchor)
            if root_diagnostics is not None:accumulate_root(root_diagnostics,row)
            new=None
            if 'proposed' in row:
                proposed=poses(row['proposed']);new=[proposed[selected.index(label)] for label in members]
                require(row['canonical_proposed']==new,'Canonical proposed projection differs')
            projected=copy.deepcopy(row);projected['members']=members;projected['old']=canonical_old
            if new is not None:projected['proposed']=new
            yield projected
        else:
            if row['kind']=='local':
                require(row['status'] in ('hard_rejected','completed'),'Incomplete local attempt')
                if row['status']=='hard_rejected':
                    require(not any(k in row for k in ('bath','log_u','log_acceptance_ratio')),'Hard rejection consumed bath')
                else:
                    scalar.close(row['log_acceptance_ratio'],scalar.bath_weight(row['bath'],config),'Local bath ratio differs')
            if row['kind'] in ('initial','retained_block'):poses(row['selected'])
            yield row


def validate_journal(rows,config,job,initial,terminal=None,cloud=None,root_diagnostics=None):
    return previous.validate_journal(canonical_rows(rows,config,job,initial,cloud,root_diagnostics),config,job,initial,terminal)


def external_activity(trace,warmup,cpu):
    """All production transitions, with last warmup as boundary; O(N) returns."""
    require(math.isfinite(cpu) and cpu>0 and len(trace)>warmup+1,'Invalid external activity allocation')
    window=trace[warmup:];values=[tuple(sorted(tuple(e) for e in row['external_edges'])) for row in window]
    last_exit={};returns=[];empty_prefix=[0];direct=enter=leave=0
    for v in values:empty_prefix.append(empty_prefix[-1]+(not v))
    for i in range(1,len(values)):
        a,b=values[i-1:i+1]
        if a==b:continue
        direct+=bool(a and b);enter+=bool(not a and b);leave+=bool(a and not b)
        last_exit[a]=i
        if b and b in last_exit:
            departed=last_exit.pop(b)
            returns.append(dict(environment=list(b),departure_block=window[departed]['block'],return_block=window[i]['block'],
                                passed_through_empty=empty_prefix[i]-empty_prefix[departed]>0))
    return dict(production_samples=len(values)-1,nonempty_fraction=sum(bool(v) for v in values[1:])/(len(values)-1),
        direct_nonempty_changes=direct,entering_nonempty=enter,leaving_nonempty=leave,
        completed_nonempty_returns=returns,direct_nonempty_changes_per_full_CPU_second=direct/cpu,
        completed_nonempty_returns_per_full_CPU_second=len(returns)/cpu,full_sampler_cpu_seconds=cpu,
        scope='Observed external edge-set events; gain/loss may change sets without exchanging every partner. No native registry or equilibrium interpretation.')


def bind_complete_inputs(base,config):
    """Authenticate all eight completed journals and sixteen cached controls first."""
    import run_native_class_physical_campaign as driver
    base=Path(base).resolve();inputs={}
    def bind(ref):
        path=Path(ref['path']).resolve();require(sha(path)==ref['sha256'],'Changed input '+str(path))
        require(str(path) not in inputs or inputs[str(path)]==ref['sha256'],'Conflicting input binding')
        inputs[str(path)]=ref['sha256'];return path
    def path_ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
    for name in ('config.json','run-binding.json','analysis-plan.json','protocol.json','execution-plan.json',
                 'execution/claim.json','execution/status.json','execution/summary.json'):
        bind(path_ref(base/name))
    plan=read(base/'execution-plan.json');status=read(base/'execution/status.json');summary=read(base/'execution/summary.json')
    require(all(status[k]==summary[k] for k in ('complete','passed','failure','active','unstarted','completed','plan_sha256'))
            and status['complete'] is True and status['passed'] is True and status['failure'] is None
            and status['active'] is None and status['unstarted']==[] and len(status['completed'])==len(plan['jobs'])==8
            and not (base/'execution/failure.json').exists(),'All eight sampler jobs must complete cleanly')
    driver.verify_plan(base/'execution-plan.json',plan,inputs[str(base/'execution-plan.json')])
    for path,digest in plan['files'].items():bind(dict(path=path,sha256=digest))
    binding=read(base/'run-binding.json');require(binding['config_sha256']==inputs[str(base/'config.json')],'Run config binding differs')
    prepared=read(bind(binding['prepared_manifest']));require(prepared['complete'] is True and prepared['passed'] is True,'Incomplete inherited preparation')
    controls=config['control_analysis'];old=read(bind(controls['analysis']));cs=read(bind(controls['summary']));cm=read(bind(controls['manifest']))
    require(cs['complete'] is True and cs['passed'] is True and cm['complete'] is True and old['complete'] is True
            and cs['analysis']['sha256']==controls['analysis']['sha256'] and cs['manifest_sha256']==controls['manifest']['sha256'],
            'Cached control receipt chain differs')
    old_chains={c['job']['id']:c for c in old['chains']}
    require(len(old_chains)==len(old['chains']),'Duplicate old control identity')
    mapping={m['job_id']:m for m in config['control_mapping']}
    require(len(mapping)==len(config['control_mapping'])==8 and set(mapping)=={j['id'] for j in config['jobs']},'Changed matched control inventory')
    new=[];cached=[];used=set()
    for job in config['jobs']:
        terminal_path=Path(config['output'])/f"job-{job['id']:03}"/'terminal.json'
        matching=[x for x in plan['jobs'] if x['terminal']['path']==str(terminal_path)]
        require(len(matching)==1,'Sampler terminal absent from frozen execution')
        tj=matching[0];terminal_ref=driver.completed_terminal(base,plan,tj['id']);terminal=read(bind(terminal_ref))
        ordinal=plan['jobs'].index(tj);directory=driver.job_directory(base,ordinal,tj)
        require(read(directory/'success.json')==status['completed'][ordinal],'Completed driver list differs')
        for name in ('attempt.json','process.json','exit.json','success.json'):bind(path_ref(directory/name))
        require(terminal['complete'] is True and terminal['conditional_target'] is True and terminal['job']==job
                and terminal['blocks']==4608 and terminal['config_sha256']==inputs[str(base/'config.json')]
                and terminal['binding_sha256']==inputs[str(base/'run-binding.json')]
                and not (terminal_path.parent/'failure.json').exists(),'Incomplete/mismatched sampler terminal')
        trajectory=bind(terminal['trajectory']);require(trajectory==terminal_path.parent/'trajectory.jsonl','Unexpected journal path')
        banks=[b for b in prepared['cloud_banks'] if all(b[k]==job[k] for k in ('context_index','initialization','stream'))]
        require(len(banks)==1,'Missing/ambiguous inherited cloud bank');bank=banks[0]
        bind(bank['raw']);bind(bank['metadata'])
        start=None
        if job['initialization']=='proposal_prepared':
            starts=[s for s in prepared['alternative_starts'] if all(s[k]==job[k] for k in ('context_index','stream'))]
            require(len(starts)==1,'Missing/ambiguous inherited start');start=read(bind(starts[0]['record']))['selected']
            if 'ledger' in starts[0]:bind(starts[0]['ledger'])
        new.append((job,terminal,trajectory,bank,start))
        link=mapping[job['id']];require(set(link['controls'])=={'local','m4'} and all(link[k]==job[k] for k in ('context_index','initialization','stream')),
                                   'Matched control metadata differs')
        for arm,old_id in link['controls'].items():
            require(old_id not in used,'Control reused as an independent replicate');used.add(old_id)
            control=old_chains[old_id];cj=control['job']
            require(cj['arm']==arm and all(cj[k]==job[k] for k in ('context_index','initialization','stream')),'Wrong matched cached control')
            ref=controls['observations'][str(old_id)];path=bind(ref)
            require(cm['files'][path.name]==ref['sha256'],'Cached observations lack original manifest binding')
            cached.append((control,path))
    require(len(cached)==16 and {identity(c['job']) for c,_ in cached}==
            {(0,a,i,s) for a in ('local','m4') for i in ('source','proposal_prepared') for s in range(4)},'Incomplete cached sixteen-chain inventory')
    return inputs,new,cached,bind


def analyze(base,output):
    from prepare_two_root_dimer_benchmark import verify
    base,output=Path(base).resolve(),Path(output).resolve();config=verify(base);validate_inventory(config)
    require(not output.exists(),'Fresh analysis output required')
    require(read(base/'analysis-plan.json')==analysis_plan(),'Frozen analysis plan differs')
    protocol=read(base/'protocol.json')
    for name in SOURCE_FILES:require(sha(Path(__file__).parents[1]/name)==protocol['source_files'][name],'Observer source changed '+name)
    inputs,new,cached,bind=bind_complete_inputs(base,config)
    shape=read(bind(config['shape']));patch=read(bind(config['patch_map']));source=read(bind(config['source_frame']))['poses']
    require(len(source)==264 and patch['shape_sha256']==config['shape']['sha256'] and len(patch['atom_patch_ids'])==len(shape['atoms']),
            'Source/shape/patch identity differs')
    observer_config=dict(boundary=dict(kind='spherical',radius=config['physical']['wall_radius']),
                         box_lengths=[2*config['physical']['wall_radius']]*3,depletant_radius=config['physical']['depletant_radius'])
    output.mkdir();started=time.process_time();chains=[];endpoints=0
    write(output/'input-binding.json',dict(input_sha256=inputs,analysis_plan=analysis_plan(),
        source_sha256={n:protocol['source_files'][n] for n in SOURCE_FILES}))
    try:
        for job,terminal,path,cloud,start in new:
            tick=time.process_time();context=config['contexts'][job['context_index']];members=[context['root'],context['child']]
            initial=[source[i] for i in members] if start is None else start
            def rows(stream):
                for line in stream:
                    require(line.endswith('\n') and len(line)<=analysis_plan()['observer_limits']['max_record_bytes'],'Truncated/oversized journal row')
                    yield json.loads(line,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
            root_diagnostics=new_root_diagnostics(config,job)
            with path.open() as stream:points=validate_journal(rows(stream),config,job,initial,terminal,cloud,root_diagnostics)
            observer=previous.ConditionalObserver(shape,patch['atom_patch_ids'],observer_config,source,members)
            trace=[];prior=None;warmup=config['allocation']['warmup_blocks']
            with (output/f"job-{job['id']:03}-observations.jsonl").open('x') as handle:
                for point in points:
                    observation=observer.classify(point['selected']);tokens=set(observation['patch_tokens'])
                    old=set(prior['patch_tokens']) if prior else tokens;union=old|tokens
                    observation.update(block=point['block'],production=point['block']>warmup,
                        sampler_cpu_seconds=point['sampler_cpu_seconds'],jaccard_from_previous=1-len(old&tokens)/len(union) if union else 0.)
                    handle.write(canonical(observation)+'\n');trace.append(observation);prior=observation
            metrics=previous.summarize_trace(trace,warmup,terminal['cpu_seconds'])
            metrics['production_cpu_seconds']=points[-1]['sampler_cpu_seconds']-points[warmup]['sampler_cpu_seconds']
            metrics['external_only']=external.external_metrics(trace,warmup,terminal['cpu_seconds'],members)
            metrics['external_activity']=external_activity(trace,warmup,terminal['cpu_seconds'])
            chains.append(dict(job=job,metrics=metrics,reused_control=False,trajectory=terminal['trajectory'],counts=terminal['counts'],
                root_diagnostics=root_diagnostics,
                observer_cpu_seconds=time.process_time()-tick,pair_classifications=observer.calls,exact_pair_cache_hits=observer.hits,
                geometry_load_cpu_seconds=terminal.get('geometry_load_cpu_seconds')));endpoints+=len(points)
        for control,path in cached:
            c=copy.deepcopy(control);job=c['job'];context=config['contexts'][job['context_index']];members=[context['root'],context['child']]
            trace=external.read_cached_trace(path,inputs[str(path)],512,4608);cpu=c['metrics']['full_sampler_cpu_seconds']
            c['metrics']['external_only']=external.external_metrics(trace,512,cpu,members)
            c['metrics']['external_activity']=external_activity(trace,512,cpu)
            c.update(reused_control=True,new_geometry_queries=0);chains.append(c)
        require(endpoints==analysis_plan()['new_retained_initial_observations'] and len(chains)==24,'Incomplete analysis inventory')
        require(sum(c['pair_classifications'] for c in chains if not c['reused_control'])<=analysis_plan()['maximum_new_pair_classifications'],
                'Observer query allocation exceeded')
        for path,digest in inputs.items():require(sha(path)==digest,'Input changed during observation '+path)
        for name in SOURCE_FILES:require(sha(Path(__file__).parents[1]/name)==protocol['source_files'][name],'Observer source changed '+name)
        comparisons=external.comparison_summaries(chains)
        for pair in comparisons['descriptive_paired_comparisons']:
            same_start=pair['left']['initialization']==pair['right']['initialization']
            pair['control_global_rng_roles_paired']=same_start and {pair['left']['arm'],pair['right']['arm']}<={'m4',ARM}
            pair['local_rng_roles_paired']=same_start
            pair['root_order_rng_shared_with_control']=False
        result=dict(schema='two-root-dimer-analysis-v1',complete=True,new_chains=8,reused_control_chains=16,
            chains=chains,comparisons=comparisons,input_sha256=inputs,
            source_sha256={n:protocol['source_files'][n] for n in SOURCE_FILES},analysis_plan=analysis_plan(),
            costs=dict(new_analysis_cpu_seconds=time.process_time()-started,
                new_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in chains if not c['reused_control']),
                reused_control_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in chains if c['reused_control']),
                shared_preparation_not_repeated=True),new_geometry_endpoints=endpoints,old_geometry_queries=0,new_physical_draws=0,native_observer=False)
        write(output/'analysis.json',result)
        write(output/'manifest.json',dict(complete=True,files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        return result
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error=repr(error),completed_chains=len(chains),completed_endpoints=endpoints,
                                        input_sha256=inputs,retries=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.base,args.output);print(json.dumps({k:result[k] for k in ('complete','new_chains','reused_control_chains')}))
