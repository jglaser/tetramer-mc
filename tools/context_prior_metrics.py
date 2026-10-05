"""Pure retained-contact metrics for the frozen fixed-context prior pilot.

No geometry, native classifier, source selection, or physical sampling.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np
from analyze_evolving_dimer_benchmark import categorical_summary


def require(ok, message):
    if not ok:raise ValueError(message)


def fingerprint(tokens):
    text=json.dumps(sorted(map(tuple,tokens)),separators=(',',':'),allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def set_ess(values, cpu, chunk_size=32):
    """Same vector covariance/Geyer estimator, accumulated in bounded FFT chunks."""
    require(math.isfinite(cpu) and cpu>0,'Positive full CPU required')
    require(type(chunk_size) is int and chunk_size>0,'Positive fixed chunk size required')
    values=[set(v) for v in values];n=len(values)
    common=dict(samples=n,sampling_CPU_seconds=cpu,
        scope='Sample-centered finite-record descriptor; stationarity, coverage and equilibrium ESS not established.')
    absent=dict(common,iact_samples=None,apparent_ess=None,apparent_ess_per_sampling_CPU_second=None)
    if n<2:return dict(absent,reason='Fewer than two observed endpoints; mixing unresolved.')
    entries=defaultdict(list)
    for row,labels in enumerate(values):
        for label in labels:entries[label].append(row)
    labels=sorted(entries);size=1<<(2*n-1).bit_length();power=np.zeros(size//2+1)
    for start in range(0,len(labels),chunk_size):
        selected=labels[start:start+chunk_size];matrix=np.zeros((n,len(selected)))
        for k,label in enumerate(selected):matrix[entries[label],k]=1.
        matrix-=matrix.mean(axis=0)
        power+=np.square(np.abs(np.fft.rfft(matrix,n=size,axis=0))).sum(axis=1)
    covariance=np.fft.irfft(power,n=size)[:n]/n
    if covariance[0]<1e-24:return dict(absent,reason='Constant observed descriptor; unresolved mixing, not infinite ESS.')
    rho=covariance/covariance[0];pairs=[];last=math.inf
    for lag in range(0,min(len(rho)-1,max(1,n//2)),2):
        pair=float(rho[lag]+rho[lag+1])
        if pair<=0:break
        pair=min(pair,last);pairs.append(pair);last=pair
    raw=-1+2*sum(pairs);tau=max(1.,raw);ess=n/tau
    return dict(common,iact_samples=tau,unfloored_iact_samples=raw,apparent_ess=ess,
        apparent_ess_per_sampling_CPU_second=ess/cpu,last_included_lag=2*len(pairs)-1,
        descriptor_components=len(labels),fft_chunk_size=chunk_size,
        estimator='Biased sample-centered FFT vector covariance, fixed32-component accumulation; positive monotone lag pairs, half-record cap, IACT floor1.')


def runs(labels, cycles):
    require(len(labels)==len(cycles),'Run index mismatch')
    result=[];start=0
    for end in range(1,len(labels)+1):
        if end<len(labels) and labels[end]==labels[start]:continue
        result.append(dict(label=labels[start],first_index=start,last_index=end-1,
            start_cycle=cycles[start],last_cycle=cycles[end-1],observations=end-start,
            exit_cycle=cycles[end] if end<len(cycles) else None,
            left_censored=start==0,right_censored=end==len(labels)))
        start=end
    return result


def qualified_visits(labels, cycles, dwell=5, keep=lambda _x:True):
    qualified=[]
    for row in runs(labels,cycles):
        if row['observations']>=dwell and keep(row['label']):
            row=dict(row,confirmation_cycle=cycles[row['first_index']+dwell-1])
            if not qualified or qualified[-1]['label']!=row['label']:qualified.append(row)
    return qualified


def visit_inventory(labels,cycles,dwell=5):
    """Retain all runs, including empty, repeated and censored visits."""
    return [dict(row,qualifies=row['observations']>=dwell,
        confirmation_cycle=cycles[row['first_index']+dwell-1] if row['observations']>=dwell else None)
        for row in runs(labels,cycles)]


def returns(visits):
    result=[dict(source=visits[i-2]['label'],via=visits[i-1]['label'],
        departure_confirmation_cycle=visits[i-2]['confirmation_cycle'],
        return_confirmation_cycle=visits[i]['confirmation_cycle'])
        for i in range(2,len(visits)) if visits[i]['label']==visits[i-2]['label']]
    disjoint=[];i=0
    while i+2<len(visits):
        if visits[i]['label']==visits[i+2]['label']:
            disjoint.append([visits[i]['confirmation_cycle'],visits[i+2]['confirmation_cycle']]);i+=2
        else:i+=1
    return dict(completed_returns=result,nonoverlapping_roundtrips=disjoint)


def persistent_neighbors(sets, cycles):
    labels=[tuple(sorted(v)) for v in sets]
    visits=qualified_visits(labels,cycles,keep=bool)
    transitions=[]
    for a,b in zip(visits,visits[1:]):
        lost=sorted(set(a['label'])-set(b['label']));gained=sorted(set(b['label'])-set(a['label']))
        kind='replacement' if lost and gained else ('attachment' if gained else 'detachment')
        transitions.append(dict(source=a['label'],target=b['label'],lost=lost,gained=gained,kind=kind,
            source_confirmation=a['confirmation_cycle'],target_confirmation=b['confirmation_cycle']))
    return dict(dwell_cycles=5,all_runs=visit_inventory(labels,cycles),visits=visits,transitions=transitions,
        completed_partner_exchanges=sum(t['kind']=='replacement' for t in transitions),**returns(visits))


def ab_label(neighbors):
    key=tuple(sorted(neighbors))
    return 'A' if key==(16,217) else ('B' if key==(16,56) else 'Other')


def ab_diagnostics(instantaneous, cycle_rows):
    labels=[ab_label(r['neighbor_labels']) for r in cycle_rows];cycles=[r['cycle'] for r in cycle_rows]
    visits=qualified_visits(labels,cycles,keep=lambda label:label in ('A','B'))
    passages=[]
    for a,b in zip(visits,visits[1:]):
        path=range(a['first_index'],b['first_index']+5)
        passages.append(dict(source=a['label'],target=b['label'],source_confirmation=a['confirmation_cycle'],
            target_confirmation=b['confirmation_cycle'],intermediate_path=[dict(cycle=cycles[k],label=labels[k]) for k in path]))
    def occupancy(rows):
        counts=Counter(ab_label(r['neighbor_labels']) for r in rows)
        return {name:dict(count=counts[name],fraction=counts[name]/len(rows)) for name in ('A','B','Other')}
    return dict(instantaneous_unconditional_occupancy=occupancy(instantaneous),cycle_occupancy=occupancy(cycle_rows),
        dwell_cycles=5,all_runs=visit_inventory(labels,cycles),confirmed_visits=visits,completed_passages=passages,
        directional_counts={a+'->'+b:sum(p['source']==a and p['target']==b for p in passages) for a,b in [('A','B'),('B','A')]},
        **returns(visits),scope='Other retains all physical weight; projecting visits changes only the secondary persistence diagnostic.')


def sample_summary(rows, cpu):
    require(rows,'Empty production trace')
    patches=[set(map(tuple,r['patch_tokens'])) for r in rows]
    neighbors=[set(r['neighbor_labels']) for r in rows]
    labels=[fingerprint(p) for p in patches];indices=[r['attempt_index'] for r in rows]
    pc=Counter(t for p in patches for t in p);nc=Counter(t for p in neighbors for t in p)
    occupancy=Counter(labels)
    return dict(samples=len(rows),full_sampler_cpu_seconds=cpu,
        patch_ess=set_ess(patches,cpu),neighbor_ess=set_ess(neighbors,cpu),
        fingerprint_ess=set_ess([{label} for label in labels],cpu),
        fingerprint=categorical_summary(labels,indices),
        neighbor_sets=categorical_summary([fingerprint([(v,) for v in s]) for s in neighbors],indices),
        patch_occupancy=[dict(token=t,count=n,fraction=n/len(rows)) for t,n in sorted(pc.items())],
        neighbor_occupancy=[dict(label=t,count=n,fraction=n/len(rows)) for t,n in sorted(nc.items())],
        unique_fingerprints=len(occupancy),singleton_fingerprint_fraction=sum(n==1 for n in occupancy.values())/len(rows),
        constant_fingerprint=len(occupancy)==1,empty_contact_fraction=sum(not x for x in neighbors)/len(rows))


def distribution(values):
    a=np.asarray(values,float)
    require(a.ndim==1 and np.isfinite(a).all(),'Invalid finite diagnostic values')
    q=[0.,.01,.05,.25,.5,.75,.95,.99,1.]
    return dict(n=len(a),quantile_probabilities=q,quantiles=np.quantile(a,q).tolist() if len(a) else None,
        mean=float(a.mean()) if len(a) else None)


def factor_summary(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['slot'],row['branch'],row['accepted'])].append(row)
    output=[]
    for (slot,branch,accepted),population in sorted(groups.items()):
        output.append(dict(slot=slot,branch=branch,accepted=accepted,attempts=len(population),
            dispositions=dict(Counter(r['status'] for r in population)),
            log_proposal_reverse_forward=distribution([r['logq'] for r in population]),
            bath_log_weight=distribution([r['bath_log_weight'] for r in population if r['bath_log_weight'] is not None]),
            raw_log_acceptance=distribution([r['raw_log_acceptance'] for r in population if r['raw_log_acceptance'] is not None])))
    return output


def contact_changes(rows,previous=None):
    """All attempted changes; no threshold or acceptance filtering."""
    groups=defaultdict(list);neighbor_changes=Counter();patch_changes=Counter();transitions=[]
    for row in rows:
        if previous is None:previous=row;continue
        old=set(map(tuple,previous['patch_tokens']));new=set(map(tuple,row['patch_tokens']))
        union=old|new;distance=len(old^new)/len(union) if union else 0.
        groups[(row['slot'],row['status'])].append(distance)
        oldn=set(previous['neighbor_labels']);newn=set(row['neighbor_labels'])
        for label in oldn-newn:neighbor_changes[(label,'removal')]+=1
        for label in newn-oldn:neighbor_changes[(label,'addition')]+=1
        # Classify patch changes only while the SAME neighbor remains present.
        for label in oldn&newn:
            a={t for t in old if label in t[:2]};b={t for t in new if label in t[:2]}
            lost,gained=a-b,b-a
            if lost or gained:
                kind='replacement' if lost and gained else ('addition' if gained else 'removal')
                patch_changes[(label,kind)]+=1
        if old!=new:
            transitions.append(dict(attempt_index=row['attempt_index'],cycle=row['cycle'],slot=row['slot'],
                lost=sorted(old-new),gained=sorted(new-old),jaccard_distance=distance))
        previous=row
    return dict(consecutive_pairs=sum(map(len,groups.values())),
        patch_jaccard_by_slot_disposition=[dict(slot=s,status=k,**distribution(v)) for (s,k),v in sorted(groups.items())],
        neighbor_additions_removals=[dict(label=label,kind=kind,count=n) for (label,kind),n in sorted(neighbor_changes.items())],
        same_neighbor_patch_changes=[dict(label=label,kind=kind,count=n) for (label,kind),n in sorted(patch_changes.items())],
        changed_patch_sets=transitions)


def persistent_patches(rows):
    cycles=[r['cycle'] for r in rows];labels=[fingerprint(r['patch_tokens']) for r in rows]
    dictionary={fingerprint(r['patch_tokens']):set(map(tuple,r['patch_tokens'])) for r in rows}
    visits=qualified_visits(labels,cycles);transitions=[]
    for a,b in zip(visits,visits[1:]):
        old,new=dictionary[a['label']],dictionary[b['label']];lost,gained=old-new,new-old
        kind='replacement' if lost and gained else ('addition' if gained else 'removal')
        transitions.append(dict(source=a['label'],target=b['label'],kind=kind,lost=sorted(lost),gained=sorted(gained),
            source_confirmation=a['confirmation_cycle'],target_confirmation=b['confirmation_cycle']))
    return dict(all_runs=visit_inventory(labels,cycles),visits=visits,changes=len(transitions),transitions=transitions,
        **returns(visits),scope='Persistent threshold patch-set changes, not native registry or basin transitions.')


def summarize(rows, full_cpu):
    production=[r for r in rows if r['production']]
    cycles=[r for r in production if r['slot']==4]
    require(len(production)==10240 and len(cycles)==2048,'Changed frozen production inventory')
    boundary=next((r for r in reversed(rows) if not r['production']),None)
    return dict(all_attempts=sample_summary(production,full_cpu),cycle_endpoints=sample_summary(cycles,full_cpu),
        persistent_neighbors=persistent_neighbors([r['neighbor_labels'] for r in cycles],[r['cycle'] for r in cycles]),
        persistent_patch_changes=persistent_patches(cycles),
        instantaneous_contact_changes=contact_changes(production,boundary),
        certified_environment_exchanges=ab_diagnostics(production,cycles),
        proposal_and_bath_factors=factor_summary(production))
