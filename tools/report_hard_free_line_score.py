#!/usr/bin/env python3
"""Derive x/y/z guides from one audited xyz trace, without geometry queries."""
import argparse
from collections import Counter
import math
from pathlib import Path
import statistics
from prepare_hard_free_line_score import ARMS,read,rows,sha,write,require,validate_inputs


def lse(values):
    values=list(values)
    if not values or max(values)==-math.inf:return -math.inf
    largest=max(values)
    return largest+math.log(math.fsum(math.exp(x-largest) for x in values))


def number(value):return -math.inf if value is None else float(value)
def nullable(value):return value if math.isfinite(value) else None


def same_log(actual,expected,message):
    actual=number(actual)
    require(actual==expected if not math.isfinite(expected) else math.isfinite(actual) and abs(actual-expected)<1e-9,message)
    return abs(actual-expected) if math.isfinite(expected) else 0.


def derive(row,guide,log_volume):
    """Use only saved g/mass/fallback terms; no CDF or geometry recomputation."""
    require(guide['schema']=='defensive-hard-free-line-guide-v1' and guide['raw_translation_axes']==[0,1,2] and
            guide['conditional_probability']==1. and guide['minimum_conditional_mass']==1e-12 and
            guide['defensive_uniform_shell_probability']==.5,'Unexpected frozen xyz law')
    weights=[c['weight'] for c in guide['gaussian_components']];normalizer=math.fsum(weights)
    weights=[w/normalizer for w in weights]
    require(all(w>0 for w in weights),'Nonpositive component weight')
    details=row['density_details'];axes=details['axes']
    require([a['axis'] for a in axes]==[0,1,2] and details['component_branches']==3*len(weights),
            'Incomplete axis/component trace')
    uniform=math.log(.5)-log_volume if row['shell_valid'] else -math.inf
    output={};gaussian_logs=None;fallbacks=0;max_error=0.;stats={}
    for name,axis in zip(('x','y','z'),axes):
        components=axis['components']
        require([c['component'] for c in components]==list(range(len(weights))),'Missing/reordered component')
        current=[c['gaussian_log_density'] for c in components]
        require(all(math.isfinite(g) for g in current),'Nonfinite component Gaussian density')
        if gaussian_logs is None:gaussian_logs=current
        require(current==gaussian_logs,'Component Gaussian density changes across axes')
        terms=[uniform];count=Counter()
        for weight,c in zip(weights,components):
            z=c['conditional_mass'];fallback=z<=1e-12;allowed=c['query_coordinate_allowed']
            require(math.isfinite(z) and 0<=z<=1+1e-12 and c['conditional_sigma']>0 and
                    math.isfinite(c['conditional_mean']) and c['fallback']==fallback and
                    type(allowed) is bool,'Conditional trace changed or invalid')
            count.update(fallback=int(fallback),query_allowed=int(allowed),zero_conditional_mass=int(z==0),
                         positive_mass_floor_fallback=int(0<z<=1e-12))
            fallbacks+=int(fallback)
            correction=0. if fallback else (-math.log(z) if allowed else -math.inf)
            terms.append(math.log(.5)+math.log(weight)+c['gaussian_log_density']+correction)
        q=lse(terms);max_error=max(max_error,same_log(axis['axis_log_proposal_density'],q,'Saved axis density differs'))
        output[name]=q;stats[name]=dict(count,geometry_CPU_seconds=axis['geometry_cpu_seconds'],
            interval_count=len(axis['hard_free_intervals']),empty_reason=axis.get('empty_reason'))
    old=lse([uniform,*[math.log(.5)+math.log(w)+g for w,g in zip(weights,gaussian_logs)]])
    xyz=lse(output.values())-math.log(3)
    max_error=max(max_error,same_log(row['baseline_log_density'],old,'Original Gaussian density differs'),
                  same_log(row['log_proposal_density'],xyz,'xyz is not the complete axis-mixture average'))
    require(fallbacks==details['fallback_component_branches'],'Fallback count differs')
    output['xyz']=xyz
    if row['hard_valid'] and row['shell_valid'] and row['capture_valid']:
        require(all(q>=old-1e-9 for q in output.values()),'Conditioning lowered density on physical support')
    return dict(log_q={name:nullable(q) for name,q in output.items()},baseline_log_q=nullable(old),
                maximum_algebra_error=max_error,axes=stats)


def moments(logterms):
    total=lse(logterms)
    return dict(log_second_moment=total,contribution_ESS=math.exp(2*total-lse(2*t for t in logterms)),
                largest_contribution=math.exp(max(logterms)-total),rows=len(logterms))


def distribution(values):
    """Represent zero-density ratios explicitly, never silently discard them."""
    finite=[v for v in values if math.isfinite(v)];zeros=sum(v==-math.inf for v in values)
    require(len(values)==len(finite)+zeros,'NaN or positive-infinite density ratio')
    ordered=sorted(values)
    return dict(rows=len(values),zero_density_rows=zeros,
        minimum=nullable(min(values)),minimum_is_negative_infinity=bool(zeros),
        median=nullable(statistics.median(ordered)),maximum=nullable(max(values)),
        finite_log_ratio_values=finite)


def report(preparation,scored,audit_path):
    preparation,scored,audit_path=map(lambda p:Path(p).resolve(),(preparation,scored,audit_path))
    bindings={}
    def bind(path,expected=None):
        digest=sha(path);require(expected is None or digest==expected,'Saved artifact changed: '+str(path))
        bindings[str(path)]=digest;return path
    frozen=read(bind(preparation/'freeze.json'))
    for name,digest in frozen['files'].items():bind(preparation/name,digest)
    probes=rows(preparation/'score-probes.jsonl');saved=read(preparation/'saved-scores.json')['rows']
    validate_inputs(probes,saved)
    audit=read(bind(audit_path));require(audit['complete'] and audit['probes']==206,'Independent audit incomplete')
    for path,digest in audit['input_sha256'].items():bind(Path(path),digest)
    require(audit['input_sha256'].get(str(scored/'probes.jsonl'))==sha(scored/'probes.jsonl'),
            'Independent audit does not bind these scored rows')
    manifest=read(bind(scored/'manifest.json'));summary=read(bind(scored/'summary.json'))
    require(manifest['schema']=='hard-free-line-guide-audit-v1' and manifest['samples']==0 and
            manifest['physical_jobs']==0 and manifest['guide_sha256']==sha(preparation/'guides/xyz.json') and
            manifest['probes_sha256']==sha(preparation/'score-probes.jsonl') and
            summary['complete'] and summary['probes']==206 and summary['samples']==0 and
            summary['manifest']==manifest,'Score manifest/allocation changed')
    require((scored/'samples.jsonl').read_bytes()==b'','Score-only run generated new poses')
    records=rows(bind(scored/'probes.jsonl',summary['probes_sha256']))
    require(len(records)==206 and [r['id'] for r in records]==[p['id'] for p in probes],'Saved score identities changed')
    guide=read(preparation/'guides/xyz.json');derived=[]
    for previous,row in zip(saved,records):
        require(row['latent']==previous['latent'] and row['kind']=='probe' and row['draw'] is None,
                'Saved query was changed or resampled')
        for flag in ('hard_valid','shell_valid','capture_valid'):
            require(row[flag]==previous['saved_geometry'][flag],'Original saved flag changed')
        require(abs(row['log_physical_jacobian']-previous['saved_geometry']['log_physical_jacobian'])<1e-12,
                'Physical Jacobian changed')
        q=derive(row,guide,manifest['log_latent_ball_volume'])
        same_log(q['baseline_log_q'],previous['old_log_q']['baseline92'],'Original saved old92 density changed')
        derived.append(dict(id=row['id'],latent=row['latent'],**q))
    critical={}
    for source,count in [('baseline',4),('expanded',74)]:
        indices=[i for i,p in enumerate(saved) if p['group']=='critical' and p['metadata']['source']['arm']==source]
        require(len(indices)==count,'Critical original source changed')
        values={name:[] for name in ('old92',*ARMS)}
        for i in indices:
            src=saved[i]['metadata']['source']
            numerator=sum(src['paired_log_weights'])+src['log_q']-math.log(src['source_attempted_draws'])
            values['old92'].append(numerator-saved[i]['old_log_q']['baseline92'])
            for name in ARMS:
                require(derived[i]['log_q'][name] is not None,'Critical physical pose has zero density')
                values[name].append(numerator-derived[i]['log_q'][name])
        laws={name:moments(v) for name,v in values.items()}
        for name in ARMS: laws[name]['second_moment_ratio_vs_old92']=math.exp(laws[name]['log_second_moment']-laws['old92']['log_second_moment'])
        critical[source]=dict(original_source_components=84 if source=='baseline' else 92,rows=count,laws=laws)
    breadth={}
    for kind in ('native_R5','native_complement','competing','invalid'):
        indices=[i for i,p in enumerate(saved) if p['group']=='breadth' and p['metadata']['coverage_class']==kind]
        require(len(indices)==32,'Breadth allocation changed')
        breadth[kind]={name:distribution([number(derived[i]['log_q'][name])-saved[i]['old_log_q']['baseline92'] for i in indices])
                       for name in ARMS}
    geometry={name:dict(geometry_CPU_seconds=math.fsum(p['axes'][name]['geometry_CPU_seconds'] for p in derived),
        empty_reasons=dict(Counter(p['axes'][name]['empty_reason'] for p in derived if p['axes'][name]['empty_reason'])),
        fallback_component_branches=sum(p['axes'][name]['fallback'] for p in derived),
        zero_conditional_mass_branches=sum(p['axes'][name]['zero_conditional_mass'] for p in derived),
        positive_mass_floor_fallback_branches=sum(p['axes'][name]['positive_mass_floor_fallback'] for p in derived),
        intervals=sum(p['axes'][name]['interval_count'] for p in derived)) for name in ('x','y','z')}
    cpu=math.fsum(r['density_cpu_seconds'] for r in records)
    result=dict(schema='hard-free-line-score-review-v1',complete=True,source_sha256=bindings,queries=206,
        reported_densities=824,primary_arm='xyz',secondary_arms=['x','y','z'],critical_source_diagnostics=critical,
        breadth=breadth,geometry=geometry,full_xyz_density_CPU_seconds=cpu,full_xyz_ms_per_query=1000*cpu/206,
        whole_probe_CPU_seconds=summary['probe_total_cpu_seconds'],
        maximum_axis_algebra_error=max(p['maximum_algebra_error'] for p in derived),
        maximum_independent_audit_errors=audit['maximum_errors'],
        rows=derived,new_pose_draws=0,new_Poisson_clouds=0,new_physical_mass_estimates=0,
        physical_gates_unchanged=True,physical_campaign_ready=False,
        scope='Retrospective complete-density scores only. Axis laws derived without geometry queries; no fresh proposal retention, mixing efficiency, physical mass or assembly conclusion.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('preparation','scored','audit','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();write(args.out,report(args.preparation,args.scored,args.audit))
