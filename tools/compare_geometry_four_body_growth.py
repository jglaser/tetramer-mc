#!/usr/bin/env python3
"""Compare completed geometry-only N4 controls with the archived positive controls.

Read-only downstream analysis: no replay of the physical campaigns or raw
auditors, no pooling of proposal laws, and no stationary efficiency inference.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from native_graph_consistency import NativeGraphConsistency

BASELINE_SHA='9a07ccafa1d905a9b588fdb169a4e41ffc0e548b395f4b350213e2af327ac868'
GEOMETRY_MODEL_SHA='e90a7c85c5bdb2a587071949f0ae6527dba080594229434080d0761e4125983b'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def physical_config(config):
    result=copy.deepcopy(config)
    for k in ('metadata','seed','shape','monomer_shape'):result.pop(k,None)
    return result


def compact_run(result):
    keys=('arm','id','start','replicate','seed','sampler_cpu_seconds','first_D_contact',
        'first_D_native_entry','final_native_component_size','final_D_strict_supported_bonds',
        'final_D_retained_supported_bonds','maximum_D_strict_supported_bonds','final_registered_keys')
    output={k:result[k]for k in keys}
    output['catalogue_consistency']=result['catalogue_consistency']
    output['native_candidates']=result['native_candidates']
    output['branch_counts']=result['branch_counts']
    output['outcomes']={name:{k:value for k,value in outcome.items()if k not in ('tracked_body',)}
                        for name,outcome in result['outcomes'].items()}
    output['native_informed_proposal']=result['arm']!='geometry'
    return output


def compare(campaign,baseline,out):
    campaign,baseline,out=map(lambda p:Path(p).resolve(),(campaign,baseline,out))
    require(not out.exists(),'Use a fresh comparison output')
    old=read(baseline/'analysis.json');require(old['complete']and sha(baseline/'analysis.json')==BASELINE_SHA,'Baseline changed')
    for name,digest in read(baseline/'freeze.json').items():require(sha(baseline/name)==digest,'Frozen baseline changed')
    parent,status=read(campaign/'protocol.json'),read(campaign/'status.json')
    require(parent['schema']=='geometry-mobile-four-body-growth-controller-v1'and status['complete']and status['phase']=='complete','Geometry campaign incomplete')
    require(status['protocol_sha256']==sha(campaign/'protocol.json'),'Terminal protocol changed')
    require(parent['native_informed_proposal']is False and parent['native_initial_scaffold']is True,'Geometry/initialization scope changed')
    for name,digest in read(campaign/'freeze.json')['files'].items():require(sha(campaign/name)==digest,'Frozen geometry campaign changed')
    folder=campaign/'geometry';manifest=read(folder/'manifest.json');assessment=read(folder/'assessment/analysis.json')
    require(manifest['schema']=='mobile-geometry-four-body-growth-campaign-v1'and manifest['model_sha256']==GEOMETRY_MODEL_SHA,'Wrong geometry arm')
    require(assessment['complete']and len(assessment['runs'])==len(status['jobs'])==len(manifest['jobs'])==4,'Missing geometry runs')
    require(status['audits']['geometry']['returncode']==0 and status['audits']['geometry']['analysis_sha256']==sha(folder/'assessment/analysis.json'),'Raw audit changed')
    require(assessment['manifest_sha256']==sha(folder/'manifest.json')and assessment['terminal_status_sha256']==sha(folder/'status.json')
        and assessment['analyzer_sha256']==manifest['observer_sha256'],'Observer source/terminal binding changed')
    positive=Path(old['campaign']);sources={str(baseline/'analysis.json'):BASELINE_SHA,
        str(folder/'assessment/analysis.json'):sha(folder/'assessment/analysis.json')}
    require(old['status_sha256']==sha(positive/'status.json'),'Positive campaign terminal changed')
    for path,digest in old['source_sha256'].items():require(sha(path)==digest,'Positive raw evidence changed')
    controls=read(positive/'original/manifest.json')
    require(manifest['shape_sha256']==controls['shape_sha256']and manifest['binary_sha256']==controls['binary_sha256']
        and manifest['source_bundle_sha256']==controls['source_bundle_sha256'],'Physical shape or executable changed')
    catalogue=folder/'provenance/native-pair-motifs.json'
    require(sha(catalogue)==manifest['input_sha256']['native-pair-motifs.json']==controls['input_sha256']['native-pair-motifs.json'],'Native observer catalogue changed')
    checker=NativeGraphConsistency(read(catalogue)['motifs'],4)
    reports=[compact_run(r)for r in old['runs']];traces=copy.deepcopy(old['traces'])
    for result,job in zip(assessment['runs'],manifest['jobs']):
        require(result['passed']and result['id']==job['id']and result['seed']==job['seed'],'Run allocation changed')
        terminal=next(j for j in status['jobs']if j['id']==job['id'])
        require(terminal['status']=='complete'and terminal['returncode']==0,'Failed geometry run')
        for name,digest in result['source_sha256'].items():
            path=Path(job['directory'])/name
            require(sha(path)==digest==terminal['output']['files'][name],'Geometry raw output changed');sources[str(path)]=digest
        previous=next(j for j in controls['jobs']if j['start']==job['start']and j['replicate']==job['replicate'])
        require(physical_config(read(job['config']))==physical_config(read(previous['config'])),'Physical parameters or starts differ')
        require(job['seed']not in [r['seed']for r in old['runs']],'Geometry seed reuses a positive-control stream')
        g=result['four_body_growth'];support=g['supported_monomer_bonds_by_saved_sweep'];coherent={}
        require(len(result['rows'])==2001 and result['audit']['all_move_records']==12000,'Incomplete sweep/update axis')
        for label,field in [('retained','registered_keys'),('entry','instantaneous_entry_keys')]:
            first=None;incompatible={};count=0
            for event in result['graph_history']:
                c=checker.check(event[field])
                if c['consistent_connected']and first is None:
                    first=dict(sweep=event['sweep'],serial=event['serial'],source=event['source'],initial_state=event['serial']==-1)
                if not c['consistent']:
                    count+=1;incompatible.setdefault(str(event[field]),dict(first_serial=event['serial'],keys=event[field],check=c))
            coherent[label]=dict(first_consistent_connected_four=first,inconsistent_observations=count,
                incompatible_graph_witnesses=list(incompatible.values()),initial=checker.check(result['graph_history'][0][field]),
                final=checker.check(result['graph_history'][-1][field]))
        raw=dict(arm='geometry',id=job['id'],start=job['start'],replicate=job['replicate'],seed=job['seed'],
            sampler_cpu_seconds=result['sampler_cpu_seconds'],first_D_contact=g['outcomes']['nonspecific']['first_observed']['tracked_attached'],
            first_D_native_entry=g['outcomes']['instantaneous_entry']['first_observed']['tracked_attached'],
            final_native_component_size=result['rows'][-1]['native']['largest_component_size'],
            final_D_strict_supported_bonds=support[-1]['tracked_entry_count'],final_D_retained_supported_bonds=support[-1]['tracked_retained_count'],
            maximum_D_strict_supported_bonds=max(s['tracked_entry_count']for s in support),
            final_registered_keys=result['rows'][-1]['registered_keys'],catalogue_consistency=coherent,
            native_candidates=result['native_candidates'],branch_counts=result['branch_counts'],outcomes=g['outcomes'])
        reports.append(compact_run(raw))
        traces.append(dict(arm='geometry',id=job['id'],start=job['start'],replicate=job['replicate'],
            sweep=[r['sweep']for r in result['rows']],native_size=[r['native']['largest_component_size']for r in result['rows']],
            contact_size=[r['nonspecific']['largest_component_size']for r in result['rows']],
            D_strict_bonds=[r['tracked_entry_count']for r in support],D_retained_bonds=[r['tracked_retained_count']for r in support]))
    scope=('Matched physical starts, bath, wall and move schedule. Geometry-only atlas versus separately retained native-informed positive controls. '
        'Native initial scaffold is supplied in every arm. Four fresh geometry streams; two per start. No equilibrium, speedup or de novo assembly claim.')
    output=dict(schema='geometry-four-body-growth-comparison-v1',complete=True,campaign=str(campaign),baseline=str(baseline),
        protocol_sha256=sha(campaign/'protocol.json'),status_sha256=sha(campaign/'status.json'),source_sha256=sources,
        runs=reports,traces=traces,scope=scope)
    out.mkdir();write(out/'analysis.json',output)
    def first(x):return 'not observed'if x is None else 'initial'if x['initial_state']else str(x['sweep'])
    lines=['# Geometry-only versus native-informed four-body growth','',scope,'',
        '| Atlas | Start | Repeat | First D contact | First strict native entry | First consistent entry4 | Final native group | D monomer bonds | CPU s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        lines.append('| '+' | '.join([r['arm'],r['start'],str(r['replicate']),first(r['first_D_contact']),first(r['first_D_native_entry']),
            first(r['catalogue_consistency']['entry']['first_consistent_connected_four']),str(r['final_native_component_size']),
            str(r['final_D_strict_supported_bonds']),f"{r['sampler_cpu_seconds']:.2f}"])+' |')
    lines+=['','All rejected attempts and initial occupancy are retained. Pair-native cycle consistency is a catalogue compatibility check, not a global crystallinity measure.',
        '','![All twelve trajectories](geometry-growth.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n');plot(traces,out)
    for name in (Path(__file__).name,'native_graph_consistency.py'):(out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    write(out/'freeze.json',{p.name:sha(p)for p in out.iterdir()if p.is_file()})
    print(json.dumps(dict(complete=True,out=str(out),analysis_sha256=sha(out/'analysis.json'))));return output


def plot(traces,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(15,10),sharex=True,sharey=True,layout='constrained')
    for r in traces:
        row=2*int(r['start']=='retained_motif8')+r['replicate'];column=('original','coverage','geometry').index(r['arm']);ax=axes[row,column]
        ax.step(r['sweep'],r['contact_size'],where='post',color='#999999',lw=1,label='exclusion group')
        ax.step(r['sweep'],r['native_size'],where='post',color='#2459a6',lw=1,label='native group')
        ax.plot(r['sweep'],r['D_strict_bonds'],color='#cb6a20',lw=.8,label='D native monomer bonds')
        ax.set_title(f"{r['arm']} · {r['start']} · {r['replicate']}",fontsize=9);ax.grid(axis='y',alpha=.2)
        if column==0:ax.set_ylabel('Body count / bonds')
        if row==3:ax.set_xlabel('MC sweep')
    ymax=max(max(r[k])for r in traces for k in ('contact_size','native_size','D_strict_bonds'))
    axes[0,0].set_ylim(0,ymax+.4);axes[0,0].legend(loc='upper right',fontsize=7)
    fig.suptitle('Identical seeded growth test: native-informed and geometry-only proposals\nAll bodies mobile; finite-time accessibility, not equilibrium',fontsize=12)
    fig.savefig(out/'geometry-growth.png',dpi=170);fig.savefig(out/'geometry-growth.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('runs/mobile-four-body-growth-comparison-v2-20260921'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();compare(a.campaign,a.baseline,a.out)
