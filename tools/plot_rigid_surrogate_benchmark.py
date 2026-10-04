#!/usr/bin/env python3
"""Present saved, whole-benchmark metrics; no trajectory or geometry access.

Metric choices are fixed before reading outcomes. Driver completion evidence is
joined by metadata and terminal hashes; its published input validation is
inherited, not repeated by scanning scientific journals or observation caches.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ARMS=('local','m4','rigid_surrogate_1','rigid_surrogate_8','rigid_surrogate_flat8')
LABELS=('local','m4','guided\n1 step','guided\n8 steps','unguided\n8 steps')
STARTS=('source','proposal_prepared')
METRICS=(
    ('contact_fingerprint_ess_per_cpu',('fingerprint_ess','apparent_ess_per_sampling_CPU_second'),
     'Whole contact-fingerprint ESS / CPU s\n(internal + external)'),
    ('external_edge_presence_ess_per_cpu',('external_only','ess','apparent_ess_per_sampling_CPU_second'),
     'External-edge presence ESS / CPU s'),
    ('nonempty_external_returns_per_cpu',('external_activity','completed_nonempty_returns_per_full_CPU_second'),
     'Completed nonempty external returns / CPU s'),
    ('any_external_contact_fraction',('external_only','any_contact_fraction'),
     'Any external contact: production fraction'))


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def record(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path))


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def authenticate(root):
    """Require a separate, successful one-job controller before reading metrics."""
    refs={}
    def load(relative):
        path=(root/relative).resolve();require(path.is_relative_to(root),'Escaping completion path')
        refs[str(path)]=sha(path)
        return json.loads(path.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
    plan=load('execution-plan.json');claim=load('execution/claim.json')
    summary=load('execution/summary.json');status=load('execution/status.json')
    require(plan['schema']=='native-class-physical-execution-v1' and plan['root']==str(root)
        and plan['maximum_workers']==plan['threads']==1 and len(plan['jobs'])==1,'Wrong analysis controller plan')
    plan_hash=refs[str(root/'execution-plan.json')]
    require(claim['schema']==plan['schema'] and claim['plan_sha256']==plan_hash
        and claim['maximum_workers']==claim['threads']==1 and claim['retries']==claim['replacements']==0,
        'Controller claim differs')
    keys=('complete','passed','failure','active','unstarted','completed','plan_sha256')
    require(all(summary[k]==status[k] for k in keys) and summary['complete'] is True and summary['passed'] is True
        and summary['failure'] is None and summary['active'] is None and summary['unstarted']==[]
        and summary['plan_sha256']==plan_hash and len(summary['completed'])==1
        and not (root/'execution/failure.json').exists() and not (root/'analysis/failure.json').exists(),
        'Independent analysis execution is incomplete, failed or undrained')
    job=plan['jobs'][0]
    require(job['id']=='whole-contact-analysis' and job['phase']=='statistics'
        and job['terminal']==dict(path=str(root/'analysis/analysis.json'),success_contract='complete'),
        'Wrong declared analysis terminal')
    directory=Path('execution/jobs')/('000-'+job['id'])
    attempt=load(directory/'attempt.json');process=load(directory/'process.json')
    done=load(directory/'success.json');exit_record=load(directory/'exit.json')
    require(done==exit_record==summary['completed'][0] and attempt['job']==job
        and all(done[k]==job[k] for k in ('id','population','phase','argv'))
        and done['success'] is True and done['child_started'] is True and done['child_drained'] is True
        and done['returncode']==0 and done['error'] is None and done['timeout'] is False
        and done['retries']==done['replacements']==0 and done['success_contract']=='complete'
        and type(done['pid']) is int and done['pid']>0
        and all(process[k]==done[k] for k in ('id','pid','birth_ticks','argv')),
        'Analysis job lifecycle differs')
    protocol=load('protocol.json');prepared=load('preparation.json')
    require(protocol['schema']=='rigid-surrogate-analysis-dispatch-v1'
        and prepared['complete'] is True and prepared['launched'] is False
        and prepared['execution_plan']==dict(path=str(root/'execution-plan.json'),sha256=plan_hash)
        and prepared['protocol']==dict(path=str(root/'protocol.json'),sha256=refs[str(root/'protocol.json')])
        and plan['files'].get(str(root/'protocol.json'))==refs[str(root/'protocol.json')]
        and claim['preparation_receipt']==dict(path=str(root/'preparation.json'),sha256=refs[str(root/'preparation.json')])
        and not (root/'preparation-failure.json').exists(),'Analysis preparation provenance differs')
    result=load('analysis/analysis.json');manifest=load('analysis/manifest.json')
    require(done['terminal']==dict(path=str(root/'analysis/analysis.json'),sha256=refs[str(root/'analysis/analysis.json')])
        and manifest['complete'] is True and manifest['files']['analysis.json']==done['terminal']['sha256'],
        'Completed result hash differs')
    require(result['schema']=='rigid-surrogate-analysis-v1' and result['complete'] is True
        and result['new_chains']==24 and result['reused_control_chains']==16 and len(result['chains'])==40
        and result['new_geometry_endpoints']==110616 and result['old_geometry_queries']==result['new_physical_draws']==0
        and result['native_observer'] is False and protocol['new_chains']==24 and protocol['cached_control_chains']==16,
        'Incomplete or unexpected whole-benchmark result')
    declaration=protocol['analysis_plan'];path=Path(declaration['path']).resolve()
    require(sha(path)==declaration['sha256']==plan['files'].get(str(path)),
        'Unbound frozen analysis declaration')
    refs[str(path)]=declaration['sha256']
    require(json.loads(path.read_text())==result['analysis_plan'],
        'Result differs from frozen analysis declaration')
    return result,refs


def saved_values(result):
    rows=[];seen=set()
    for chain in result['chains']:
        job=chain['job'];identity=(job['context_index'],job['arm'],job['initialization'],job['stream'])
        require(identity not in seen,'Duplicate stream identity');seen.add(identity)
        require(chain['reused_control'] is (job['arm'] in ('local','m4')),'Wrong new/cached stream origin')
        metrics=chain['metrics'];cpu=metrics['full_sampler_cpu_seconds']
        require(metrics['production_samples']==4096 and math.isfinite(cpu) and cpu>0
            and metrics['external_activity']['production_samples']==4096
            and metrics['external_activity']['full_sampler_cpu_seconds']==cpu,'Wrong sample count or CPU denominator')
        values={}
        for name,path,_ in METRICS:
            value=metrics
            for key in path:value=value[key]
            if value is None:require('ess' in name,'Only undefined ESS may be null')
            else:require(type(value) in (float,int) and math.isfinite(value) and value>=0,'Invalid saved metric')
            values[name]=value
        require(values['any_external_contact_fraction']<=1,'Invalid saved contact fraction')
        singletons=metrics['singleton_fingerprint_fraction']
        require(type(singletons) in (int,float) and math.isfinite(singletons)
            and 0<=singletons<=1,'Invalid singleton-fingerprint fraction')
        rows.append(dict(job=job,reused_control=chain['reused_control'],full_sampler_cpu_seconds=cpu,
            **values,whole_patch_fingerprint_ess=metrics['fingerprint_ess'],
            singleton_fingerprint_fraction=singletons))
    expected={(0,arm,start,stream) for arm in ARMS for start in STARTS for stream in range(4)}
    require(seen==expected and len(rows)==40,'All four streams, five arms and two starts must remain present')
    return sorted(rows,key=lambda r:(STARTS.index(r['job']['initialization']),r['job']['stream'],ARMS.index(r['job']['arm'])))


def saved_agreement(result):
    """Copy the predeclared between-start comparisons; do not pool trajectories."""
    rows=[];seen=set()
    for pair in result['comparisons']['descriptive_paired_comparisons']:
        left,right=pair['left'],pair['right']
        if left['arm']!=right['arm']:continue
        require(left['context_index']==right['context_index']==0
            and left['stream']==right['stream']
            and {left['initialization'],right['initialization']}==set(STARTS),
            'Mismatched initial-condition comparison')
        key=(left['arm'],left['stream']);require(key not in seen,'Duplicate start comparison');seen.add(key)
        values=pair['external_only']
        row=dict(arm=left['arm'],stream=left['stream'],
            external_environment_TV=values['environment_total_variation'],
            maximum_external_edge_occupancy_difference=values['edge_occupancy_max_difference'])
        for name in ('external_environment_TV','maximum_external_edge_occupancy_difference'):
            value=row[name]
            require(type(value) in (float,int) and math.isfinite(value) and 0<=value<=1,
                'Invalid saved initial-condition difference')
        rows.append(row)
    require(seen=={(arm,stream) for arm in ARMS for stream in range(4)}
        and len(rows)==20,'All twenty between-start comparisons must remain present')
    return sorted(rows,key=lambda r:(ARMS.index(r['arm']),r['stream']))


def plot(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    require(not output.exists(),'Fresh plot directory required; no overwrite')
    source=record(__file__);result,inputs=authenticate(root);rows=saved_values(result)
    agreement=saved_agreement(result)
    # Imported only after the independent whole-analysis completion check.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    output.mkdir(parents=True)
    try:
        colors=('#0072B2','#D55E00','#009E73','#CC79A7');markers=('o','s','^','D')
        fig,axes=plt.subplots(2,4,figsize=(19,9.6),sharey='col')
        for row_index,start in enumerate(STARTS):
            group=[r for r in rows if r['job']['initialization']==start]
            for col,(name,_,title) in enumerate(METRICS):
                ax=axes[row_index,col]
                for stream in range(4):
                    points={r['job']['arm']:r[name] for r in group if r['job']['stream']==stream}
                    ys=[np.nan if points[arm] is None else points[arm] for arm in ARMS]
                    ax.plot(range(len(ARMS)),ys,color=colors[stream],marker=markers[stream],lw=.9,ms=5,alpha=.85)
                for x,arm in enumerate(ARMS):
                    missing=[str(r['job']['stream']) for r in group if r['job']['arm']==arm and r[name] is None]
                    if missing:
                        ax.text(x,-.21,'undefined\n'+','.join(missing),transform=ax.get_xaxis_transform(),
                            ha='center',va='top',fontsize=8,color='#555555',clip_on=False)
                ax.set_xticks(range(len(ARMS)),LABELS);ax.set_xlim(-.3,len(ARMS)-.7)
                ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
                if row_index==0:ax.set_title(title,fontsize=10,pad=13)
                if col==0:ax.set_ylabel('Source start' if start=='source' else 'Proposal-prepared start',fontsize=11)
                defined=[r[name] for r in group if r[name] is not None]
                if not defined:
                    ax.text(.5,.5,'ESS undefined for all streams',transform=ax.transAxes,ha='center',fontsize=9,color='#555555')
                elif max(defined)==0:
                    label='All defined fractions are zero' if name.endswith('fraction') else 'All defined rates are zero'
                    ax.text(.5,.5,label,transform=ax.transAxes,ha='center',fontsize=9,color='#555555')
        # Shared-column axes need both starts populated before fixed limits.
        # These are display bounds only; nulls never enter them as zero values.
        for col,(name,_,_) in enumerate(METRICS):
            defined=[r[name] for r in rows if r[name] is not None]
            if name.endswith('fraction'):axes[0,col].set_ylim(-.025,1.025)
            elif name=='external_edge_presence_ess_per_cpu' and defined and min(defined)>0:
                axes[0,col].set_yscale('log')
                axes[0,col].set_ylim(.6*min(defined),1.4*max(defined))
                axes[0,col].set_title(METRICS[col][2]+' · log scale',fontsize=10,pad=13)
            else:
                maximum=max(defined) if defined else None
                upper=1.12*maximum if maximum is not None and maximum>0 else 1.
                axes[0,col].set_ylim(-.04*upper,upper)
                if maximum is None:axes[0,col].set_yticks([])
        fig.suptitle('Corrected rigid-surrogate comparison · each stream retained separately',fontsize=15,y=.98)
        handles=[Line2D([0],[0],color=colors[s],marker=markers[s],lw=.9,label=f'Stream {s}') for s in range(4)]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.935),ncol=4,frameon=False)
        fig.subplots_adjust(left=.075,right=.985,top=.825,bottom=.25,hspace=.58,wspace=.29)
        fig.text(.075,.045,'Saved production metrics; rates use full sampler CPU, including warmup and rejected work.\n'
            'Lines join matching streams across arms; starts are separate. Undefined ESS is not zero.\n'
            'Whole-fingerprint ESS may reflect unique local contact labels; it does not certify independent basins.\n'
            'Rare-contact presence ESS can be high after a single occupied frame; read alongside returns and occupancies.\n'
            'Conditional growth-state test: 2 mobile + 262 fixed tetramers, rd 1.4 Å, z 0.0275 Å⁻³. No native-stability inference.',fontsize=10,va='bottom')
        fig.savefig(output/'contact-efficiency.png',dpi=180,facecolor='white')
        fig.savefig(output/'contact-efficiency.svg',facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True)
        definitions=(('external_environment_TV','Whole external contact-set occupancy'),
            ('maximum_external_edge_occupancy_difference','Largest external-edge occupancy difference'))
        for ax,(name,title) in zip(axes,definitions):
            for stream in range(4):
                by_arm={r['arm']:r[name] for r in agreement if r['stream']==stream}
                ax.plot(range(len(ARMS)),[by_arm[a] for a in ARMS],color=colors[stream],
                    marker=markers[stream],lw=.9,ms=5)
            ax.set_xticks(range(len(ARMS)),LABELS);ax.set_xlim(-.3,len(ARMS)-.7)
            ax.set_ylim(-.025,1.025);ax.set_title(title,fontsize=11)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Difference between source and prepared starts')
        fig.suptitle('Initialization agreement · same stream, two starts',fontsize=14,y=.98)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=4,frameon=False)
        fig.subplots_adjust(left=.075,right=.975,top=.80,bottom=.24,wspace=.18)
        fig.text(.075,.06,'Four streams remain separate. Zero difference can also mean both traces stayed unbound.\n'
            'Read together with contact occupancy, nonempty returns, and undefined ESS; agreement alone is not convergence.',fontsize=10)
        fig.savefig(output/'initialization-agreement.png',dpi=180,facecolor='white')
        fig.savefig(output/'initialization-agreement.svg',facecolor='white');plt.close(fig)
        write(output/'plotted-values.json',dict(schema='rigid-surrogate-contact-presentation-values-v1',
            metric_paths={name:['metrics',*path] for name,path,_ in METRICS},rows=rows,initialization_agreement=agreement,
            note='Values copied from completed analysis. Whole-fingerprint and external-edge-presence ESS are distinct finite-record descriptors. No metrics re-estimated.'))
        for path,digest in inputs.items():require(sha(path)==digest,'Completion input changed during rendering')
        require(record(__file__)==source,'Plot source changed during rendering')
        receipt=dict(schema='rigid-surrogate-contact-plot-v1',complete=True,source=source,input_sha256=inputs,
            outputs={p.name:record(p) for p in output.iterdir() if p.is_file()},streams=40,
            metric_paths={name:['metrics',*path] for name,path,_ in METRICS},
            null_ess_policy='Undefined retained as null in table, gaps/labels in panels; never replaced by zero.',
            matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
            new_geometry_queries=0,new_statistics_estimated=0,scientific_journals_read=0,
            scope='Presentation of inherited completed evidence only. No starts or streams pooled; no new sampling gate.')
        write(output/'plot-receipt.json',receipt);return receipt
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error=repr(error),source=source,input_sha256=inputs));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(plot(args.root,args.output),indent=2))
