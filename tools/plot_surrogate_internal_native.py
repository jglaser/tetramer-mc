#!/usr/bin/env python3
"""Plot completed internal-native scalar summaries; never open scientific journals."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'

import analyze_surrogate_internal_native as observer
from plot_flexible_surrogate_benchmark import ARMS, LABELS, STARTS, require, number, count, close, write

CACHED = ('local', 'm4')
METRICS = (
    ('native_fraction', 'Internal native occupancy'),
    ('native_frames', 'Native occupied frames / 4096'),
    ('presence_ess_per_cpu', 'Motif-presence ESS / full CPU s'),
    ('nonempty_returns_per_cpu', 'Nonempty motif-set returns / full CPU s'),
    ('nonempty_returns', 'Completed nonempty motif-set returns'))
SCOPE = ('Internal pair27/132 only, two mobile bodies and262 fixed spectators. Instantaneous labels may flicker. '
    '4096 production endpoints per chain; last warmup512 supplies transition baseline only. '
    'Full sampler CPU includes warmup; unchanged/rejected residence remains included. '
    'Finite-record ESS is not equilibrium ESS. Read ESS alongside occupied frames and return counts. '
    'No pooling, external native inference, assembly, physical kinetics or equilibrium claim.')


class MetadataBindings:
    """Bounded metadata/source reads for the existing lifecycle verifier."""
    def __init__(self):
        self.files = {}; self.bytes = 0

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        require(path.suffix in ('.json', '.py'), 'Only metadata/source files may be read')
        size = path.stat().st_size
        require(size <= 128*1024**2, 'Oversized metadata file')
        if str(path) not in self.files:
            self.bytes += size; require(self.bytes <= 256*1024**2, 'Metadata budget exceeded')
        with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(expected is None or digest == expected, 'Changed bound metadata: '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest, 'Metadata changed during binding')
        self.files[str(path)] = digest
        return dict(path=str(path), sha256=digest)

    def load(self, ref):
        require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Exact BoundFile required')
        path = Path(self.bind(ref['path'], ref['sha256'])['path'])
        return observer.driver.read(path)

    def read(self, path): return self.load(self.bind(path))


def authenticate(root):
    """Drain gate first, then reuse metadata-only completed_report; no raw hashes."""
    root = Path(root).resolve(); b = MetadataBindings()
    status = b.read(root/'execution/status.json'); terminal_status = b.read(root/'execution/summary.json')
    fields = ('complete', 'passed', 'failure', 'active', 'unstarted', 'completed', 'plan_sha256')
    require(all(status[k] == terminal_status[k] for k in fields)
        and status['complete'] is True and status['passed'] is True and status['failure'] is None
        and status['active'] is None and status['unstarted'] == [] and len(status['completed']) == 1,
        'Internal-native observer is incomplete, failed or undrained')
    refs = dict(execution_plan=b.bind(root/'execution-plan.json'), summary=status['completed'][0]['terminal'])
    plan = b.load(refs['execution_plan']); job = plan['jobs'][0]
    require(plan['root'] == str(root) and job['id'] == 'whole-internal-native-analysis'
        and job['phase'] == 'geometry' and job['population'] == 'context0-48-new-16-cached'
        and job['terminal'] == dict(path=str(root/'analysis/summary.json'), success_contract='complete_and_passed')
        and status['completed'][0]['success_contract'] == 'complete_and_passed', 'Wrong internal-native terminal contract')
    result = observer.completed_report(refs, b, native=True)
    require(result['schema'] == observer.SCHEMA and result['allocation'] == observer.ALLOCATION
        and result['external_native_observer'] is False and result['cycle_observer'] is False
        and result['assembly_gate_open'] is False, 'Wrong native observer scope')
    protocol = b.read(root/'protocol.json'); prepared = b.read(root/'preparation.json')
    claim = b.read(root/'execution/claim.json')
    require(protocol['schema'] == 'surrogate-internal-native-dispatch-v1'
        and protocol['root'] == str(root) and protocol['allocation'] == observer.ALLOCATION
        and prepared['schema'] == 'surrogate-internal-native-preparation-v1'
        and prepared['complete'] is True and prepared['launched'] is False
        and prepared['new_chains'] == 48 and prepared['cached_chains'] == 16
        and prepared['execution_plan'] == refs['execution_plan']
        and prepared['protocol'] == b.bind(root/'protocol.json')
        and claim['preparation_receipt'] == b.bind(root/'preparation.json')
        and plan['files'].get(str(root/'protocol.json')) == prepared['protocol']['sha256'], 'Preparation identity differs')
    audit_ref = protocol['audit_plan']; audit = b.load(audit_ref)
    require(audit_ref == prepared['audit_plan'] and audit_ref['path'] == str(root/'audit-plan.json')
        and plan['files'].get(audit_ref['path']) == audit_ref['sha256'] == result['plan_sha256']
        and audit['schema'] == observer.PLAN_SCHEMA and audit['root'] == str(root)
        and audit['output'] == str(root/'analysis') and audit['allocation'] == observer.ALLOCATION
        and audit['source_sha256'] == result['source_sha256'] == protocol['source_sha256']
        and audit['input_sha256'] == result['input_sha256']
        and all(plan['files'].get(p) == h for p,h in audit['input_sha256'].items()),
        'Result inputs/source differ from frozen audit declaration')
    # Compare raw-input identities; their completed audit remains the authority.
    # These paths are deliberately not opened or hashed here.
    declared = {observer.identity(c['job']):c for c in audit['chains']}
    require(len(declared) == 48 and set(declared) == observer.expected(observer.FLEXIBLE+observer.RIGID), 'Wrong frozen new-chain inventory')
    require(protocol['new_jobs'] == [c['job'] for c in audit['chains']], 'Dispatch new jobs differ')
    cached = {observer.identity(j):j for j in protocol['cached_control_jobs']}
    require(len(cached) == 16 and set(cached) == observer.expected(CACHED), 'Wrong cached-control inventory')
    for chain in result['chains']:
        key = observer.identity(chain['job'])
        if key in declared:
            require(chain['job'] == declared[key]['job'] and chain['trajectory'] == declared[key]['trajectory']
                and chain['terminal'] == declared[key]['terminal'] and chain['reused_native_control'] is False,
                'New-chain identity differs')
        else:
            require(key in cached and chain['job'] == cached[key] and chain['reused_native_control'] is True,
                'Cached-chain identity differs')
    config = b.load(audit['config'])
    require(audit['input_sha256'].get(audit['config']['path']) == audit['config']['sha256']
        and config['physical']['depletant_radius'] == 1.4 and config['physical']['activity'] == .0275
        and [config['contexts'][0][k] for k in ('root','child')] == [27,132], 'Conditional physical target differs')
    result.pop('_execution_files')
    return result, b.files


def motif_ids(keys):
    require(type(keys) is list, 'Motif keys must be a list')
    ids = []
    for key in keys:
        require(type(key) is list and len(key) == 3 and key[:2] == [27,132]
            and all(type(x) is int for x in key) and 0 <= key[2] < 14, 'Wrong internal motif key')
        ids.append(key[2])
    require(len(ids) == len(set(ids)), 'Repeated motif key')
    return ids


def saved_values(result):
    rows = []; seen = set()
    for chain in result['chains']:
        job = chain['job']; key = observer.identity(job)
        require(type(job['context_index']) is type(job['stream']) is int and key not in seen, 'Duplicate/invalid chain identity')
        seen.add(key); require(chain['reused_native_control'] is (job['arm'] in CACHED), 'Wrong cached origin')
        m = chain['metrics']; cpu = number(m['full_sampler_cpu_seconds'], 'full sampler CPU')
        require(m['schema'] == 'internal-native-metrics-v1' and cpu > 0
            and type(m['production_samples']) is int and m['production_samples'] == 4096
            and type(m['retained_endpoints']) is int and m['retained_endpoints'] == 4609, 'Wrong residence/CPU inventory')
        require(number(m['production_window_cpu_seconds'], 'production CPU') <= cpu, 'Production CPU exceeds full CPU')
        fraction = number(m['internal_native_attachment_fraction'], 'native occupancy', True)
        d = m['internal_motifs']; env = d['environments']; values = d['environment_values']
        occupancy = {k:number(v, 'motif-set occupancy', True) for k,v in env['occupancy'].items()}
        require(occupancy and all(k in values for k in occupancy), 'Missing environment dictionary')
        ids = {k:motif_ids(v) for k,v in values.items()}
        close(math.fsum(occupancy.values()), 1., 'environment occupancy total')
        close(math.fsum(v for k,v in occupancy.items() if ids[k]), fraction, 'native occupancy')
        frames = fraction*4096; close(frames, round(frames), 'occupied-frame count')
        marginal = [0.]*14; seen_motifs = set()
        for entry in d['marginal_occupancy']:
            motif, = motif_ids([entry['key']]); require(motif not in seen_motifs, 'Duplicate motif marginal'); seen_motifs.add(motif)
            marginal[motif] = number(entry['fraction'], 'motif occupancy', True)
        for motif in range(14):
            close(marginal[motif], math.fsum(v for k,v in occupancy.items() if motif in ids[k]), 'motif marginal')
        for v in occupancy.values(): close(v*4096, round(v*4096), 'category residence count')
        counts = d['counts']; rates = d['rates_per_full_sampler_cpu_second']
        require(set(counts) == set(rates), 'Event-rate inventory differs')
        for name,value in counts.items():
            count(value, name); close(number(rates[name], name+'/CPU'), value/cpu, name+' CPU denominator')
        require(counts['completed_passages'] == counts['enter_nonempty']+counts['leave_nonempty']+counts['direct_nonempty_changes'], 'Passage counter differs')
        returns = env['completed_returns']
        require(counts['completed_returns'] == len(returns) and counts['nonempty_returns'] == sum(e['returned_environment_nonempty'] for e in returns), 'Return counter differs')
        for event in returns:
            require(type(event['returned_environment_nonempty']) is bool and type(event['passed_through_empty']) is bool
                and type(event['departure_block']) is type(event['return_block']) is int
                and 513 <= event['departure_block'] < event['return_block'] <= 4608,
                'Return is outside production/baseline convention')
        ess = d['presence_ess']; value, rate = ess['apparent_ess'], ess['apparent_ess_per_sampling_CPU_second']
        require(ess['samples'] == 4096 and ess['sampling_CPU_seconds'] == cpu and (value is None) == (rate is None), 'ESS scope/null mismatch')
        if value is not None:
            require(0 < number(value, 'ESS') <= 4096, 'ESS outside retained count')
            close(number(rate, 'ESS/CPU'), value/cpu, 'ESS CPU denominator')
        rows.append(dict(job=job, reused_native_control=chain['reused_native_control'], full_sampler_cpu_seconds=cpu,
            production_samples=4096, native_fraction=fraction, native_frames=round(frames), presence_ess=value,
            presence_ess_per_cpu=rate, null_ess_reason=ess.get('reason') if value is None else None,
            nonempty_returns=counts['nonempty_returns'], nonempty_returns_per_cpu=rates['nonempty_returns'],
            nonempty_returns_through_empty=sum(e['returned_environment_nonempty'] and e['passed_through_empty'] for e in returns),
            direct_nonempty_changes=counts['direct_nonempty_changes'], motif_fractions=marginal,
            motif_set_occupancy=occupancy, event_counts=dict(counts)))
    require(len(rows) == 64 and seen == observer.expected(ARMS), 'All64 separate chains required')
    return sorted(rows, key=lambda r:(STARTS.index(r['job']['initialization']), ARMS.index(r['job']['arm']), r['job']['stream']))


def initialization_contrasts(rows):
    index = {(r['job']['arm'],r['job']['stream'],r['job']['initialization']):r for r in rows}
    require(len(index) == 64, 'Initialization comparison needs all64 chains')
    result = []
    for arm in ARMS:
        for stream in range(4):
            source,prepared = (index[arm,stream,start] for start in STARTS)
            a,b = source['motif_set_occupancy'],prepared['motif_set_occupancy']
            result.append(dict(arm=arm,stream=stream,source_native_fraction=source['native_fraction'],
                prepared_native_fraction=prepared['native_fraction'],
                prepared_minus_source_native_fraction=prepared['native_fraction']-source['native_fraction'],
                maximum_motif_fraction_difference=max(abs(x-y) for x,y in zip(source['motif_fractions'],prepared['motif_fractions'])),
                motif_set_occupancy_TV=.5*math.fsum(abs(a.get(k,0.)-b.get(k,0.)) for k in a.keys()|b.keys())))
    return result


def plot(root, output):
    output = Path(output).resolve(); require(not output.exists(), 'Fresh plot output required')
    result,inputs = authenticate(root); rows = saved_values(result); contrasts = initialization_contrasts(rows)
    b = MetadataBindings(); source = b.bind(__file__)
    for p in observer.local_sources(__file__).values(): b.bind(p)
    sources = dict(b.files)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    colors = ('#0072B2','#D55E00','#009E73','#CC79A7'); markers = ('o','s','^','D')
    legend = [Line2D([0],[0],color=colors[s],marker=markers[s],lw=.8,label=f'Stream {s}') for s in range(4)]
    def decorate(ax):
        for lo,hi,color in ((-.5,1.5,'#eeeeee'),(1.5,4.5,'#f6efe6'),(4.5,7.5,'#eaf3fa')): ax.axvspan(lo,hi,color=color,zorder=0)
        ax.set_xticks(range(8),LABELS,fontsize=8);ax.set_xlim(-.4,7.4);ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    def draw(ax,values,stream):
        for group in (range(2),range(2,5),range(5,8)):
            ax.plot([i+(stream-1.5)*.1 for i in group],[np.nan if values[ARMS[i]] is None else values[ARMS[i]] for i in group],
                    color=colors[stream],marker=markers[stream],ms=4,lw=.7)
    output.mkdir(parents=True)
    try:
        fig,axes=plt.subplots(2,5,figsize=(25,10),sharey='col')
        for ri,start in enumerate(STARTS):
            group=[r for r in rows if r['job']['initialization']==start]
            for ci,(key,label) in enumerate(METRICS):
                ax=axes[ri,ci];decorate(ax)
                for stream in range(4):draw(ax,{r['job']['arm']:r[key] for r in group if r['job']['stream']==stream},stream)
                if ri==0:ax.set_title(label,fontsize=10)
                if ci==0:ax.set_ylabel('Source start' if ri==0 else 'Proposal-prepared start')
                if key=='presence_ess_per_cpu':
                    for x,arm in enumerate(ARMS):
                        missing=[str(r['job']['stream']) for r in group if r['job']['arm']==arm and r[key] is None]
                        if missing:ax.text(x,-.22,'undefined\n'+','.join(missing),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7)
        for ci,(key,label) in enumerate(METRICS):
            defined=[r[key] for r in rows if r[key] is not None]
            if key=='presence_ess_per_cpu' and defined:
                axes[0,ci].set_yscale('log')
                axes[0,ci].set_ylim(.6*min(defined),1.5*max(defined))
                axes[0,ci].set_title(label+' · log scale',fontsize=10)
            else:
                upper=1 if key=='native_fraction' else max(defined or [0]) or 1
                axes[0,ci].set_ylim(-.03*upper,1.08*upper)
                if key=='presence_ess_per_cpu':
                    for ax in axes[:,ci]:
                        ax.set_yticks([])
                        ax.text(.5,.5,'ESS undefined for all streams',transform=ax.transAxes,ha='center',fontsize=9)
        fig.suptitle('Internal native registry · all64 chains kept separate',fontsize=16,y=.985)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
        fig.subplots_adjust(top=.84,bottom=.25,left=.045,right=.99,wspace=.3,hspace=.66)
        fig.text(.045,.04,'Gray:16 cached native controls; tan:24 newly labelled rigid chains; blue:24 newly labelled flexible chains.\n'
            'ESS gaps remain undefined. A rare occupied frame can produce a large presence ESS; read the occupancy, frame and return-count panels together.\n'+SCOPE,
            fontsize=9,wrap=True)
        for ext in ('png','svg'):fig.savefig(output/f'internal-native-efficiency.{ext}',dpi=170,facecolor='white')
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,13),sharey=True)
        for ax,start in zip(axes,STARTS):
            ordered=[next(r for r in rows if r['job']['arm']==arm and r['job']['stream']==s and r['job']['initialization']==start) for arm in ARMS for s in range(4)]
            im=ax.imshow([r['motif_fractions'] for r in ordered],vmin=0,vmax=1,aspect='auto',cmap='viridis')
            ax.set_xticks(range(14),range(14));ax.set_xlabel('Directed motif ID');ax.set_title(start.replace('_',' '))
            ax.set_yticks(range(32),[f'{arm} · s{s}' for arm in ARMS for s in range(4)],fontsize=8)
            for y in (7.5,19.5):ax.axhline(y,color='white',lw=1.2)
        fig.colorbar(im,ax=axes,label='Marginal production occupancy',fraction=.025,pad=.03)
        fig.suptitle('Every motif and every stream · overlapping motif marginals are not disjoint probabilities',fontsize=12)
        fig.subplots_adjust(left=.18,right=.85,top=.94,bottom=.08,wspace=.15)
        fig.text(.18,.025,'Zero denotes absent motif in that finite record, not zero equilibrium weight. Starts and streams are not pooled.',fontsize=9)
        for ext in ('png','svg'):fig.savefig(output/f'internal-native-motifs.{ext}',dpi=170,facecolor='white')
        plt.close(fig)
        definitions=(('prepared_minus_source_native_fraction','Prepared − source native occupancy'),
            ('maximum_motif_fraction_difference','Maximum motif marginal difference'),('motif_set_occupancy_TV','Motif-set occupancy total variation'))
        fig,axes=plt.subplots(1,3,figsize=(18,6))
        for ax,(key,label) in zip(axes,definitions):
            decorate(ax)
            for stream in range(4):draw(ax,{r['arm']:r[key] for r in contrasts if r['stream']==stream},stream)
            ax.set_ylim(-1.03 if key.startswith('prepared_') else -.03,1.03);ax.set_title(label,fontsize=11);ax.axhline(0,color='#555',lw=.6)
        fig.suptitle('Same-stream initialization contrasts ·32 separate comparisons',fontsize=14)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False)
        fig.subplots_adjust(top=.77,bottom=.25,left=.05,right=.98,wspace=.2)
        fig.text(.05,.06,'Differences of saved occupancies only; no trajectory concatenation or statistical-error estimate.\n'
            'Agreement can mean both starts remained without registry. Shared RNG roles do not make arms independent replicates.',fontsize=10)
        for ext in ('png','svg'):fig.savefig(output/f'initialization-differences.{ext}',dpi=170,facecolor='white')
        plt.close(fig)
        write(output/'plotted-values.json',dict(schema='surrogate-internal-native-presentation-v1',rows=rows,
            initialization_contrasts=contrasts,cached_controls=16,new_rigid_native_labels=24,new_flexible_native_labels=24,scope=SCOPE))
        for p,h in {**inputs,**sources}.items():b.bind(p,h)
        receipt=dict(schema='surrogate-internal-native-plot-v1',complete=True,passed=True,source=source,source_sha256=sources,
            input_sha256=inputs,outputs={p.name:b.bind(p) for p in output.iterdir() if p.suffix=='.json'},
            chains=64,cached_native_controls=16,new_native_chains=48,initialization_comparisons=32,
            native_registry=True,external_native_observer=False,cycle_observer=False,assembly_gate_open=False,
            null_ess_policy='Undefined stays null and renders as a labelled gap; never zero-filled.',
            scientific_journals_read=0,new_geometry_queries=0,new_statistics_estimated=0,
            streams_or_starts_pooled=False,matplotlib_version=matplotlib.__version__,numpy_version=np.__version__)
        # Artifact hashes are output-only, outside the metadata-input reader.
        for p in output.iterdir():
            if p.suffix in ('.png','.svg'):
                with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
                receipt['outputs'][p.name]=dict(path=str(p),sha256=digest)
        write(output/'plot-receipt.json',receipt);return receipt
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error=repr(error),source=source,input_sha256=inputs));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(plot(args.root,args.output),indent=2))
