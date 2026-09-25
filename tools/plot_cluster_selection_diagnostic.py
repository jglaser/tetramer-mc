#!/usr/bin/env python3
"""Plot frozen passive cluster diagnostics; never reads or advances live runs."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    args = parser.parse_args()
    root = args.diagnostic
    summary = json.loads((root / 'replay/summary.json').read_text())
    native = json.loads((root / 'native/matched-native-summary.json').read_text())
    events = [json.loads(x) for x in (root / 'replay/events.jsonl').read_text().splitlines()]
    census = [json.loads(x) for x in (root / 'replay/phase-census.jsonl').read_text().splitlines()]
    docking = [e for e in events if e['whole_component'] and e['size'] == 2
               and e['hard_valid'] and e['gained_contacts'] and not e['lost_contacts']]
    assert len(events) == summary['overall']['attempted']
    assert all(e['sweep'] <= summary['last_sweep'] for e in events)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(2, 2, figsize=(12.7, 8.2))
    fig.subplots_adjust(left=.085, right=.975, top=.88, bottom=.11, hspace=.48, wspace=.32)
    blue, orange, green = '#277DA1', '#E17C37', '#49936A'
    fig.suptitle('Which oligomers do the new moves mobilize?', fontsize=19, x=.085, ha='left', y=.965)
    fig.text(.085, .918, 'Frozen sweeps 0–600  •  8 seed + 256 free tetramers  •  500 μM  •  rᵈ = 1.4 Å, z = 0.0275 Å⁻³', color='#444444')
    for name, label, color in [('control','Earlier control',blue), ('cluster','With cluster phase',orange)]:
        ax[0,0].plot([r[name]['sampler_cpu_seconds']/60 for r in native['rows']],
                     [r[name]['seed_component_tetramers'] for r in native['rows']],
                     'o-', label=label, color=color, ms=4)
    ax[0,0].set(xlabel='Sampler CPU minutes', ylabel='Native seed-connected tetramers',
                title='A   Greater early growth; one stream per arm', ylim=(7,18))
    ax[0,0].legend(frameon=False, loc='upper left', fontsize=9)
    ax[0,0].text(.98,.04,'Different initial random configurations', transform=ax[0,0].transAxes,
                 ha='right', fontsize=9, color='#555555')
    classes=[('whole','solution','Whole solution\noligomer'),
             ('embedded','solution','Subset in solution\naggregate'),
             ('embedded','seed_connected','Subset connected\nto seed')]
    for offset, branch, color in [(-.19,'local',blue),(.19,'transport',orange)]:
        vals=[]; labels=[]
        for topo,env,_ in classes:
            rows=[v for k,v in summary['groups'].items() if k.split('|')[1:]==[topo,env,branch]]
            total=sum(v['attempted'] for v in rows); accepted=sum(v['accepted'] for v in rows)
            vals.append(100*accepted/total if total else 0); labels.append(f'{accepted}/{total}')
        xs=np.arange(3)+offset
        ax[0,1].bar(xs,vals,width=.36,color=color,label='Local rigid' if branch=='local' else 'Transport')
        for x,v,l in zip(xs,vals,labels): ax[0,1].text(x,v+3,l,ha='center',fontsize=9)
    ax[0,1].set(xticks=np.arange(3),xticklabels=[c[2] for c in classes],
                ylabel='Accepted / all attempts (%)',ylim=(0,119),title='B   Whole-oligomer motion is readily accepted')
    ax[0,1].legend(frameon=False, fontsize=9, loc='upper right')
    hist=census[-1]['solution_component_size_histogram']; sizes=range(2,max(map(int,hist))+1)
    vals=[hist.get(str(s),0) for s in sizes]
    bars=ax[1,0].bar(list(sizes),vals,color=[blue if s<=3 else orange for s in sizes],width=.65)
    for b,v in zip(bars,vals): ax[1,0].text(b.get_x()+b.get_width()/2,v+.5,str(v),ha='center')
    ax[1,0].set(xlabel='Tetramers per whole solution oligomer',ylabel='Number of oligomers',
                xticks=list(sizes),ylim=(0,max(vals)*1.2),title='C   Some solution oligomers exceed size 3')
    ax[1,0].text(.98,.94,'Phase 600 start; excludes seed component\nIsolated tetramers omitted',
                transform=ax[1,0].transAxes,ha='right',va='top',fontsize=9,color='#555555')
    x=np.arange(1,len(docking)+1)
    for values,label,color,marker in [
        ([e['gate']['log_weight'] for e in docking],'Sampled depletion factor',green,'o'),
        ([e['map_log_reverse_forward'] for e in docking],'Map correction',orange,'s'),
        ([e['log_acceptance'] for e in docking],'Combined log acceptance',blue,'x')]:
        ax[1,1].plot(x,values,marker+'-',label=label,color=color,ms=4,lw=1)
    ax[1,1].axhline(0,color='#AAAAAA',lw=.8,zorder=0)
    ax[1,1].set(xlabel='Hard-valid whole-dimer attachment trial',ylabel='Log acceptance contribution',
                xticks=x,title='D   Docking is limited by the current map')
    ax[1,1].legend(frameon=False,loc='center left',bbox_to_anchor=(0,.63),fontsize=9)
    for a in ax.flat: a.grid(axis='y',alpha=.15); a.set_axisbelow(True)
    fig.text(.085,.035,f"{summary['overall']['accepted']} / {summary['overall']['attempted']} cluster moves accepted; "
             f"{summary['overall']['accepted_attachments'] + summary['overall']['accepted_detachments'] + summary['overall']['accepted_exchanges']} changed exclusion-contact partners. Growth difference is descriptive, not a speedup estimate.",fontsize=10)
    for suffix in ('png','svg'): fig.savefig(root/f'cluster-selection-diagnostic.{suffix}',dpi=180)
    sources=['replay/summary.json','replay/events.jsonl','replay/phase-census.jsonl','native/matched-native-summary.json']
    (root/'figure-provenance.json').write_text(json.dumps({'sources':{s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')

if __name__ == '__main__': main()
