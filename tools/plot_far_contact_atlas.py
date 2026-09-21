#!/usr/bin/env python3
"""Plot fixed-prefix sensitivity and the full far-region positive partition."""
import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_expanded_contact_atlas import point, limits, read, sha
from prepare_cayley_rms_cover import write, require


def plot(path, out):
    require(not out.exists(), 'Fresh figure directory required')
    result = read(path); require(result['complete'], 'Completed audited campaign required')
    require(result['original_q_window'] == dict(minimum=5., maximum=37., lower_inclusive=True, upper_inclusive=False), 'Different far window')
    for name,digest in result['archived_sha256'].items(): require(sha(path.parent/'provenance'/name) == digest, 'Archive changed')
    for name,digest in result['source_sha256'].items(): require(sha(name) == digest, 'Analysis input changed')
    protocol_path = path.parent/'provenance/atlas-protocol.json'; protocol = read(protocol_path)
    cfg = protocol['physical']; require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and len(cfg['fixed_poses']) == 2, 'Different physical target')
    arms = {c['arm']: c for c in result['campaigns']}; require(set(arms) == {'narrow','broad'}, 'Missing width arm')
    population_counts = {len(c['populations']) for c in arms.values()}
    require(len(population_counts) == 1, 'Population counts differ across width arms')
    population_count = population_counts.pop()
    colors = dict(narrow='#16735b', broad='#bb7429'); gray = '#657181'
    prefixes = {name: [p['physical']['full'] for p in c['prefixes']] for name,c in arms.items()}
    budgets = [p['draws'] for p in prefixes['narrow']]
    require(budgets == [p['draws'] for p in prefixes['broad']], 'Prefix counts differ')
    keys = ('radial_0','radial_1','radial_2','outside2')
    references = [arms['narrow']['local_calibration'][k]['physical']['fresh'] for k in keys[:3]]
    for k,r in zip(keys, references): require(r == arms['broad']['local_calibration'][k]['physical']['fresh'], 'Different calibration reference')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig, axes = plt.subplots(1,2,figsize=(12.5,6.7),gridspec_kw={'width_ratios':[1,1.2]})
    panels = [[r for rows in prefixes.values() for r in rows], [c['physical'][k] for c in arms.values() for k in keys]+references]
    for ax, rows in zip(axes, panels):
        ax.set_ylim(*limits(rows)); ax.grid(axis='y', alpha=.18); ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel(r'$\log[Q/(1\,\mathrm{Å}^3)]$')
    for name,offset in (('narrow',-.045),('broad',.045)):
        rows = prefixes[name]; x = np.arange(len(rows))+offset
        axes[0].plot(x,[r['logQ'] for r in rows],color=colors[name],alpha=.65,lw=1.4)
        for xx,row in zip(x,rows): point(axes[0],xx,row,colors[name],axes[0].get_ylim()[0])
    axes[0].set_xticks(range(len(budgets)),[f'{n:,}' for n in budgets]); axes[0].set_xlim(-.35,len(budgets)-.6)
    axes[0].set_xlabel(f'Unconditional draws per width, across {population_count} populations')
    axes[0].set_title('Complete far region: fixed proposal, increasing sample count',fontsize=11,pad=24)
    axes[0].text(.5,1.019,'Prefixes within an arm share rows; width arms have separate streams.',transform=axes[0].transAxes,ha='center',fontsize=8.2,color=gray)
    for name,offset in (('narrow',-.14),('broad',.14)):
        for i,key in enumerate(keys): point(axes[1],i+offset,arms[name]['physical'][key],colors[name],axes[1].get_ylim()[0])
    for i,row in enumerate(references): point(axes[1],i,row,gray,axes[1].get_ylim()[0],marker='D',markersize=5)
    axes[1].set_xticks(range(4),['ρ ≤ 0.5','0.5 < ρ ≤ 1','1 < ρ ≤ 2','Outside ρ ≤ 2']); axes[1].set_xlim(-.45,3.45)
    axes[1].set_xlabel('Fixed geometric shells (Å-equivalent); all far poses retained')
    axes[1].set_title('Four disjoint contributions sum to the full far integral',fontsize=11,pad=24)
    axes[1].text(.5,1.019,'Gray: earlier finite-region calibration; no reference for all outside poses.',transform=axes[1].transAxes,ha='center',fontsize=8.2,color=gray)
    legend = [Line2D([],[],marker='o',lw=1.3,color=colors[n],label=f'{n.capitalize()}: geometric σ={s} Å')
              for n,s in [('narrow','.2'),('broad','.4')]]
    legend.append(Line2D([],[],marker='D',lw=0,color=gray,label='Local calibration used in proposal construction'))
    fig.suptitle('Calibrated contact atlas: independent full-far sampling',fontsize=14,y=.98)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.168,'Both AB neighbors fixed • depletant radius 1.5 Å • activity 0.035 Å⁻³ • capture radius 18 Å • original q ≥ 5',ha='center',fontsize=9)
    fig.text(.5,.117,'Bars: log(Q ± one observed row SE). Prefix comparisons retain shared-row covariance; nested regions are not added.',ha='center',fontsize=8.7)
    fig.text(.5,.067,'Two frozen 180-component guides; only 59 geometric covariances differ. No refitting or stopping during these populations.',ha='center',fontsize=8.7)
    fig.text(.5,.023,'This measures integration precision. Agreement and observed errors do not exclude unseen contacts or establish trajectory mixing.',ha='center',fontsize=8.7)
    fig.tight_layout(rect=(0,.215,1,.86),w_pad=2.5); out.mkdir(parents=True)
    for ext in ('png','svg'): fig.savefig(out/f'far-contact-atlas.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    sources = {'analysis.json':path,'protocol.json':protocol_path,'plotter.py':Path(__file__),
        'plot_helpers.py':Path(__file__).with_name('plot_expanded_contact_atlas.py')}
    for n,p in sources.items(): shutil.copy2(p,out/n)
    write(out/'provenance.json',dict(input_sha256={str(p):sha(p) for p in sources.values()},archived_sha256={n:sha(out/n) for n in sources},
        output_sha256={f'far-contact-atlas.{e}':sha(out/f'far-contact-atlas.{e}') for e in ('png','svg')},
        plotted=dict(prefixes=prefixes,partition={n:{k:c['physical'][k] for k in keys} for n,c in arms.items()},calibration=references),
        scope='Separate original full-density importance estimators, fixed correlated prefixes, and unchanged calibration masks; no unseen-tail or mixing claim.'))
    print(out/'far-contact-atlas.png')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plot(args.analysis.resolve(),args.out.resolve())
