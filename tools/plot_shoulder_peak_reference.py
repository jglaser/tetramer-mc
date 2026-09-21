#!/usr/bin/env python3
"""Show identical local masks and explicitly uncalibrated historical remainders."""
import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_expanded_contact_atlas import point, limits
from prepare_cayley_rms_cover import read, write, sha, require


def plot(path, out):
    require(not out.exists(), 'Fresh figure output required')
    result = read(path); require(result['complete'], 'Completed independent audit required')
    require(result['original_q_window'] == dict(minimum=1., maximum=1.1, lower_inclusive=False, upper_inclusive=False), 'Different physical window')
    for name,digest in result['archived_sha256'].items(): require(sha(path.parent/'provenance'/name) == digest, 'Executed archive changed')
    for name,digest in result['source_sha256'].items(): require(sha(Path(name)) == digest, 'Analysis input changed')
    histories = {h['name']:h for h in result['histories']}
    order = ('direct-inner','mixture-confirmation','geometry-confirmation')
    require(set(histories) == set(order), 'Historical controls differ')
    fresh = {c['radius_A']:c for c in result['campaigns']}
    require(set(fresh) == {.25,.5}, 'Frozen radii differ')
    refs = [fresh[.25]['physical']['ball0p25'], result['independent_shell_sum']['physical']['ball0p5']]
    colors = dict(zip(order,('#647080','#16735b','#bc7627'))); blue = '#226bb1'
    labels = dict(zip(order,('Historical direct inner cover','Historical mixture guide','Historical geometry guide')))
    local_keys = ('ball0p25','ball0p5'); outside_keys = ('outside0p25','outside0p5')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig, axes = plt.subplots(1,2,figsize=(12.5,6.7))
    panels = [[h['physical'][k] for h in histories.values() for k in local_keys]+refs,
              [h['physical'][k] for h in histories.values() for k in outside_keys]]
    for ax, rows in zip(axes,panels):
        ax.set_ylim(*limits(rows)); ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
        ax.set_ylabel(r'$\log[Q/(1\,\mathrm{Å}^3)]$');ax.set_xlim(-.42,1.42)
    for name,offset in zip(order,(-.21,-.07,.07)):
        for ax,keys in zip(axes,(local_keys,outside_keys)):
            for i,key in enumerate(keys):point(ax,i+offset,histories[name]['physical'][key],colors[name],ax.get_ylim()[0])
    for i,row in enumerate(refs):point(axes[0],i+.21,row,blue,axes[0].get_ylim()[0],marker='D',markersize=6)
    axes[0].set_xticks([0,1],['ρ ≤ 0.25 Å','ρ ≤ 0.5 Å'])
    axes[1].set_xticks([0,1],['ρ > 0.25 Å','ρ > 0.5 Å'])
    axes[0].set_title('Identical finite regions under different sampling laws',fontsize=11,pad=22)
    axes[1].set_title('Historical inner-shoulder weight outside each ball',fontsize=11,pad=22)
    axes[0].text(.5,1.017,'Blue radius-0.5 value: prespecified independent disjoint-shell sum.',ha='center',transform=axes[0].transAxes,fontsize=8.2,color='#657181')
    axes[1].text(.5,1.017,'No fresh whole-remainder reference: missing blue points are not zeros.',ha='center',transform=axes[1].transAxes,fontsize=8.2,color='#657181')
    for ax in axes:ax.set_xlabel('Fixed geometric chart radius around the recorded peak')
    legend=[Line2D([],[],marker='o',lw=0,color=colors[n],label=labels[n]) for n in order]
    legend.append(Line2D([],[],marker='D',lw=0,color=blue,label='Fresh uniform geometric reference'))
    fig.suptitle('Inner-shoulder contact: calibrating a concentrated reference estimate',fontsize=14,y=.98)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.168,'Both AB neighbors fixed • radius 1.5 Å • activity 0.035 Å⁻³ • capture 18 Å • original 1 < q < 1.1',ha='center',fontsize=9)
    fig.text(.5,.116,'All methods retain original full N and their own normalized proposal density. Bars show log(Q ± one observed row SE).',ha='center',fontsize=8.6)
    fig.text(.5,.068,'Nested balls and their complements share rows; they are not independent estimates. Fresh whole balls are not added.',ha='center',fontsize=8.6)
    fig.text(.5,.023,'The center was selected from historical extremes. This is local calibration, not whole-shoulder convergence or assembly.',ha='center',fontsize=8.6)
    fig.tight_layout(rect=(0,.215,1,.825),w_pad=2.7);out.mkdir(parents=True)
    for ext in ('png','svg'):fig.savefig(out/f'inner-shoulder-local-reference.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    sources={'analysis.json':path,'plotter.py':Path(__file__),'plot_helpers.py':Path(__file__).with_name('plot_expanded_contact_atlas.py')}
    for name,p in sources.items():shutil.copy2(p,out/name)
    write(out/'provenance.json',dict(input_sha256={str(p):sha(p) for p in sources.values()},archived_sha256={n:sha(out/n) for n in sources},
        output_sha256={f'inner-shoulder-local-reference.{ext}':sha(out/f'inner-shoulder-local-reference.{ext}') for ext in ('png','svg')},
        plotted=dict(local_reference=refs,history={n:{k:h['physical'][k] for k in (*local_keys,*outside_keys)} for n,h in histories.items()}),
        scope='Identical local masks, original full denominators, explicit unmeasured remainders. Local selection used historical data; no independent significance, unseen-tail or mixing claim.'))
    print(out/'inner-shoulder-local-reference.png')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.analysis.resolve(),a.out.resolve())
