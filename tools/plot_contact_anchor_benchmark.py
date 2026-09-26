#!/usr/bin/env python3
"""Plot completed fixed-allocation contact-anchor diagnostics, retaining failures."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(__doc__)
p.add_argument('--benchmark',type=Path,required=True)
a=p.parse_args(); root=a.benchmark
raw=(root/'analysis.json').read_bytes(); data=json.loads(raw)
names=['uniform-anchor','contact-anchor']; labels=['Uniform primary','Contact-aware primary']
colors=['#277DA1','#E17C37']; arms=[data['arms'][n] for n in names]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,3,figsize=(13.7,4.5))
fig.subplots_adjust(left=.065,right=.98,bottom=.23,top=.74,wspace=.38)
fig.suptitle('Does recognizing the current interface improve oligomer docking?',x=.065,ha='left',y=.96,fontsize=17)
fig.text(.065,.86,'Same frozen 264-tetramer state • 4 streams × 128 reset phases per arm • rᵈ = 1.4 Å, z = 0.0275 Å⁻³',color='#444444')
embedded=[x['groups'].get('branch=involution/topology=embedded',{}) for x in arms]
coverage=[100*g.get('pool_contains_direct_partner',0)/max(g.get('pool_recorded',0),1) for g in embedded]
ax[0].bar([0,1],coverage,color=colors,width=.55)
for i,(v,g) in enumerate(zip(coverage,embedded)):
 ax[0].text(i,v+3,f"{g.get('pool_contains_direct_partner',0)}/{g.get('pool_recorded',0)}",ha='center',fontsize=9)
ax[0].set(title='A   Existing interface represented',ylabel='Embedded learned trials with partner in pool (%)',ylim=(0,117))
learned=[x['groups'].get('branch=involution',{}) for x in arms]
valid=[100*g.get('hard_valid',0)/max(g.get('attempted',0),1) for g in learned]
ax[1].bar([0,1],valid,color=colors,width=.55)
for i,(v,g) in enumerate(zip(valid,learned)):
 ax[1].text(i,v+1,f"{g.get('hard_valid',0)}/{g.get('attempted',0)}",ha='center',fontsize=9)
ax[1].set(title='B   Whole oligomer fits',ylabel='All learned trials that are hard-valid (%)',ylim=(0,max(10,max(valid)*1.35)))
counts=[g.get('accepted_contact_changing',0) for g in learned]
rates=[100*c/x['kernel_cpu_seconds'] for c,x in zip(counts,arms)]
ax[2].bar([0,1],rates,color=colors,width=.55)
for i,(v,c,g) in enumerate(zip(rates,counts,learned)):
 ax[2].text(i,v+max(.02,max(rates)*.04),f'{c} events',ha='center',fontsize=9)
ax[2].set(title='C   Accepted interface changes',ylabel='Learned contact changes / 100 kernel CPU s',ylim=(0,max(.3,max(rates)*1.35)))
for x in ax:
 x.set_xticks([0,1],labels);x.tick_params(axis='x',labelsize=9);x.grid(axis='y',alpha=.16);x.set_axisbelow(True)
cpus=[x['kernel_cpu_seconds'] for x in arms]
fig.text(.065,.115,f'Kernel CPU: {cpus[0]:.1f} s uniform, {cpus[1]:.1f} s contact-aware. All rejected trials retained. Contacts are exclusion overlaps, not native-registry labels.',fontsize=9)
fig.text(.065,.055,'This is a conditional proposal benchmark from one fixed state; it does not measure equilibrium, growth or contact ESS.',fontsize=9,color='#444444')
for ext in ['png','svg']:fig.savefig(root/f'contact-anchor-comparison.{ext}',dpi=180)
(root/'figure-provenance.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(raw).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
