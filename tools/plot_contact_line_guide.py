#!/usr/bin/env python3
"""Explain the contact-line guide using a completed, independently audited line.

The selected line is reconstructed from projected atom KD trees. No proposal,
Poisson point cloud, native classification or physical calculation is rerun.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm
from analyze_contact_line_audit import Reconstructor,interval_masses,read,sha,rotation,require


def merged(intervals):
    result=[]
    for a,b in sorted(intervals):
        if a>=b:continue
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def intersection(a,b):
    return merged([(max(x,u),min(y,v)) for x,y in a for u,v in b if max(x,u)<min(y,v)])


def difference(a,b):
    result=[]
    for lo,hi in a:
        cursor=lo
        for x,y in b:
            if y<=cursor or x>=hi:continue
            if x>cursor:result.append([cursor,min(x,hi)])
            cursor=max(cursor,y)
            if cursor>=hi:break
        if cursor<hi:result.append([cursor,hi])
    return result


def reconstruct(shape,origin,direction,segment,fixed,widths,contacts):
    """Independent infinite-line 2D broadphase; exact leaf interval unions."""
    atoms=np.asarray([a['center'] for a in shape['atoms']]);r=np.asarray([a['radius'] for a in shape['atoms']])
    position=np.asarray(origin['position']);moving=atoms@rotation(origin['orientation']).T+position
    direction=np.asarray(direction);speed=np.linalg.norm(direction);unit=direction/speed
    transverse=np.linalg.svd(unit[None,:])[2][1:].T
    projected=(moving-position)@transverse;tree=cKDTree(projected)
    core=[];by_neighbor=[];pair_count=0
    for pose in fixed:
        stationary=atoms@rotation(pose['orientation']).T+pose['position']
        neighbors=tree.query_ball_tree(cKDTree((stationary-position)@transverse),2*r.max()+max(widths)+1e-9)
        intervals=[[] for _ in [0.,*widths]]
        for i,indices in enumerate(neighbors):
            if not indices:continue
            j=np.asarray(indices);delta=stationary[j]-moving[i];along=delta@unit
            perpendicular=np.linalg.norm(np.cross(delta,unit),axis=1);pair_count+=len(j)
            for wi,w in enumerate([0.,*widths]):
                radius=r[i]+r[j]+w;take=perpendicular<radius
                p=perpendicular[take];rr=radius[take];s=along[take]
                half=rr*np.sqrt((1-p/rr)*(1+p/rr))
                lower=np.maximum(segment[0],(s-half)/speed);upper=np.minimum(segment[1],(s+half)/speed)
                intervals[wi].extend(zip(lower[lower<upper].tolist(),upper[lower<upper].tolist()))
        sets=[merged(v) for v in intervals];core.extend(sets[0]);by_neighbor.append(sets[1:])
    core=merged(core);free=difference([segment],core);joint=[]
    for wi in range(len(widths)):
        selected=free
        for index in contacts:selected=intersection(selected,by_neighbor[index][wi])
        joint.append(selected)
    return dict(core=core,hard_free=free,contacts=by_neighbor,feasible=joint,projected_pair_candidates=pair_count)


def projected_mask(points,radii,axes,x,y):
    mask=np.zeros((len(y),len(x)),bool)
    for p,r in zip(points[:,axes],radii):
        i=max(0,np.searchsorted(x,p[0]-r));j=min(len(x),np.searchsorted(x,p[0]+r)+1)
        k=max(0,np.searchsorted(y,p[1]-r));l=min(len(y),np.searchsorted(y,p[1]+r)+1)
        if i<j and k<l:mask[k:l,i:j]|=(x[i:j][None,:]-p[0])**2+(y[k:l][:,None]-p[1])**2<=r*r
    return mask


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();root=a.root.resolve();out=a.out.resolve();require(not out.exists(),'Output must be new')
    receipt=read(a.audit);require(receipt['complete'] and Path(receipt['root']).resolve()==root,'Wrong independent audit')
    for p,digest in receipt['input_sha256'].items():require(sha(p)==digest,'Audited input changed')
    summary=read(root/'summary.json');require(summary['complete'],'Incomplete recorded output')
    region=read(root/'provenance/region.json');guide=read(root/'provenance/importance-guide.json')
    shape=read(root/'provenance/shape.json');config=read(root/'provenance/config.json')
    recon=Reconstructor(region,guide,config,shape)
    require(recon.axes==[0] and len(config['fixed_poses'])==2,'Figure is scoped to the frozen x-axis/two-scaffold pilot')
    records=[json.loads(s) for name in ('samples','probes') for s in (root/(name+'.jsonl')).read_text().splitlines()]
    candidates=[]
    for row in records:
        for axis in row['density_details'].get('axes',[]):
            for wi,width in enumerate(axis.get('widths',[])):
                positive=[r for r in width['intervals'] if r['lower']<r['upper']]
                if not positive:continue
                draw=row.get('draw') or {}
                actual=(draw.get('conditional') and not draw.get('fallback') and draw['axis']==axis['axis'] and draw['width_index']==wi)
                candidates.append((len(positive),bool(actual),row,axis,wi))
    require(candidates,'No recorded positive contact line is available; do not invent an example')
    _,actual,row,axis,wi=max(candidates,key=lambda c:(c[0],c[1]))
    selected_intervals=axis['widths'][wi]['intervals'];x=np.asarray(row['raw_coordinates'])
    means,sigmas=recon.conditional(x,axis['axis']);masses=interval_masses(selected_intervals,means,sigmas)
    if actual:component=row['draw']['component']
    else:
        eligible=np.flatnonzero(masses>recon.floor);require(len(eligible)>0,'All component masses fall below the floor')
        component=int(eligible[np.argmax(recon.gaussian_logs(row['latent'])[eligible])])
    m,s,mass=float(means[component]),float(sigmas[component]),float(masses[component])
    interval_probabilities=[float(interval_masses([interval],np.asarray([m]),np.asarray([s]))[0]/mass) for interval in selected_intervals]
    geometry=reconstruct(shape,axis['origin'],axis['direction'],axis['segment'],config['fixed_poses'],recon.widths,recon.contact_indices)
    discrepancy=[]
    for i,d in enumerate(axis['widths']):
        recorded=merged([[r['lower'],r['upper']] for r in d['intervals']])
        generated=geometry['feasible'][i]
        symmetric=difference(recorded,generated)+difference(generated,recorded)
        error=sum(b-a for a,b in symmetric);require(error<1e-8,'Independent interval reconstruction differs')
        discrepancy.append(error)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(16,9.8));grid=fig.add_gridspec(2,3,left=.055,right=.965,bottom=.19,top=.835,height_ratios=[1.25,1.],wspace=.3,hspace=.42)
    blue,green,gold,red='#377ca6','#36876b','#cf8c25','#b95668'
    ax=fig.add_subplot(grid[0,0]);Rf=recon.Rf;center=np.asarray(recon.fixed['position'])
    atoms=recon.atoms;radii=recon.radii
    worlds=[(atoms@rotation(p['orientation']).T+p['position']-center)@Rf for p in config['fixed_poses']]
    worlds.append((atoms@rotation(row['pose']['orientation']).T+row['pose']['position']-center)@Rf)
    points=np.concatenate(worlds);low=points.min(axis=0)-radii.max()-3;high=points.max(axis=0)+radii.max()+3
    px=np.arange(low[0],high[0]+.3,.3);py=np.arange(low[1],high[1]+.3,.3)
    for i,(world,color) in enumerate(zip(worlds,[blue,green,gold])):
        mask=projected_mask(world,radii,(0,1),px,py)
        ax.contourf(px,py,mask.astype(float),levels=[.5,1.5],colors=[color],alpha=.26,zorder=i+1)
        ax.contour(px,py,mask.astype(float),levels=[.5],colors=[color],linewidths=.8,zorder=i+1)
    moving_center=(np.asarray(row['pose']['position'])-center)@Rf
    line0=(np.asarray(axis['origin']['position'])-center)@Rf
    d=np.asarray(axis['direction'])@Rf
    ends=np.array([line0+t*d for t in axis['segment']])
    ax.plot(ends[:,0],ends[:,1],color='#333',linewidth=2.2,zorder=8)
    ax.plot(moving_center[0],moving_center[1],'o',color=gold,markersize=5,zorder=9)
    ax.set_aspect('equal');ax.set_xlabel('anchor-frame x (Å)');ax.set_ylabel('y (Å)');ax.set_title('A  Recorded protein neighborhood',loc='left',weight='bold')
    ax.legend(handles=[Patch(fc=blue,alpha=.4,label='Fixed 0'),Patch(fc=green,alpha=.4,label='Fixed 1'),Patch(fc=gold,alpha=.4,label='Moving')],loc='upper left',fontsize=8,frameon=False,ncol=3)
    ax.text(0,-.24,'Atom-sphere silhouettes; projection overlap\ndoes not determine a 3D clash.',transform=ax.transAxes,fontsize=9,color='#555')
    ax=fig.add_subplot(grid[0,1:]);pos=ax.get_position();ax.set_position([.49,pos.y0,.475,pos.height]);segment=axis['segment'];contact=geometry['contacts'];width=recon.widths[wi]
    rows=[('R4 ∩ center capture',[segment],'#b4b9c1'),('Any hard-core overlap',geometry['core'],red),
        (f'Contact fixed 0: gap < {width:g} Å',contact[recon.contact_indices[0]][wi],blue),
        (f'Contact fixed 1: gap < {width:g} Å',contact[recon.contact_indices[1]][wi],green),
        ('Both contacts, all cores excluded',geometry['feasible'][wi],gold)]
    for k,(label,intervals,color) in enumerate(rows):
        y=len(rows)-k-1;ax.hlines(y,*segment,color='#eceef0',linewidth=10,zorder=0)
        for lo,hi in intervals:ax.plot([lo,hi],[y,y],color=color,linewidth=10,solid_capstyle='butt')
    ax.set_yticks(range(len(rows)),[r[0] for r in rows][::-1],fontsize=9);ax.set_ylim(-.65,len(rows)-.35)
    ax.set_xlim(segment);ax.set_xlabel('Raw chart translation s = tₓ (Å)');ax.set_title('B  Exact feasible intervals on this fixed line',loc='left',weight='bold')
    ax.grid(axis='x',alpha=.2);ax.spines['left'].set_visible(False)
    ax=fig.add_subplot(grid[1,0]);ax.axis('off')
    ax.text(0,1.,'C  Retain five coordinates',fontsize=12,weight='bold',va='top')
    ax.text(0,.79,r'$x=(s,\,v),\quad v=(t_y,t_z,a_1,a_2,a_3)$',fontsize=13,va='top')
    ax.text(0,.60,'Draw the existing correlated Gaussian.\nKeep v fixed; only replace s.\nOrientation stays fixed on the line.',fontsize=9.7,linespacing=1.3,va='top')
    ax.text(0,.27,r'$s\mid v,k\sim\mathcal{N}(m_k(v),\sigma_k^2)$',fontsize=13,va='top')
    ax.text(0,.09,f'Illustrated component {component}: m = {m:.5f} Å\nσ = {s:.5f} Å; interval mass Z = {mass:.5g}',fontsize=9.6,linespacing=1.3,va='top')
    ax=fig.add_subplot(grid[1,1:]);pos=ax.get_position();ax.set_position([.49,pos.y0,.475,pos.height]);span=segment[1]-segment[0];xx=np.linspace(*segment,4000);base=norm.pdf(xx,m,s)
    ax.plot(xx,base,color='#5a6572',linewidth=1.5,label='Original conditional Normal')
    for index,(lo,hi) in enumerate(geometry['feasible'][wi]):
        sx=np.linspace(lo,hi,180);yy=norm.pdf(sx,m,s)/mass
        ax.fill_between(sx,0,yy,color=gold,alpha=.25)
        ax.plot(sx,yy,color=gold,linewidth=1.8,label='Normal restricted to feasible intervals' if index==0 else None)
    ax.set_xlim(segment);ax.set_ylabel('Conditional density (Å⁻¹)');ax.set_xlabel('Raw chart translation s = tₓ (Å)')
    ax.set_title('D  Select intervals by Gaussian mass, then invert within them',loc='left',weight='bold')
    ax.legend(frameon=False,fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.15)
    ax.text(.02,.94,'Interval probabilities (left to right):\n'+', '.join(f'{100*p:.4f}%' for p in interval_probabilities),transform=ax.transAxes,ha='left',va='top',fontsize=9,color='#76531d')
    fig.suptitle('Exact line conditioning: preserve the outer marginal, concentrate the translation',x=.055,y=.975,ha='left',fontsize=17,weight='bold')
    kind='recorded successful conditional draw' if actual else 'conditional reconstructed for a recorded line; no extra draw'
    fig.text(.055,.940,f'{root.name} · {row["kind"]} id {row["id"]} · {kind}',fontsize=10.5,color='#444')
    fig.text(.055,.907,f'{len(geometry["feasible"][wi])} disjoint positive-length interval(s) · independent atom reconstruction agrees within {max(discrepancy):.2g} Å of symmetric-difference length',fontsize=10.5)
    fig.text(.055,.148,'Empty intervals or Z ≤ 10⁻¹²: retain the original s. Never redraw the five retained coordinates.',fontsize=11,weight='bold')
    fig.text(.055,.115,r'Full density: $q = 0.50\,U_{R4}+0.25\,G+0.25\,H$;  H sums every component and width, including each fallback branch.',fontsize=11)
    fig.text(.055,.079,'The selected width is only this illustration. Production widths are 0.02, 0.1, 0.5 Å with equal probabilities; the fixed passive pilot uses raw x only.',fontsize=9.5)
    fig.text(.055,.048,'Gaussian fallback tails remain normalized on all R6. Physical weights use the complete J·W/q with unchanged R4/capture/hard indicators.',fontsize=9.5)
    fig.text(.055,.023,'Proposal-only geometry and density diagnostic; no new depletants, native labels, physical mass estimate or assembly conclusion.',fontsize=9.3,color='#555')
    out.mkdir(parents=True)
    for ext in ('png','svg','pdf'):fig.savefig(out/('contact-line-guide.'+ext),dpi=190)
    plt.close(fig)
    data=dict(scope='Scientific visualization of recorded proposal geometry; no new sampling',source_row=row,
        axis=axis['axis'],width_index=wi,component=component,actual_selected_draw=bool(actual),
        conditional_mean=m,conditional_sigma=s,conditional_mass=mass,conditional_interval_probabilities=interval_probabilities,independent_geometry=geometry,
        symmetric_difference_lengths_A=discrepancy,projection='xy in fixed region anchor body frame',
        selection='Prefer most disconnected positive feasible intervals, then an actually selected successful conditioning branch.')
    (out/'figure-data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    archive=out/'provenance';archive.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('analyze_contact_line_audit.py'),a.audit]:shutil.copy2(p,archive/p.name)
    sources=[root/'manifest.json',root/'summary.json',root/'samples.jsonl',root/'probes.jsonl',a.audit,*[root/'provenance'/n for n in ['region.json','importance-guide.json','config.json','shape.json']]]
    reconstruction_receipt=dict(complete=True,scope='One recorded protein line independently reconstructed from atom-sphere quadratics; no proposal, depletant or physical-weight draws.',
        source_kind=row['kind'],source_id=row['id'],axis=axis['axis'],raw_line_origin=axis['origin'],raw_line_direction=axis['direction'],segment=axis['segment'],
        widths_A=recon.widths,contact_neighbor_indices=recon.contact_indices,geometry=geometry,
        recorded_intervals=[d['intervals'] for d in axis['widths']],symmetric_difference_lengths_A=discrepancy,
        maximum_symmetric_difference_length_A=max(discrepancy),input_sha256={str(p):sha(p) for p in sources},
        reconstruction_code_sha256=sha(Path(__file__)),limitation='Agreement concerns interval volumes on this line; endpoint closure is separately checked by the interval-algebra unit tests.')
    (out/'interval-reconstruction.json').write_text(json.dumps(reconstruction_receipt,indent=2,allow_nan=False)+'\n')
    files=[p for p in out.rglob('*') if p.is_file()]
    (out/'manifest.json').write_text(json.dumps(dict(complete=True,input_sha256={str(p):sha(p) for p in sources},output_sha256={str(p.relative_to(out)):sha(p) for p in files}),indent=2)+'\n')
    print(out/'contact-line-guide.png')


if __name__=='__main__':main()
