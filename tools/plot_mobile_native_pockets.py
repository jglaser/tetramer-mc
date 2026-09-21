#!/usr/bin/env python3
"""Shared projections of two native sites on one unchanged observed scaffold."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from diagnose_mobile_native_remainder import AtomicGeometry

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def centers(p,members):
    r=Rotation.from_quat(np.array(p['orientation'])[[1,2,3,0]]).as_matrix()
    return np.array([m['position']for m in members])@r.T+p['position']

def run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    out=Path(out).resolve()
    if out.exists():raise ValueError('Use a fresh figure directory')
    diagnostic=ROOT/'runs/mobile-native-remainder-diagnostic-20260921'
    for name,digest in read(diagnostic/'freeze.json').items():
        if sha(diagnostic/name)!=digest:raise ValueError('Saved geometric diagnostic changed')
    data=read(diagnostic/'analysis.json');key=data['global_top16'][0]
    selected=next(r for r in data['selected_rows']if all(r[k]==key[k]for k in ('arm','population','draw')))
    if selected['site_key']!='body2:7|body1:4':raise ValueError('Unexpected illustrative site')
    if not selected['independent_atomic_geometry']['hard_valid_at_zero_tolerance']:raise ValueError('Invalid saved illustration')
    source=ROOT/'runs/mobile-native-region-definition-20260921/inputs'
    config=read(source/'physical-config.json');shape=read(source/'tetramer-shape.json')
    snapshot=read(source/'fixed-snapshot.json');wall=dict(center=snapshot['original_global']['wall_center'],radius=snapshot['original_global']['wall_radius'])
    fixed=config['fixed_poses'];old=data['original_native_pose'];new=selected['pose'];members=shape['rigid_members']
    motifs={m['id']:m for m in read(source/'native-pair-motifs.json')['motifs']}
    audit=AtomicGeometry(shape,[old],wall)
    pair_gap=audit.gap(audit.placed(new),0)
    body_centers=[centers(p,members)for p in fixed]+[centers(old,members),centers(new,members)]
    out.mkdir();shutil.copy2(__file__,out/'plot.py')
    colors=['#24769a','#786491','#d77c21'];fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    titles=['Original site: motifs 4 / 10\n3 prescribed monomer contacts','Alternative site: motifs 7 / 4\n4 prescribed monomer contacts']
    for column,(moving,motif_ids)in enumerate(zip(body_centers[2:],((4,10),(7,4)))):
        bodies=body_centers[:2]+[moving]
        for row,dims in enumerate(((0,1),(0,2))):
            ax=axes[row,column]
            for index,points in enumerate(bodies):
                p=points[:,dims]
                try:
                    hull=ConvexHull(p);outline=p[hull.vertices]
                    ax.fill(outline[:,0],outline[:,1],color=colors[index],alpha=.1)
                except Exception:pass
                ax.scatter(p[:,0],p[:,1],s=100,c=colors[index],edgecolors='white',linewidths=.7,zorder=4,
                    label=('fixed A','fixed B','mobile tetramer')[index])
                for i,v in enumerate(p):ax.annotate(str(i),v,ha='center',va='center',fontsize=7,color='white',zorder=5)
            for anchor,motif_id in enumerate(motif_ids):
                for contact in motifs[motif_id]['member_contacts']:
                    a=body_centers[anchor][contact['member_i'],list(dims)];b=moving[contact['member_j'],list(dims)]
                    ax.plot([a[0],b[0]],[a[1],b[1]],color='#b2463a',linewidth=2,alpha=.85,zorder=2)
            allpoints=np.concatenate(body_centers)
            ax.set_xlim(allpoints[:,dims[0]].min()-12,allpoints[:,dims[0]].max()+12)
            ax.set_ylim(allpoints[:,dims[1]].min()-12,allpoints[:,dims[1]].max()+12)
            ax.set_aspect('equal');ax.set_xlabel('x (Å)'if row else'');ax.set_ylabel(('y','z')[row]+' (Å)');ax.grid(alpha=.13)
            if row==0:ax.set_title(titles[column],fontsize=11)
    axes[0,0].legend(loc='best',fontsize=8)
    fig.suptitle('Two native placements around the same fixed tetramers',fontsize=14)
    fig.text(.5,.025,'Points: monomer centers; shading: tetramer grouping; red: catalogue-prescribed contacts.\n'
             'Matched axes within each view. Alternative uses a clash-free sampled pose; this is a geometry schematic.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,bottom=.14,top=.85,wspace=.25,hspace=.35)
    fig.savefig(out/'native-pockets.png',dpi=180);fig.savefig(out/'native-pockets.svg');plt.close(fig)
    result=dict(scope='Geometry only, no bath weights or physical sampling. Grouping polygons are not particle surfaces.',
        illustrative_saved_row={k:selected[k]for k in ('arm','population','seed','draw','pose','site_key')},
        original_pose=old,fixed_poses=fixed,prescribed_motif_ids=[[4,10],[7,4]],
        original_alternative_minimum_atomic_surface_gap_A=pair_gap['minimum_surface_gap_A'],
        coexistence_hard_valid=pair_gap['minimum_surface_gap_A']>=0,
        coexistence_scope='Tests these exact two illustrative mobile poses together only; no packing relaxation or weight calculation.',
        source_sha256={str(p):sha(p)for p in [Path(__file__),diagnostic/'analysis.json',source/'physical-config.json',
            source/'tetramer-shape.json',source/'native-pair-motifs.json',source/'fixed-snapshot.json']})
    (out/'geometry.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
