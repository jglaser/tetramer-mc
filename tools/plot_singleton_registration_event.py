#!/usr/bin/env python3
"""Render a recorded native-registration event, without any simulation or classifier run.

Orthographic ray casting draws the front surface of the actual atom-sphere union.
Contact atom outlines are projections and may include occluded surface atoms.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def matrix(pose):return Rotation.from_quat(np.asarray(pose['orientation'])[[1,2,3,0]]).as_matrix()


def union_image(points,radii,colors,axes,x,y):
    """Exact frontmost sphere intersection at each orthographic pixel center."""
    depth_axis=({0,1,2}-set(axes)).pop()
    depth=np.full((len(y),len(x)),-np.inf)
    rgb=np.ones((len(y),len(x),3))
    light=np.array([-.30,.45,.84]);light/=np.linalg.norm(light)
    for center,radius,color in zip(points,radii,colors):
        c=center[list(axes)]
        lo=np.maximum(0,[np.searchsorted(x,c[0]-radius),np.searchsorted(y,c[1]-radius)])
        hi=np.minimum([len(x),len(y)],[np.searchsorted(x,c[0]+radius)+1,np.searchsorted(y,c[1]+radius)+1])
        if np.any(lo>=hi):continue
        dx=x[lo[0]:hi[0]][None,:]-c[0];dy=y[lo[1]:hi[1]][:,None]-c[1]
        rr=radius*radius-dx*dx-dy*dy;inside=rr>=0
        dz=np.sqrt(np.maximum(rr,0));front=center[depth_axis]+dz
        sl=np.s_[lo[1]:hi[1],lo[0]:hi[0]]
        update=inside&(front>depth[sl])
        intensity=.49+.51*np.maximum(0,(dx*light[0]+dy*light[1]+dz*light[2])/radius)
        shaded=intensity[...,None]*np.asarray(to_rgb(color))[None,None,:]
        depth[sl][update]=front[update];rgb[sl][update]=shaded[update]
    return rgb,np.isfinite(depth)


def projection_mask(points,radii,axes,x,y):
    mask=np.zeros((len(y),len(x)),bool)
    for p,r in zip(points[:,axes],radii):
        lo=np.maximum(0,[np.searchsorted(x,p[0]-r),np.searchsorted(y,p[1]-r)])
        hi=np.minimum([len(x),len(y)],[np.searchsorted(x,p[0]+r)+1,np.searchsorted(y,p[1]+r)+1])
        if np.any(lo>=hi):continue
        mask[lo[1]:hi[1],lo[0]:hi[0]]|=(x[lo[0]:hi[0]][None,:]-p[0])**2+(y[lo[1]:hi[1]][:,None]-p[1])**2<=r*r
    return mask


def contacts(anchor,mobile,radii,rd,patch_ids):
    candidates=cKDTree(anchor).query_ball_tree(cKDTree(mobile),2*radii.max()+2*rd+1e-8)
    left,right,pairs=[],[],[];minimum=float('inf')
    for i,js in enumerate(candidates):
        if not js:continue
        js=np.asarray(js);gap=np.linalg.norm(mobile[js]-anchor[i],axis=1)-radii[i]-radii[js]
        minimum=min(minimum,float(gap.min()))
        for j in js[gap<=2*rd]:
            left.append(i);right.append(int(j));pairs.append((patch_ids[i],patch_ids[j]))
    require(minimum>=-1e-8,'Rendered pair has a hard clash')
    return dict(anchor_atoms=sorted(set(left)),mobile_atoms=sorted(set(right)),patch_pairs=sorted(set(pairs)),minimum_atom_gap_A=minimum)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--patch-map',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();pilot=a.pilot.resolve();out=a.out.resolve()
    require(not out.exists(),'Refusing to overwrite existing figure artifacts')
    protocol=read(pilot/'protocol.json');analysis=read(pilot/'analysis.json')
    require(read(pilot/'status.json')['complete'] and analysis['complete'],'Completed pilot and native analysis required')
    require(analysis['protocol_sha256']==sha(pilot/'protocol.json'),'Changed protocol')
    for name,digest in read(pilot/'freeze.json').items():require(sha(pilot/name)==digest,'Changed frozen input: '+name)
    job=next(j for j in protocol['jobs'] if j['id']=='fused-p2')
    config_path=Path(job['config']);config=read(config_path);directory=Path(job['directory'])
    require(config['boundary']['kind']=='spherical','This recorded-event renderer expects the frozen spherical control')
    summary=read(directory/'summary.json');events_path=directory/'events.jsonl'
    require(summary['events_sha256']==sha(events_path),'Changed event record')
    rows=[r for r in map(json.loads,events_path.read_text().splitlines()) if r['phase']==11 and r['kind']=='cluster_event']
    state=copy.deepcopy(config['initial_poses']);record=None
    for row in rows:
        moving=row['members'][0];require(state[moving]==row['old_poses'][0],'Event replay mismatch')
        if row['event']==3:record=row;break
        state[moving]=row['retained_poses'][0]
    require(record and record['members']==[213] and record['accepted'] and record['hard_valid'],'Wrong target event')
    require(record['proposed_poses']==record['retained_poses'],'Accepted pose mismatch')
    q=record['proposal'];require(q['raw_trial_count']==8 and q['target_label']['kind']=='single' and q['target_label']['branch']==5,'Wrong target chart')
    require(q['oligomer']['fused_components']==0 and all(not r['hard_valid'] for r in q['raw_trials'][:7]) and q['raw_trials'][7]['hard_valid'],'Wrong retry history')
    native=next(p for p in analysis['populations'] if p['id']==job['id'])['native']['events']
    observed=next(e for e in native if e['phase']==11 and e['event_time']==record['event_time'] and e['member']==213)
    require(observed['gained']==[[70,213,7]] and observed['lost']==[],'Native observation changed')
    patch_analysis_path=pilot/'patch-analysis/patch-changes.json';patch_analysis=read(patch_analysis_path)
    patch_event=next(e for e in patch_analysis['accepted_events'] if e['job']==job['id'] and e['phase']==11 and e['event']==3)
    require(patch_event['old_partners']==patch_event['new_partners']==[70] and patch_event['jaccard_distance']==1,'Patch evidence differs')
    shape_path=pilot/'provenance/shape.json';shape=read(shape_path);patch_path=a.patch_map.resolve();patch=read(patch_path)
    require(sha(shape_path)==protocol['shape_sha256']==patch['shape_sha256'],'Shape/patch identity mismatch')
    require(sha(patch_path)==patch_analysis['patch_map_sha256'],'Patch map changed')
    atoms=np.asarray([p['center'] for p in shape['atoms']]);radii=np.asarray([p['radius'] for p in shape['atoms']])
    anchor=state[70];anchor_matrix=matrix(anchor);origin=np.asarray(anchor['position'])
    def anchor_frame(pose):
        return (atoms@matrix(pose).T+np.asarray(pose['position'])-origin)@anchor_matrix
    require(np.max(abs(anchor_frame(anchor)-atoms))<1e-10,'Anchor-frame roundtrip failed')
    world={'anchor':atoms,'before':anchor_frame(record['old_poses'][0]),'after':anchor_frame(record['retained_poses'][0])}
    rd=protocol['depletant_radius'];patch_ids=patch['atom_patch_ids']
    contact={k:contacts(atoms,world[k],radii,rd,patch_ids) for k in ('before','after')}
    for k,field in [('before','old_patch_tokens'),('after','new_patch_tokens')]:
        require(contact[k]['patch_pairs']==[tuple(e[2:]) for e in patch_event[field]],'Exact atom contact footprints differ from observer tokens')
    motifs_path=pilot/'provenance/native/inputs/native-pair-motifs.json'
    motif=next(m for m in read(motifs_path)['motifs'] if m['id']==7)
    reference_pose=dict(position=motif['relative_position'],orientation=motif['relative_orientation'])
    world['reference']=atoms@matrix(reference_pose).T+np.asarray(reference_pose['position'])
    # Display axes are fixed to the anchor body, not fitted separately at either endpoint.
    axes_list=[(0,1),(0,2)];resolution=.16
    allpoints=np.concatenate(list(world.values()));low=(allpoints-radii.max()).min(axis=0)-5;high=(allpoints+radii.max()).max(axis=0)+5
    span=float(max(high-low));middle=(low+high)/2;low=middle-span/2;high=middle+span/2
    colors=dict(anchor='#90bccd',mobile='#e6c697',old='#b23b72',new='#137c63',reference='#3b4048')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(13.5,12))
    grid=fig.add_gridspec(2,2,left=.08,right=.96,bottom=.235,top=.82,wspace=.08,hspace=.42)
    raster=[]
    for row,axes in enumerate(axes_list):
        x=np.arange(low[axes[0]],high[axes[0]]+resolution,resolution);y=np.arange(low[axes[1]],high[axes[1]]+resolution,resolution)
        for col,endpoint in enumerate(('before','after')):
            ax=fig.add_subplot(grid[row,col]);sites=contact[endpoint];accent=colors['old' if endpoint=='before' else 'new']
            cs=np.array([colors['anchor']]*len(atoms)+[colors['mobile']]*len(atoms))
            cs[sites['anchor_atoms']]=accent;cs[len(atoms)+np.asarray(sites['mobile_atoms'])]=accent
            points=np.concatenate((atoms,world[endpoint]));rr=np.tile(radii,2)
            rgb,mask=union_image(points,rr,cs,axes,x,y)
            ax.imshow(rgb,origin='lower',extent=[x[0]-.5*resolution,x[-1]+.5*resolution,y[0]-.5*resolution,y[-1]+.5*resolution],interpolation='nearest',zorder=2)
            footprint=np.concatenate((atoms[sites['anchor_atoms']],world[endpoint][sites['mobile_atoms']]))
            footprint_radii=np.concatenate((radii[sites['anchor_atoms']],radii[sites['mobile_atoms']]))
            fm=projection_mask(footprint,footprint_radii,axes,x,y)
            ax.contour(x,y,fm.astype(float),levels=[.5],colors=[accent],linewidths=.7,zorder=4)
            if endpoint=='after':
                reference_mask=projection_mask(world['reference'],radii,axes,x,y)
                ax.contour(x,y,reference_mask.astype(float),levels=[.5],colors=[colors['reference']],linestyles='dashed',linewidths=.9,zorder=5)
            center=(np.asarray(record['old_poses' if endpoint=='before' else 'retained_poses'][0]['position'])-origin)@anchor_matrix
            ax.plot(0,0,'+',color='#165064',markersize=7,zorder=6)
            ax.plot(center[axes[0]],center[axes[1]],'+',color='#85623a',markersize=7,zorder=6)
            ax.set_xlim(low[axes[0]],high[axes[0]]);ax.set_ylim(low[axes[1]],high[axes[1]])
            ax.set_aspect('equal');ax.set_xlabel('anchor-frame x (Å)');ax.set_ylabel(('y' if axes[1]==1 else 'z')+' (Å)')
            ax.set_title(('Before: competing registry' if col==0 else 'After: native motif 7 observed')+(' · x–y' if row==0 else ' · x–z'),loc='left',fontsize=11.3,pad=9)
            raster.append(dict(endpoint=endpoint,axes=list(axes),resolution_A=resolution,grid_shape=list(mask.shape),image_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    fig.suptitle('A large change of native registration with the same contact partner',x=.08,y=.975,ha='left',fontsize=17,weight='bold')
    fig.text(.08,.941,'Recorded accepted endpoint · moving tetramer 213, anchor tetramer 70 · identical body frame and scale in all panels',fontsize=11,color='#444')
    fig.text(.08,.913,f'Center displacement {patch_event["displacement_A"]:.2f} Å    |    rotation {patch_event["rotation_degrees"]:.2f}°    |    14 old patch-pair labels → 14 new labels; none shared',fontsize=10.5)
    fig.legend(handles=[Patch(fc=colors['anchor'],label='Anchor 70'),Patch(fc=colors['mobile'],label='Moving 213'),
        Line2D([0],[0],color=colors['old'],lw=2,label='Old contact footprint'),Line2D([0],[0],color=colors['new'],lw=2,label='New contact footprint'),
        Line2D([0],[0],color=colors['reference'],ls='--',label='Native motif 7 reference')],
        loc='upper left',bbox_to_anchor=(.074,.895),ncol=5,frameon=False,fontsize=9.3,columnspacing=1.3)
    fig.text(.08,.182,'How this endpoint was generated',fontsize=12,weight='bold')
    fig.text(.08,.154,'Cap 8 · ordinary single chart 5 · zero fused components in this retained context · first hard-valid candidate on trial 8',fontsize=10.4)
    fig.text(.08,.126,f'Full-guide log ratio {q["map_log_reverse_forward"]:+.3f}  +  pool ratio {q["anchor_log_reverse_forward"]:+.3f}  +  Poisson log factor {record["gate"]["log_weight"]:+.3f}  →  acceptance probability 1',fontsize=10.4)
    fig.text(.08,.088,'Hard atom-sphere unions are rendered at 0.16 Å per pixel. Contact outlines mark projected atoms whose exclusion shells overlap (gap ≤ 2r_d).',fontsize=9,color='#444')
    fig.text(.08,.065,'Native reference and fixed 32-patch labels are passive diagnostics. One accepted reset event establishes accessibility; it does not establish assembly or mixing.',fontsize=9,color='#444')
    fig.text(.08,.037,f'Cap-8 fused-p2 · reset phase 11, event 3 · r_d = {rd:g} Å, z = {protocol["depletant_activity"]:g} Å⁻³ · wall and other tetramers omitted only from display',fontsize=9,color='#444')
    out.mkdir(parents=True)
    for ext in ('png','svg','pdf'):fig.savefig(out/('native-registration-event.'+ext),dpi=200)
    plt.close(fig)
    provenance=out/'provenance';provenance.mkdir();shutil.copy2(__file__,provenance/Path(__file__).name)
    for source,name in [(shape_path,'shape.json'),(patch_path,'patch-map.json'),(motifs_path,'native-pair-motifs.json')]:shutil.copy2(source,provenance/name)
    data=dict(scope='Passive visualization of one recorded accepted event; no physical sampling or native classifier execution.',
        event=record,anchor_world_pose=anchor,patch_event=patch_event,native_observation=observed,
        contact_atoms=contact,native_reference_pose=reference_pose,
        hard_atom_count_per_body=len(atoms),anchor_frame_atoms={k:v.tolist() for k,v in world.items()},
        raster=raster,axis_limits=dict(low=low.tolist(),high=high.tolist()),
        rendering='Orthographic nearest front sphere surface per pixel, using all hard atom radii; no voxel smoothing or outer envelope.',
        native_overlay='Projected outline of exact motif 7 in the frozen reference catalogue; passive only.')
    (out/'figure-data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    sources=[pilot/'protocol.json',pilot/'analysis.json',events_path,directory/'summary.json',config_path,patch_analysis_path,shape_path,patch_path,motifs_path,Path(__file__)]
    files=[p for p in out.rglob('*') if p.is_file()]
    (out/'manifest.json').write_text(json.dumps(dict(complete=True,input_sha256={str(p):sha(p) for p in sources},
        output_sha256={str(p.relative_to(out)):sha(p) for p in files}),indent=2)+'\n')
    print(out/'native-registration-event.png')


if __name__=='__main__':main()
