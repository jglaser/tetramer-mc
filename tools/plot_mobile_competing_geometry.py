#!/usr/bin/env python3
"""Plot the exact mobile trap and an alternative native pose in the AB gauge.

Projected atom-sphere unions are visualization only, never collision tests.
No trajectory, contact audit, or physical sampler is run by this script.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'A':'#1687a0', 'B':'#b47e18', 'mobile':'#923e8a'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world_atoms(atoms, pose):
    matrix = Rotation.from_quat(np.asarray(pose['orientation'])[[1,2,3,0]]).as_matrix()
    return atoms@matrix.T+np.asarray(pose['position'])


def projected_union(points, radii, axes, x, y):
    """Rasterize the exact projected disks at regular sample points."""
    mask = np.zeros((len(y),len(x)), dtype=bool)
    for center,radius in zip(points[:,axes],radii):
        ix0,ix1 = np.searchsorted(x,[center[0]-radius,center[0]+radius],side='left')
        iy0,iy1 = np.searchsorted(y,[center[1]-radius,center[1]+radius],side='left')
        ix0,ix1 = max(0,ix0),min(len(x),ix1+1)
        iy0,iy1 = max(0,iy0),min(len(y),iy1+1)
        if ix0<ix1 and iy0<iy1:
            mask[iy0:iy1,ix0:ix1] |= ((x[ix0:ix1]-center[0])**2)[None,:]+((y[iy0:iy1]-center[1])**2)[:,None] <= radius**2
    return mask


def draw_union(ax, mask, x, y, color, alpha, linestyle='solid', hatch=None, zorder=2):
    ax.contourf(x,y,mask.astype(float),levels=[.5,1.5],colors=[to_rgba(color,alpha)],
        hatches=[hatch],zorder=zorder)
    ax.contour(x,y,mask.astype(float),levels=[.5],colors=[color],linewidths=1.05,
        linestyles=linestyle,zorder=zorder+.1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,default=ROOT/'runs/mobile-competing-geometry-20260921')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    geometry,out=args.geometry.resolve(),args.out.resolve()
    assert not out.exists(), 'Refusing to overwrite an existing figure artifact'
    analysis,snapshot=read(geometry/'analysis.json'),read(geometry/'fixed-snapshot.json')
    assert analysis['schema']=='mobile-competing-geometry-v1' and snapshot['schema']=='mobile-fixed-snapshot-geometry-v1'
    final=analysis['snapshots'][-1]
    assert final['sweep']==snapshot['sweep']==2000
    assert snapshot['ab_gauge']==final['ab_gauge']
    assert snapshot['original_global']['poses']==final['original_poses']
    verified={}
    for name,record in analysis['source'].items():
        assert sha(record['path'])==record['sha256'],('Source changed',name)
        verified[name]=record
    shape=read(analysis['source']['shape']['path'])
    motifs=read(analysis['source']['motifs']['path'])['motifs']
    atoms=np.asarray([a['center'] for a in shape['atoms']])
    radii=np.asarray([a['radius'] for a in shape['atoms']])
    poses={'A':snapshot['physical_fixed_neighbors'][0], 'B':snapshot['physical_fixed_neighbors'][1],
        'native':snapshot['native_candidate'], 'observed':snapshot['trapped_pose']}
    world={name:world_atoms(atoms,value) for name,value in poses.items()}
    # Classification labels are the frozen catalogue's classes at these motifs.
    assert {c['family'] for c in motifs[4]['member_contacts']}=={'C3'}
    assert {c['family'] for c in motifs[12]['member_contacts']}=={'C4'}
    assert {c['family'] for c in motifs[3]['member_contacts']}=={'C1','C2'}
    assert final['registered_motifs'][0]['bodies']==[1,2]
    assert len(final['registered_motifs'])==1
    native_gaps=final['original_native_alternative']['gaps_to_body1_and_2_A']
    observed_gaps=final['pair_gaps_A']
    assert min(native_gaps)>0 and min(observed_gaps.values())>0
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white','hatch.linewidth':.4})
    fig=plt.figure(figsize=(14,9.3))
    grid=fig.add_gridspec(2,2,height_ratios=[4.5,2.0],hspace=.37,wspace=.16,
        left=.065,right=.965,bottom=.065,top=.80)
    projections=[(0,1),(0,2)]
    resolution=.20
    raster_records=[]
    for panel,axes in enumerate(projections):
        ax=fig.add_subplot(grid[0,panel])
        projected=np.concatenate([v[:,axes] for v in world.values()])
        low=np.min(projected-radii.max(),axis=0)-7
        high=np.max(projected+radii.max(),axis=0)+7
        x=np.arange(low[0],high[0]+resolution,resolution)
        y=np.arange(low[1],high[1]+resolution,resolution)
        for name,color,alpha,line,hatch,order in (
                ('native',COLORS['mobile'],.09,'dashed','///',1),
                ('A',COLORS['A'],.35,'solid',None,2),
                ('B',COLORS['B'],.35,'solid',None,3),
                ('observed',COLORS['mobile'],.35,'solid',None,4)):
            mask=projected_union(world[name],radii,axes,x,y)
            draw_union(ax,mask,x,y,color,alpha,line,hatch,order)
            raster_records.append(dict(projection=list(axes),body=name,grid_shape=list(mask.shape),
                projected_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        ax.add_patch(Circle((0,0),snapshot['ab_gauge']['original_capture_radius_A'],
            fill=False,edgecolor='#45454f',linestyle=(0,(5,3)),linewidth=1.5,zorder=8))
        for name in ('native','A','B','observed'):
            center=np.asarray(poses[name]['position'])[list(axes)]
            color=COLORS.get(name,COLORS['mobile'])
            ax.plot(*center,'o',markersize=3.5,color=color,zorder=10)
        # Labels point to centers and distinguish two mutually exclusive body0 positions.
        offsets=({'native':(-23,27),'A':(20,20),'B':(-15,-30),'observed':(1,28)} if panel==0 else
                 {'native':(-23,30),'A':(13,21),'B':(-10,-30),'observed':(0,35)})
        for name in ('native','A','B','observed'):
            center=np.asarray(poses[name]['position'])[list(axes)]
            label={'native':'0: native alternative','A':'Fixed A = body 2','B':'Fixed B = body 1','observed':'0: observed competitor'}[name]
            ax.annotate(label,center,xytext=offsets[name],textcoords='offset points',ha='center',va='center',
                fontsize=9.4,color=COLORS.get(name,COLORS['mobile']),zorder=12,
                bbox=dict(boxstyle='round,pad=.22',fc='white',ec='none',alpha=.93),
                arrowprops=dict(arrowstyle='-',color='#777777',lw=.6))
        ax.annotate('Old center capture: 18 Å',xy=(-12.73,-12.73),xytext=(-25,-31),
            textcoords='data',fontsize=9,ha='center',color='#44444e',zorder=12,
            arrowprops=dict(arrowstyle='-',color='#44444e',lw=.7),
            bbox=dict(boxstyle='round,pad=.15',fc='white',ec='none',alpha=.9))
        ax.set_aspect('equal')
        ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1])
        ax.set_xlabel('x in the original AB frame (Å)')
        ax.set_ylabel(('y' if panel==0 else 'z')+' (Å)')
        ax.set_title(('a  Top projection (x–y)' if panel==0 else 'b  Side projection (x–z)'),loc='left',fontsize=12,pad=10)
        ax.grid(alpha=.12,lw=.6)
    left=fig.add_subplot(grid[1,0]);left.axis('off')
    left.text(0,1.04,'Same fixed scaffold; two alternative positions of body 0',fontsize=11.5,weight='bold',va='top')
    rows=[['Gap to A (Å)',f'{native_gaps[1]:.3f}  ·  C3',f"{observed_gaps['0-2']:.3f}  ·  unregistered"],
          ['Gap to B (Å)',f'{native_gaps[0]:.3f}  ·  C4',f"{observed_gaps['0-1']:.3f}  ·  beyond depletion range"],
          ['Distance from old center', '0 Å',f"{snapshot['ab_gauge']['trapped_distance_from_original_capture_A']:.2f} Å (3D)"]]
    table=left.table(cellText=rows,colLabels=['Exact 3D geometry','Native alternative','Observed competitor'],
        cellLoc='left',colLoc='left',bbox=[0,.24,1,.61],colWidths=[.34,.27,.39])
    table.auto_set_font_size(False);table.set_fontsize(8.7)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#dedee4');cell.set_linewidth(.55);cell.PAD=.045
        if r==0:cell.set_facecolor('#f0eef5');cell.set_text_props(weight='bold')
        elif c==2:cell.set_facecolor('#fbf5fa')
    left.text(0,.09,'A–B is a native C1/C2 scaffold. Minimum atom-surface gap: 0.0015 Å.',fontsize=9,color='#45454f')
    right=fig.add_subplot(grid[1,1]);right.axis('off')
    right.text(0,1.04,'What this snapshot establishes',fontsize=11.5,weight='bold',va='top')
    lines=[
        'The observed competitor is outside the earlier 18 Å capture domain.',
        'Earlier AB regional weights did not include this attachment site.',
        'Both alternative poses satisfy exact atomic hard-core checks.',
        'Their statistical weights remain to be measured on this scaffold.'
    ]
    for k,line in enumerate(lines):right.text(0,.80-.17*k,line,fontsize=9.5,va='top')
    fig.suptitle('A persistent competing attachment lies outside the old capture domain',
        x=.065,y=.975,ha='left',fontsize=17,weight='bold')
    fig.text(.065,.934,'Dispersed r01 · correlated transport c = 0.9 · final sweep 2000 · depletant radius 1.5 Å · activity 0.035 Å⁻³',fontsize=10.4,color='#45454f')
    handles=[Patch(fc=to_rgba(COLORS['A'],.35),ec=COLORS['A'],label='Fixed A'),
        Patch(fc=to_rgba(COLORS['B'],.35),ec=COLORS['B'],label='Fixed B'),
        Patch(fc=to_rgba(COLORS['mobile'],.35),ec=COLORS['mobile'],label='Observed body 0'),
        Patch(fc=to_rgba(COLORS['mobile'],.09),ec=COLORS['mobile'],ls='--',hatch='///',label='Alternative body 0 (not simultaneous)'),
        Line2D([0],[0],color='#45454f',ls='--',label='Old center constraint')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.060,.907),ncol=5,frameon=False,fontsize=9,
        handlelength=2,columnspacing=1.5)
    fig.text(.065,.845,'Silhouettes are projections of the atom-sphere unions; overlap in a projection does not imply a 3D clash.',fontsize=10,color='#45454f')
    fig.text(.065,.018,'Body labels are preserved. A common proper isometry aligns body 2 with A; the observed B displacement is retained (member RMS 0.084 Å). '
        'The finite spherical wall is omitted from these local views.',fontsize=8.8,color='#45454f')
    out.mkdir(parents=True)
    for suffix in ('png','svg','pdf'):
        fig.savefig(out/f'mobile-competing-geometry.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    provenance=out/'provenance';provenance.mkdir()
    shutil.copy2(__file__,provenance/'plotter.py')
    for name in ('analysis.json','fixed-snapshot.json'):
        shutil.copy2(geometry/name,provenance/name)
    plot_data=dict(poses=poses,body_atom_count=len(atoms),projection_axes=[list(v) for v in projections],
        raster_spacing_A=resolution,raster_records=raster_records,
        native_gaps_to_B_A_Angstrom=native_gaps,observed_pair_gaps_A=observed_gaps,
        native_classes={'A':'C3','B':'C4'},fixed_pair_classes=['C1','C2'],
        capture_center=[0,0,0],capture_radius_A=snapshot['ab_gauge']['original_capture_radius_A'],
        observed_distance_from_capture_center_A=snapshot['ab_gauge']['trapped_distance_from_original_capture_A'],
        scope='Alternative poses of one body, not a four-body state. Raster silhouettes use the hard atom radii, not expanded depletant radii. '
              '2D projections do not determine clashes; table values come from the previously verified exact 3D geometry diagnostic. '
              'The wall is omitted visually, not removed from the physical target. No inference of statistical weight or equilibrium is made.')
    (out/'plot-data.json').write_text(json.dumps(plot_data,indent=2,allow_nan=False)+'\n')
    report=dict(complete=True,geometry_analysis=dict(path=str(geometry/'analysis.json'),sha256=sha(geometry/'analysis.json')),
        exact_snapshot=dict(path=str(geometry/'fixed-snapshot.json'),sha256=sha(geometry/'fixed-snapshot.json')),
        source=verified,plotter_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        provenance_files={p.name:sha(p) for p in provenance.iterdir()},scope=plot_data['scope'])
    (out/'provenance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(out/'mobile-competing-geometry.png')


if __name__=='__main__':
    main()
