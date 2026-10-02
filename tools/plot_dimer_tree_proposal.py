#!/usr/bin/env python3
"""Coordinate schematic, with illustrative rigid bodies and no simulation data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
COLORS={'A':'#858d9a','R':'#207eac','C':'#cf7637'}
INK='#243348'


def body(ax, xy, angle, label):
    """Four disks denote one rigid object, not an actual protein geometry."""
    offsets=np.array([[-.17,-.14],[.15,-.12],[-.13,.16],[.16,.15]])
    theta=np.deg2rad(angle)
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    for center,radius in zip(offsets@rotation.T+xy,[.16,.145,.14,.115]):
        ax.add_patch(Circle(center,radius,facecolor=COLORS[label],edgecolor='white',lw=.7,zorder=5))
    ax.text(*xy,label,ha='center',va='center',color='white',weight='bold',fontsize=12,zorder=6)


def edge(ax, a, b, label, bend=0., label_xy=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
        color=INK,lw=1.5,shrinkA=23,shrinkB=23,connectionstyle=f'arc3,rad={bend}'))
    midpoint=(np.array(a)+b)/2
    label_xy=label_xy or (midpoint[0],midpoint[1]+.14)
    ax.text(*label_xy,label,ha='center',color=INK,fontsize=12,zorder=10,
        bbox=dict(facecolor='white',edgecolor='none',pad=2))


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK})
    fig=plt.figure(figsize=(13.2,6.1),facecolor='white')
    panels=[fig.add_axes([.035,.29,.28,.51]),fig.add_axes([.35,.29,.30,.51]),
            fig.add_axes([.685,.29,.28,.51])]
    for ax in panels:
        ax.set(xlim=(0,3.2),ylim=(0,3.1),aspect='equal');ax.axis('off')
    fig.text(.04,.935,'Docking and internal contact update in one trial',fontsize=20,weight='bold')
    fig.text(.04,.888,'Two relative poses share a moving root; each tetramer stays rigid.',fontsize=12)
    old,maps,new=panels
    old.set_title('1  Encode the old tree',loc='left',fontsize=13,weight='bold',pad=12)
    a=(.5,.7);r=(1.55,1.45);c=(2.6,2.05)
    edge(old,a,r,r'$h_0=A^{-1}g_R$',-.04,label_xy=(.65,1.46))
    edge(old,r,c,r'$h_1=g_R^{-1}g_C$',-.08,label_xy=(1.83,2.44))
    body(old,a,0,'A');body(old,r,15,'R');body(old,c,-35,'C')
    old.text(.5,.17,'fixed spectator',ha='center',fontsize=10)
    old.text(2.14,2.92,'selected dimer',ha='center',fontsize=11)
    maps.set_title('2  Map both old edges',loc='left',fontsize=13,weight='bold',pad=12)
    for y,color,title,equation in [(2.17,COLORS['R'],'docking coordinates',r'$h_0\ \longleftrightarrow\ h_0^{\prime}$'),
                                   (1.04,COLORS['C'],'internal coordinates',r'$h_1\ \longleftrightarrow\ h_1^{\prime}$')]:
        maps.add_patch(FancyBboxPatch((.15,y-.35),2.9,.79,boxstyle='round,pad=.06',
            facecolor=color+'14',edgecolor=color,lw=1.3))
        maps.text(1.6,y+.21,title,ha='center',fontsize=11,color=color,weight='bold')
        maps.text(1.6,y-.10,equation,ha='center',fontsize=19)
    maps.text(1.6,.29,'Retain both inverse traces.\nAdd both proposal corrections.',ha='center',va='center',fontsize=11)
    new.set_title('3  Decode one joint endpoint',loc='left',fontsize=13,weight='bold',pad=12)
    r1=(1.4,1.56);c1=(1.43,2.5)
    edge(new,a,r1,r'$h_0^{\prime}$',.04,label_xy=(.65,1.38))
    edge(new,r1,c1,r'$h_1^{\prime}$',-.15,label_xy=(1.90,2.04))
    body(new,a,0,'A');body(new,r1,50,'R');body(new,c1,75,'C')
    new.text(2.43,1.03,r'$g_R^{\prime}=Ah_0^{\prime}$'+'\n'+r'$g_C^{\prime}=g_R^{\prime}h_1^{\prime}$',
        ha='center',va='center',fontsize=12)
    new.text(.5,.17,'same spectator',ha='center',fontsize=10)
    fig.add_artist(FancyBboxPatch((.04,.13),.92,.105,boxstyle='round,pad=.008',
        transform=fig.transFigure,facecolor='#f5f0e5',edgecolor='#d7c39b'))
    fig.text(.055,.201,'One endpoint acceptance decision',fontsize=12,weight='bold',va='center')
    fig.text(.055,.162,'Hard geometry + complete old/new excluded unions + both proposal corrections',fontsize=12,va='center')
    fig.text(.04,.064,'Schematic coordinates and bodies. The bath correction includes changes of internal and spectator overlap.',fontsize=10)
    out=ROOT/'docs/assets';out.mkdir(exist_ok=True)
    for suffix in ('svg','png'):
        fig.savefig(out/('dimer-tree-proposal.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
