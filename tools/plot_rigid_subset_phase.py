#!/usr/bin/env python3
"""Scientific schematic for reversible fixed-duration rigid-subset moves.

This diagram is explanatory geometry, not a protein trajectory or a performance
result. Regenerate with the project's scientific Python environment.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Ellipse
import numpy as np

NAVY = "#172D43"
BLUE = "#267EB5"
TEAL = "#168B86"
GOLD = "#D49A35"
RED = "#C45A48"
GRAY = "#8795A2"
LIGHT = "#F4F7FA"


def body(ax, x, y, color, size=.29, angle=0, label=None, alpha=1):
    """A four-lobed icon represents ONE rigid tetramer."""
    cs, sn = np.cos(angle), np.sin(angle)
    for dx, dy in [(-.13,-.1),(.13,-.1),(-.1,.14),(.13,.15)]:
        ax.add_patch(Circle((x+cs*dx-sn*dy,y+sn*dx+cs*dy),
                            size*.66, facecolor=color, edgecolor="white",
                            lw=1.0, alpha=alpha, zorder=4))
    if label:
        ax.text(x,y,label,color="white",ha="center",va="center",
                fontsize=10,fontweight="bold",zorder=5)


def arrow(ax, start, end, color=NAVY, style="-|>", lw=1.8, rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,
                                mutation_scale=15,lw=lw,color=color,
                                connectionstyle=f"arc3,rad={rad}"))


def link(ax, x1,y1,x2,y2, color=GRAY, ls="-", lw=2):
    ax.plot([x1,x2],[y1,y2],color=color,lw=lw,ls=ls,zorder=2)


def panel(ax, letter, title):
    ax.set(xlim=(0,10),ylim=(0,6.4),aspect="equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((.05,.05),9.9,6.3,
                               boxstyle="round,pad=0.01,rounding_size=.14",
                               facecolor="white",edgecolor="#D8E0E8",lw=1.2))
    ax.text(.38,5.85,letter,color=TEAL,fontsize=15,fontweight="bold",va="center")
    ax.text(.9,5.85,title,color=NAVY,fontsize=14,fontweight="bold",va="center")


def make_figure(out: Path):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "text.color":NAVY,"mathtext.fontset":"dejavusans",
                         "svg.fonttype":"none"})
    fig,axs=plt.subplots(2,2,figsize=(16,11.4))
    fig.patch.set_facecolor(LIGHT)
    fig.subplots_adjust(left=.025,right=.975,top=.875,bottom=.1,wspace=.025,hspace=.035)
    fig.text(.045,.963,"Rigid oligomer proposals without a recruitment penalty",
             fontsize=23,fontweight="bold",color=NAVY)
    fig.text(.045,.924,"Internal geometry determines the subset rate; algorithmic residence time corrects the changing number of choices.",
             fontsize=12.5,color="#435669")

    ax=axs[0,0];panel(ax,"A","Select a subset, even inside a larger aggregate")
    # left: disconnected dimer and seed; right: docked dimer, subset unchanged
    for offset,docked in [(0,False),(5.1,True)]:
        a=(1.15+offset,3.9 if not docked else 3.5)
        b=(2.0+offset,3.9 if not docked else 3.5)
        seed=[(2.0+offset,2.55),(2.85+offset,2.55),(2.43+offset,1.8)]
        link(ax,*a,*b,color=BLUE,lw=3)
        for u,v in [(seed[0],seed[1]),(seed[1],seed[2]),(seed[2],seed[0])]:link(ax,*u,*v,color=GOLD)
        if docked: link(ax,*b,*seed[0],color=RED,lw=2.5)
        for x,y in seed:body(ax,x,y,GOLD)
        body(ax,*a,BLUE,label="a");body(ax,*b,BLUE,label="b")
        ax.add_patch(Ellipse(((a[0]+b[0])/2,a[1]),1.8,.92,fill=False,
                             edgecolor=BLUE,lw=1.8,ls=(0,(4,3)),zorder=7))
        ax.text(1.9+offset,4.8,"Before: X" if not docked else "After: Y",
                ha="center",fontweight="bold",fontsize=12)
    arrow(ax,(3.5,3.4),(5.5,3.4))
    ax.text(4.5,3.7,"dock",ha="center",fontsize=10)
    arrow(ax,(5.5,2.95),(3.5,2.95),color=GRAY)
    ax.text(4.5,2.52,"detach",ha="center",fontsize=10,color=GRAY)
    ax.text(8.9,3.03,"new external\ncontact",ha="center",fontsize=9.5,color=RED)
    ax.text(5,.88,r"$\lambda_S(X)=\kappa_{|S|}\,\mathbf{1}\{S\ \mathrm{internally\ connected}\}=\lambda_S(Y)$",
            ha="center",fontsize=13)
    ax.text(5,.37,"The internal a–b geometry is unchanged; joining the seed does not erase the reverse move.",
            ha="center",fontsize=9.8)

    ax=axs[0,1];panel(ax,"B","Map one handle; carry the selected oligomer")
    a=(1.55,3.7); b=(2.5,3.7); c=(2.25,4.5)
    aa=(7.45,3.5); bb=(7.45,4.45); cc=(6.65,4.2)
    for pts,col in [([a,b,c],BLUE),([aa,bb,cc],TEAL)]:
        for i,j in [(0,1),(1,2)]:link(ax,*pts[i],*pts[j],color=col,lw=2.5)
        for p,l in zip(pts,["h","b","c"]):body(ax,*p,col,label=l)
    arrow(ax,(3.25,4.05),(5.95,4.05),rad=-.1)
    ax.text(4.8,4.65,"six-dimensional pose map",ha="center",fontsize=10.5)
    ax.text(4.8,3.4,r"$H=g'_h g_h^{-1}$",ha="center",fontsize=16)
    ax.text(5,2.35,r"$g'_i=H g_i\quad (i\in S)$",ha="center",fontsize=18)
    ax.text(5,1.55,"One proper rotation + translation; every internal pose is preserved.",ha="center",fontsize=11)
    ax.text(5,.9,"Use the one-handle proposal ratio and one chart Jacobian.",ha="center",fontsize=11,fontweight="bold")
    ax.text(5,.38,"Check all carried bodies against the wall and stationary bodies; hard failure is a self-loop.",
            ha="center",fontsize=9.8)

    ax=axs[1,0];panel(ax,"C","Keep the full many-body depletion gate")
    # Illustrative exclusion disks; artist does not calculate their exact areas.
    for x,y,r in [(3.5,3.1,1.12),(4.5,3.3,1.1),(4.1,2.25,1.0)]:
        ax.add_patch(Circle((x,y),r,facecolor="#BCC5CE",edgecolor="#758492",alpha=.65,lw=1.1))
    for x,y in [(2.25,3.5),(2.25,2.55)]:
        ax.add_patch(Circle((x,y),.83,facecolor=BLUE,edgecolor=BLUE,alpha=.26,lw=1.7))
        body(ax,x,y,BLUE,size=.2)
    for x,y in [(5.35,3.5),(5.35,2.55)]:
        ax.add_patch(Circle((x,y),.83,facecolor=TEAL,edgecolor=TEAL,alpha=.26,lw=1.7))
        body(ax,x,y,TEAL,size=.2)
    ax.text(1.15,4.65,r"old union $B_X$",color=BLUE,fontsize=11)
    ax.text(5.5,4.65,r"new union $B_Y$",color=TEAL,fontsize=11)
    ax.text(3.9,1.07,r"stationary union $E$",color="#596977",ha="center",fontsize=11)
    ax.text(8.03,3.68,"Union overlap,\nnot a sum of\npair contacts",ha="center",fontsize=11,fontweight="bold")
    ax.text(8.03,2.35,"Internal excluded\nvolume cancels\nunder rigid motion",ha="center",fontsize=10.5)
    ax.text(5,.53,r"$\log[\pi(Y)/\pi(X)]=z\,[\,|B_Y\cap E|-|B_X\cap E|\,]$",ha="center",fontsize=14)

    ax=axs[1,1];panel(ax,"D","Stop at fixed algorithmic time, not event count")
    ax.text(5,5.05,r"$\Lambda(X)=\sum_S\lambda_S(X),\quad \Delta t\sim\mathrm{Exp}(\Lambda(X))$",ha="center",fontsize=14)
    ax.text(5,4.39,r"Choose $S$ with probability $\lambda_S(X)/\Lambda(X)$; apply its corrected move.",ha="center",fontsize=10.8)
    y=2.8;start=.7;end=9.4;T=8.0
    ax.plot([start,end],[y,y],color=NAVY,lw=2)
    ax.plot([T,T],[1.62,3.7],color=RED,lw=1.8,ls=(0,(4,3)))
    ax.text(T,3.86,r"fixed duration $\tau$",color=RED,ha="center",fontsize=11)
    eventxs=[2.0,4.05,5.1,9.0]
    for x,col,lab,yy in [(2.0,TEAL,"accepted",2.05),(4.05,RED,"rejected",1.45),(5.1,TEAL,"accepted",2.05),(9.0,GRAY,"beyond end:\nno proposal",1.85)]:
        ax.plot(x,y,"o",ms=9,color=col,zorder=5)
        ax.text(x,yy,lab,color=col,ha="center",fontsize=10)
    ax.plot(start,y,"|",ms=16,color=NAVY);ax.text(start,2.1,"0",ha="center",fontsize=11)
    for a,b in [(start,2.0),(2.0,4.05),(4.05,5.1),(5.1,9.0)]:
        arrow(ax,(a,3.2),(b,3.2),color=GRAY,style="<->",lw=1)
    ax.text(5,.74,"Rejected events consume time. The final waiting interval contributes residence time.",ha="center",fontsize=10.2,fontweight="bold")
    ax.text(5,.33,"Saving fixed-time endpoints preserves physical weights; saving only events generally does not.",ha="center",fontsize=9.8)

    fig.text(.047,.055,"Scope",color=TEAL,fontweight="bold",fontsize=11)
    fig.text(.095,.055,"Schematic only • connected dimers/trimers • exact physical and map corrections remain • no speedup claimed",fontsize=10.7)
    fig.text(.047,.027,"Why the clock matters: configurations with more eligible subsets generate more events, but receive proportionally shorter holding times.",fontsize=10.7,color="#435669")
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/"rigid-subset-phase.svg",facecolor=fig.get_facecolor())
    fig.savefig(out/"rigid-subset-phase.png",dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("runs/rigid-subset-schematic-20260925"))
    make_figure(parser.parse_args().out)


if __name__=="__main__":
    main()
