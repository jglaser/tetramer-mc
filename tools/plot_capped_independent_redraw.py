#!/usr/bin/env python3
"""Render the proposed finite-cap independent hard-conditioned proposal.

Scientific schematic only: this does not execute or validate a physical move.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(16,9));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    fig.patch.set_facecolor('#fafbfc')
    blue='#245f96';orange='#b95b1f';green='#207966';ink='#182b3b';gray='#65717b'
    def box(x,y,w,h,face='#edf3f9',edge=blue):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.012',
                            facecolor=face,edgecolor=edge,linewidth=1.6);ax.add_patch(patch);return patch
    def txt(x,y,s,size=12,weight='normal',color=ink,ha='left',va='top'):
        ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,
                linespacing=1.5,transform=ax.transAxes)
    def arrow(a,b,color=blue,style='-|>',connection='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=16,
                     linewidth=1.8,color=color,connectionstyle=connection))
    txt(.02,.98,'Hard-conditioned independent redraw',24,'bold')
    txt(.02,.925,'Amortize one frozen catalogue over several cheap hard-geometry trials; preserve the physical acceptance rule.',12)
    box(.735,.918,.245,.058,face='#fff1dd',edge=orange)
    txt(.8575,.947,'Implementation in progress',11,'bold',orange,'center','center')

    box(.02,.545,.215,.305)
    txt(.035,.833,'1  Retain context C',15,'bold',blue)
    txt(.035,.785,'Moving subset and internal shape\nOrdered anchor pool A\nAll spectator poses + wall\nFrozen, normalized mixture G',11)
    txt(.035,.622,'Same context at both endpoints.\nBuild once; do not redraw A.',10,'bold')
    arrow((.238,.714),(.28,.714))

    box(.285,.605,.232,.245,face='#f2f6fa')
    txt(.30,.831,'2  Fresh raw trial',15,'bold',blue)
    txt(.30,.779,r'$i\sim w_i,\quad \xi\sim N(0,I_6)$',16)
    txt(.30,.734,r'$g=\mathrm{decode}_i(\xi)$',16)
    txt(.30,.678,'Redraw across the full mixture:\nnew label AND new latent.',11,'bold')
    arrow((.52,.714),(.555,.714))

    diamond=Polygon([[.60,.815],[.654,.714],[.60,.613],[.546,.714]],
                    closed=True,facecolor='#e6f3ee',edgecolor=green,linewidth=1.7)
    ax.add_patch(diamond)
    txt(.60,.738,'Hard-valid?',12,'bold',green,'center','center')
    txt(.60,.695,r'$H_C(g)=1$',13,'normal',green,'center','center')
    arrow((.657,.714),(.72,.714),green)
    txt(.688,.757,'FIRST\nvalid',10,'bold',green,'center','center')

    box(.725,.545,.255,.305,face='#eef7f2',edge=green)
    txt(.74,.833,'3  One physical decision',15,'bold',green)
    txt(.74,.785,'Evaluate complete G at x and y.\nAdd the full ordered-pool ratio.\nApply the existing exact\nmany-body Poisson gate\n(and frozen bias, if enabled).',11)
    txt(.74,.60,'Accept y, or retain x.\nSTOP after this decision.',11,'bold')

    # Failed raw trials return to a completely fresh independent draw.
    ax.plot([.60,.60,.402],[.61,.455,.455],color=orange,linewidth=1.8)
    arrow((.402,.455),(.402,.601),orange)
    txt(.503,.511,'Clash + budget remains',11,'bold',orange,'center','center')
    txt(.503,.478,'increment k; try again',10,'normal',orange,'center','center')
    # All-failed cap is a retained-state outcome, not a dropped attempt.
    arrow((.59,.455),(.687,.366),orange)
    box(.53,.285,.45,.105,face='#fff5eb',edge=orange)
    txt(.755,.34,'Cap exhausted OR MH rejected: retain x',13,'bold',orange,'center','center')
    arrow((.86,.541),(.86,.396),green)
    txt(.872,.468,'If rejected:\nretain x; no retry',10,'bold',green,'left','center')
    box(.02,.302,.452,.106,face='#f1f3f5',edge=gray)
    txt(.035,.385,'Control parameters',12,'bold',gray)
    txt(.035,.348,'T fixed in advance (or a function of C only).\nKeep all nulls, raw failures and rejected endpoint states.',10)

    box(.02,.092,.96,.158,face='white',edge=blue)
    txt(.038,.235,'Why the unknown hard-valid probability cancels',14,'bold',blue)
    txt(.039,.181,r'$Z_C=\int G_C(g)H_C(g)\,dg,\quad B_T(Z_C)=\sum_{k=0}^{T-1}(1-Z_C)^k$',15)
    txt(.039,.127,r'$q_T(y\mid x,C)=B_T(Z_C)G_C(y)H_C(y)$',15)
    txt(.54,.177,r'$\frac{q_T(x\mid y,C)}{q_T(y\mid x,C)}=\frac{G_C(x)}{G_C(y)}$',19)
    txt(.775,.178,'Same B at x and y.\nPool ratio is separate.',10)
    txt(.02,.06,'Uniform defense is chosen once and keeps its existing symmetric kernel; only the learned branch is shown.',10,color=gray)
    txt(.02,.029,'Independent redraw is essential: retrying one Gaussian, a correlated transport, or after MH rejection has a different law.',10,'bold',orange)
    fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02)
    for ext in ('png','svg','pdf'):
        fig.savefig(args.out/('capped-independent-redraw.'+ext),dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__':
    main()
