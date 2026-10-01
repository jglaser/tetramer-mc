#!/usr/bin/env python3
"""Scientific schematic of the two-distance guide using exact sphere geometry.

This is a constructed geometric illustration, not a simulation result. It uses
the independently validated reference functions without altering the frozen
audit implementation. No proposals or Poisson clouds are sampled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon,Rectangle
import numpy as np
from contact_distance_reference import polygon,transverse_basis,circle_coordinates,wrapped_cauchy_log_density


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():raise ValueError('Figure output must be new')
    fixed_radius=1.;moving_radius=1.;width=.5;D=4.2;r1,r2=2.2,2.25
    c1=np.array([-D/2,0.,0.]);c2=np.array([D/2,0.,0.]);d=2.
    vertices,area=polygon(d,d,D,width);h,rho=circle_coordinates(r1,r2,D);center=c1[0]+h
    phi_mean=2.1;conditional_tangent_sd=.15;gamma=conditional_tangent_sd/rho;b=.9
    spectator_radius=.8;spectator_offset=1.9;exclusion_radius=moving_radius+spectator_radius
    cosine=(spectator_offset**2+rho**2-exclusion_radius**2)/(2*spectator_offset*rho)
    forbidden_half_angle=math.acos(cosine)
    phi=np.linspace(0,2*math.pi,2000)
    density=(1-b)/(2*math.pi)+b*np.exp(wrapped_cauchy_log_density(phi,phi_mean,gamma))
    assert abs((r1*r1-r2*r2+D*D)/(2*D)-h)<1e-12
    assert abs(math.hypot(h,rho)-r1)<1e-12 and abs(math.hypot(D-h,rho)-r2)<1e-12
    assert r1>=2 and r2>=2 and 0<forbidden_half_angle<math.pi
    blue,green,gold,red,gray='#307da3','#2c8767','#c78625','#b94d66','#65717b'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(14,10.1));grid=fig.add_gridspec(2,2,left=.068,right=.97,top=.865,bottom=.225,wspace=.25,hspace=.53)
    fig.suptitle('Two distances choose a contact circle; the azimuth chooses a pose',x=.068,y=.972,ha='left',fontsize=17,weight='bold')
    fig.text(.068,.937,'Constructed sphere geometry, not simulation data. Fixed orientation throughout; all three translation coordinates may change.',fontsize=10.5,color='#444')
    fig.text(.068,.909,'Shown width w = 0.5 Å; moving and selected fixed spheres have radius 1 Å. Two selected atom pairs do not certify the whole particle.',fontsize=10.5)

    ax=fig.add_subplot(grid[0,0]);ax.set_aspect('equal')
    for x,color,label in [(c1[0],blue,'Fixed 1'),(c2[0],green,'Fixed 2')]:
        ax.add_patch(Circle((x,0),fixed_radius,facecolor=color,edgecolor=color,alpha=.23,lw=1.5))
        ax.plot(x,0,'o',color=color,ms=4);ax.text(x,-1.23,label,ha='center',color=color,fontsize=10)
    ax.add_patch(Circle((center,rho),moving_radius,facecolor=gold,edgecolor=gold,alpha=.25,lw=1.5))
    ax.plot(center,rho,'o',color=gold,ms=5);ax.text(center,1.94,'Moving sphere',ha='center',color='#785018')
    for x,lab,dx in [(c1[0],r'$r_1=2.20$ Å',-.14),(c2[0],r'$r_2=2.25$ Å',.13)]:
        ax.plot([x,center],[0,rho],color=gray,lw=1.2)
        ax.text((x+center)/2+dx,rho/2+.17,lab,ha='center',fontsize=10)
    ax.plot([c1[0],c2[0]],[0,0],ls=':',color=gray,lw=1.)
    ax.plot([center,center],[-rho,rho],color=gold,lw=2.,ls='--')
    ax.annotate('',xy=(c1[0],-1.72),xytext=(c2[0],-1.72),arrowprops=dict(arrowstyle='<->',color=gray))
    ax.text(0,-1.92,'D = 4.20 Å',ha='center',color=gray)
    ax.set_xlim(-3.5,3.5);ax.set_ylim(-2.07,2.21);ax.axis('off')
    ax.set_title('A  Two selected surface gaps at fixed orientation',loc='left',weight='bold',fontsize=11.5,pad=12)
    ax.text(.02,-.05,r'$c_j=f_j-Rb_j$; here $b_j=0$.  Gaps: $r_j-2$ Å.',transform=ax.transAxes,fontsize=10)

    ax=fig.add_subplot(grid[0,1]);ax.set_aspect('equal')
    ax.add_patch(Rectangle((d,d),width,width,facecolor=red,alpha=.12,edgecolor='none'))
    ax.add_patch(Polygon(vertices,closed=True,facecolor=gold,edgecolor=gold,alpha=.35,lw=2))
    ax.plot([2,2.2],[2.2,2],color=red,ls='--',lw=1.5)
    ax.plot(r1,r2,'o',color='#7a521b',ms=6);ax.text(r1+.025,r2+.025,'Shown circle',fontsize=9,color='#795118')
    ax.text(2.06,2.39,'Uniform in this polygon',fontsize=10,color='#795118')
    ax.text(2.014,2.025,'Impossible\ntriangle',fontsize=8.5,color=red)
    ax.set_xlim(1.965,2.53);ax.set_ylim(1.965,2.53);ax.set_xticks([2.,2.1,2.2,2.3,2.4,2.5]);ax.set_yticks([2.,2.1,2.2,2.3,2.4,2.5])
    ax.set_xlabel(r'$r_1$ (Å)');ax.set_ylabel(r'$r_2$ (Å)');ax.grid(alpha=.15)
    ax.set_title('B  Sample the two radii in a clipped rectangle',loc='left',weight='bold',fontsize=11.5,pad=12)
    ax.text(1.13,.97,r'$2\leq r_j\leq2+w$'+'\n\n'+r'$|r_1-r_2|\leq D$'+'\n'+r'$r_1+r_2\geq D$'+'\n\n'+f'Area A = {area:.2f} Å²',transform=ax.transAxes,va='top',fontsize=10)
    # Move the compact square left inside its wide panel, reserving text space.
    pos=ax.get_position();ax.set_position([.56,pos.y0,.225,pos.height])

    ax=fig.add_subplot(grid[1,0]);ax.set_aspect('equal')
    ax.add_patch(Circle((spectator_offset,0),exclusion_radius,fc=red,alpha=.08,ec=red,ls='--',lw=1.4))
    ax.add_patch(Circle((spectator_offset,0),spectator_radius,fc=gray,ec=gray,alpha=.25,lw=1.))
    ax.plot(rho*np.cos(phi),rho*np.sin(phi),color=gold,lw=3)
    bad=np.linspace(-forbidden_half_angle,forbidden_half_angle,300)
    ax.plot(rho*np.cos(bad),rho*np.sin(bad),color=red,lw=4)
    ax.plot(0,0,'+',color=gray,ms=8);ax.plot([0,rho*math.cos(phi_mean)],[0,rho*math.sin(phi_mean)],color=blue,lw=1.)
    point=np.array([rho*math.cos(phi_mean),rho*math.sin(phi_mean)]);ax.plot(*point,'o',color=blue,ms=5)
    ax.plot(rho,0,'x',color=red,ms=7,mew=1.8)
    ax.annotate('Same selected radii;\na spectator can cause overlap',xy=(rho,0),xytext=(-1.8,-1.12),fontsize=9,color=red,arrowprops=dict(arrowstyle='->',color=red,connectionstyle='arc3,rad=.1'))
    ax.text(spectator_offset,.04,'Unselected\nspectator',ha='center',va='center',fontsize=9,color='#47515a')
    ax.text(-1.75,1.07,f'Circle radius ρ = {rho:.3f} Å',fontsize=10,color='#795118')
    ax.text(1.95,-1.13,'Dashed: moving-center\nexclusion, radius 1.8 Å',fontsize=8.5,ha='center',color=red)
    ax.set_xlim(-1.9,3.05);ax.set_ylim(-1.36,1.35);ax.set_xlabel('Transverse coordinate ξ₁ (Å)');ax.set_ylabel('ξ₂ (Å)')
    ax.set_title('C  The intersection circle does not exclude other cores',loc='left',weight='bold',fontsize=11.5,pad=12)
    ax.spines['left'].set_color('#aaa');ax.spines['bottom'].set_color('#aaa')

    ax=fig.add_subplot(grid[1,1])
    ax.axvspan(0,forbidden_half_angle,color=red,alpha=.1);ax.axvspan(2*math.pi-forbidden_half_angle,2*math.pi,color=red,alpha=.1)
    ax.axhline(1/(2*math.pi),color=gray,ls='--',lw=1.5,label='Uniform azimuth')
    ax.plot(phi,density,color=blue,lw=2,label='90% localized + 10% uniform')
    ax.axvline(phi_mean,color=blue,alpha=.25,lw=1.);ax.text(phi_mean+.08,max(density)*.86,r'$\phi_0$',color=blue)
    ax.set_xlim(0,2*math.pi);ax.set_ylim(0,max(density)*1.18);ax.set_xticks([0,math.pi/2,math.pi,3*math.pi/2,2*math.pi],[r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
    ax.set_xlabel('Azimuth φ (radians)');ax.set_ylabel('Conditional density (rad⁻¹)');ax.legend(frameon=False,loc='upper right',fontsize=8.7)
    ax.set_title('D  Localize the azimuth while retaining its full support',loc='left',weight='bold',fontsize=11.5,pad=12)
    ax.grid(axis='y',alpha=.15);ax.text(.015,.70,'Red sectors:\noverlap with spectator',transform=ax.transAxes,fontsize=9,color=red)
    ax.text(.96,.54,r'$f(t\mid a,k,w)=\frac{D}{A\,r_1r_2}\,p_\phi(\phi)$'+'\n'+f'Shown γ = {gamma:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=11.5,linespacing=1.8)

    fig.text(.068,.159,r'Complete proposal: $q=0.50\,U_{R4}+0.25\,G+0.25\,H$;  $H$ sums every Gaussian component and width.',fontsize=12,weight='bold')
    fig.text(.068,.124,'Retain the Gaussian angular marginal; widths 0.02, 0.1 and 0.5 Å are equal. Coincident centers or empty/tiny polygons retain the original draw.',fontsize=10)
    fig.text(.068,.089,'Evaluate the full mixture density, including fallback branches. Whole-union, R4 or capture failure gives zero J·W/q; never retry an invalid endpoint.',fontsize=10)
    fig.text(.068,.054,'Localized azimuth uses the component’s conditional translation mean and covariance. It is a normalized guide, not the physical depletion weight.',fontsize=9.7,color='#444')
    fig.text(.068,.022,'This figure illustrates geometry and probability only: no protein sampling, Poisson clouds, equilibrium weights or assembly outcome.',fontsize=9.7,color='#555')
    out.mkdir(parents=True)
    for ext in ('png','svg','pdf'):fig.savefig(out/('contact-distance-guide.'+ext),dpi=190)
    plt.close(fig)
    data=dict(scope='Constructed exact sphere-geometry illustration; no simulation data or random draws.',fixed_centers=[c1.tolist(),c2.tolist()],
        fixed_radius=fixed_radius,moving_radius=moving_radius,width_A=width,radii=[r1,r2],polygon_vertices=vertices.tolist(),polygon_area_A2=area,
        circle_center=[center,0.,0.],circle_radius_A=rho,azimuth=dict(mode=phi_mean,gamma=gamma,localized_probability=b,conditional_tangent_sd_A=conditional_tangent_sd),
        spectator=dict(transverse_center=[spectator_offset,0.],radius_A=spectator_radius,moving_center_exclusion_radius_A=exclusion_radius,forbidden_half_angle=forbidden_half_angle),
        basis='ξ1 and ξ2 span the plane perpendicular to the selected fixed-sphere center axis.')
    (out/'figure-data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    archive=out/'provenance';archive.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('contact_distance_reference.py')]:shutil.copy2(p,archive/p.name)
    files=[p for p in out.rglob('*') if p.is_file()]
    (out/'manifest.json').write_text(json.dumps(dict(complete=True,output_sha256={str(p.relative_to(out)):sha(p) for p in files}),indent=2)+'\n')
    print(out/'contact-distance-guide.png')


if __name__=='__main__':main()
